#!/usr/bin/env python3
from __future__ import annotations

# ✅ FIX(A): "python scripts/xxx.py" 실행에서도 scripts.* import 되도록 repo root를 sys.path에 추가
import sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from scripts.feature_spec import read_feature_cols_meta, get_feature_cols


# ✅ FIX: 루트 data/가 아니라 data/labels/ 로
DATA_PATH = Path("data/labels/strategy_raw_data.csv")

TAIL_MODEL_PATH = Path("app/tail_model.pkl")
TAIL_SCALER_PATH = Path("app/tail_scaler.pkl")


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    uniq = np.unique(y_true)
    if len(uniq) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def parse_csv_list(s: str) -> list[str]:
    items = [x.strip() for x in str(s or "").split(",")]
    return [x for x in items if x]


def resolve_feature_cols(args_features: str) -> tuple[list[str], str]:
    """
    Priority:
      1) --features (explicit override)
      2) data/meta/feature_cols.json (SSOT written by build_features.py)
      3) fallback SSOT default (sector disabled)
    """
    override = parse_csv_list(args_features)
    if override:
        return override, "--features"

    cols_meta, _sector_enabled = read_feature_cols_meta()
    if cols_meta:
        return cols_meta, "data/meta/feature_cols.json"

    return get_feature_cols(sector_enabled=False), "feature_spec.py (fallback)"


def ensure_feature_columns_strict(df: pd.DataFrame, feat_cols: list[str], source_hint: str = "") -> None:
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        hint = f" (src={source_hint})" if source_hint else ""
        raise ValueError(
            f"Missing feature columns{hint}: {missing}\n"
            f"-> build_strategy_labels가 SSOT feature_cols로 strategy_raw_data를 만들도록 먼저 맞춰져 있어야 함."
        )


def coerce_features_numeric(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feat_cols:
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train strategy models (tail).")
    ap.add_argument("--profit-target", type=float, required=True)   # 로그/재현용
    ap.add_argument("--max-days", type=int, required=True)          # 로그/재현용
    ap.add_argument("--stop-level", type=float, required=True)      # 로그/재현용
    ap.add_argument("--max-extend-days", type=int, required=True)   # 로그/재현용
    ap.add_argument("--tail-threshold", type=float, default=-0.30, help="라벨 생성 시 사용한 값 기록용")

    ap.add_argument("--data", type=str, default=str(DATA_PATH))
    ap.add_argument("--target-col", type=str, default="Tail")

    # ✅ 기본은 SSOT(meta) 사용. 필요하면 override.
    ap.add_argument("--features", type=str, default="", help="comma-separated feature cols (override SSOT/meta)")

    ap.add_argument("--out-model", type=str, default=str(TAIL_MODEL_PATH))
    ap.add_argument("--out-scaler", type=str, default=str(TAIL_SCALER_PATH))
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=800)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run scripts/build_strategy_labels.py first.")

    df = pd.read_csv(data_path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in {data_path}")

    feat_cols, feat_src = resolve_feature_cols(args.features)
    feat_cols = [c.strip() for c in feat_cols if c.strip()]

    # ✅ STRICT: feature 누락이면 즉시 에러
    ensure_feature_columns_strict(df, feat_cols, source_hint=f"{feat_src}, data={data_path}")

    df = coerce_features_numeric(df, feat_cols)

    df = df.dropna(subset=feat_cols + [args.target_col]).reset_index(drop=True)

    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = pd.to_numeric(df[args.target_col], errors="coerce").fillna(0).astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        raise RuntimeError("Tail target has only one class. tail_threshold/horizon might be off.")

    n = len(df)
    split_idx = int(n * float(args.train_ratio))
    split_idx = max(50, min(split_idx, n - 50))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # n_splits safety
    n_splits = min(int(args.n_splits), 5)
    if split_idx < 200:
        n_splits = min(n_splits, 2)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    base = LogisticRegression(max_iter=int(args.max_iter))

    # ✅ sklearn 버전 호환: estimator= (new) / base_estimator= (old)
    try:
        model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=tscv)
    except TypeError:
        model = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=tscv)

    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:, 1]
    auc = safe_auc(y_test, probs)

    print("=" * 60)
    print("[TRAIN] strategy tail model")
    print("Tail Test ROC-AUC:", round(auc, 6) if auc == auc else "nan")
    print("Base Tail Rate:", round(float(np.mean(y_test)), 6))
    print("Predicted Mean Probability:", round(float(np.mean(probs)), 6))
    print("Rows:", n, "Train:", len(y_train), "Test:", len(y_test))
    print("feature_cols_source:", feat_src)
    print("feature_cols:", feat_cols)
    print("Args:", {
        "profit_target": args.profit_target,
        "max_days": args.max_days,
        "stop_level": args.stop_level,
        "max_extend_days": args.max_extend_days,
        "tail_threshold": args.tail_threshold,
        "data": str(data_path),
    })
    print("=" * 60)

    out_model = Path(args.out_model)
    out_scaler = Path(args.out_scaler)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)
    print(f"[DONE] saved: {out_model} / {out_scaler}")


if __name__ == "__main__":
    main()