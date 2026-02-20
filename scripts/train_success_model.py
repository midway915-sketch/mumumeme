#!/usr/bin/env python3
from __future__ import annotations

# ✅ FIX(A): "python scripts/xxx.py" 실행에서도 scripts.* import 되도록 repo root를 sys.path에 추가
import sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from scripts.feature_spec import read_feature_cols_meta, get_feature_cols


DATA_DIR = Path("data")
LABELS_DIR = DATA_DIR / "labels"
META_DIR = DATA_DIR / "meta"
APP_DIR = Path("app")

# ✅ 파이프라인 기본(너 workflow 기준)
DEFAULT_LABELS_PARQ = LABELS_DIR / "labels_model.parquet"
DEFAULT_LABELS_CSV = LABELS_DIR / "labels_model.csv"

DEFAULT_OUT_MODEL = APP_DIR / "model.pkl"
DEFAULT_OUT_SCALER = APP_DIR / "scaler.pkl"
DEFAULT_REPORT = META_DIR / "train_model_report.json"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


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
            f"-> build_labels/build_strategy_labels 단계가 SSOT feature_cols를 포함하도록 먼저 맞춰져 있어야 함."
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


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train p_success (Success) model with SSOT feature set.")
    ap.add_argument("--labels-parq", type=str, default=str(DEFAULT_LABELS_PARQ))
    ap.add_argument("--labels-csv", type=str, default=str(DEFAULT_LABELS_CSV))

    ap.add_argument("--target-col", type=str, default="Success")
    ap.add_argument("--date-col", type=str, default="Date")

    # ✅ 기본은 SSOT(meta). 필요할 때만 override.
    ap.add_argument("--features", type=str, default="", help="comma-separated feature cols (override SSOT/meta)")

    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=500)

    ap.add_argument("--out-model", type=str, default=str(DEFAULT_OUT_MODEL))
    ap.add_argument("--out-scaler", type=str, default=str(DEFAULT_OUT_SCALER))
    ap.add_argument("--out-report", type=str, default=str(DEFAULT_REPORT))
    args = ap.parse_args()

    labels_parq = Path(args.labels_parq)
    labels_csv = Path(args.labels_csv)

    df = read_table(labels_parq, labels_csv).copy()
    labels_src = str(labels_parq if labels_parq.exists() else labels_csv)

    if args.date_col not in df.columns:
        raise ValueError(f"labels missing date column: {args.date_col} (src={labels_src})")
    if args.target_col not in df.columns:
        raise ValueError(f"labels missing target column: {args.target_col} (src={labels_src})")

    # normalize + sort by time
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=[args.date_col]).sort_values(args.date_col).reset_index(drop=True)

    feat_cols, feat_src = resolve_feature_cols(args.features)
    feat_cols = [c.strip() for c in feat_cols if c.strip()]

    # ✅ STRICT: feature 누락이면 즉시 에러
    ensure_feature_columns_strict(df, feat_cols, source_hint=f"{feat_src}, labels_src={labels_src}")

    df = coerce_features_numeric(df, feat_cols)

    use = df.dropna(subset=feat_cols + [args.target_col]).copy()
    y = pd.to_numeric(use[args.target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    X = use[feat_cols].to_numpy(dtype=float)

    n = len(use)
    if n < 200:
        raise RuntimeError(f"Not enough training rows: {n} (labels_src={labels_src})")

    split_idx = int(n * float(args.train_ratio))
    split_idx = max(50, min(split_idx, n - 50))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base = LogisticRegression(max_iter=int(args.max_iter))
    tscv = TimeSeriesSplit(n_splits=int(args.n_splits))

    # ✅ sklearn 호환: estimator/base_estimator
    try:
        model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=tscv)
    except TypeError:
        model = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=tscv)

    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:, 1]
    auc = safe_auc(y_test, probs)

    APP_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    out_model = Path(args.out_model)
    out_scaler = Path(args.out_scaler)
    out_report = Path(args.out_report)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)

    report = {
        "type": "p_success",
        "labels_src": labels_src,
        "target_col": args.target_col,
        "date_col": args.date_col,
        "feature_cols_source": feat_src,
        "feature_cols": feat_cols,
        "rows_total": int(n),
        "rows_train": int(len(y_train)),
        "rows_test": int(len(y_test)),
        "auc": (round(float(auc), 6) if np.isfinite(auc) else None),
        "base_rate_test": float(np.mean(y_test)) if len(y_test) else None,
        "pred_mean_test": float(np.mean(probs)) if len(probs) else None,
        "out_model": str(out_model),
        "out_scaler": str(out_scaler),
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("[DONE] train_success_model.py (p_success)")
    print("labels_src:", labels_src)
    print("AUC:", report["auc"])
    print("feature_cols_source:", feat_src)
    print("features:", feat_cols)
    print("[DONE] saved model ->", out_model)
    print("[DONE] saved scaler ->", out_scaler)
    print("[DONE] wrote report ->", out_report)
    print("=" * 60)


if __name__ == "__main__":
    main()