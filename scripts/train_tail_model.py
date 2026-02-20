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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from scripts.feature_spec import read_feature_cols_meta, get_feature_cols


APP_DIR = Path("app")
META_DIR = Path("data/meta")
LABELS_DIR = Path("data/labels")


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def parse_csv_list(s: str) -> list[str]:
    if s is None:
        return []
    items = [x.strip() for x in str(s).split(",")]
    return [x for x in items if x]


def resolve_feature_cols(args_features: str) -> tuple[list[str], str]:
    """
    Priority:
      1) --features (explicit override)
      2) data/meta/feature_cols.json (SSOT written by build_features.py)
      3) fallback SSOT default (sector disabled)
    Returns: (feature_cols, source_string)
    """
    override = parse_csv_list(args_features)
    if override:
        return [str(c).strip() for c in override if str(c).strip()], "--features"

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
            f"-> labels_tail 생성 단계(build_strategy_labels/build_tail_labels)가 "
            f"SSOT feature_cols를 포함하도록 먼저 맞춰져 있어야 함."
        )


def coerce_features_numeric(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in feat_cols:
        df[c] = (
            pd.to_numeric(df[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    return df


def _fit_model(X_train, y_train, n_splits: int, max_iter: int):
    base = LogisticRegression(max_iter=max_iter)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    try:
        clf = CalibratedClassifierCV(estimator=base, method="isotonic", cv=tscv)
    except TypeError:
        clf = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=tscv)
    clf.fit(X_train, y_train)
    return clf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    # default: SSOT/meta. override only if you really want.
    ap.add_argument("--features", default="", type=str, help="comma-separated feature cols (override SSOT/meta)")

    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=500, type=int)
    args = ap.parse_args()

    pt_tag = int(round(args.profit_target * 100))
    sl_tag = int(round(abs(args.stop_level) * 100))
    tag = f"pt{pt_tag}_h{args.max_days}_sl{sl_tag}_ex{args.max_extend_days}"

    parq = LABELS_DIR / f"labels_tail_{tag}.parquet"
    csv = LABELS_DIR / f"labels_tail_{tag}.csv"

    if parq.exists() or csv.exists():
        df = read_table(parq, csv).copy()
        src = str(parq if parq.exists() else csv)
    else:
        # fallback only if TailTarget already exists there
        parq2 = LABELS_DIR / "labels_model.parquet"
        csv2 = LABELS_DIR / "labels_model.csv"
        df = read_table(parq2, csv2).copy()
        src = str(parq2 if parq2.exists() else csv2)
        if "TailTarget" not in df.columns:
            raise FileNotFoundError(
                f"Missing tail labels. Provide {parq}/{csv} or include TailTarget in labels_model."
            )

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if "TailTarget" not in df.columns:
        raise ValueError(f"TailTarget column missing (src={src})")

    feat_cols, feat_src = resolve_feature_cols(args.features)
    feat_cols = [str(c).strip() for c in feat_cols if str(c).strip()]

    # STRICT: missing features -> fail fast
    ensure_feature_columns_strict(df, feat_cols, source_hint=f"{feat_src}, labels_src={src}")

    df = coerce_features_numeric(df, feat_cols)

    use = df.dropna(subset=feat_cols + ["TailTarget"]).copy()
    y = pd.to_numeric(use["TailTarget"], errors="coerce").fillna(0).astype(int).to_numpy()
    X = use[feat_cols].to_numpy(dtype=float)

    if len(use) < 200:
        raise RuntimeError(f"Not enough training rows: {len(use)} (labels_src={src})")

    split_idx = int(len(use) * float(args.train_ratio))
    split_idx = max(50, min(split_idx, len(use) - 50))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = _fit_model(X_train_s, y_train, n_splits=int(args.n_splits), max_iter=int(args.max_iter))

    probs = model.predict_proba(X_test_s)[:, 1]
    auc = float("nan")
    if len(np.unique(y_test)) > 1:
        auc = float(roc_auc_score(y_test, probs))

    APP_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    out_model = APP_DIR / "tail_model.pkl"
    out_scaler = APP_DIR / "tail_scaler.pkl"
    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)

    report = {
        "tag": tag,
        "rows_used": int(len(use)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_cols_source": feat_src,
        "feature_cols": feat_cols,
        "auc": (round(auc, 6) if np.isfinite(auc) else None),
        "labels_src": src,
        "paths": {"model": str(out_model), "scaler": str(out_scaler)},
    }
    (META_DIR / f"train_tail_report_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print("[DONE] train_tail_model.py")
    print("tag:", tag)
    print("AUC:", report["auc"])
    print("feature_cols_source:", feat_src)
    print("features:", feat_cols)
    print("=" * 60)


if __name__ == "__main__":
    main()