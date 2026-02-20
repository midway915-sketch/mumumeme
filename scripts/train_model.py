#!/usr/bin/env python3
from __future__ import annotations

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


DATA_DIR = Path("data")
LABELS_PARQ = DATA_DIR / "labels" / "labels_model.parquet"
LABELS_CSV = DATA_DIR / "labels" / "labels_model.csv"
APP_DIR = Path("app")
META_DIR = DATA_DIR / "meta"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Training data not found: {parq} (or {csv})")


def parse_csv_list(s: str) -> list[str]:
    if s is None:
        return []
    items = [x.strip() for x in str(s).split(",")]
    return [x for x in items if x]


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


def ensure_feature_columns_strict(df: pd.DataFrame, feat_cols: list[str], source_hint: str = "") -> None:
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        hint = f" (src={source_hint})" if source_hint else ""
        raise ValueError(f"Missing feature columns{hint}: {missing}")


def write_train_report(tag: str, report: dict) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    p = META_DIR / (f"train_model_report_{tag}.json" if tag else "train_model_report.json")
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote train report -> {p}")


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

    cols_meta, sector_enabled = read_feature_cols_meta()
    if cols_meta:
        return cols_meta, "data/meta/feature_cols.json"

    # fallback: sector OFF
    return get_feature_cols(sector_enabled=False), "feature_spec.py (fallback)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", type=str, help="optional tag suffix for saving model files + meta report")

    ap.add_argument("--target-col", default="Success", type=str)
    ap.add_argument("--date-col", default="Date", type=str)
    ap.add_argument("--ticker-col", default="Ticker", type=str)

    # ✅ 기본은 SSOT(meta) 사용. 필요할 때만 override.
    ap.add_argument("--features", default="", type=str, help="comma-separated feature cols (override SSOT/meta)")

    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=500, type=int)

    ap.add_argument("--out-model", default="", type=str)
    ap.add_argument("--out-scaler", default="", type=str)

    args = ap.parse_args()

    df = read_table(LABELS_PARQ, LABELS_CSV).copy()

    date_col = args.date_col
    target_col = args.target_col
    ticker_col = args.ticker_col

    for c in [date_col, target_col]:
        if c not in df.columns:
            raise ValueError(f"labels_model missing required column: {c}")

    # normalize date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=[date_col]).copy()
    df = df.sort_values(date_col).reset_index(drop=True)

    feat_cols, feat_src = resolve_feature_cols(args.features)
    feat_cols = [str(c).strip() for c in feat_cols if str(c).strip()]

    # ✅ STRICT: 누락 피처는 바로 에러
    ensure_feature_columns_strict(df, feat_cols, source_hint=feat_src)

    # numeric coercion
    df = coerce_features_numeric(df, feat_cols)

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    X = df[feat_cols].to_numpy(dtype=float)

    if len(y) < 200:
        raise RuntimeError(f"Not enough training rows: {len(y)}")

    split_idx = int(len(y) * float(args.train_ratio))
    split_idx = max(50, min(split_idx, len(y) - 50))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base = LogisticRegression(max_iter=int(args.max_iter))
    tscv = TimeSeriesSplit(n_splits=int(args.n_splits))

    # sklearn 버전 차이 방어 (estimator vs base_estimator)
    try:
        model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=tscv)
    except TypeError:
        model = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=tscv)

    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:, 1]
    auc = float("nan")
    if len(np.unique(y_test)) > 1:
        auc = float(roc_auc_score(y_test, probs))

    base_rate = float(np.mean(y_test)) if len(y_test) else float("nan")
    pred_mean = float(np.mean(probs)) if len(probs) else float("nan")

    print("=" * 60)
    print("[TRAIN] p_success model")
    print("rows:", len(y), "train:", len(y_train), "test:", len(y_test))
    print("AUC:", (round(auc, 6) if np.isfinite(auc) else "nan"))
    print("base_rate:", (round(base_rate, 6) if np.isfinite(base_rate) else "nan"))
    print("pred_mean:", (round(pred_mean, 6) if np.isfinite(pred_mean) else "nan"))
    print("feature_cols_source:", feat_src)
    print("feature_cols:", feat_cols)
    print("=" * 60)

    APP_DIR.mkdir(parents=True, exist_ok=True)
    tag = (args.tag or "").strip()

    out_model = Path(args.out_model) if args.out_model else (APP_DIR / (f"model_{tag}.pkl" if tag else "model.pkl"))
    out_scaler = Path(args.out_scaler) if args.out_scaler else (APP_DIR / (f"scaler_{tag}.pkl" if tag else "scaler.pkl"))

    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)

    print(f"[DONE] saved model -> {out_model}")
    print(f"[DONE] saved scaler -> {out_scaler}")

    report = {
        "tag": tag,
        "target_col": target_col,
        "date_col": date_col,
        "ticker_col": ticker_col if ticker_col in df.columns else "",
        "feature_cols_source": feat_src,
        "feature_cols": feat_cols,
        "rows_total": int(len(y)),
        "rows_train": int(len(y_train)),
        "rows_test": int(len(y_test)),
        "auc": float(auc) if np.isfinite(auc) else None,
        "base_rate_test": float(base_rate) if np.isfinite(base_rate) else None,
        "pred_mean_test": float(pred_mean) if np.isfinite(pred_mean) else None,
        "out_model": str(out_model),
        "out_scaler": str(out_scaler),
    }
    write_train_report(tag, report)


if __name__ == "__main__":
    main()