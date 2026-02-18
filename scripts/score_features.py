# scripts/score_features.py
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def ensure_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def coerce_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _model_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-z))
    raise TypeError("Model does not support probability prediction")


def load_joblib(path: str):
    if joblib is None:
        raise RuntimeError("joblib is not available. pip install joblib")
    return joblib.load(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build features_scored by attaching p_success/p_tail predictions to features_model.")
    ap.add_argument("--features-parq", default="data/features/features_model.parquet", type=str)
    ap.add_argument("--features-csv", default="data/features/features_model.csv", type=str)

    ap.add_argument("--out-parq", default="data/features/features_scored.parquet", type=str)
    ap.add_argument("--out-csv", default="data/features/features_scored.csv", type=str)

    ap.add_argument("--success-model", default="app/model.pkl", type=str)
    ap.add_argument("--success-scaler", default="app/scaler.pkl", type=str)

    ap.add_argument("--tail-model", default="app/tail_model.pkl", type=str)

    ap.add_argument("--require-success", action="store_true", help="Fail if success model/scaler missing.")
    ap.add_argument("--require-tail", action="store_true", help="Fail if tail model missing.")

    ap.add_argument("--lambda-tail", default=0.05, type=float, help="utility = ret_score - lambda_tail*p_tail")
    ap.add_argument("--meta-out", default="data/features/features_scored_meta.json", type=str)
    args = ap.parse_args()

    feats = read_table(args.features_parq, args.features_cvv if False else args.features_cvv)  # noqa