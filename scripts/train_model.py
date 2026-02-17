# scripts/train_model.py
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


DATA_DIR = Path("data")
LABELS_PARQ = DATA_DIR / "labels" / "labels_model.parquet"
LABELS_CSV = DATA_DIR / "labels" / "labels_model.csv"

APP_DIR = Path("app")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", type=str, help="optional tag suffix for saving model files")

    ap.add_argument("--target-col", default="Success", type=str)
    ap.add_argument("--date-col", default="Date", type=str)
    ap.add_argument("--ticker-col", default="Ticker", type=str)

    ap.add_argument(
        "--features",
        default="Drawdown_252,Drawdown_60,ATR_ratio,Z_score,MACD_hist,MA20_slope,Market_Drawdown,Market_ATR_ratio",
        type=str,
        help="comma-separated feature column names",
    )

    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=500, type=int)

    ap.add_argument("--out-model", default="", type=str, help="override model output path")
    ap.add_argument("--out-scaler", default="", type=str, help="override scaler output path")

    args = ap.parse_args()

    df = read_table(LABELS_PARQ, LABELS_CSV).copy()

    date_col = args.date_col
    ticker_col = args.ticker_col
    target_col = args.target_col
    feat_cols = parse_csv_list(args.features)

    # basic schema checks
    for c in [date_col, ticker_col, target_col]:
        if c not in df.columns:
            raise ValueError(f"labels_model missing required column: {c}")

    missing_feats = [c for c in feat_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(f"labels_model missing feature columns: {missing_feats}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col] + feat_cols + [target_col]).copy()
    df = df.sort_values(date_col).reset_index(drop=True)

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")

    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].to_numpy(dtype=float)
    y = y.loc[mask].to_numpy(dtype=int)

    if len(y) < 200:
        raise RuntimeError(f"Not enough training rows after cleaning: {len(y)}")

    split_idx = int(len(y) * float(args.train_ratio))
    split_idx = max(50, min(split_idx, len(y) - 50))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base = LogisticRegression(max_iter=int(args.max_iter))
    tscv = TimeSeriesSplit(n_splits=int(args.n_splits))

    # sklearn 버전에 따라 estimator 파라미터명이 다를 수 있는데,
    # CalibratedClassifierCV는 최근 버전에서 estimator= 를 씀.
    model = CalibratedClassifierCV(
        estimator=base,
        method="isotonic",
        cv=tscv,
    )

    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")

    print("=" * 60)
    print("[TRAIN] success model")
    print("rows:", len(y), "train:", len(y_train), "test:", len(y_test))
    print("AUC:", (round(float(auc), 4) if np.isfinite(auc) else "nan"))
    print("base_rate:", round(float(np.mean(y_test)), 4) if len(y_test) else "nan")
    print("pred_mean:", round(float(np.mean(probs)), 4) if len(probs) else "nan")
    print("=" * 60)

    APP_DIR.mkdir(parents=True, exist_ok=True)

    tag = (args.tag or "").strip()
    if args.out_model:
        out_model = Path(args.out_model)
    else:
        out_model = APP_DIR / (f"model_{tag}.pkl" if tag else "model.pkl")

    if args.out_scaler:
        out_scaler = Path(args.out_scaler)
    else:
        out_scaler = APP_DIR / (f"scaler_{tag}.pkl" if tag else "scaler.pkl")

    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)

    print(f"[DONE] saved model -> {out_model}")
    print(f"[DONE] saved scaler -> {out_scaler}")


if __name__ == "__main__":
    main()