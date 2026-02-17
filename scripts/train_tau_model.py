# scripts/train_tau_model.py
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


DATA_DIR = Path("data")
LABEL_DIR = DATA_DIR / "labels"
IN_PARQ = LABEL_DIR / "tau_labels.parquet"
IN_CSV = LABEL_DIR / "tau_labels.csv"

MODEL_PATH = Path("app/tau_model.pkl")
SCALER_PATH = Path("app/tau_scaler.pkl")

FEATURES = [
    "Drawdown_252",
    "Drawdown_60",
    "ATR_ratio",
    "Z_score",
    "MACD_hist",
    "MA20_slope",
    "Market_Drawdown",
    "Market_ATR_ratio",
]

def read_table() -> pd.DataFrame:
    if IN_PARQ.exists():
        return pd.read_parquet(IN_PARQ)
    if IN_CSV.exists():
        return pd.read_csv(IN_CSV)
    raise FileNotFoundError(f"Missing tau labels: {IN_PARQ} (or {IN_CSV})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--max-tau", type=int, default=365,
                    help="clip Tau_Days to reduce outlier impact (success-only)")
    args = ap.parse_args()

    df = read_table()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # success-only regression target
    df = df[df["Success"] == 1].copy()
    df = df.dropna(subset=FEATURES + ["Tau_Days"])

    if df.empty:
        raise RuntimeError("No success samples for tau model. Check label build / profit target / horizon.")

    X = df[FEATURES].astype(float)
    y = pd.to_numeric(df["Tau_Days"], errors="coerce").astype(float)
    y = y.clip(lower=1, upper=float(args.max_tau))

    # split by time 80/20 (latest as test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # HGBR works well, no heavy deps, fast for Actions
    base = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        random_state=42,
        loss="absolute_error",  # robust MAE
    )

    # time-series CV fit: train on expanding windows
    tscv = TimeSeriesSplit(n_splits=int(args.n_splits))
    # 간단히 전체 train에 fit (CV는 지표만 보고 튜닝 없이 진행)
    base.fit(X_train_s, y_train)

    pred = base.predict(X_test_s)
    mae = mean_absolute_error(y_test, pred)

    print("=" * 60)
    print("TAU MODEL (success-only regression)")
    print("Test MAE (days):", round(mae, 3))
    print("y_test mean:", round(float(np.mean(y_test)), 3))
    print("pred mean:", round(float(np.mean(pred)), 3))
    print("=" * 60)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(base, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[DONE] wrote {MODEL_PATH} / {SCALER_PATH}")


if __name__ == "__main__":
    main()