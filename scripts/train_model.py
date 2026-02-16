# scripts/train_model.py
from __future__ import annotations

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


DEFAULT_DATA_PATH = Path("data/raw_data.csv")
MODEL_PATH = Path("app/model.pkl")
SCALER_PATH = Path("app/scaler.pkl")

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


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    uniq = np.unique(y_true)
    if len(uniq) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train success model (Calibrated Logistic).")
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--target-col", type=str, default="Success")
    ap.add_argument("--out-model", type=str, default=str(MODEL_PATH))
    ap.add_argument("--out-scaler", type=str, default=str(SCALER_PATH))
    ap.add_argument("--n-splits", type=int, default=3)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    else:
        df = df.reset_index(drop=True)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in {data_path}")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in training data: {missing}")

    df = df.dropna(subset=FEATURES + [args.target_col]).reset_index(drop=True)

    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = df[args.target_col].to_numpy(dtype=int)

    if len(np.unique(y)) < 2:
        raise RuntimeError("Target has only one class. Labels might be wrong or too few rows.")

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base = LogisticRegression(max_iter=800)

    # TimeSeriesSplit: 데이터 너무 적으면 splits 줄임
    n_splits = min(int(args.n_splits), 5)
    if split_idx < 200:
        n_splits = min(n_splits, 2)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    model = CalibratedClassifierCV(
        base_estimator=base,
        method="isotonic",
        cv=tscv,
    )
    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:, 1]
    auc = safe_auc(y_test, probs)

    print("=" * 60)
    print("Test ROC-AUC:", round(auc, 4) if auc == auc else "nan")
    print("Base Success Rate:", round(float(y_test.mean()), 4))
    print("Predicted Mean Probability:", round(float(probs.mean()), 4))
    print("Rows:", len(df), "Train:", len(y_train), "Test:", len(y_test))
    print("=" * 60)

    out_model = Path(args.out_model)
    out_scaler = Path(args.out_scaler)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)
    print(f"✅ saved: {out_model} / {out_scaler}")


if __name__ == "__main__":
    main()
