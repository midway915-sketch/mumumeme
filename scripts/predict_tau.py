# scripts/predict_tau.py
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


DATA_DIR = Path("data")
FEATURE_DIR = DATA_DIR / "features"
SIGNALS_DIR = DATA_DIR / "signals"
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

FEATS_PARQ = FEATURE_DIR / "features_model.parquet"
FEATS_CSV = FEATURE_DIR / "features_model.csv"

MODEL_PATH = Path("app/tau_model.pkl")
SCALER_PATH = Path("app/tau_scaler.pkl")

OUT_PARQ = SIGNALS_DIR / "tau_pred.parquet"
OUT_CSV = SIGNALS_DIR / "tau_pred.csv"

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


def read_feats() -> pd.DataFrame:
    if FEATS_PARQ.exists():
        return pd.read_parquet(FEATS_PARQ)
    if FEATS_CSV.exists():
        return pd.read_csv(FEATS_CSV)
    raise FileNotFoundError(f"Missing features: {FEATS_PARQ} (or {FEATS_CSV})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-min", type=float, default=1.0)
    ap.add_argument("--clip-max", type=float, default=999.0)
    args = ap.parse_args()

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Missing tau model/scaler. Run train_tau_model.py first.")

    df = read_feats()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X = df[FEATURES].astype(float)
    Xs = scaler.transform(X)

    tau_hat = model.predict(Xs)
    tau_hat = pd.Series(tau_hat).clip(lower=float(args.clip_min), upper=float(args.clip_max))

    out = df[["Date", "Ticker"]].copy()
    out["tau_hat"] = tau_hat.values

    out.to_parquet(OUT_PARQ, index=False)
    out.to_csv(OUT_CSV, index=False)
    print(f"[DONE] wrote {OUT_PARQ} rows={len(out)}")


if __name__ == "__main__":
    main()