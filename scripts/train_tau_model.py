# scripts/train_tau_model.py
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score


FEATURES_PARQ = "data/features/features_model.parquet"
FEATURES_CSV  = "data/features/features_model.csv"

TAU_LABELS_PARQ = "data/labels/labels_tau.parquet"
TAU_LABELS_CSV  = "data/labels/labels_tau.csv"

OUT_MODEL  = "app/tau_model.pkl"
OUT_SCALER = "app/tau_scaler.pkl"


DEFAULT_FEATURES = [
    "Drawdown_252",
    "Drawdown_60",
    "ATR_ratio",
    "Z_score",
    "MACD_hist",
    "MA20_slope",
    "Market_Drawdown",
    "Market_ATR_ratio",
]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES), type=str)
    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=600, type=int)
    args = ap.parse_args()

    feats = read_table(FEATURES_PARQ, FEATURES_CSV).copy()
    lab   = read_table(TAU_LABELS_PARQ, TAU_LABELS_CSV).copy()

    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise KeyError("features_model must have Date,Ticker")
    if "Date" not in lab.columns or "Ticker" not in lab.columns or "TauClass" not in lab.columns:
        raise KeyError("labels_tau must have Date,Ticker,TauClass")

    feats["Date"] = norm_date(feats["Date"])
    lab["Date"]   = norm_date(lab["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    lab["Ticker"]   = lab["Ticker"].astype(str).str.upper().str.strip()

    features = [x.strip() for x in args.features.split(",") if x.strip()]
    missing = [c for c in features if c not in feats.columns]
    if missing:
        raise KeyError(f"features_model missing: {missing}")

    merged = feats.merge(lab[["Date", "Ticker", "TauClass"]], on=["Date", "Ticker"], how="inner")
    merged = merged.dropna(subset=features + ["TauClass"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    if merged.empty:
        raise RuntimeError("No training rows after merge. Check labels/features overlap.")

    X = merged[features].astype(float)
    y = pd.to_numeric(merged["TauClass"], errors="coerce").astype("Int64")
    merged = merged.loc[y.notna()].copy()
    X = merged[features].astype(float)
    y = merged["TauClass"].astype(int)

    # time split by rows (already date-sorted)
    n = len(merged)
    split = int(n * float(args.train_ratio))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    base = LogisticRegression(
        max_iter=int(args.max_iter),
        multi_class="multinomial",
        n_jobs=1,
    )

    # Calibrate with sigmoid (isotonic is heavy + multi-class can be unstable)
    tscv = TimeSeriesSplit(n_splits=int(args.n_splits))
    model = CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=tscv
    )

    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)
    pred = np.argmax(probs, axis=1)

    ll = log_loss(y_test, probs, labels=sorted(np.unique(y)))
    acc = accuracy_score(y_test, pred)

    print("=" * 60)
    print("[TAU MODEL] Test logloss:", round(float(ll), 4))
    print("[TAU MODEL] Test accuracy:", round(float(acc), 4))
    print("[TAU MODEL] Class dist (test):", pd.Series(y_test).value_counts(normalize=True).sort_index().to_dict())
    print("=" * 60)

    Path("app").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_MODEL)
    joblib.dump(scaler, OUT_SCALER)
    print(f"[DONE] wrote: {OUT_MODEL} / {OUT_SCALER}")


if __name__ == "__main__":
    main()