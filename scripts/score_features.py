# scripts/score_features.py
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def write_table(df: pd.DataFrame, parq: str, csv: str) -> None:
    Path(parq).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parq, index=False)
    df.to_csv(csv, index=False)


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def pick_feature_cols(df: pd.DataFrame, scaler, model) -> list[str]:
    # 1) scaler feature_names_in_ (best)
    cols = getattr(scaler, "feature_names_in_", None)
    if cols is not None:
        cols = [c for c in cols if c in df.columns]
        if cols:
            return cols

    # 2) model feature_names_in_
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        cols = [c for c in cols if c in df.columns]
        if cols:
            return cols

    # 3) fallback: numeric columns except obvious non-features
    blacklist = {
        "Date", "Ticker", "Group",
        "Open", "High", "Low", "Close", "Adj Close",
        "Volume",
        "p_tail", "p_success", "utility",
        "TailTarget", "Target", "TauClass", "TauDays",
    }
    num_cols = []
    for c in df.columns:
        if c in blacklist:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    return num_cols


def predict_proba_safe(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        # binary => [:,1]
        if isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
    # fallback
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        z = np.asarray(z, dtype=float)
        # logistic squash
        return 1.0 / (1.0 + np.exp(-z))
    # ultimate fallback
    y = model.predict(X)
    return np.asarray(y, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach p_success/p_tail to features -> features_scored")
    ap.add_argument("--in-parq", default="data/features/features_model.parquet", type=str)
    ap.add_argument("--in-csv", default="data/features/features_model.csv", type=str)
    ap.add_argument("--out-parq", default="data/features/features_scored.parquet", type=str)
    ap.add_argument("--out-csv", default="data/features/features_scored.csv", type=str)

    ap.add_argument("--success-model", default="app/model.pkl", type=str)
    ap.add_argument("--success-scaler", default="app/scaler.pkl", type=str)

    ap.add_argument("--tail-model", default="app/tail_model.pkl", type=str)
    ap.add_argument("--tail-scaler", default="app/tail_scaler.pkl", type=str)

    ap.add_argument("--require-success", action="store_true", help="fail if success model missing")
    ap.add_argument("--require-tail", action="store_true", help="fail if tail model missing")
    args = ap.parse_args()

    df = read_table(args.in_parq, args.in_csv).copy()
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError(f"features must have Date,Ticker. cols={list(df.columns)[:50]}")

    df["Date"] = norm_date(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # ---------- p_success
    success_ok = Path(args.success_model).exists() and Path(args.success_scaler).exists()
    if (not success_ok) and args.require_success:
        raise FileNotFoundError(f"Missing success model/scaler: {args.success_model}, {args.success_scaler}")

    if success_ok:
        sm = joblib.load(args.success_model)
        ss = joblib.load(args.success_scaler)
        feat_cols = pick_feature_cols(df, ss, sm)

        X = df.reindex(columns=feat_cols).copy()
        # fill missing columns to 0, coerce numeric
        for c in feat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(0.0)

        Xs = ss.transform(X.to_numpy(dtype=float))
        df["p_success"] = predict_proba_safe(sm, Xs).astype(float)
    else:
        df["p_success"] = 0.0

    # ---------- p_tail
    tail_ok = Path(args.tail_model).exists() and Path(args.tail_scaler).exists()
    if (not tail_ok) and args.require_tail:
        raise FileNotFoundError(f"Missing tail model/scaler: {args.tail_model}, {args.tail_scaler}")

    if tail_ok:
        tm = joblib.load(args.tail_model)
        ts = joblib.load(args.tail_scaler)
        feat_cols = pick_feature_cols(df, ts, tm)

        X = df.reindex(columns=feat_cols).copy()
        for c in feat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(0.0)

        Xs = ts.transform(X.to_numpy(dtype=float))
        df["p_tail"] = predict_proba_safe(tm, Xs).astype(float)
    else:
        df["p_tail"] = 0.0

    # clamp
    df["p_success"] = pd.to_numeric(df["p_success"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["p_tail"] = pd.to_numeric(df["p_tail"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    write_table(df, args.out_parq, args.out_csv)
    print(f"[DONE] wrote scored features: {args.out_parq} rows={len(df)} cols={len(df.columns)}")
    print(f"[INFO] models: success={'OK' if success_ok else 'MISSING->0'} tail={'OK' if tail_ok else 'MISSING->0'}")


if __name__ == "__main__":
    main()