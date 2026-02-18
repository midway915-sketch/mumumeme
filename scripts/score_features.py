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


def write_table(df: pd.DataFrame, out_parq: str, out_csv: str) -> None:
    Path(out_parq).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parq, index=False)
    df.to_csv(out_csv, index=False)


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def ensure_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba_any(model, X: np.ndarray) -> np.ndarray:
    """
    Return p(class=1) for binary classification.
    Supports sklearn models with predict_proba or decision_function.
    """
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        # shape (n,2) or (n,) depending on model
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1].astype(float)
        return np.asarray(p).astype(float).reshape(-1)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = np.asarray(s).astype(float).reshape(-1)
        return sigmoid(s)
    # fallback: predict returns 0/1
    y = model.predict(X)
    y = np.asarray(y).astype(float).reshape(-1)
    return np.clip(y, 0.0, 1.0)


def pick_feature_columns(df: pd.DataFrame, scaler=None, model=None) -> list[str]:
    """
    Prefer:
      1) scaler.feature_names_in_ (if present)
      2) model.feature_names_in_ (if present)
      3) all numeric columns excluding common non-features
    """
    for obj in (scaler, model):
        if obj is not None and hasattr(obj, "feature_names_in_"):
            cols = [c for c in list(obj.feature_names_in_) if c in df.columns]
            if cols:
                return cols

    # fallback: numeric columns minus identifiers/targets
    exclude = set([
        "Date", "Ticker",
        "Open", "High", "Low", "Close", "Volume",
        "Target", "TailTarget", "TauClass",
        "p_success", "p_tail",
    ])
    num_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    return num_cols


def score_one(
    feats: pd.DataFrame,
    model_path: str,
    scaler_path: str | None,
    out_col: str,
) -> pd.Series:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if (scaler_path and Path(scaler_path).exists()) else None

    cols = pick_feature_columns(feats, scaler=scaler, model=model)
    if not cols:
        raise ValueError(f"No feature columns found for scoring {out_col}. Check scaler/model feature_names_in_.")

    X = feats[cols].copy()

    # coerce numeric
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # fill NaN (models hate NaN)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Xv = X.to_numpy(dtype=float)

    if scaler is not None:
        Xv = scaler.transform(Xv)

    p = predict_proba_any(model, Xv)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)

    return pd.Series(p, index=feats.index, name=out_col)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score features_model -> features_scored (p_success, p_tail).")
    ap.add_argument("--in-parq", default="data/features/features_model.parquet", type=str)
    ap.add_argument("--in-csv", default="data/features/features_model.csv", type=str)
    ap.add_argument("--out-parq", default="data/features/features_scored.parquet", type=str)
    ap.add_argument("--out-csv", default="data/features/features_scored.csv", type=str)

    ap.add_argument("--success-model", default="app/model.pkl", type=str)
    ap.add_argument("--success-scaler", default="app/scaler.pkl", type=str)
    ap.add_argument("--tail-model", default="app/tail_model.pkl", type=str)
    ap.add_argument("--tail-scaler", default="app/tail_scaler.pkl", type=str)

    ap.add_argument("--require-success", action="store_true", help="fail if success model files missing")
    ap.add_argument("--require-tail", action="store_true", help="fail if tail model files missing")

    args = ap.parse_args()

    feats = read_table(args.in_parq, args.in_csv).copy()
    ensure_cols(feats, ["Date", "Ticker"], "features_model")
    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # ----- success score
    if args.require_success:
        if (not Path(args.success_model).exists()) or (not Path(args.success_scaler).exists()):
            raise FileNotFoundError(f"Missing success model/scaler: {args.success_model}, {args.success_scaler}")
    if Path(args.success_model).exists():
        scaler_path = args.success_scaler if Path(args.success_scaler).exists() else None
        feats["p_success"] = score_one(
            feats=feats,
            model_path=args.success_model,
            scaler_path=scaler_path,
            out_col="p_success",
        )
    else:
        feats["p_success"] = 0.0

    # ----- tail score
    if args.require_tail:
        if (not Path(args.tail_model).exists()) or (not Path(args.tail_scaler).exists()):
            raise FileNotFoundError(f"Missing tail model/scaler: {args.tail_model}, {args.tail_scaler}")
    if Path(args.tail_model).exists():
        scaler_path = args.tail_scaler if Path(args.tail_scaler).exists() else None
        feats["p_tail"] = score_one(
            feats=feats,
            model_path=args.tail_model,
            scaler_path=scaler_path,
            out_col="p_tail",
        )
    else:
        feats["p_tail"] = 0.0

    # sanity: not all-zeros unless truly missing models
    ps_u = feats["p_success"].dropna().unique()
    pt_u = feats["p_tail"].dropna().unique()
    print("[INFO] p_success unique head:", ps_u[:10])
    print("[INFO] p_tail    unique head:", pt_u[:10])

    write_table(feats, args.out_parq, args.out_csv)
    print(f"[DONE] wrote: {args.out_parq} rows={len(feats)}")
    if len(feats):
        dmin = pd.to_datetime(feats["Date"]).min().date()
        dmax = pd.to_datetime(feats["Date"]).max().date()
        print(f"[INFO] range: {dmin}..{dmax}")


if __name__ == "__main__":
    main()