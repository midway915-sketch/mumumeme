# scripts/predict_gate.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


FEATURES_PARQ = "data/features/features_model.parquet"
FEATURES_CSV  = "data/features/features_model.csv"

OUT_DIR_DEFAULT = "data/signals"


REQUIRED_BASE_COLS = ["Date", "Ticker"]


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


def parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def safe_float(x, default=np.nan) -> float:
    try:
        v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
        return float(v) if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def attach_tau_predictions(picks: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """
    If app/tau_model.pkl & app/tau_scaler.pkl exist, attach:
      TauClassPred (0/1/2/3)
      TauP10 = P(class=0)
      TauP20 = P(class<=1)  (0 or 1)
      TauPH  = P(class<=2)  (0 or 1 or 2)
    """
    model_path = Path("app/tau_model.pkl")
    scaler_path = Path("app/tau_scaler.pkl")
    if not model_path.exists() or not scaler_path.exists():
        return picks

    # Use the same feature list as train_tau_model default
    feature_cols = [
        "Drawdown_252",
        "Drawdown_60",
        "ATR_ratio",
        "Z_score",
        "MACD_hist",
        "MA20_slope",
        "Market_Drawdown",
        "Market_ATR_ratio",
    ]
    missing = [c for c in feature_cols if c not in feats.columns]
    if missing:
        print(f"[WARN] tau features missing: {missing} -> skip tau attach")
        return picks

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # merge picks with features to get feature row
    tmp = picks.merge(feats[["Date", "Ticker"] + feature_cols], on=["Date", "Ticker"], how="left")
    X = tmp[feature_cols].astype(float)
    Xs = scaler.transform(X)

    proba = model.predict_proba(Xs)
    # columns correspond to sorted classes observed in training; assume 0..3
    # Make robust mapping
    classes = list(getattr(model, "classes_", [0, 1, 2, 3]))
    cls_to_idx = {int(c): i for i, c in enumerate(classes)}

    def p_of(k: int) -> np.ndarray:
        if k in cls_to_idx:
            return proba[:, cls_to_idx[k]]
        return np.zeros(len(tmp), dtype=float)

    p0 = p_of(0)
    p1 = p_of(1)
    p2 = p_of(2)
    p3 = p_of(3)

    pred = np.argmax(proba, axis=1)
    pred_class = np.array([classes[i] for i in pred], dtype=int)

    tmp["TauClassPred"] = pred_class
    tmp["TauP10"] = p0
    tmp["TauP20"] = (p0 + p1)
    tmp["TauPH"]  = (p0 + p1 + p2)
    # keep only added cols on picks
    keep = list(picks.columns) + ["TauClassPred", "TauP10", "TauP20", "TauPH"]
    return tmp[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--mode", required=True, choices=["none", "tail", "utility", "tail_utility"])
    ap.add_argument("--tag", required=True, type=str)
    ap.add_argument("--suffix", required=True, type=str)

    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT, type=str)
    ap.add_argument("--features-parq", default=FEATURES_PARQ, type=str)
    ap.add_argument("--features-csv", default=FEATURES_CSV, type=str)

    ap.add_argument("--tail-threshold", required=True, type=float)
    ap.add_argument("--utility-quantile", required=True, type=float)
    ap.add_argument("--rank-by", required=True, choices=["utility", "ret_score", "p_success"])
    ap.add_argument("--lambda-tail", required=True, type=float)

    ap.add_argument("--require-files", default="", type=str, help="comma-separated files that must exist before running")
    args = ap.parse_args()

    # require-files
    req = [x for x in parse_csv_list(args.require_files) if x]
    for f in req:
        if not Path(f).exists():
            raise FileNotFoundError(f"Required file missing: {f}")

    feats = read_table(args.features_parq, args.features_csv).copy()
    for c in REQUIRED_BASE_COLS:
        if c not in feats.columns:
            raise KeyError(f"features_model missing required column: {c}")

    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Expect these columns exist from your pipeline
    # - p_tail (0~1), utility, ret_score, p_success (optional)
    for c in ["p_tail", "utility", "ret_score"]:
        if c not in feats.columns:
            # allow empty fallback
            feats[c] = np.nan

    if "p_success" not in feats.columns:
        feats["p_success"] = np.nan

    # Gate filter
    use = feats.copy()
    if args.mode in ("tail", "tail_utility"):
        use = use[pd.to_numeric(use["p_tail"], errors="coerce").fillna(1.0) <= float(args.tail_threshold)].copy()
    if args.mode in ("utility", "tail_utility"):
        uq = float(args.utility_quantile)
        cut = pd.to_numeric(feats["utility"], errors="coerce").quantile(uq) if len(feats) else np.nan
        use = use[pd.to_numeric(use["utility"], errors="coerce").fillna(-np.inf) >= cut].copy()

    # Rank metric
    rank_col = args.rank_by
    rk = pd.to_numeric(use.get(rank_col, np.nan), errors="coerce").fillna(-np.inf)

    # If utility ranking and tail penalty lambda > 0, apply penalty
    if rank_col == "utility":
        lam = float(args.lambda_tail)
        ptail = pd.to_numeric(use["p_tail"], errors="coerce").fillna(0.0)
        rk = rk - lam * ptail

    use["_rank"] = rk

    # Pick top-1 per date
    picks = (
        use.sort_values(["Date", "_rank"], ascending=[True, False])
           .drop_duplicates(["Date"], keep="first")[["Date", "Ticker", "_rank"]]
           .rename(columns={"_rank": "RankValue"})
           .reset_index(drop=True)
    )

    # Attach tau predictions if tau model exists
    picks = attach_tau_predictions(picks, feats)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
    picks.to_csv(out_path, index=False)
    print(f"[DONE] wrote picks: {out_path} rows={len(picks)}")


if __name__ == "__main__":
    main()