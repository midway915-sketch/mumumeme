# scripts/predict_gate.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def _to_dt(x):
    return pd.to_datetime(x, errors="coerce")


def _pick_feature_cols(model, fallback: list[str], df_cols: list[str]) -> list[str]:
    cols = set(df_cols)
    # sklearn sometimes has feature_names_in_
    if hasattr(model, "feature_names_in_"):
        feats = [c for c in list(getattr(model, "feature_names_in_")) if c in cols]
        if feats:
            return feats
    feats = [c for c in fallback if c in cols]
    if feats:
        return feats
    # last resort: numeric cols
    banned = {"Date", "Ticker"}
    num = [c for c in df_cols if c not in banned and pd.api.types.is_numeric_dtype(pd.Series(dtype=float))]
    return [c for c in df_cols if c not in banned][:8]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--mode", required=True, type=str, choices=["none", "tail", "utility", "tail_utility"])
    ap.add_argument("--tag", required=True, type=str)
    ap.add_argument("--suffix", required=True, type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    ap.add_argument("--features-parq", default="data/features/features_model.parquet", type=str)
    ap.add_argument("--features-csv", default="data/features/features_model.csv", type=str)

    ap.add_argument("--tail-threshold", required=True, type=float)
    ap.add_argument("--utility-quantile", required=True, type=float)
    ap.add_argument("--rank-by", required=True, type=str, choices=["utility", "ret_score", "p_success"])
    ap.add_argument("--lambda-tail", required=True, type=float)

    ap.add_argument("--require-files", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = read_table(args.features_parq, args.features_csv).copy()
    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_model must contain Date and Ticker")
    feats["Date"] = _to_dt(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Load models
    # success
    if args.require_files:
        for p in ["app/model.pkl", "app/scaler.pkl", "app/tail_model.pkl", "app/tail_scaler.pkl", "app/tau_cdf_models.pkl", "app/tau_scaler.pkl"]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing required model file: {p}")

    success_model = joblib.load("app/model.pkl")
    success_scaler = joblib.load("app/scaler.pkl")

    tail_model = joblib.load("app/tail_model.pkl")
    tail_scaler = joblib.load("app/tail_scaler.pkl")

    tau_models = joblib.load("app/tau_cdf_models.pkl")  # dict: TauLE10/20/40 -> model
    tau_scaler = joblib.load("app/tau_scaler.pkl")

    # Choose feature columns per model
    fallback_feats = [
        "Drawdown_252","Drawdown_60","ATR_ratio","Z_score","MACD_hist","MA20_slope","Market_Drawdown","Market_ATR_ratio"
    ]
    cols = list(feats.columns)

    succ_feats = _pick_feature_cols(success_model, fallback_feats, cols)
    tail_feats = _pick_feature_cols(tail_model, fallback_feats, cols)
    # tau models share same scaler features ideally; use TauLE40 model as reference
    ref_tau_model = tau_models.get("TauLE40", next(iter(tau_models.values())))
    tau_feats = _pick_feature_cols(ref_tau_model, fallback_feats, cols)

    # Prepare matrices (fill NaN with 0 for inference)
    Xs = feats[succ_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    Xt = feats[tail_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    Xtau = feats[tau_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    Xs_s = success_scaler.transform(Xs)
    Xt_s = tail_scaler.transform(Xt)
    Xtau_s = tau_scaler.transform(Xtau)

    p_success = success_model.predict_proba(Xs_s)[:, 1]
    p_tail = tail_model.predict_proba(Xt_s)[:, 1]

    p_le10 = tau_models["TauLE10"].predict_proba(Xtau_s)[:, 1]
    p_le20 = tau_models["TauLE20"].predict_proba(Xtau_s)[:, 1]
    p_le40 = tau_models["TauLE40"].predict_proba(Xtau_s)[:, 1]

    # enforce monotonic cdf (optional clamp)
    p_le10 = np.clip(p_le10, 0.0, 1.0)
    p_le20 = np.clip(np.maximum(p_le20, p_le10), 0.0, 1.0)
    p_le40 = np.clip(np.maximum(p_le40, p_le20), 0.0, 1.0)

    p10 = p_le10
    p20 = np.clip(p_le20 - p_le10, 0.0, 1.0)
    p40 = np.clip(p_le40 - p_le20, 0.0, 1.0)
    pgt40 = np.clip(1.0 - p_le40, 0.0, 1.0)

    # expected tau (representative mids)
    tau_exp = (7.0 * p10) + (15.0 * p20) + (30.0 * p40) + (55.0 * pgt40)

    out = feats[["Date", "Ticker"]].copy()
    out["p_success"] = pd.to_numeric(p_success, errors="coerce")
    out["p_tail"] = pd.to_numeric(p_tail, errors="coerce")
    out["p_tau10"] = pd.to_numeric(p10, errors="coerce")
    out["p_tau20"] = pd.to_numeric(p20, errors="coerce")
    out["p_tau40"] = pd.to_numeric(p40, errors="coerce")
    out["tau_exp"] = pd.to_numeric(tau_exp, errors="coerce")

    # ret_score: if exists in features use it, else proxy by (p_success - p_tail)
    if "ret_score" in feats.columns:
        out["ret_score"] = pd.to_numeric(feats["ret_score"], errors="coerce").fillna(0.0)
    else:
        out["ret_score"] = out["p_success"].fillna(0.0) - out["p_tail"].fillna(0.0)

    # utility: success - lambda*tail (simple, consistent)
    lam = float(args.lambda_tail)
    out["utility"] = out["p_success"].fillna(0.0) - lam * out["p_tail"].fillna(0.0)

    # Gate filters
    use = out.copy()
    use["Skipped"] = 0

    if args.mode in ("tail", "tail_utility"):
        ok_tail = (use["p_tail"].fillna(1.0) <= float(args.tail_threshold))
    else:
        ok_tail = pd.Series([True] * len(use))

    if args.mode in ("utility", "tail_utility"):
        # daily quantile cut on utility
        # compute per day threshold
        q = float(args.utility_quantile)
        cut = use.groupby("Date")["utility"].transform(lambda s: s.quantile(q))
        ok_util = (use["utility"] >= cut)
    else:
        ok_util = pd.Series([True] * len(use))

    ok = ok_tail & ok_util
    use.loc[~ok, "Skipped"] = 1

    # Rank metric
    rank_col = args.rank_by
    if rank_col not in use.columns:
        raise ValueError(f"rank_by column missing: {rank_col}")

    # pick top-1 among non-skipped per day
    def pick_one(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        cand = g[g["Skipped"] == 0]
        if len(cand) == 0:
            # no pick
            g["Pick"] = 0
            return g
        # highest rank
        idx = cand[rank_col].astype(float).idxmax()
        g["Pick"] = 0
        g.loc[idx, "Pick"] = 1
        return g

    use = use.groupby("Date", group_keys=False).apply(pick_one)
    # output picks file: one row per day with chosen ticker (or NaN if none)
    picks = use[use["Pick"] == 1].copy()
    # ensure 1 row per day: if multiple (ties), keep first
    picks = picks.sort_values(["Date", rank_col], ascending=[True, False]).drop_duplicates(["Date"], keep="first")

    # create per-day file expected by simulator
    # If no pick, still emit a file with Date rows and Skipped=1 for debugging? keep only picks is okay if sim expects picks per day.
    # We'll emit full daily table to be safe.
    out_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
    full_path = out_dir / f"picks_full_{args.tag}_gate_{args.suffix}.csv"

    use = use.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    use.to_csv(full_path, index=False)

    picks_out = picks[["Date", "Ticker", "p_success", "p_tail", "utility", "ret_score", "tau_exp", "p_tau10", "p_tau20", "p_tau40"]].copy()
    picks_out.to_csv(out_path, index=False)

    print(f"[DONE] wrote picks: {out_path} (rows={len(picks_out)})")
    print(f"[INFO] full daily table: {full_path} (rows={len(use)})")


if __name__ == "__main__":
    main()