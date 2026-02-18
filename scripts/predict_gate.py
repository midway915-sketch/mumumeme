# scripts/predict_gate.py
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib


# -------------------------
# IO helpers
# -------------------------
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


def read_universe(universe_csv: str) -> list[str]:
    path = Path(universe_csv)
    if not path.exists():
        raise FileNotFoundError(f"Missing universe file: {universe_csv}")

    uni = pd.read_csv(path)
    if "Ticker" not in uni.columns:
        raise ValueError("universe.csv must contain 'Ticker' column")

    if "Enabled" in uni.columns:
        uni = uni[uni["Enabled"] == True]  # noqa: E712

    tickers = (
        uni["Ticker"].astype(str).str.upper().str.strip().dropna().unique().tolist()
    )
    return tickers


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"features_model missing required columns: {missing}")


def require_files(spec: str) -> None:
    if not spec:
        return
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    missing = [p for p in parts if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


# -------------------------
# scoring
# -------------------------
def coerce_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def build_utility(df: pd.DataFrame, lambda_tail: float) -> pd.Series:
    ret_score = coerce_num(df, "ret_score", 0.0)
    p_tail = coerce_num(df, "p_tail", 0.0)
    return (ret_score - float(lambda_tail) * p_tail).astype(float)


def rank_pick_one_per_day(df: pd.DataFrame, rank_by: str) -> pd.DataFrame:
    if df.empty:
        return df

    if rank_by not in ("utility", "ret_score", "p_success"):
        raise ValueError(f"rank_by must be one of utility|ret_score|p_success (got {rank_by})")

    ev = coerce_num(df, "EV", 0.0) if "EV" in df.columns else pd.Series([0.0] * len(df), index=df.index)
    close = coerce_num(df, "Close", 0.0) if "Close" in df.columns else pd.Series([0.0] * len(df), index=df.index)

    rank_col = coerce_num(df, rank_by, 0.0)
    df = df.copy()
    df["_rank"] = rank_col
    df["_ev"] = ev
    df["_close"] = close

    df = df.sort_values(["Date", "_rank", "_ev", "_close"], ascending=[True, False, False, False])
    out = df.drop_duplicates(["Date"], keep="first").drop(columns=["_rank", "_ev", "_close"])
    return out


def maybe_predict_tau(feats: pd.DataFrame, tau_model_path: str, tau_scaler_path: str, feature_cols: list[str]) -> pd.Series:
    mp = Path(tau_model_path)
    sp = Path(tau_scaler_path)
    if (not mp.exists()) or (not sp.exists()):
        return pd.Series([np.nan] * len(feats), index=feats.index, dtype=float)

    model = joblib.load(mp)
    scaler = joblib.load(sp)

    X = feats.copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_cols]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    Xs = scaler.transform(X.values)

    # regression: predict days
    # (If you later switch to bucket classifier, you can adjust here)
    try:
        pred = model.predict(Xs)
        pred = pd.to_numeric(pd.Series(pred, index=feats.index), errors="coerce")
        return pred.astype(float)
    except Exception:
        return pd.Series([np.nan] * len(feats), index=feats.index, dtype=float)


# -------------------------
# main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Predict gate picks (Top-1 per day) with universe-only candidates.")
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--mode", required=True, choices=["none", "tail", "utility", "tail_utility"])
    ap.add_argument("--tag", required=True, type=str)
    ap.add_argument("--suffix", required=True, type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    ap.add_argument("--features-parq", default="data/features/features_model.parquet", type=str)
    ap.add_argument("--features-csv", default="data/features/features_model.csv", type=str)

    ap.add_argument("--universe-csv", default="data/universe.csv", type=str)
    ap.add_argument("--exclude-tickers", default="SPY,^VIX", type=str, help="comma-separated tickers to force-exclude")

    ap.add_argument("--tail-threshold", required=True, type=float)
    ap.add_argument("--utility-quantile", required=True, type=float)
    ap.add_argument("--rank-by", required=True, choices=["utility", "ret_score", "p_success"])
    ap.add_argument("--lambda-tail", required=True, type=float)

    ap.add_argument("--p-success-min", default=0.0, type=float, help="p_success must be >= this to be eligible (0 disables)")

    ap.add_argument("--require-files", default="", type=str, help="comma-separated file paths that must exist")

    # τ model (optional)
    ap.add_argument("--tau-model-path", default="app/tau_model.pkl", type=str)
    ap.add_argument("--tau-scaler-path", default="app/tau_scaler.pkl", type=str)
    ap.add_argument("--tau-features", default="", type=str, help="comma-separated feature cols for tau model (blank => use common defaults)")

    args = ap.parse_args()

    if args.require_files:
        require_files(args.require_files)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = read_table(args.features_parq, args.features_csv).copy()
    ensure_cols(feats, ["Date", "Ticker"])
    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    universe = set(read_universe(args.universe_csv))
    excludes = set([t.strip().upper() for t in args.exclude_tickers.split(",") if t.strip()])

    before = len(feats)
    feats = feats[feats["Ticker"].isin(universe)].copy()
    feats = feats[~feats["Ticker"].isin(excludes)].copy()
    after = len(feats)

    if after == 0:
        raise RuntimeError(
            f"No rows left after universe filter. before={before} after={after} "
            f"(check universe.csv / features_model tickers)"
        )

    feats["p_tail"] = coerce_num(feats, "p_tail", 0.0)
    feats["p_success"] = coerce_num(feats, "p_success", 0.0)
    feats["ret_score"] = coerce_num(feats, "ret_score", 0.0)
    feats["utility"] = build_utility(feats, lambda_tail=float(args.lambda_tail))

    # τ_pred (optional)
    if args.tau_features.strip():
        tau_cols = [c.strip() for c in args.tau_features.split(",") if c.strip()]
    else:
        # safe defaults (you can override from workflow later)
        tau_cols = [
            "Drawdown_252", "Drawdown_60", "ATR_ratio", "Z_score", "MACD_hist", "MA20_slope",
            "Market_Drawdown", "Market_ATR_ratio"
        ]
    feats["tau_pred"] = maybe_predict_tau(feats, args.tau_model_path, args.tau_scaler_path, tau_cols)

    # gate conditions
    tail_ok = feats["p_tail"] <= float(args.tail_threshold)

    q = float(args.utility_quantile)
    if not (0.0 <= q <= 1.0):
        raise ValueError("--utility-quantile must be in [0,1]")

    util_cut = feats.groupby("Date")["utility"].transform(lambda s: float(s.quantile(q)) if len(s) else np.nan)
    util_ok = (feats["utility"] >= util_cut).fillna(False)

    ps_min = float(args.p_success_min)
    ps_ok = feats["p_success"] >= ps_min

    if args.mode == "none":
        eligible = feats[ps_ok].copy() if ps_min > 0 else feats
    elif args.mode == "tail":
        eligible = feats[tail_ok & ps_ok].copy()
    elif args.mode == "utility":
        eligible = feats[util_ok & ps_ok].copy()
    elif args.mode == "tail_utility":
        eligible = feats[tail_ok & util_ok & ps_ok].copy()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    picks = rank_pick_one_per_day(eligible, rank_by=args.rank_by)

    out_cols = ["Date", "Ticker", "p_tail", "p_success", "ret_score", "utility", "tau_pred"]
    extra_cols = [c for c in ["EV", "Close", "Volume"] if c in picks.columns]
    keep = [c for c in out_cols + extra_cols if c in picks.columns]
    picks_out = picks[keep].copy()

    picks_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
    picks_out.to_csv(picks_path, index=False)

    meta = {
        "tag": args.tag,
        "suffix": args.suffix,
        "mode": args.mode,
        "rank_by": args.rank_by,
        "tail_threshold": float(args.tail_threshold),
        "utility_quantile": float(args.utility_quantile),
        "lambda_tail": float(args.lambda_tail),
        "p_success_min": float(args.p_success_min),
        "tau_model_path": args.tau_model_path,
        "tau_scaler_path": args.tau_scaler_path,
        "tau_features": tau_cols,
        "universe_csv": args.universe_csv,
        "universe_size": int(len(universe)),
        "excluded": sorted(list(excludes)),
        "rows_features_after_filter": int(len(feats)),
        "rows_eligible": int(len(eligible)),
        "picks_days": int(picks_out["Date"].nunique()) if not picks_out.empty else 0,
        "picks_rows": int(len(picks_out)),
        "profit_target": float(args.profit_target),
        "max_days": int(args.max_days),
        "stop_level": float(args.stop_level),
        "max_extend_days": int(args.max_extend_days),
        "features_src": args.features_parq if Path(args.features_parq).exists() else args.features_csv,
    }
    meta_path = out_dir / f"picks_meta_{args.tag}_gate_{args.suffix}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"[DONE] wrote picks: {picks_path} rows={len(picks_out)} days={meta['picks_days']}")
    print(f"[DONE] wrote meta : {meta_path}")
    print(f"[INFO] mode={args.mode} tail_max={args.tail_threshold} u_q={args.utility_quantile} rank_by={args.rank_by} lambda_tail={args.lambda_tail} p_success_min={args.p_success_min}")
    print(f"[INFO] universe_only rows: {before} -> {after} (excluded {sorted(list(excludes))})")
    if picks_out.empty:
        print("[WARN] picks_out is empty. Gate may be too strict for this period.")
    else:
        bad = set(picks_out["Ticker"].astype(str).str.upper().unique().tolist()) & excludes
        if bad:
            raise RuntimeError(f"[BUG] excluded tickers still present in picks: {sorted(list(bad))}")
    print("=" * 60)


if __name__ == "__main__":
    main()