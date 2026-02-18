# scripts/predict_gate.py
from __future__ import annotations

import argparse
from pathlib import Path
import json
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

    tickers = uni["Ticker"].astype(str).str.upper().str.strip().dropna().unique().tolist()
    return tickers


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"features_scored missing required columns: {missing}")


def coerce_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def build_utility(df: pd.DataFrame, lambda_tail: float) -> pd.Series:
    ret_score = coerce_num(df, "ret_score", 0.0)
    p_tail = coerce_num(df, "p_tail", 0.0)
    return (ret_score - float(lambda_tail) * p_tail).astype(float)


def rank_pick_topk_per_day(df: pd.DataFrame, rank_by: str, topk: int) -> pd.DataFrame:
    if df.empty:
        return df
    if rank_by not in ("utility", "ret_score", "p_success"):
        raise ValueError(f"rank_by must be one of utility|ret_score|p_success (got {rank_by})")

    ev = coerce_num(df, "EV", 0.0) if "EV" in df.columns else pd.Series([0.0] * len(df), index=df.index)
    close = coerce_num(df, "Close", 0.0) if "Close" in df.columns else pd.Series([0.0] * len(df), index=df.index)

    rank_col = coerce_num(df, rank_by, 0.0)
    x = df.copy()
    x["_rank"] = rank_col
    x["_ev"] = ev
    x["_close"] = close

    x = x.sort_values(["Date", "_rank", "_ev", "_close"], ascending=[True, False, False, False])
    x["RankIdx"] = x.groupby("Date").cumcount() + 1
    x = x.loc[x["RankIdx"] <= int(topk)].copy()
    x = x.drop(columns=["_rank", "_ev", "_close"])
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict gate picks (TopK per day) with universe-only candidates.")
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--mode", required=True, choices=["none", "tail", "utility", "tail_utility"])
    ap.add_argument("--tag", required=True, type=str)
    ap.add_argument("--suffix", required=True, type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    ap.add_argument("--features-parq", default="data/features/features_scored.parquet", type=str)
    ap.add_argument("--features-csv", default="data/features/features_scored.csv", type=str)

    ap.add_argument("--universe-csv", default="data/universe.csv", type=str)
    ap.add_argument("--exclude-tickers", default="SPY,^VIX", type=str)

    ap.add_argument("--tail-threshold", required=True, type=float)
    ap.add_argument("--utility-quantile", required=True, type=float)
    ap.add_argument("--rank-by", required=True, choices=["utility", "ret_score", "p_success"])
    ap.add_argument("--lambda-tail", required=True, type=float)

    ap.add_argument("--ps-min", default=0.0, type=float, help="minimum p_success to allow entry (0 disables filter)")
    ap.add_argument("--topk", default=1, type=int, help="number of picks per day to output (1 or 2)")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = read_table(args.features_parq, args.features_cvs if hasattr(args, "features_cvs") else args.features_csv).copy()
    ensure_cols(feats, ["Date", "Ticker", "p_tail", "p_success", "ret_score"])

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
        raise RuntimeError(f"No rows after universe filter. before={before} after={after}")

    feats["p_tail"] = coerce_num(feats, "p_tail", 0.0)
    feats["p_success"] = coerce_num(feats, "p_success", 0.0)
    feats["ret_score"] = coerce_num(feats, "ret_score", 0.0)
    feats["utility"] = build_utility(feats, lambda_tail=float(args.lambda_tail))

    tail_ok = feats["p_tail"] <= float(args.tail_threshold)

    q = float(args.utility_quantile)
    util_cut = feats.groupby("Date")["utility"].transform(lambda s: float(s.quantile(q)) if len(s) else np.nan)
    util_ok = (feats["utility"] >= util_cut).fillna(False)

    ps_min = float(args.ps_min)
    ps_ok = feats["p_success"] >= ps_min if ps_min > 0 else pd.Series([True] * len(feats), index=feats.index)

    if args.mode == "none":
        eligible = feats[ps_ok].copy()
    elif args.mode == "tail":
        eligible = feats[tail_ok & ps_ok].copy()
    elif args.mode == "utility":
        eligible = feats[util_ok & ps_ok].copy()
    elif args.mode == "tail_utility":
        eligible = feats[tail_ok & util_ok & ps_ok].copy()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    topk = int(args.topk)
    if topk < 1 or topk > 2:
        raise ValueError("--topk must be 1 or 2")

    picks = rank_pick_topk_per_day(eligible, rank_by=args.rank_by, topk=topk)

    keep = [c for c in ["Date", "Ticker", "RankIdx", "p_tail", "p_success", "ret_score", "utility", "EV", "Close", "Volume"] if c in picks.columns]
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
        "ps_min": float(args.ps_min),
        "topk": int(args.topk),
        "universe_size": int(len(universe)),
        "excluded": sorted(list(excludes)),
        "rows_features_after_filter": int(len(feats)),
        "rows_eligible": int(len(eligible)),
        "picks_days": int(picks_out["Date"].nunique()) if not picks_out.empty else 0,
        "picks_rows": int(len(picks_out)),
    }
    meta_path = out_dir / f"picks_meta_{args.tag}_gate_{args.suffix}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"[DONE] wrote picks: {picks_path} rows={len(picks_out)}")
    print(f"[DONE] wrote meta : {meta_path}")
    print(f"[INFO] universe_only rows: {before} -> {after} (excluded {sorted(list(excludes))})")
    print("=" * 60)


if __name__ == "__main__":
    main()