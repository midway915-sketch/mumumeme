# scripts/predict_gate.py
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


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

    tickers = uni["Ticker"].astype(str).str.upper().str.strip().dropna().unique().tolist()
    return tickers


def require_files(spec: str) -> None:
    if not spec:
        return
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    missing = [p for p in parts if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def force_columns_from_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    If Date/Ticker are in index (including MultiIndex), reset_index().
    """
    out = df
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    else:
        if getattr(out.index, "name", None) in ("Date", "Ticker"):
            out = out.reset_index()
    return out


def canonicalize_date_ticker_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee columns 'Date' and 'Ticker' exist by:
    - resetting index if needed
    - renaming common variants (date/Datetime/index/level_0 etc.)
    - renaming ticker variants (ticker/symbol/level_1 etc.)
    """
    out = force_columns_from_index(df).copy()
    cols = list(out.columns)

    # ---- Date
    if "Date" not in out.columns:
        candidates = ["date", "Datetime", "datetime", "DATE", "index", "level_0"]
        found = None
        for c in candidates:
            if c in out.columns:
                found = c
                break
        if found is not None:
            out = out.rename(columns={found: "Date"})

    # ---- Ticker
    if "Ticker" not in out.columns:
        candidates = ["ticker", "TICKER", "Symbol", "symbol", "SYMBOL", "level_1"]
        found = None
        for c in candidates:
            if c in out.columns:
                found = c
                break
        if found is not None:
            out = out.rename(columns={found: "Ticker"})

    # If still missing, die with rich debug
    if "Date" not in out.columns or "Ticker" not in out.columns:
        raise ValueError(
            "[predict_gate] Cannot find required columns Date/Ticker.\n"
            f"Columns={list(out.columns)[:80]}\n"
            f"IndexType={type(out.index)} IndexName={getattr(out.index,'name',None)}"
        )

    return out


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

    out = df.copy()
    out["_rank"] = rank_col
    out["_ev"] = ev
    out["_close"] = close

    out = out.sort_values(["Date", "_rank", "_ev", "_close"], ascending=[True, False, False, False])
    out = out.drop_duplicates(["Date"], keep="first").drop(columns=["_rank", "_ev", "_close"])
    return out


def rank_pick_topk_per_day(df: pd.DataFrame, rank_by: str, topk: int) -> pd.DataFrame:
    if df.empty:
        return df
    if topk <= 0:
        return df.iloc[0:0].copy()
    if rank_by not in ("utility", "ret_score", "p_success"):
        raise ValueError(f"rank_by must be one of utility|ret_score|p_success (got {rank_by})")
    if "Date" not in df.columns:
        raise ValueError(f"[predict_gate] rank_pick_topk_per_day requires Date column. cols={list(df.columns)[:50]}")

    ev = coerce_num(df, "EV", 0.0) if "EV" in df.columns else pd.Series([0.0] * len(df), index=df.index)
    close = coerce_num(df, "Close", 0.0) if "Close" in df.columns else pd.Series([0.0] * len(df), index=df.index)
    rank_col = coerce_num(df, rank_by, 0.0)

    out = df.copy()
    out["_rank"] = rank_col
    out["_ev"] = ev
    out["_close"] = close

    out = out.sort_values(["Date", "_rank", "_ev", "_close"], ascending=[True, False, False, False])
    out["RankInDay"] = out.groupby("Date").cumcount() + 1
    out = out[out["RankInDay"] <= int(topk)].drop(columns=["_rank", "_ev", "_close"])
    return out


def parse_weights(s: str, k: int) -> list[float]:
    if k <= 0:
        return []
    parts = [x.strip() for x in (s or "").split(",") if x.strip()]
    if len(parts) != k:
        return [1.0 / k] * k
    try:
        w = [float(x) for x in parts]
        wsum = sum(w)
        if wsum <= 0:
            return [1.0 / k] * k
        return [x / wsum for x in w]
    except Exception:
        return [1.0 / k] * k


# -------------------------
# main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Predict gate picks (Top-1 or Top-K per day) with universe-only candidates.")
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
    ap.add_argument("--exclude-tickers", default="SPY,^VIX", type=str)

    ap.add_argument("--tail-threshold", required=True, type=float)
    ap.add_argument("--utility-quantile", required=True, type=float)
    ap.add_argument("--rank-by", required=True, choices=["utility", "ret_score", "p_success"])
    ap.add_argument("--lambda-tail", required=True, type=float)

    ap.add_argument("--topk", default=1, type=int)
    ap.add_argument("--topk-weights", default="", type=str)

    ap.add_argument("--require-files", default="", type=str)

    args = ap.parse_args()

    if args.require_files:
        require_files(args.require_files)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- load + canonicalize features
    feats = read_table(args.features_parq, args.features_csv)
    feats = canonicalize_date_ticker_columns(feats)

    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # ---------- universe filter
    universe = set(read_universe(args.universe_csv))
    excludes = set([t.strip().upper() for t in args.exclude_tickers.split(",") if t.strip()])

    before = len(feats)
    feats = feats[feats["Ticker"].isin(universe)].copy()
    feats = feats[~feats["Ticker"].isin(excludes)].copy()
    after = len(feats)

    if after == 0:
        raise RuntimeError(
            f"No rows left after universe filter. before={before} after={after}. "
            f"Check universe.csv and features_model tickers."
        )

    # ---------- numeric columns
    feats["p_tail"] = coerce_num(feats, "p_tail", 0.0)
    feats["p_success"] = coerce_num(feats, "p_success", 0.0)
    feats["ret_score"] = coerce_num(feats, "ret_score", 0.0)
    feats["utility"] = build_utility(feats, lambda_tail=float(args.lambda_tail))

    # ---------- gate
    tail_ok = feats["p_tail"] <= float(args.tail_threshold)

    q = float(args.utility_quantile)
    if not (0.0 <= q <= 1.0):
        raise ValueError("--utility-quantile must be in [0,1]")

    util_cut = feats.groupby("Date")["utility"].transform(lambda s: float(s.quantile(q)) if len(s) else np.nan)
    util_ok = (feats["utility"] >= util_cut).fillna(False)

    if args.mode == "none":
        eligible = feats
    elif args.mode == "tail":
        eligible = feats[tail_ok].copy()
    elif args.mode == "utility":
        eligible = feats[util_ok].copy()
    elif args.mode == "tail_utility":
        eligible = feats[tail_ok & util_ok].copy()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # ---------- pick
    topk = int(args.topk)
    if topk <= 1:
        picks = rank_pick_one_per_day(eligible, rank_by=args.rank_by)
    else:
        picks = rank_pick_topk_per_day(eligible, rank_by=args.rank_by, topk=topk)

    # ---------- output
    if picks.empty:
        picks_out = picks.copy()
    else:
        picks_out = picks.copy()
        if topk <= 1:
            picks_out["RankInDay"] = 1
            picks_out["Weight"] = 1.0
        else:
            weights = parse_weights(args.topk_weights, k=topk)
            if "RankInDay" not in picks_out.columns:
                picks_out["RankInDay"] = picks_out.groupby("Date").cumcount() + 1
            picks_out["Weight"] = picks_out["RankInDay"].apply(
                lambda r: float(weights[int(r) - 1]) if 1 <= int(r) <= len(weights) else 0.0
            )

    keep = ["Date", "Ticker", "RankInDay", "Weight", "p_tail", "p_success", "ret_score", "utility"]
    keep += [c for c in ["EV", "Close", "Volume"] if c in picks_out.columns]
    picks_out = picks_out[[c for c in keep if c in picks_out.columns]].copy()

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
        "topk": int(args.topk),
        "topk_weights": args.topk_weights,
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
    print(f"[INFO] mode={args.mode} tail_max={args.tail_threshold} u_q={args.utility_quantile} rank_by={args.rank_by} lambda_tail={args.lambda_tail} topk={args.topk}")
    print(f"[INFO] universe_only rows: {before} -> {after} (excluded {sorted(list(excludes))})")

    if not picks_out.empty:
        bad = set(picks_out["Ticker"].astype(str).str.upper().unique().tolist()) & excludes
        if bad:
            raise RuntimeError(f"[BUG] excluded tickers still present in picks: {sorted(list(bad))}")

    if picks_out.empty:
        print("[WARN] picks_out is empty. Gate may be too strict.")
    print("=" * 60)


if __name__ == "__main__":
    main()