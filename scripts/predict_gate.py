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


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"features_model missing required columns: {missing} (have={list(df.columns)[:50]})")


def require_files(spec: str) -> None:
    """
    spec example: "data/features/features_model.parquet,app/model.pkl"
    """
    if not spec:
        return
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    missing = [p for p in parts if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def force_columns_from_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the most common cause of pandas groupby KeyError:
    Date/Ticker ended up in index (including MultiIndex).
    This function tries to restore Date/Ticker as normal columns.
    """
    out = df

    # MultiIndex with Date/Ticker
    if isinstance(out.index, pd.MultiIndex):
        names = [n for n in out.index.names if n in ("Date", "Ticker")]
        if names:
            out = out.reset_index()

    # single index named Date/Ticker
    if "Date" not in out.columns and getattr(out.index, "name", None) == "Date":
        out = out.reset_index()
    if "Ticker" not in out.columns and getattr(out.index, "name", None) == "Ticker":
        out = out.reset_index()

    # sometimes index columns come back as "level_0"/etc. If Date exists but is non-datetime, we'll handle later.
    return out


# -------------------------
# scoring
# -------------------------
def coerce_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def build_utility(df: pd.DataFrame, lambda_tail: float) -> pd.Series:
    """
    Utility = ret_score - lambda_tail * p_tail
    """
    ret_score = coerce_num(df, "ret_score", 0.0)
    p_tail = coerce_num(df, "p_tail", 0.0)
    return (ret_score - float(lambda_tail) * p_tail).astype(float)


def rank_pick_one_per_day(df: pd.DataFrame, rank_by: str) -> pd.DataFrame:
    """
    df already filtered to eligible candidates.
    Returns one row per Date (top-1).
    """
    if df.empty:
        return df

    if rank_by not in ("utility", "ret_score", "p_success"):
        raise ValueError(f"rank_by must be one of utility|ret_score|p_success (got {rank_by})")

    # stable tie-breakers: rank -> EV -> Close
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
    """
    Returns up to topk rows per Date (ranked).
    Requires Date column.
    """
    if df.empty:
        return df
    if topk <= 0:
        return df.iloc[0:0].copy()
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
    out["RankInDay"] = out.groupby("Date").cumcount() + 1
    out = out[out["RankInDay"] <= int(topk)].drop(columns=["_rank", "_ev", "_close"])
    return out


def parse_weights(s: str, k: int) -> list[float]:
    """
    "0.7,0.3" -> [0.7,0.3]
    If empty or invalid -> equal weights.
    Normalize to sum=1.
    """
    if k <= 0:
        return []
    parts = [x.strip() for x in (s or "").split(",") if x.strip()]
    if len(parts) != k:
        w = [1.0 / k] * k
        return w
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
    ap.add_argument("--exclude-tickers", default="SPY,^VIX", type=str, help="comma-separated tickers to force-exclude")

    ap.add_argument("--tail-threshold", required=True, type=float)
    ap.add_argument("--utility-quantile", required=True, type=float)
    ap.add_argument("--rank-by", required=True, choices=["utility", "ret_score", "p_success"])
    ap.add_argument("--lambda-tail", required=True, type=float)

    # Top-K support (for Top-2 split comparison)
    ap.add_argument("--topk", default=1, type=int, help="number of picks per day (1=Top-1)")
    ap.add_argument("--topk-weights", default="", type=str, help="comma-separated weights for TopK (e.g. 0.7,0.3)")

    # optional: fail early if some files must exist
    ap.add_argument("--require-files", default="", type=str, help="comma-separated file paths that must exist")

    args = ap.parse_args()

    if args.require_files:
        require_files(args.require_files)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- load features
    feats = read_table(args.features_parq, args.features_csv).copy()
    feats = force_columns_from_index(feats)

    # now we can require Date/Ticker
    ensure_cols(feats, ["Date", "Ticker"])

    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # ---------- universe-only filter (prevents ^VIX/SPY leaking into candidates)
    universe = set(read_universe(args.universe_csv))
    excludes = set([t.strip().upper() for t in args.exclude_tickers.split(",") if t.strip()])

    before = len(feats)
    feats = feats[feats["Ticker"].isin(universe)].copy()
    feats = feats[~feats["Ticker"].isin(excludes)].copy()
    after = len(feats)

    if after == 0:
        raise RuntimeError(
            f"No rows left after universe filter. "
            f"Check universe.csv and features_model tickers. before={before} after={after}"
        )

    # ---------- compute utility + sanitize numeric columns
    feats["p_tail"] = coerce_num(feats, "p_tail", 0.0)
    feats["p_success"] = coerce_num(feats, "p_success", 0.0)
    feats["ret_score"] = coerce_num(feats, "ret_score", 0.0)
    feats["utility"] = build_utility(feats, lambda_tail=float(args.lambda_tail))

    # ---------- gate conditions
    tail_ok = feats["p_tail"] <= float(args.tail_threshold)

    q = float(args.utility_quantile)
    if not (0.0 <= q <= 1.0):
        raise ValueError("--utility-quantile must be in [0,1]")

    # per-date quantile cutoff
    # IMPORTANT: Date must be a column (force_columns_from_index handles this)
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

    # ---------- pick topk per day
    topk = int(args.topk)
    if topk <= 1:
        picks = rank_pick_one_per_day(eligible, rank_by=args.rank_by)
    else:
        picks = rank_pick_topk_per_day(eligible, rank_by=args.rank_by, topk=topk)

    # ---------- output
    if picks.empty:
        picks_out = picks.copy()
    else:
        weights = parse_weights(args.topk_weights, k=topk if topk > 1 else 1)
        picks_out = picks.copy()
        # attach weight per row if TopK
        if topk > 1:
            # RankInDay exists in rank_pick_topk_per_day
            if "RankInDay" not in picks_out.columns:
                picks_out["RankInDay"] = picks_out.groupby("Date").cumcount() + 1
            picks_out["Weight"] = picks_out["RankInDay"].apply(lambda r: float(weights[int(r)-1]) if 1 <= int(r) <= len(weights) else 0.0)
        else:
            picks_out["RankInDay"] = 1
            picks_out["Weight"] = 1.0

    # IMPORTANT: simulate engine expects Date/Ticker; we also keep Weight/RankInDay for Top-2 split
    base_cols = ["Date", "Ticker", "RankInDay", "Weight", "p_tail", "p_success", "ret_score", "utility"]
    extra_cols = [c for c in ["EV", "Close", "Volume"] if c in picks_out.columns]
    keep = [c for c in base_cols + extra_cols if c in picks_out.columns]
    picks_out = picks_out[keep].copy()

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

    # quick sanity: ensure no excluded leak
    if not picks_out.empty:
        bad = set(picks_out["Ticker"].astype(str).str.upper().unique().tolist()) & excludes
        if bad:
            raise RuntimeError(f"[BUG] excluded tickers still present in picks: {sorted(list(bad))}")

    if picks_out.empty:
        print("[WARN] picks_out is empty. Gate may be too strict for this period.")
    print("=" * 60)


if __name__ == "__main__":
    main()