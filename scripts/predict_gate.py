#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# IO helpers
# ----------------------------
def read_table(parq_path: Path, csv_path: Path) -> pd.DataFrame:
    if parq_path.exists():
        return pd.read_parquet(parq_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing file: {parq_path} (or {csv_path})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def coerce_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return x.astype(float)


def parse_require_files(s: str) -> List[Path]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    return [Path(p) for p in parts]


def ensure_required_files_exist(require_files: List[Path]) -> None:
    missing = [str(p) for p in require_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


# ----------------------------
# Column normalization
# ----------------------------
def ensure_date_ticker_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'Date' and 'Ticker' exist as columns.
    Handles:
      - Date is an index level named date/datetime/timestamp/Date
      - column names date/datetime/timestamp -> Date
      - ticker/symbol -> Ticker
      - groupby/apply artefacts where Date becomes index level 'Date' or 'level_0'
    """
    out = df.copy()

    # If Date is in index name (exact or common variants), reset it into columns
    idx_name = (out.index.name or "")
    if "Date" not in out.columns and idx_name.lower() in ("date", "datetime", "timestamp"):
        out = out.reset_index()

    # If Date exists as an index level in a MultiIndex (common after groupby/apply), bring it back
    if "Date" not in out.columns and isinstance(out.index, pd.MultiIndex):
        names = [str(n) for n in out.index.names]
        if "Date" in names:
            out = out.reset_index(level="Date")
        elif "date" in [n.lower() for n in names if n is not None]:
            # find actual name
            for n in out.index.names:
                if n is not None and str(n).lower() == "date":
                    out = out.reset_index(level=n)
                    break
        elif "level_0" in names:
            # some pandas versions use level_0 for group key
            out = out.reset_index(level=0)

    # Normalize column names to Date/Ticker
    colmap = {c.lower(): c for c in out.columns}

    # Date candidates
    if "date" in colmap and "Date" not in out.columns:
        out = out.rename(columns={colmap["date"]: "Date"})
    if "datetime" in colmap and "Date" not in out.columns:
        out = out.rename(columns={colmap["datetime"]: "Date"})
    if "timestamp" in colmap and "Date" not in out.columns:
        out = out.rename(columns={colmap["timestamp"]: "Date"})
    # If reset_index produced "level_0" and it's actually dates, map to Date
    if "level_0" in out.columns and "Date" not in out.columns:
        out = out.rename(columns={"level_0": "Date"})

    # Ticker candidates
    if "ticker" in colmap and "Ticker" not in out.columns:
        out = out.rename(columns={colmap["ticker"]: "Ticker"})
    if "symbol" in colmap and "Ticker" not in out.columns:
        out = out.rename(columns={colmap["symbol"]: "Ticker"})

    if "Date" not in out.columns:
        raise KeyError(
            f"Date column missing. cols(head)={list(out.columns)[:30]} index.name={out.index.name} "
            f"index.names={getattr(out.index, 'names', None)}"
        )
    if "Ticker" not in out.columns:
        raise KeyError(
            f"Ticker column missing. cols(head)={list(out.columns)[:30]} index.name={out.index.name} "
            f"index.names={getattr(out.index, 'names', None)}"
        )

    return out


def apply_exclude_tickers(df: pd.DataFrame, exclude_csv: str) -> pd.DataFrame:
    ex = [t.strip().upper() for t in (exclude_csv or "").split(",") if t.strip()]
    if not ex:
        return df
    return df[~df["Ticker"].isin(ex)].copy()


# ----------------------------
# Features source selection
# ----------------------------
def pick_features_source(require_files: List[Path]) -> Tuple[Path, Path]:
    """
    Prefer features_scored if present in require_files.
    Otherwise default to scored path; caller will fallback to features_model if not found.
    Returns (parq_candidate, csv_candidate)
    """
    for p in require_files:
        name = p.name
        if "features_scored" in name and (name.endswith(".parquet") or name.endswith(".csv")):
            if name.endswith(".parquet"):
                return p, p.with_suffix(".csv")
            return p.with_suffix(".parquet"), p

    return Path("data/features/features_scored.parquet"), Path("data/features/features_scored.csv")


# ----------------------------
# Utility / gating / ranking
# ----------------------------
def build_utility(df: pd.DataFrame, lambda_tail: float) -> pd.Series:
    # utility = ret_score - lambda * p_tail
    ret = coerce_num(df, "ret_score", 0.0)
    p_tail = coerce_num(df, "p_tail", 0.0)
    return ret - float(lambda_tail) * p_tail


def _post_group_apply_fix(res: pd.DataFrame) -> pd.DataFrame:
    """
    After groupby/apply, some pandas versions move group key to index.
    This function guarantees Date becomes a column again.
    """
    if "Date" in res.columns:
        return res.reset_index(drop=True)

    # Try recovering from index level names
    if isinstance(res.index, pd.MultiIndex):
        names = [str(n) for n in res.index.names]
        if "Date" in names:
            res = res.reset_index(level="Date")
        elif "level_0" in names:
            res = res.reset_index(level=0)
        else:
            # generic: bring first level back
            res = res.reset_index(level=0)
    else:
        # single index: reset to a column (might become 'index')
        res = res.reset_index()

    # Normalize possible names
    if "level_0" in res.columns and "Date" not in res.columns:
        res = res.rename(columns={"level_0": "Date"})
    if "index" in res.columns and "Date" not in res.columns:
        # last resort: if index column looks like datetime, treat as Date
        res = res.rename(columns={"index": "Date"})

    # Ensure Date/Ticker columns exist now
    res = ensure_date_ticker_columns(res)
    return res.reset_index(drop=True)


def gate_filter(
    df: pd.DataFrame,
    mode: str,
    tail_threshold: float,
    utility_quantile: float,
) -> pd.DataFrame:
    mode = (mode or "none").strip().lower()
    d = ensure_date_ticker_columns(df)

    if mode == "none":
        return d.copy()

    if mode == "tail":
        return d[d["p_tail"] <= float(tail_threshold)].copy()

    if mode == "utility":
        q = float(utility_quantile)

        def _f(g: pd.DataFrame) -> pd.DataFrame:
            g = ensure_date_ticker_columns(g)
            if len(g) == 0:
                return g
            thr = g["utility"].quantile(q)
            return g[g["utility"] >= thr]

        res = d.groupby("Date", group_keys=True).apply(_f)
        return _post_group_apply_fix(res)

    if mode == "tail_utility":
        d2 = d[d["p_tail"] <= float(tail_threshold)].copy()
        q = float(utility_quantile)

        def _f(g: pd.DataFrame) -> pd.DataFrame:
            g = ensure_date_ticker_columns(g)
            if len(g) == 0:
                return g
            thr = g["utility"].quantile(q)
            return g[g["utility"] >= thr]

        res = d2.groupby("Date", group_keys=True).apply(_f)
        return _post_group_apply_fix(res)

    raise ValueError(f"Unknown mode: {mode} (expected none|tail|utility|tail_utility)")


def rank_topk_per_day(df: pd.DataFrame, rank_by: str, topk: int) -> pd.DataFrame:
    d = ensure_date_ticker_columns(df)

    if d.empty:
        return d

    rb = (rank_by or "utility").strip().lower()
    if rb == "utility":
        metric = d["utility"]
    elif rb == "ret_score":
        metric = d["ret_score"]
    elif rb == "p_success":
        metric = d["p_success"]
    else:
        raise ValueError(f"Unknown rank_by: {rank_by} (expected utility|ret_score|p_success)")

    d["_metric"] = metric
    d = d.sort_values(["Date", "_metric"], ascending=[True, False])

    picks = (
        d.groupby("Date", as_index=False, group_keys=False)
        .head(int(topk))
        .drop(columns=["_metric"])
        .reset_index(drop=True)
    )
    return picks


# ----------------------------
# main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Gate picks generator (reads features_scored if available).")

    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--mode", type=str, required=True, help="none|tail|utility|tail_utility")
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--suffix", type=str, required=True)

    ap.add_argument("--tail-threshold", type=float, required=True)
    ap.add_argument("--utility-quantile", type=float, required=True)
    ap.add_argument("--rank-by", type=str, required=True, help="utility|ret_score|p_success")
    ap.add_argument("--lambda-tail", type=float, required=True)

    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--ps-min", type=float, required=True)

    ap.add_argument("--exclude-tickers", type=str, default="")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--require-files", type=str, required=True)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    require_files = parse_require_files(args.require_files)
    ensure_required_files_exist(require_files)

    # decide which features file to read (prefer scored)
    f_parq, f_csv = pick_features_source(require_files)

    # fallback: if scored not found, try features_model
    if (not f_parq.exists()) and (not f_csv.exists()):
        f_parq = Path("data/features/features_model.parquet")
        f_csv = Path("data/features/features_model.csv")

    feats = read_table(f_parq, f_csv).copy()
    features_src = str(f_parq) if f_parq.exists() else str(f_csv)

    feats = ensure_date_ticker_columns(feats)

    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    feats["p_success"] = coerce_num(feats, "p_success", 0.0)
    feats["p_tail"] = coerce_num(feats, "p_tail", 0.0)
    feats["ret_score"] = coerce_num(feats, "ret_score", 0.0)

    feats = apply_exclude_tickers(feats, args.exclude_tickers)

    # base gate: p_success min
    feats = feats[feats["p_success"] >= float(args.ps_min)].copy()

    # utility column
    feats["utility"] = build_utility(feats, float(args.lambda_tail))

    # mode gate
    gated = gate_filter(
        feats,
        mode=args.mode,
        tail_threshold=float(args.tail_threshold),
        utility_quantile=float(args.utility_quantile),
    )

    # rank + topk per day
    picks = rank_topk_per_day(gated, rank_by=args.rank_by, topk=int(args.topk))

    # output paths MUST match run_grid_workflow.sh
    picks_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
    meta_path = out_dir / f"picks_meta_{args.tag}_gate_{args.suffix}.json"

    keep_cols = [c for c in ["Date", "Ticker", "p_success", "p_tail", "ret_score", "utility"] if c in picks.columns]
    picks_out = picks[keep_cols].copy()

    picks_out.to_csv(picks_path, index=False)

    meta = {
        "tag": args.tag,
        "suffix": args.suffix,
        "mode": args.mode,
        "rank_by": args.rank_by,
        "topk": int(args.topk),
        "ps_min": float(args.ps_min),
        "tail_threshold": float(args.tail_threshold),
        "utility_quantile": float(args.utility_quantile),
        "lambda_tail": float(args.lambda_tail),
        "exclude_tickers": args.exclude_tickers,
        "features_src": features_src,
        "rows_in_features": int(len(feats)),
        "rows_after_gate": int(len(gated)),
        "rows_picks": int(len(picks_out)),
        "profit_target": float(args.profit_target),
        "max_days": int(args.max_days),
        "stop_level": float(args.stop_level),
        "max_extend_days": int(args.max_extend_days),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] wrote: {picks_path} rows={len(picks_out)}")
    print(f"[DONE] wrote: {meta_path}")


if __name__ == "__main__":
    main()