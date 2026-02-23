#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_csv_list(s: str) -> list[str]:
    items = [x.strip() for x in str(s or "").split(",")]
    return [x for x in items if x]


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate predictor: produce daily picks CSV based on scored features.")

    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--mode", required=True, type=str)  # none|tail|utility|tail_utility
    ap.add_argument("--tail-threshold", required=True, type=float)
    ap.add_argument("--utility-quantile", required=True, type=float)
    ap.add_argument("--rank-by", required=True, type=str)  # utility|ret_score|p_success
    ap.add_argument("--lambda-tail", required=True, type=float)

    ap.add_argument("--topk", default=1, type=int)
    ap.add_argument("--ps-min", default=0.0, type=float)
    ap.add_argument("--badexit-max", default=1.0, type=float)

    ap.add_argument("--tag", required=True, type=str)
    ap.add_argument("--suffix", required=True, type=str)

    ap.add_argument("--exclude-tickers", default="", type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str, help="output directory for picks/summaries")
    ap.add_argument(
        "--require-files",
        default="",
        type=str,
        help="comma-separated paths that must exist before running (fail-fast).",
    )

    ap.add_argument("--features-path", default="", type=str)

    args = ap.parse_args()

    # fail-fast required files (workflow compatibility)
    reqs = parse_csv_list(getattr(args, "require_files", ""))
    missing = [p for p in reqs if p and (not Path(p).exists())]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    tag = str(args.tag).strip()
    suffix = str(args.suffix).strip()
    if not tag or not suffix:
        raise ValueError("--tag and --suffix are required and must be non-empty.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load scored features
    if args.features_path:
        p = Path(args.features_path)
        feats = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_parquet(p)
    else:
        # default scored features location
        p_parq = Path("data/features/features_scored.parquet")
        p_csv = Path("data/features/features_scored.csv")
        if p_parq.exists():
            feats = pd.read_parquet(p_parq)
        elif p_csv.exists():
            feats = pd.read_csv(p_csv)
        else:
            raise FileNotFoundError("Missing scored features: data/features/features_scored.(parquet|csv)")

    # basic columns
    required_cols = {"Date", "Ticker"}
    if not required_cols.issubset(set(feats.columns)):
        raise ValueError(f"features_scored must contain {sorted(required_cols)}. cols={list(feats.columns)[:50]}")

    # normalize
    feats = feats.copy()
    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).reset_index(drop=True)

    # exclude tickers
    exclude = set([x.upper().strip() for x in parse_csv_list(args.exclude_tickers)])
    if exclude:
        feats = feats.loc[~feats["Ticker"].isin(exclude)].copy()

    # ensure needed score columns exist (fill safe defaults)
    for c, default in [("p_success", 0.0), ("p_tail", 1.0), ("p_badexit", 0.0)]:
        if c not in feats.columns:
            feats[c] = float(default)
        feats[c] = (
            pd.to_numeric(feats[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(float(default))
            .astype(float)
        )

    # -------- gate logic (simple/robust baseline) --------
    # filter by thresholds
    feats = feats.loc[feats["p_success"] >= float(args.ps_min)].copy()
    feats = feats.loc[feats["p_badexit"] <= float(args.badexit_max)].copy()

    mode = str(args.mode).lower().strip()
    if mode not in ("none", "tail", "utility", "tail_utility"):
        raise ValueError(f"--mode must be one of none|tail|utility|tail_utility (got {args.mode})")

    tail_thr = float(args.tail_threshold)
    uq = float(args.utility_quantile)
    lam = float(args.lambda_tail)

    # utility score (always computed for safety)
    # - higher is better
    # - tail penalty: lam * max(0, p_tail - tail_thr)
    feats["utility"] = feats["p_success"] - lam * np.maximum(0.0, feats["p_tail"] - tail_thr)

    # apply mode-specific filters
    if mode in ("tail", "tail_utility"):
        feats = feats.loc[feats["p_tail"] <= tail_thr].copy()

    if mode in ("utility", "tail_utility"):
        # keep only top utility quantile per day
        def _topq(g: pd.DataFrame) -> pd.DataFrame:
            if len(g) <= 1:
                return g
            qv = float(g["utility"].quantile(uq))
            return g.loc[g["utility"] >= qv]

        feats = feats.groupby("Date", group_keys=False).apply(_topq).reset_index(drop=True)

    rank_by = str(args.rank_by).lower().strip()
    if rank_by not in ("utility", "ret_score", "p_success"):
        raise ValueError("--rank-by must be one of utility|ret_score|p_success")

    if rank_by not in feats.columns:
        # ret_score 없을 수도 있어서 안전하게 utility로 폴백
        if rank_by == "ret_score":
            rank_by = "utility"
        else:
            raise ValueError(f"rank column missing: {rank_by}")

    # pick topk per day
    topk = int(args.topk)
    if topk < 1:
        raise ValueError("--topk must be >= 1")

    feats = feats.sort_values(["Date", rank_by], ascending=[True, False]).reset_index(drop=True)
    picks = feats.groupby("Date", group_keys=False).head(topk).reset_index(drop=True)

    # output
    out_path = out_dir / f"picks_{tag}_gate_{suffix}.csv"
    picks[["Date", "Ticker", "p_success", "p_tail", "p_badexit", "utility"]].to_csv(out_path, index=False)
    print(f"[DONE] wrote picks: {out_path} rows={len(picks)}")


if __name__ == "__main__":
    main()