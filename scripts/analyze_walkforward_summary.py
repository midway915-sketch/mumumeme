#!/usr/bin/env python3
# scripts/analyze_walkforward_summary.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to _summary_walkforward.csv")
    ap.add_argument("--out", default="_analysis_walkforward.csv", help="Output analysis CSV")
    ap.add_argument("--topn", type=int, default=20, help="Top N configs to print by score")
    args = ap.parse_args()

    p = Path(args.summary)
    if not p.exists():
        raise FileNotFoundError(f"Missing summary file: {p}")

    df = pd.read_csv(p)

    if df.empty:
        raise RuntimeError("summary is empty")

    # ---- common columns (be tolerant to your aggregator's exact naming)
    col_period = _find_col(df, ["period", "WF_PERIOD"])
    col_tag = _find_col(df, ["tag", "TAG"])
    col_suffix = _find_col(df, ["suffix", "SUFFIX"])

    # performance-ish columns (names may differ depending on what you aggregated)
    col_seed = _find_col(df, ["SeedMultiple", "seed_multiple", "seed_mult", "seed"])
    col_maxdd = _find_col(df, ["MaxDD", "max_dd", "maxdd", "MaxDrawdown"])
    col_trades = _find_col(df, ["n_trades", "trades", "num_trades", "NTrades"])
    col_psmin = _find_col(df, ["ps_min", "PS_MIN", "psmin"])

    # numeric conversions if present
    _to_numeric(df, [c for c in [col_seed, col_maxdd, col_trades, col_psmin] if c])

    # ---- sanity / warnings
    warnings: list[str] = []

    # (A) ps_min meaningfulness check
    # If p_success column was filled with 0.0 (missing models), ps_min filter is meaningless.
    # We detect that indirectly by: (1) ps_min exists, (2) many rows had ps_min>0 but still empty picks,
    # or user explicitly observed p_success filled 0.
    # Here: we just warn when ps_min exists AND (seed/maxdd/trades) are all NaN a lot,
    # because it often indicates "no trades" / "empty picks".
    if col_psmin:
        frac_nan_seed = float(df[col_seed].isna().mean()) if col_seed else 1.0
        frac_nan_trades = float(df[col_trades].isna().mean()) if col_trades else 1.0
        if frac_nan_seed > 0.5 and frac_nan_trades > 0.5:
            warnings.append(
                "Many rows have no trades/metrics. If p_success was filled as 0.0 (missing models), "
                "ps_min filtering can zero-out picks unless you set ps_min=0.00 or train models."
            )

    # ---- score definition
    # We want a single sortable score even if your columns vary.
    # If SeedMultiple exists and MaxDD exists: score = SeedMultiple / (1 + abs(MaxDD))
    # Else: fallback to SeedMultiple, else fallback to -MaxDD.
    score = None
    if col_seed and col_maxdd:
        score = df[col_seed] / (1.0 + df[col_maxdd].abs())
    elif col_seed:
        score = df[col_seed]
    elif col_maxdd:
        score = -df[col_maxdd].abs()
    else:
        # last resort: trade count
        if col_trades:
            score = df[col_trades]
        else:
            score = pd.Series([0.0] * len(df))

    df["_score"] = pd.to_numeric(score, errors="coerce")

    # group key: config identity (suffix usually uniquely encodes params)
    key_cols = [c for c in [col_suffix] if c]
    if not key_cols:
        # fallback: use all columns except period/tag
        key_cols = [c for c in df.columns if c not in set([col_period, col_tag]) and not c.startswith("_")]

    # aggregate across half-years per config
    agg = df.groupby(key_cols, dropna=False).agg(
        periods=("__dummy__", "size") if "__dummy__" in df.columns else (df.columns[0], "size"),
        score_mean=("_score", "mean"),
        score_med=("_score", "median"),
    )
    agg = agg.reset_index()

    if col_seed and col_seed in df.columns:
        agg["seed_mean"] = df.groupby(key_cols)[col_seed].mean().values
        agg["seed_med"] = df.groupby(key_cols)[col_seed].median().values
    if col_maxdd and col_maxdd in df.columns:
        agg["maxdd_mean"] = df.groupby(key_cols)[col_maxdd].mean().values
        agg["maxdd_worst"] = df.groupby(key_cols)[col_maxdd].max().values  # max dd (less negative if signed?) depends on your convention
    if col_trades and col_trades in df.columns:
        agg["trades_mean"] = df.groupby(key_cols)[col_trades].mean().values
        agg["trades_sum"] = df.groupby(key_cols)[col_trades].sum().values

    agg = agg.sort_values(["score_mean", "score_med"], ascending=False).reset_index(drop=True)

    out = Path(args.out)
    agg.to_csv(out, index=False)

    # ---- print quick report
    print("=" * 72)
    print("[ANALYZE] walkforward summary analysis")
    print(f"input : {p}")
    print(f"output: {out}")
    print(f"rows  : {len(df)} (raw), {len(agg)} (configs)")
    if warnings:
        print("-" * 72)
        for w in warnings:
            print(f"[WARN] {w}")

    print("-" * 72)
    topn = min(int(args.topn), len(agg))
    if topn <= 0:
        topn = 10
    show_cols = [c for c in agg.columns if c in set(key_cols + ["periods", "score_mean", "seed_mean", "maxdd_mean", "trades_mean", "trades_sum"])]
    if not show_cols:
        show_cols = agg.columns.tolist()[: min(12, len(agg.columns))]

    print(f"[TOP {topn}] by score_mean")
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(agg[show_cols].head(topn).to_string(index=False))
    print("=" * 72)


if __name__ == "__main__":
    # 작은 안전장치: pandas 옵션/버전 차이로 인한 경고는 무시
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)