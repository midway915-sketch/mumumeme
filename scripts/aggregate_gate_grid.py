# scripts/aggregate_gate_grid.py
from __future__ import annotations

import argparse
from pathlib import Path
import glob
import numpy as np
import pandas as pd


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float(default)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-dir", default="data/signals", type=str)
    ap.add_argument("--pattern", default="gate_summary_*.csv", type=str)
    ap.add_argument("--out-aggregate", default="data/signals/gate_grid_aggregate.csv", type=str)
    ap.add_argument("--out-top", default="data/signals/gate_grid_top_by_recent10y.csv", type=str)
    ap.add_argument("--topn", default=30, type=int)

    ap.add_argument("--lev-cap", default=1.0, type=float, help="Filter: Max_LeveragePct_Closed <= lev-cap (1.0=100%).")
    ap.add_argument("--prefer-adjusted", default="true", type=str, help="true|false: rank by Adj_Recent10Y_SeedMultiple if present.")
    args = ap.parse_args()

    sig = Path(args.signals_dir)
    paths = sorted(glob.glob(str(sig / "**" / args.pattern), recursive=True))
    if not paths:
        raise SystemExit(f"[ERROR] no {args.pattern} found under {sig}")

    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if len(df) == 0:
                continue
            df["__source__"] = str(p)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")

    if not dfs:
        raise SystemExit("[ERROR] summary csv files found but none readable.")

    out = pd.concat(dfs, ignore_index=True)

    # normalize numeric cols
    num_cols = [
        "SeedMultiple", "Recent10Y_SeedMultiple",
        "Adj_SeedMultiple", "Adj_Recent10Y_SeedMultiple",
        "MaxHoldingDaysObserved", "Max_Extend_Over_MaxDays",
        "SuccessRate", "CycleCount", "ClosedCycleCount",
        "Max_LeveragePct_Closed",
        "ProfitTarget", "MaxHoldingDays", "StopLevel",
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # leverage cap filter
    lev_cap = float(args.lev_cap)
    if "Max_LeveragePct_Closed" in out.columns:
        lev = out["Max_LeveragePct_Closed"].fillna(0.0)
        out = out.loc[lev <= lev_cap].copy()
    else:
        # if missing, keep all
        pass

    if out.empty:
        raise SystemExit(f"[ERROR] After lev-cap filter <= {lev_cap}, no rows remain.")

    prefer_adj = str(args.prefer_adjusted).lower() == "true"

    rank_col = None
    if prefer_adj and "Adj_Recent10Y_SeedMultiple" in out.columns and out["Adj_Recent10Y_SeedMultiple"].notna().any():
        rank_col = "Adj_Recent10Y_SeedMultiple"
    elif "Recent10Y_SeedMultiple" in out.columns and out["Recent10Y_SeedMultiple"].notna().any():
        rank_col = "Recent10Y_SeedMultiple"
    elif prefer_adj and "Adj_SeedMultiple" in out.columns and out["Adj_SeedMultiple"].notna().any():
        rank_col = "Adj_SeedMultiple"
    elif "SeedMultiple" in out.columns and out["SeedMultiple"].notna().any():
        rank_col = "SeedMultiple"
    else:
        # fallback: success then seed
        if "SuccessRate" in out.columns:
            rank_col = "SuccessRate"

    out["RankBy"] = rank_col if rank_col else ""

    # sort
    if rank_col and rank_col in out.columns:
        out_sorted = out.sort_values([rank_col], ascending=False).reset_index(drop=True)
    else:
        out_sorted = out.reset_index(drop=True)

    Path(args.out_aggregate).parent.mkdir(parents=True, exist_ok=True)
    out_sorted.to_csv(args.out_aggregate, index=False)

    top = out_sorted.head(int(args.topn)).copy()
    top.to_csv(args.out_top, index=False)

    print(f"[DONE] aggregate: {args.out_aggregate} rows={len(out_sorted)} rank_by={rank_col}")
    print(f"[DONE] top      : {args.out_top} rows={len(top)}")


if __name__ == "__main__":
    main()