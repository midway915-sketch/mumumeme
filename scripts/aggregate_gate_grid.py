# scripts/aggregate_gate_grid.py
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import pandas as pd
import numpy as np


def to_float(x, default: float = 0.0) -> float:
    """
    Robust scalar converter.
    Accepts scalar, Series, list-like; returns float.
    """
    try:
        if x is None:
            return float(default)
        if isinstance(x, (int, float, np.number)):
            v = float(x)
            return v if np.isfinite(v) else float(default)
        if isinstance(x, pd.Series):
            v = pd.to_numeric(x, errors="coerce").dropna()
            if len(v) == 0:
                return float(default)
            vv = float(v.iloc[-1])
            return vv if np.isfinite(vv) else float(default)
        # list/array
        s = pd.Series(x)
        v = pd.to_numeric(s, errors="coerce").dropna()
        if len(v) == 0:
            return float(default)
        vv = float(v.iloc[-1])
        return vv if np.isfinite(vv) else float(default)
    except Exception:
        return float(default)


def read_csv_safe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate gate_summary_*.csv into one table + top selection.")
    ap.add_argument("--signals-dir", type=str, default="data/signals")
    ap.add_argument("--out-aggregate", type=str, default="data/signals/gate_grid_aggregate.csv")
    ap.add_argument("--out-top", type=str, default="data/signals/gate_grid_top_by_recent10y.csv")
    ap.add_argument("--pattern", type=str, default="gate_summary_*.csv")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--penalty-alpha", type=float, default=0.0, help="return_penalty = alpha * leverage_pct")
    args = ap.parse_args()

    sig = Path(args.signals_dir)
    paths = sorted(glob.glob(str(sig / args.pattern)))
    if not paths:
        raise SystemExit(f"[ERROR] no {args.pattern} found in {sig}")

    rows = []
    for p in paths:
        try:
            df = read_csv_safe(p)
            if df.empty:
                continue
            r = df.iloc[0].to_dict()
            r["_path"] = p
            rows.append(r)
        except Exception as e:
            print(f"[WARN] failed reading {p}: {e}")

    if not rows:
        raise SystemExit("[ERROR] all gate_summary files unreadable/empty")

    out = pd.DataFrame(rows)

    # --- normalize expected columns (if missing, create) ---
    for c in [
        "Recent10Y_SeedMultiple",
        "SeedMultiple",
        "MaxExtendDaysObserved",
        "MaxHoldingDaysObserved",
        "CycleCount",
        "SuccessRate",
        "MaxLeveragePct",
        "Max_LeveragePct_Closed",
        "Max_Extend_Over_MaxDays",
    ]:
        if c not in out.columns:
            out[c] = np.nan

    # Prefer new names from summarize_sim_trades.py if present
    # We'll compute "effective" leverage/extend columns for scoring
    lev_vals = []
    ext_vals = []
    succ_vals = []
    cyc_vals = []

    for _, row in out.iterrows():
        # leverage
        lev = to_float(row.get("MaxLeveragePct", np.nan), default=np.nan)
        if not np.isfinite(lev):
            lev = to_float(row.get("Max_LeveragePct_Closed", np.nan), default=0.0)
        lev_vals.append(lev if np.isfinite(lev) else 0.0)

        # extend over maxdays
        ext = to_float(row.get("MaxExtendDaysObserved", np.nan), default=np.nan)
        if not np.isfinite(ext):
            ext = to_float(row.get("Max_Extend_Over_MaxDays", np.nan), default=0.0)
        ext_vals.append(ext if np.isfinite(ext) else 0.0)

        # success rate
        succ = to_float(row.get("SuccessRate", np.nan), default=0.0)
        succ_vals.append(succ)

        # cycle count
        cyc = to_float(row.get("CycleCount", np.nan), default=0.0)
        cyc_vals.append(cyc)

    out["LeveragePct_eff"] = lev_vals
    out["ExtendOverMax_eff"] = ext_vals
    out["SuccessRate_eff"] = succ_vals
    out["CycleCount_eff"] = cyc_vals

    # --- scoring: prioritize recent 10y seed multiple, optionally penalize leverage ---
    out["Recent10Y_SeedMultiple_num"] = pd.to_numeric(out["Recent10Y_SeedMultiple"], errors="coerce")
    out["SeedMultiple_num"] = pd.to_numeric(out["SeedMultiple"], errors="coerce")

    # choose a primary metric (recent10y if available else overall)
    out["PrimarySeedMultiple"] = out["Recent10Y_SeedMultiple_num"].combine_first(out["SeedMultiple_num"])

    alpha = float(args.penalty_alpha)
    out["AdjustedScore"] = out["PrimarySeedMultiple"] - alpha * (out["LeveragePct_eff"] / 100.0)

    # save aggregate
    Path(args.out_aggregate).parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(["AdjustedScore", "PrimarySeedMultiple"], ascending=False).to_csv(args.out_aggregate, index=False)
    print(f"[DONE] wrote aggregate: {args.out_aggregate} rows={len(out)}")

    # top table
    top = out.sort_values(["AdjustedScore", "PrimarySeedMultiple"], ascending=False).head(int(args.topn)).copy()
    top.to_csv(args.out_top, index=False)
    print(f"[DONE] wrote top: {args.out_top} rows={len(top)}")


if __name__ == "__main__":
    main()