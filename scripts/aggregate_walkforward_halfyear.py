#!/usr/bin/env python3
# scripts/aggregate_walkforward_halfyear.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing curve file: {parq} (or {csv})")


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(dd.min())  # negative


def _compute_window_metrics(curve: pd.DataFrame, date_from: str, date_to: str) -> dict:
    c = curve.copy()
    c["Date"] = pd.to_datetime(c["Date"], errors="coerce")
    c = c.dropna(subset=["Date"])

    start = pd.to_datetime(date_from)
    end = pd.to_datetime(date_to)

    w = c[(c["Date"] >= start) & (c["Date"] <= end)].copy()
    if w.empty:
        return {"empty": True, "reason": "no_curve_rows_in_window"}

    # prefer Equity col, fallback to SeedEquity if needed
    if "Equity" in w.columns:
        eq = pd.to_numeric(w["Equity"], errors="coerce").to_numpy(dtype=float)
    elif "SeedEquity" in w.columns:
        eq = pd.to_numeric(w["SeedEquity"], errors="coerce").to_numpy(dtype=float)
    else:
        return {"empty": True, "reason": "no_equity_column"}

    eq = eq[~np.isnan(eq)]
    if eq.size == 0:
        return {"empty": True, "reason": "equity_all_nan"}

    start_eq = float(eq[0])
    end_eq = float(eq[-1])
    seed_mult = (end_eq / start_eq) if start_eq != 0 else float("nan")
    max_dd = _max_drawdown(eq)

    return {
        "empty": False,
        "start_equity": start_eq,
        "end_equity": end_eq,
        "seed_multiple_window": float(seed_mult),
        "max_dd_window": float(max_dd),
        "days": int((w["Date"].iloc[-1] - w["Date"].iloc[0]).days),
        "n_curve_rows": int(len(w)),
    }


def _parse_tag_suffix_cap_from_curve_name(name: str) -> tuple[str, str, str] | None:
    # sim_engine_curve_{TAG}_gate_{SUFFIX}_cap{cap}.(parquet|csv)
    m = re.match(r"^sim_engine_curve_(.+?)_gate_(.+?)_cap([a-z0-9]+)\.(parquet|csv)$", name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/signals/walkforward", help="walkforward root dir")
    ap.add_argument("--out", type=str, default="data/signals/walkforward/_summary_walkforward.csv", help="output csv")
    ap.add_argument("--glob", type=str, default="sim_engine_curve_*_cap*.parquet", help="curve glob (parquet)")
    ap.add_argument("--glob-csv", type=str, default="sim_engine_curve_*_cap*.csv", help="curve glob (csv)")
    args = ap.parse_args()

    root = Path(args.root)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    # period folders like 2018H1, 2018H2...
    periods = sorted([p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}H[12]", p.name)])
    if not periods:
        raise RuntimeError(f"No period folders under: {root}")

    for period_dir in periods:
        period = period_dir.name

        # debug json carries date_from/date_to and params
        debug_files = sorted(period_dir.glob("picks_*_gate_*.debug.json"))

        # build index: (tag, suffix) -> debug dict
        dbg_map: dict[tuple[str, str], dict] = {}
        for jf in debug_files:
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            tag = d.get("tag")
            suffix = d.get("suffix")
            if tag and suffix:
                dbg_map[(tag, suffix)] = d

        # collect curve files (parquet + csv)
        curve_files = list(period_dir.glob(args.glob)) + list(period_dir.glob(args.glob_csv))
        for cf in sorted(curve_files):
            parsed = _parse_tag_suffix_cap_from_curve_name(cf.name)
            if not parsed:
                continue
            tag, suffix, cap = parsed

            dbg = dbg_map.get((tag, suffix), {})
            date_from = dbg.get("date_from")
            date_to = dbg.get("date_to")

            # if no debug info, skip (we need window)
            if not date_from or not date_to:
                rows.append({
                    "period": period,
                    "tag": tag,
                    "suffix": suffix,
                    "cap": cap,
                    "empty": True,
                    "reason": "missing_debug_date_from_to",
                    "curve_file": str(cf),
                })
                continue

            # load curve (parq/csv)
            if cf.suffix.lower() == ".parquet":
                curve = pd.read_parquet(cf)
            else:
                curve = pd.read_csv(cf)

            met = _compute_window_metrics(curve, date_from, date_to)

            row = {
                "period": period,
                "tag": tag,
                "suffix": suffix,
                "cap": cap,
                "date_from": date_from,
                "date_to": date_to,
                "curve_file": str(cf),
                "ps_min": dbg.get("ps_min"),
                "tail_threshold": dbg.get("tail_threshold"),
                "utility_quantile": dbg.get("utility_quantile"),
                "lambda_tail": dbg.get("lambda_tail"),
                "topk": dbg.get("topk"),
                "badexit_max": dbg.get("badexit_max"),
            }
            row.update(met)
            rows.append(row)

        # If period has no curve files at all, still report it (common when picks=0 -> sim skipped)
        if not curve_files:
            rows.append({
                "period": period,
                "tag": None,
                "suffix": None,
                "cap": None,
                "empty": True,
                "reason": "no_curve_files_in_period",
                "curve_file": None,
            })

    df = pd.DataFrame(rows)

    # sort: non-empty first, then best seed_multiple desc
    if "seed_multiple_window" in df.columns:
        df["_seed"] = pd.to_numeric(df["seed_multiple_window"], errors="coerce")
    else:
        df["_seed"] = np.nan

    df = df.sort_values(
        by=["period", "empty", "_seed"],
        ascending=[True, True, False],
        na_position="last",
    ).drop(columns=["_seed"])

    df.to_csv(outp, index=False, encoding="utf-8-sig")
    print(f"[DONE] wrote {outp} rows={len(df)}")
    print(df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()