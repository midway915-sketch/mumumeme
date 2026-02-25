#!/usr/bin/env python3
# scripts/analyze_walkforward_summary.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _calc_cagr_from_mult_days(mult: float, days: float) -> float:
    if not np.isfinite(mult) or mult <= 0:
        return float("nan")
    if not np.isfinite(days) or days <= 0:
        return float("nan")
    years = days / 365.0
    if years <= 0:
        return float("nan")
    return float(mult ** (1.0 / years) - 1.0)


def _wavg(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    m = x.notna() & (w > 0)
    if not m.any():
        return float("nan")
    return float((x[m] * w[m]).sum() / w[m].sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--group-cols", default="suffix,cap", type=str, help="comma-separated group keys")
    args = ap.parse_args()

    summ = Path(args.summary)
    if not summ.exists():
        raise FileNotFoundError(f"missing summary: {summ}")

    df = pd.read_csv(summ)
    if df.empty:
        raise RuntimeError("summary is empty")

    # required in summary
    for c in ["suffix", "cap", "SeedMultiple_AfterWarmup", "Days_AfterWarmup"]:
        if c not in df.columns:
            df[c] = np.nan

    df["SeedMultiple_AfterWarmup"] = pd.to_numeric(df["SeedMultiple_AfterWarmup"], errors="coerce")
    df["Days_AfterWarmup"] = pd.to_numeric(df["Days_AfterWarmup"], errors="coerce")

    # weights
    df["TradeCount"] = pd.to_numeric(df.get("TradeCount", np.nan), errors="coerce").fillna(0.0)
    df["TickerWeight"] = pd.to_numeric(df.get("TradeCount", np.nan), errors="coerce").fillna(0.0)  # fallback weight
    # if we had explicit ticker-split count later, we can swap it in easily.

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    for c in group_cols:
        if c not in df.columns:
            df[c] = ""

    g = df.groupby(group_cols, dropna=False)

    rows = []
    for key, part in g:
        part = part.copy()

        # TOTAL seed multiple = product across periods (after-warmup)
        mults = part["SeedMultiple_AfterWarmup"].dropna()
        if len(mults) > 0:
            seed_total = float(np.prod(mults.values))
        else:
            seed_total = float("nan")

        days_total = float(part["Days_AfterWarmup"].dropna().sum()) if part["Days_AfterWarmup"].notna().any() else float("nan")
        cagr_total = _calc_cagr_from_mult_days(seed_total, days_total)

        # QQQ total (optional)
        qqq_mults = pd.to_numeric(part.get("QQQ_SeedMultiple_SamePeriod", np.nan), errors="coerce").dropna()
        qqq_seed_total = float(np.prod(qqq_mults.values)) if len(qqq_mults) > 0 else float("nan")
        qqq_cagr_total = _calc_cagr_from_mult_days(qqq_seed_total, days_total) if np.isfinite(days_total) else float("nan")
        excess_cagr_total = float(cagr_total - qqq_cagr_total) if np.isfinite(cagr_total) and np.isfinite(qqq_cagr_total) else float("nan")

        # badexit weighted rates
        bad_row = _wavg(part.get("BadExitRate_Row", np.nan), part["TradeCount"])
        bad_tkr = _wavg(part.get("BadExitRate_Ticker", np.nan), part["TradeCount"])
        reval_share = _wavg(part.get("BadExitReasonShare_RevalFail", np.nan), part["TradeCount"])
        grace_share = _wavg(part.get("BadExitReasonShare_GraceEnd", np.nan), part["TradeCount"])

        # dd style: max across periods (conservative)
        max_dd = pd.to_numeric(part.get("MaxDD_AfterWarmup", np.nan), errors="coerce")
        max_dd_total = float(np.nanmax(max_dd.values)) if max_dd.notna().any() else float("nan")

        uw = pd.to_numeric(part.get("MaxUnderwaterDays_AfterWarmup", np.nan), errors="coerce")
        uw_total = float(np.nanmax(uw.values)) if uw.notna().any() else float("nan")

        rec = pd.to_numeric(part.get("MaxDDRecoveryDays_AfterWarmup", np.nan), errors="coerce")
        rec_total = float(np.nanmax(rec.values)) if rec.notna().any() else float("nan")

        trade_total = float(part["TradeCount"].sum())

        # carry representative params (mode/median-ish): take first non-null
        def first_nonnull(col: str):
            if col not in part.columns:
                return np.nan
            s = part[col].dropna()
            return s.iloc[0] if len(s) else np.nan

        row = {}
        if isinstance(key, tuple):
            for i, c in enumerate(group_cols):
                row[c] = key[i]
        else:
            row[group_cols[0]] = key

        row.update({
            "Periods": int(part["period"].nunique()) if "period" in part.columns else int(len(part)),
            "DaysTotal_AfterWarmup": days_total,
            "SeedMultiple_Total": seed_total,
            "CAGR_Total": cagr_total,
            "QQQ_SeedMultiple_Total": qqq_seed_total,
            "QQQ_CAGR_Total": qqq_cagr_total,
            "ExcessCAGR_Total": excess_cagr_total,

            "TradeCount_Total": trade_total,
            "BadExitRate_Row_Total": bad_row,
            "BadExitRate_Ticker_Total": bad_tkr,
            "BadExitReasonShare_RevalFail_Total": reval_share,
            "BadExitReasonShare_GraceEnd_Total": grace_share,

            "MaxDD_Total": max_dd_total,
            "MaxUnderwaterDays_Total": uw_total,
            "MaxDDRecoveryDays_Total": rec_total,

            # helpful params
            "ps_min": first_nonnull("ps_min"),
            "tail_threshold": first_nonnull("tail_threshold"),
            "utility_quantile": first_nonnull("utility_quantile"),
            "lambda_tail": first_nonnull("lambda_tail"),
            "topk": first_nonnull("topk"),
            "badexit_max": first_nonnull("badexit_max"),
        })

        # simple ranking score
        # - prioritize CAGR_Total, penalize maxDD
        if np.isfinite(row["CAGR_Total"]) and np.isfinite(row["MaxDD_Total"]):
            row["Score"] = float(row["CAGR_Total"] / (1.0 + row["MaxDD_Total"]))
        elif np.isfinite(row["CAGR_Total"]):
            row["Score"] = float(row["CAGR_Total"])
        else:
            row["Score"] = float("nan")

        rows.append(row)

    out = pd.DataFrame(rows)
    # sort best first
    if "Score" in out.columns:
        out = out.sort_values(["Score", "CAGR_Total", "SeedMultiple_Total"], ascending=[False, False, False])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[DONE] wrote analysis: {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()