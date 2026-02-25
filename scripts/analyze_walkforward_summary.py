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
    years = days / 365.25  # more accurate than 365
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


def _first_nonnull(part: pd.DataFrame, col: str):
    if col not in part.columns:
        return np.nan
    s = part[col].dropna()
    return s.iloc[0] if len(s) else np.nan


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

    # ---- normalize core columns (support both "after-warmup" summary and current walkforward summary)
    # identifiers
    for c in ["period", "suffix", "cap"]:
        if c not in df.columns:
            df[c] = ""

    # dates (for StartDate/EndDate)
    if "date_from" in df.columns:
        df["date_from"] = pd.to_datetime(df["date_from"], errors="coerce").dt.tz_localize(None)
    else:
        df["date_from"] = pd.NaT

    if "date_to" in df.columns:
        df["date_to"] = pd.to_datetime(df["date_to"], errors="coerce").dt.tz_localize(None)
    else:
        df["date_to"] = pd.NaT

    # per-period seed multiple + days
    if "SeedMultiple_AfterWarmup" in df.columns:
        df["__mult"] = pd.to_numeric(df["SeedMultiple_AfterWarmup"], errors="coerce")
    elif "seed_multiple_window" in df.columns:
        df["__mult"] = pd.to_numeric(df["seed_multiple_window"], errors="coerce")
    elif "start_equity" in df.columns and "end_equity" in df.columns:
        se = pd.to_numeric(df["start_equity"], errors="coerce")
        ee = pd.to_numeric(df["end_equity"], errors="coerce")
        df["__mult"] = ee / se
    else:
        df["__mult"] = np.nan

    if "Days_AfterWarmup" in df.columns:
        df["__days"] = pd.to_numeric(df["Days_AfterWarmup"], errors="coerce")
    elif "days" in df.columns:
        df["__days"] = pd.to_numeric(df["days"], errors="coerce")
    else:
        # fallback from date range
        d0 = df["date_from"]
        d1 = df["date_to"]
        df["__days"] = (d1 - d0).dt.days.astype("float")

    # weights for weighted-averages (fallback: trade_count if exists, else 1)
    if "TradeCount" in df.columns:
        df["TradeCount"] = pd.to_numeric(df["TradeCount"], errors="coerce").fillna(0.0)
    elif "trade_count" in df.columns:
        df["TradeCount"] = pd.to_numeric(df["trade_count"], errors="coerce").fillna(0.0)
    else:
        df["TradeCount"] = 1.0

    # optional QQQ multiple
    if "QQQ_SeedMultiple_SamePeriod" in df.columns:
        df["__qqq_mult"] = pd.to_numeric(df["QQQ_SeedMultiple_SamePeriod"], errors="coerce")
    elif "qqq_seed_multiple_window" in df.columns:
        df["__qqq_mult"] = pd.to_numeric(df["qqq_seed_multiple_window"], errors="coerce")
    else:
        df["__qqq_mult"] = np.nan

    # ---- group keys
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    for c in group_cols:
        if c not in df.columns:
            df[c] = ""

    # ---- IMPORTANT: dedupe per (period, config params) inside each group
    # (your summary currently has duplicated rows per period/config -> makes totals absurd)
    config_keys = [
        "period", "suffix", "cap",
        "ps_min", "tail_threshold", "utility_quantile", "lambda_tail", "topk", "badexit_max",
        # if these exist later, they’ll auto participate if present
        "trail_stop", "tp1_frac", "gate_mode",
    ]
    dedupe_cols = [c for c in config_keys if c in df.columns]

    rows = []
    for key, part0 in df.groupby(group_cols, dropna=False):
        part0 = part0.copy()

        # dedupe (fixes the "DaysTotal huge" / "everything 0 or weird" symptom)
        if dedupe_cols:
            part = part0.drop_duplicates(subset=dedupe_cols, keep="first").copy()
        else:
            part = part0.drop_duplicates(keep="first").copy()

        # Start/End dates
        start_dt = part["date_from"].min()
        end_dt = part["date_to"].max()
        start_str = "" if pd.isna(start_dt) else str(pd.Timestamp(start_dt).date())
        end_str = "" if pd.isna(end_dt) else str(pd.Timestamp(end_dt).date())

        # TOTAL seed multiple = product across UNIQUE periods/configs (after-warmup or window)
        mults = pd.to_numeric(part["__mult"], errors="coerce").dropna()
        seed_total = float(np.prod(mults.values)) if len(mults) else float("nan")

        days_series = pd.to_numeric(part["__days"], errors="coerce").dropna()
        days_total = float(days_series.sum()) if len(days_series) else float("nan")

        cagr_total = _calc_cagr_from_mult_days(seed_total, days_total)

        # QQQ totals (optional)
        qqq_mults = pd.to_numeric(part["__qqq_mult"], errors="coerce").dropna()
        qqq_seed_total = float(np.prod(qqq_mults.values)) if len(qqq_mults) else float("nan")
        qqq_cagr_total = _calc_cagr_from_mult_days(qqq_seed_total, days_total) if np.isfinite(days_total) else float("nan")
        excess_cagr_total = float(cagr_total - qqq_cagr_total) if np.isfinite(cagr_total) and np.isfinite(qqq_cagr_total) else float("nan")

        # badexit weighted rates (if present)
        bad_row = _wavg(part.get("BadExitRate_Row", np.nan), part["TradeCount"])
        bad_tkr = _wavg(part.get("BadExitRate_Ticker", np.nan), part["TradeCount"])
        reval_share = _wavg(part.get("BadExitReasonShare_RevalFail", np.nan), part["TradeCount"])
        grace_share = _wavg(part.get("BadExitReasonShare_GraceEnd", np.nan), part["TradeCount"])

        # dd-style: max across periods (conservative) if present
        max_dd = pd.to_numeric(part.get("MaxDD_AfterWarmup", np.nan), errors="coerce")
        max_dd_total = float(np.nanmax(max_dd.values)) if max_dd.notna().any() else float("nan")

        uw = pd.to_numeric(part.get("MaxUnderwaterDays_AfterWarmup", np.nan), errors="coerce")
        uw_total = float(np.nanmax(uw.values)) if uw.notna().any() else float("nan")

        rec = pd.to_numeric(part.get("MaxDDRecoveryDays_AfterWarmup", np.nan), errors="coerce")
        rec_total = float(np.nanmax(rec.values)) if rec.notna().any() else float("nan")

        trade_total = float(pd.to_numeric(part["TradeCount"], errors="coerce").fillna(0.0).sum())

        row = {}
        if isinstance(key, tuple):
            for i, c in enumerate(group_cols):
                row[c] = key[i]
        else:
            row[group_cols[0]] = key

        row.update({
            "StartDate_Total": start_str,
            "EndDate_Total": end_str,
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

            # helpful params (representative: first non-null in this group)
            "ps_min": _first_nonnull(part, "ps_min"),
            "tail_threshold": _first_nonnull(part, "tail_threshold"),
            "utility_quantile": _first_nonnull(part, "utility_quantile"),
            "lambda_tail": _first_nonnull(part, "lambda_tail"),
            "topk": _first_nonnull(part, "topk"),
            "badexit_max": _first_nonnull(part, "badexit_max"),
        })

        # simple ranking score
        # - prioritize CAGR_Total, penalize maxDD if available
        if np.isfinite(row["CAGR_Total"]) and np.isfinite(row["MaxDD_Total"]):
            row["Score"] = float(row["CAGR_Total"] / (1.0 + abs(row["MaxDD_Total"])))
        elif np.isfinite(row["CAGR_Total"]):
            row["Score"] = float(row["CAGR_Total"])
        else:
            row["Score"] = float("nan")

        rows.append(row)

    out = pd.DataFrame(rows)

    # sort best first
    sort_cols = [c for c in ["Score", "CAGR_Total", "SeedMultiple_Total"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # quick sanity log (prints to Actions log)
    if len(out):
        b = out.iloc[0]
        print("=" * 70)
        print("[TOP] group=", ", ".join([f"{c}={b.get(c)}" for c in group_cols]))
        print(f"StartDate={b.get('StartDate_Total')}  EndDate={b.get('EndDate_Total')}")
        print(f"DaysTotal={b.get('DaysTotal_AfterWarmup')}  SeedTotal={b.get('SeedMultiple_Total')}  CAGR={b.get('CAGR_Total')}")
        print("=" * 70)

    print(f"[DONE] wrote analysis: {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()