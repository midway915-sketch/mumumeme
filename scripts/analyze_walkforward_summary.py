#!/usr/bin/env python3
# scripts/analyze_walkforward_summary.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _safe_to_dt(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.tz_localize(None)


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


def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _recompute_from_curve(curve_file: str, warmup_end: pd.Timestamp) -> dict:
    """
    curve_file: path in summary (parquet/csv)
    warmup_end: Timestamp
    Returns:
      StartDate_AfterWarmup, EndDate_AfterWarmup,
      Days_AfterWarmup_Recalc, SeedMultiple_AfterWarmup_Recalc, CAGR_AfterWarmup_Recalc
    """
    out = {
        "StartDate_AfterWarmup": "",
        "EndDate_AfterWarmup": "",
        "Days_AfterWarmup_Recalc": float("nan"),
        "SeedMultiple_AfterWarmup_Recalc": float("nan"),
        "CAGR_AfterWarmup_Recalc": float("nan"),
    }

    if not curve_file or not isinstance(curve_file, str):
        return out

    p = Path(curve_file)
    if not p.exists():
        return out

    try:
        c = _read_any(p)
    except Exception:
        return out

    if c.empty or ("Date" not in c.columns) or ("Equity" not in c.columns):
        return out

    c = c.copy()
    c["Date"] = _safe_to_dt(c["Date"])
    c["Equity"] = pd.to_numeric(c["Equity"], errors="coerce")
    c = c.dropna(subset=["Date", "Equity"]).sort_values("Date").reset_index(drop=True)
    if c.empty:
        return out

    # slice after warmup_end
    warm = pd.to_datetime(warmup_end, errors="coerce")
    if pd.isna(warm):
        return out

    c2 = c.loc[c["Date"] >= warm].copy()
    if c2.empty:
        return out

    start_date = pd.Timestamp(c2["Date"].iloc[0])
    end_date = pd.Timestamp(c2["Date"].iloc[-1])

    start_eq = float(c2["Equity"].iloc[0])
    end_eq = float(c2["Equity"].iloc[-1])

    days = float((end_date - start_date).days)
    mult = float(end_eq / start_eq) if np.isfinite(start_eq) and start_eq > 0 and np.isfinite(end_eq) and end_eq > 0 else float("nan")
    cagr = _calc_cagr_from_mult_days(mult, days)

    out.update(
        {
            "StartDate_AfterWarmup": str(start_date.date()),
            "EndDate_AfterWarmup": str(end_date.date()),
            "Days_AfterWarmup_Recalc": days,
            "SeedMultiple_AfterWarmup_Recalc": mult,
            "CAGR_AfterWarmup_Recalc": cagr,
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--group-cols", default="suffix,cap", type=str, help="comma-separated group keys")
    ap.add_argument(
        "--recompute-from-curve",
        action="store_true",
        help="recompute start/end/days/seed/cagr using curve_file + WarmupEndDate (recommended)",
    )
    args = ap.parse_args()

    summ = Path(args.summary)
    if not summ.exists():
        raise FileNotFoundError(f"missing summary: {summ}")

    df = pd.read_csv(summ)
    if df.empty:
        raise RuntimeError("summary is empty")

    # ensure required keys exist
    for c in ["suffix", "cap"]:
        if c not in df.columns:
            df[c] = ""

    # ensure numeric columns exist (fallback)
    for c in ["SeedMultiple_AfterWarmup", "Days_AfterWarmup", "QQQ_SeedMultiple_SamePeriod"]:
        if c not in df.columns:
            df[c] = np.nan

    # weights (use TradeCount if exists)
    df["TradeCount"] = pd.to_numeric(df.get("TradeCount", np.nan), errors="coerce").fillna(0.0)

    # Parse warmup end
    if "WarmupEndDate" in df.columns:
        df["_WarmupEndDT"] = _safe_to_dt(df["WarmupEndDate"])
    else:
        df["_WarmupEndDT"] = pd.NaT

    # Optional: recompute per-row using curve
    if args.recompute_from_curve:
        if "curve_file" not in df.columns:
            # keep going; analysis can still run with existing columns
            df["curve_file"] = ""

        cache: dict[tuple[str, str], dict] = {}

        def _row_recalc(r: pd.Series) -> pd.Series:
            cf = str(r.get("curve_file", "") or "")
            w = r.get("_WarmupEndDT", pd.NaT)
            key = (cf, "" if pd.isna(w) else str(pd.Timestamp(w).date()))
            if key in cache:
                d = cache[key]
            else:
                d = _recompute_from_curve(cf, w)
                cache[key] = d
            for k, v in d.items():
                r[k] = v
            return r

        df = df.apply(_row_recalc, axis=1)

        # choose recalced values when available
        df["SeedMultiple_AW_Use"] = pd.to_numeric(df.get("SeedMultiple_AfterWarmup_Recalc", df["SeedMultiple_AfterWarmup"]), errors="coerce")
        df["Days_AW_Use"] = pd.to_numeric(df.get("Days_AfterWarmup_Recalc", df["Days_AfterWarmup"]), errors="coerce")
        df["CAGR_AW_Use"] = pd.to_numeric(df.get("CAGR_AfterWarmup_Recalc", df.get("CAGR_AfterWarmup", np.nan)), errors="coerce")
    else:
        df["SeedMultiple_AW_Use"] = pd.to_numeric(df["SeedMultiple_AfterWarmup"], errors="coerce")
        df["Days_AW_Use"] = pd.to_numeric(df["Days_AfterWarmup"], errors="coerce")
        df["CAGR_AW_Use"] = pd.to_numeric(df.get("CAGR_AfterWarmup", np.nan), errors="coerce")

    # group keys
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    for c in group_cols:
        if c not in df.columns:
            df[c] = ""

    g = df.groupby(group_cols, dropna=False)

    rows = []
    for key, part in g:
        part = part.copy()

        # TOTAL seed multiple = product across periods (after-warmup)
        mults = pd.to_numeric(part["SeedMultiple_AW_Use"], errors="coerce").dropna()
        seed_total = float(np.prod(mults.values)) if len(mults) > 0 else float("nan")

        # TOTAL days = sum of per-period days (after-warmup)
        days_total = pd.to_numeric(part["Days_AW_Use"], errors="coerce")
        days_total = float(days_total.dropna().sum()) if days_total.notna().any() else float("nan")

        # TOTAL cagr
        cagr_total = _calc_cagr_from_mult_days(seed_total, days_total)

        # Group start/end dates (if we have recomputed per-row start/end)
        start_total = ""
        end_total = ""
        if "StartDate_AfterWarmup" in part.columns and "EndDate_AfterWarmup" in part.columns:
            sdt = _safe_to_dt(part["StartDate_AfterWarmup"])
            edt = _safe_to_dt(part["EndDate_AfterWarmup"])
            if sdt.notna().any():
                start_total = str(pd.Timestamp(sdt.min()).date())
            if edt.notna().any():
                end_total = str(pd.Timestamp(edt.max()).date())

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

        row.update(
            {
                "StartDate_Total": start_total,
                "EndDate_Total": end_total,
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
            }
        )

        # simple ranking score
        if np.isfinite(row["CAGR_Total"]) and np.isfinite(row["MaxDD_Total"]):
            row["Score"] = float(row["CAGR_Total"] / (1.0 + row["MaxDD_Total"]))
        elif np.isfinite(row["CAGR_Total"]):
            row["Score"] = float(row["CAGR_Total"])
        else:
            row["Score"] = float("nan")

        rows.append(row)

    out = pd.DataFrame(rows)
    if "Score" in out.columns:
        out = out.sort_values(["Score", "CAGR_Total", "SeedMultiple_Total"], ascending=[False, False, False])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[DONE] wrote analysis: {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()