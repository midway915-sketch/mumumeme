#!/usr/bin/env python3
# scripts/analyze_walkforward_summary.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd


# ----------------------------
# io utils
# ----------------------------
def _safe_to_dt(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.tz_localize(None)


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _curve_to_trades_path(curve_path: Path) -> Path:
    # sim_engine_curve_... -> sim_engine_trades_...
    name = curve_path.name.replace("sim_engine_curve_", "sim_engine_trades_")
    # keep same extension if possible; if missing, try both
    cand = curve_path.with_name(name)
    if cand.exists():
        return cand
    if cand.suffix.lower() == ".parquet":
        alt = cand.with_suffix(".csv")
    else:
        alt = cand.with_suffix(".parquet")
    return alt


# ----------------------------
# finance helpers
# ----------------------------
def _calc_cagr(start_equity: float, end_equity: float, days: float) -> float:
    if not np.isfinite(start_equity) or not np.isfinite(end_equity) or start_equity <= 0 or end_equity <= 0:
        return float("nan")
    if not np.isfinite(days) or days <= 0:
        return float("nan")
    years = days / 365.0
    if years <= 0:
        return float("nan")
    return float((end_equity / start_equity) ** (1.0 / years) - 1.0)


def _max_drawdown_and_durations(equity: pd.Series) -> dict:
    """
    equity: positive series indexed in time order (no gaps assumption required)
    Returns:
      MaxDD (negative number, e.g. -0.25),
      MaxUnderwaterDays (int),
      MaxDDRecoveryDays (int, peak->recovery duration for the max-dd episode; NaN if never recovered)
    """
    e = pd.to_numeric(equity, errors="coerce").dropna().astype(float)
    if e.empty:
        return {
            "MaxDD_AfterWarmup": float("nan"),
            "MaxUnderwaterDays_AfterWarmup": float("nan"),
            "MaxDDRecoveryDays_AfterWarmup": float("nan"),
        }

    peak = e.expanding().max()
    dd = e / peak - 1.0  # <= 0
    max_dd = float(dd.min())

    # underwater days: longest consecutive stretch where dd < 0
    underwater = (dd < 0).astype(int)
    max_underwater = 0
    cur = 0
    for v in underwater.values:
        if v:
            cur += 1
            if cur > max_underwater:
                max_underwater = cur
        else:
            cur = 0

    # recovery days for max-dd episode:
    # find the time of max dd (first occurrence), then find when equity reaches previous peak again
    idx_min = int(np.argmin(dd.values))
    # peak level at the time just before/at min:
    peak_level = float(peak.iloc[idx_min])
    # search forward for recovery
    rec_days = float("nan")
    e_after = e.iloc[idx_min:]
    rec_pos = np.where(e_after.values >= peak_level)[0]
    if len(rec_pos) > 0:
        # 0 means same day recovered; duration in "rows" (days count) = position index
        rec_days = int(rec_pos[0])

    return {
        "MaxDD_AfterWarmup": max_dd,
        "MaxUnderwaterDays_AfterWarmup": int(max_underwater),
        "MaxDDRecoveryDays_AfterWarmup": rec_days,
    }


def _daily_risk_stats(equity: pd.Series) -> dict:
    """
    equity -> daily returns stats:
      DailyVol (annualized, sqrt(252) * std),
      Sharpe0 (rf=0),
      Sortino0 (rf=0)
    """
    e = pd.to_numeric(equity, errors="coerce").dropna().astype(float)
    if len(e) < 3:
        return {"DailyVol": float("nan"), "Sharpe0": float("nan"), "Sortino0": float("nan")}

    r = e.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"DailyVol": float("nan"), "Sharpe0": float("nan"), "Sortino0": float("nan")}

    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    ann_mu = mu * 252.0
    ann_sd = sd * math.sqrt(252.0) if np.isfinite(sd) else float("nan")
    sharpe = ann_mu / ann_sd if ann_sd and np.isfinite(ann_sd) and ann_sd > 0 else float("nan")

    down = r.copy()
    down = down[down < 0]
    down_sd = float(down.std(ddof=1)) if len(down) >= 2 else float("nan")
    ann_down_sd = down_sd * math.sqrt(252.0) if np.isfinite(down_sd) else float("nan")
    sortino = ann_mu / ann_down_sd if ann_down_sd and np.isfinite(ann_down_sd) and ann_down_sd > 0 else float("nan")

    return {"DailyVol": ann_sd, "Sharpe0": sharpe, "Sortino0": sortino}


def _is_badexit_reason(reason: str) -> int:
    r = str(reason or "").strip().upper()
    if r.startswith("REVAL_FAIL"):
        return 1
    if r.startswith("GRACE_END_EXIT"):
        return 1
    return 0


def _badexit_reason_bucket(reason: str) -> str:
    r = str(reason or "").strip().upper()
    if r.startswith("REVAL_FAIL"):
        return "REVAL_FAIL"
    if r.startswith("GRACE_END_EXIT"):
        return "GRACE_END_EXIT"
    return "OTHER"


# ----------------------------
# per-period file parsing
# ----------------------------
@dataclass
class PeriodPiece:
    period: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    days: int
    equity: pd.Series  # index aligned to dates (same length as dates)
    dates: pd.Series
    qqq_mult: float
    qqq_days: float
    trades: pd.DataFrame


def _read_prices(prices_parq: Path, prices_csv: Path) -> pd.DataFrame:
    if prices_parq.exists():
        df = pd.read_parquet(prices_parq)
    elif prices_csv.exists():
        df = pd.read_csv(prices_csv)
    else:
        raise FileNotFoundError(f"Missing prices: {prices_parq} (or {prices_csv})")

    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("prices must have Date,Ticker")
    if "Close" not in df.columns:
        raise ValueError("prices missing Close")

    df = df.copy()
    df["Date"] = _safe_to_dt(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def _qqq_mult_for_range(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float]:
    q = prices.loc[prices["Ticker"] == "QQQ"].copy()
    if q.empty:
        return (float("nan"), float("nan"))
    q = q.sort_values("Date")

    q_start = q.loc[q["Date"] >= start]
    if q_start.empty:
        return (float("nan"), float("nan"))
    p0 = float(q_start["Close"].iloc[0])

    q_end = q.loc[q["Date"] <= end]
    if q_end.empty:
        return (float("nan"), float("nan"))
    p1 = float(q_end["Close"].iloc[-1])

    if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(p1) or p1 <= 0:
        return (float("nan"), float("nan"))
    mult = float(p1 / p0)
    days = float((end - start).days)
    return (mult, days)


def _clean_curve_df(curve: pd.DataFrame) -> pd.DataFrame:
    if curve is None or curve.empty:
        return pd.DataFrame()
    if "Date" not in curve.columns or "Equity" not in curve.columns:
        return pd.DataFrame()
    c = curve.copy()
    c["Date"] = _safe_to_dt(c["Date"])
    c["Equity"] = pd.to_numeric(c["Equity"], errors="coerce")
    c = c.dropna(subset=["Date", "Equity"]).sort_values("Date").reset_index(drop=True)
    c = c.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return c


def _warmup_end_from_trades(trades: pd.DataFrame) -> pd.Timestamp | None:
    if trades is None or trades.empty or "EntryDate" not in trades.columns:
        return None
    entry = _safe_to_dt(trades["EntryDate"]).dropna()
    if entry.empty:
        return None
    return pd.Timestamp(entry.min())


def _read_period_piece(
    period: str,
    curve_path: Path,
    prices: pd.DataFrame,
) -> PeriodPiece | None:
    if not curve_path.exists():
        return None

    trades_path = _curve_to_trades_path(curve_path)
    if not trades_path.exists():
        return None

    curve_raw = _read_any(curve_path)
    curve = _clean_curve_df(curve_raw)
    if curve.empty:
        return None

    trades = _read_any(trades_path)
    warmup_end = _warmup_end_from_trades(trades)
    if warmup_end is None:
        # no trades => treat as empty after-warmup
        return None

    cur2 = curve.loc[curve["Date"] >= warmup_end].copy()
    if cur2.empty:
        return None

    dates = cur2["Date"].reset_index(drop=True)
    equity = cur2["Equity"].reset_index(drop=True)

    start_date = pd.Timestamp(dates.iloc[0])
    end_date = pd.Timestamp(dates.iloc[-1])
    days = int((end_date - start_date).days)

    qqq_mult, qqq_days = _qqq_mult_for_range(prices, start_date, end_date)

    return PeriodPiece(
        period=str(period),
        start_date=start_date,
        end_date=end_date,
        days=days,
        equity=equity,
        dates=dates,
        qqq_mult=qqq_mult,
        qqq_days=qqq_days,
        trades=trades,
    )


def _stitch_pieces_to_equity(pieces: list[PeriodPiece]) -> tuple[pd.Series, pd.Series, float, float]:
    """
    Stitch after-warmup equity across periods by chaining daily returns.
    Returns:
      stitched_dates (pd.Series),
      stitched_equity (pd.Series starting at 1.0),
      qqq_mult_total (product of per-piece qqq mult),
      total_days (sum of per-piece days)
    """
    if not pieces:
        return (pd.Series(dtype="datetime64[ns]"), pd.Series(dtype="float64"), float("nan"), float("nan"))

    # sort by start_date
    pieces = sorted(pieces, key=lambda x: x.start_date)

    all_dates = []
    all_rets = []
    qqq_mult_total = 1.0
    total_days = 0.0

    for pc in pieces:
        e = pd.to_numeric(pc.equity, errors="coerce").astype(float)
        d = _safe_to_dt(pc.dates)
        m = min(len(e), len(d))
        e = e.iloc[:m].reset_index(drop=True)
        d = d.iloc[:m].reset_index(drop=True)

        if len(e) < 2:
            continue

        # daily pct returns inside this piece
        r = e.pct_change().replace([np.inf, -np.inf], np.nan)
        r = r.iloc[1:].fillna(0.0).astype(float)
        d2 = d.iloc[1:]

        all_rets.append(r)
        all_dates.append(d2)

        if np.isfinite(pc.qqq_mult) and pc.qqq_mult > 0:
            qqq_mult_total *= float(pc.qqq_mult)
        else:
            qqq_mult_total = float("nan")

        if np.isfinite(pc.days) and pc.days > 0:
            total_days += float(pc.days)

    if not all_rets:
        return (pd.Series(dtype="datetime64[ns]"), pd.Series(dtype="float64"), float("nan"), float("nan"))

    rets = pd.concat(all_rets, ignore_index=True)
    dates = pd.concat(all_dates, ignore_index=True)

    eq = (1.0 + rets).cumprod()
    return (dates, eq, float(qqq_mult_total), float(total_days))


def _recent_window_metrics(dates: pd.Series, equity: pd.Series, years: float) -> tuple[float, float]:
    if dates is None or equity is None or len(dates) < 3 or len(equity) < 3:
        return (float("nan"), float("nan"))
    d = _safe_to_dt(dates)
    e = pd.to_numeric(equity, errors="coerce").astype(float)
    m = min(len(d), len(e))
    d = d.iloc[:m].reset_index(drop=True)
    e = e.iloc[:m].reset_index(drop=True)

    end = pd.Timestamp(d.iloc[-1])
    start_cut = end - pd.Timedelta(days=int(round(years * 365.0)))
    mask = d >= start_cut
    d2 = d[mask].reset_index(drop=True)
    e2 = e[mask].reset_index(drop=True)
    if len(e2) < 3:
        return (float("nan"), float("nan"))
    seed_mult = float(e2.iloc[-1] / e2.iloc[0]) if e2.iloc[0] > 0 else float("nan")
    days = float((pd.Timestamp(d2.iloc[-1]) - pd.Timestamp(d2.iloc[0])).days)
    cagr = _calc_cagr(float(e2.iloc[0]), float(e2.iloc[-1]), days)
    return (seed_mult, cagr)


# ----------------------------
# per-config aggregation
# ----------------------------
def _trade_struct_stats(trades_all: pd.DataFrame, total_days: float) -> dict:
    if trades_all is None or trades_all.empty:
        return {
            "TradeCount": 0,
            "TradesPerYear": float("nan"),
            "HoldDays_Mean": float("nan"),
            "HoldDays_Median": float("nan"),
            "HoldDays_P90": float("nan"),
        }

    df = trades_all.copy()

    # TradeCount: MAIN rows if present, else all rows
    if "CycleType" in df.columns:
        main = df[df["CycleType"].astype(str).str.upper().eq("MAIN")].copy()
        use = main if len(main) else df
    else:
        use = df

    trade_count = int(len(use))
    tpy = float(trade_count / (total_days / 365.0)) if np.isfinite(total_days) and total_days > 0 else float("nan")

    hold = None
    if "HoldingDays" in use.columns:
        hold = pd.to_numeric(use["HoldingDays"], errors="coerce").dropna().astype(float)

    if hold is None or hold.empty:
        return {
            "TradeCount": trade_count,
            "TradesPerYear": tpy,
            "HoldDays_Mean": float("nan"),
            "HoldDays_Median": float("nan"),
            "HoldDays_P90": float("nan"),
        }

    return {
        "TradeCount": trade_count,
        "TradesPerYear": tpy,
        "HoldDays_Mean": float(hold.mean()),
        "HoldDays_Median": float(hold.median()),
        "HoldDays_P90": float(hold.quantile(0.90)),
    }


def _badexit_stats(trades_all: pd.DataFrame) -> dict:
    base = {
        "BadExitRate_Row": float("nan"),
        "BadExitRate_Ticker": float("nan"),
        "BadExitReasonShare_RevalFail": float("nan"),
        "BadExitReasonShare_GraceEnd": float("nan"),
        "BadExitReturnMean": float("nan"),
        "NonBadExitReturnMean": float("nan"),
        "BadExitReturnDiff": float("nan"),
    }
    if trades_all is None or trades_all.empty:
        return base
    if "Reason" not in trades_all.columns:
        return base

    df = trades_all.copy()
    df["Reason"] = df["Reason"].astype(str)
    df["IsBadExit"] = df["Reason"].apply(_is_badexit_reason).astype(int)

    # row-based
    base["BadExitRate_Row"] = float(df["IsBadExit"].mean()) if len(df) else float("nan")

    # ticker-expanded
    if "Tickers" in df.columns:
        rows = []
        for _, r in df.iterrows():
            ticks = [t.strip().upper() for t in str(r.get("Tickers", "")).split(",") if t.strip()]
            if not ticks:
                continue
            for t in ticks:
                rows.append({"Ticker": t, "IsBadExit": int(r["IsBadExit"]), "CycleReturn": r.get("CycleReturn", np.nan), "Reason": r.get("Reason", "")})
        if rows:
            ex = pd.DataFrame(rows)
            base["BadExitRate_Ticker"] = float(ex["IsBadExit"].mean())
        else:
            base["BadExitRate_Ticker"] = float("nan")

    # reason shares among badexits
    bad = df[df["IsBadExit"] == 1].copy()
    if len(bad):
        buckets = bad["Reason"].apply(_badexit_reason_bucket)
        denom = float(len(buckets))
        base["BadExitReasonShare_RevalFail"] = float((buckets == "REVAL_FAIL").sum() / denom)
        base["BadExitReasonShare_GraceEnd"] = float((buckets == "GRACE_END_EXIT").sum() / denom)

    # return diff (CycleReturn if present; else nan)
    if "CycleReturn" in df.columns:
        ret = pd.to_numeric(df["CycleReturn"], errors="coerce")
        bad_ret = ret[df["IsBadExit"] == 1].dropna()
        ok_ret = ret[df["IsBadExit"] == 0].dropna()
        if len(bad_ret):
            base["BadExitReturnMean"] = float(bad_ret.mean())
        if len(ok_ret):
            base["NonBadExitReturnMean"] = float(ok_ret.mean())
        if np.isfinite(base["BadExitReturnMean"]) and np.isfinite(base["NonBadExitReturnMean"]):
            base["BadExitReturnDiff"] = float(base["BadExitReturnMean"] - base["NonBadExitReturnMean"])

    return base


def _analyze_one_config(group: pd.DataFrame, prices: pd.DataFrame) -> dict:
    # group rows are periods for one config
    # required columns present in summary: period, curve_file, suffix, cap, ps_min, tail_threshold, utility_quantile, lambda_tail, topk, badexit_max
    pieces: list[PeriodPiece] = []
    trades_parts: list[pd.DataFrame] = []

    for _, r in group.iterrows():
        period = str(r.get("period", ""))
        curve_file = Path(str(r.get("curve_file", "")))
        pc = _read_period_piece(period=period, curve_path=curve_file, prices=prices)
        if pc is None:
            continue
        pieces.append(pc)
        if pc.trades is not None and not pc.trades.empty:
            trades_parts.append(pc.trades)

    dates, eq, qqq_mult_total, total_days = _stitch_pieces_to_equity(pieces)

    out = {}
    # identity cols (stable)
    for k in ["suffix", "cap", "ps_min", "tail_threshold", "utility_quantile", "lambda_tail", "topk", "badexit_max"]:
        out[k] = group.iloc[0].get(k, "")

    if len(eq) < 3:
        # still output identity cols with NaNs
        out.update({
            "StartDate_AfterWarmup": "",
            "EndDate_AfterWarmup": "",
            "TotalDays_AfterWarmup": float("nan"),
            "SeedMultiple_AfterWarmup": float("nan"),
            "CAGR_AfterWarmup": float("nan"),
            "QQQ_SeedMultiple_SamePeriod": float("nan"),
            "QQQ_CAGR_SamePeriod": float("nan"),
            "ExcessSeedMultiple_AfterWarmup": float("nan"),
            "ExcessCAGR_AfterWarmup": float("nan"),
            "Recent5Y_SeedMultiple_AfterWarmup": float("nan"),
            "Recent5Y_CAGR_AfterWarmup": float("nan"),
        })
        out.update(_max_drawdown_and_durations(eq))
        out.update(_daily_risk_stats(eq))
        trades_all = pd.concat(trades_parts, ignore_index=True) if trades_parts else pd.DataFrame()
        out.update(_trade_struct_stats(trades_all, total_days=float("nan")))
        out.update(_badexit_stats(trades_all))
        return out

    start_dt = pd.Timestamp(dates.iloc[0])
    end_dt = pd.Timestamp(dates.iloc[-1])

    seed_mult = float(eq.iloc[-1] / eq.iloc[0]) if float(eq.iloc[0]) > 0 else float("nan")
    cagr = _calc_cagr(float(eq.iloc[0]), float(eq.iloc[-1]), float(total_days))

    qqq_cagr = _calc_cagr(1.0, float(qqq_mult_total), float(total_days)) if np.isfinite(qqq_mult_total) else float("nan")
    excess_seed = float(seed_mult / qqq_mult_total) if np.isfinite(seed_mult) and np.isfinite(qqq_mult_total) and qqq_mult_total > 0 else float("nan")
    excess_cagr = float(cagr - qqq_cagr) if np.isfinite(cagr) and np.isfinite(qqq_cagr) else float("nan")

    r5_mult, r5_cagr = _recent_window_metrics(dates, eq, years=5.0)

    out.update({
        "StartDate_AfterWarmup": str(start_dt.date()),
        "EndDate_AfterWarmup": str(end_dt.date()),
        "TotalDays_AfterWarmup": float(total_days),
        "SeedMultiple_AfterWarmup": seed_mult,
        "CAGR_AfterWarmup": cagr,
        "QQQ_SeedMultiple_SamePeriod": float(qqq_mult_total),
        "QQQ_CAGR_SamePeriod": qqq_cagr,
        "ExcessSeedMultiple_AfterWarmup": excess_seed,
        "ExcessCAGR_AfterWarmup": excess_cagr,
        "Recent5Y_SeedMultiple_AfterWarmup": r5_mult,
        "Recent5Y_CAGR_AfterWarmup": r5_cagr,
    })

    out.update(_max_drawdown_and_durations(eq))
    out.update(_daily_risk_stats(eq))

    trades_all = pd.concat(trades_parts, ignore_index=True) if trades_parts else pd.DataFrame()
    out.update(_trade_struct_stats(trades_all, total_days=total_days))
    out.update(_badexit_stats(trades_all))

    return out


# ----------------------------
# main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze walkforward summary by config (stitch half-years into total period).")

    ap.add_argument("--summary", required=True, type=str, help="path to _summary_walkforward2.csv")
    ap.add_argument("--out", required=True, type=str, help="output analysis csv (total-period by config)")

    # optional prices (for QQQ baseline)
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    args = ap.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"summary not found: {summary_path}")

    df = pd.read_csv(summary_path)
    if df.empty:
        raise SystemExit("[ERROR] summary is empty")

    # Ensure required columns
    need = ["period", "curve_file", "suffix", "cap", "ps_min", "tail_threshold", "utility_quantile", "lambda_tail", "topk", "badexit_max"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"[ERROR] summary missing columns: {miss}")

    # normalize types
    for c in ["ps_min", "tail_threshold", "utility_quantile", "lambda_tail", "topk", "badexit_max"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["curve_file"] = df["curve_file"].astype(str)

    prices = _read_prices(Path(args.prices_parq), Path(args.prices_csv))

    group_cols = ["suffix", "cap", "ps_min", "tail_threshold", "utility_quantile", "lambda_tail", "topk", "badexit_max"]
    out_rows = []

    for _, g in df.groupby(group_cols, dropna=False):
        out_rows.append(_analyze_one_config(g, prices=prices))

    out = pd.DataFrame(out_rows)

    # scoring / sorting: prefer after-warmup total-period metrics
    sort_cols = []
    if "ExcessCAGR_AfterWarmup" in out.columns:
        sort_cols.append("ExcessCAGR_AfterWarmup")
    if "CAGR_AfterWarmup" in out.columns:
        sort_cols.append("CAGR_AfterWarmup")
    if "SeedMultiple_AfterWarmup" in out.columns:
        sort_cols.append("SeedMultiple_AfterWarmup")
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[DONE] wrote: {out_path} rows={len(out)}")

    if len(out):
        best = out.iloc[0].to_dict()
        print("=" * 70)
        print("[BEST] total-period (stitched half-years) by ExcessCAGR/CAGR/SeedMultiple")
        print(f"suffix={best.get('suffix')} cap={best.get('cap')} ps_min={best.get('ps_min')} tail={best.get('tail_threshold')} q={best.get('utility_quantile')} lam={best.get('lambda_tail')} topk={best.get('topk')} be={best.get('badexit_max')}")
        print(f"SeedMultiple_AfterWarmup={best.get('SeedMultiple_AfterWarmup')}  CAGR_AfterWarmup={best.get('CAGR_AfterWarmup')}")
        print(f"QQQ_Mult={best.get('QQQ_SeedMultiple_SamePeriod')}  QQQ_CAGR={best.get('QQQ_CAGR_SamePeriod')}")
        print(f"ExcessSeed={best.get('ExcessSeedMultiple_AfterWarmup')}  ExcessCAGR={best.get('ExcessCAGR_AfterWarmup')}")
        print(f"MaxDD={best.get('MaxDD_AfterWarmup')}  UWDays={best.get('MaxUnderwaterDays_AfterWarmup')}  RecDays={best.get('MaxDDRecoveryDays_AfterWarmup')}")
        print(f"TradeCount={best.get('TradeCount')}  TradesPerYear={best.get('TradesPerYear')}")
        print(f"BadExitRate_Row={best.get('BadExitRate_Row')}  BadExitRate_Ticker={best.get('BadExitRate_Ticker')}")
        print("=" * 70)


if __name__ == "__main__":
    main()