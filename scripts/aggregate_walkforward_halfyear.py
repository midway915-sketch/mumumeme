#!/usr/bin/env python3
# scripts/aggregate_walkforward_halfyear.py
from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


# -----------------------
# utils
# -----------------------
def _safe_to_dt(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.tz_localize(None)


def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _clean_curve(curve: pd.DataFrame) -> pd.DataFrame:
    if curve is None or curve.empty:
        return pd.DataFrame()
    if "Date" not in curve.columns or "Equity" not in curve.columns:
        return pd.DataFrame()

    c = curve.copy()
    c["Date"] = _safe_to_dt(c["Date"])
    c["Equity"] = pd.to_numeric(c["Equity"], errors="coerce")
    c = c.dropna(subset=["Date", "Equity"]).sort_values("Date").reset_index(drop=True)
    if c.empty:
        return pd.DataFrame()

    if "InCycle" in c.columns:
        inc = pd.to_numeric(c["InCycle"], errors="coerce").fillna(0).astype(int)
    elif "InPosition" in c.columns:
        inc = pd.to_numeric(c["InPosition"], errors="coerce").fillna(0).astype(int)
    else:
        inc = pd.Series([0] * len(c), index=c.index, dtype=int)

    c["InCycle"] = (inc > 0).astype(int)
    c = c.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return c


def _calc_cagr(start: float, end: float, days: float) -> float:
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0 or end <= 0:
        return float("nan")
    if not np.isfinite(days) or days <= 0:
        return float("nan")
    years = days / 365.0
    if years <= 0:
        return float("nan")
    return float((end / start) ** (1.0 / years) - 1.0)


def _seed_multiple_and_cagr(c: pd.DataFrame) -> tuple[float, float, float]:
    if c is None or c.empty:
        return (float("nan"), float("nan"), float("nan"))
    start_eq = float(c["Equity"].iloc[0])
    end_eq = float(c["Equity"].iloc[-1])
    start_dt = pd.Timestamp(c["Date"].iloc[0])
    end_dt = pd.Timestamp(c["Date"].iloc[-1])
    days = float((end_dt - start_dt).days)
    sm = float(end_eq / start_eq) if np.isfinite(start_eq) and start_eq > 0 else float("nan")
    cagr = _calc_cagr(start_eq, end_eq, days)
    return (sm, cagr, days)


def _max_dd_stats(c: pd.DataFrame) -> tuple[float, int, int]:
    """
    returns:
      MaxDD (0~1), MaxUnderwaterDays, MaxDDRecoveryDays
    """
    if c is None or c.empty or len(c) < 3:
        return (float("nan"), 0, 0)

    eq = pd.to_numeric(c["Equity"], errors="coerce").values.astype(float)
    dt = _safe_to_dt(c["Date"]).values
    if len(eq) < 3:
        return (float("nan"), 0, 0)

    peak = np.maximum.accumulate(eq)
    dd = 1.0 - (eq / peak)
    max_dd = float(np.nanmax(dd)) if np.isfinite(np.nanmax(dd)) else float("nan")

    under = dd > 1e-12
    max_under = 0
    cur = 0
    for u in under:
        if u:
            cur += 1
            max_under = max(max_under, cur)
        else:
            cur = 0

    is_new_high = eq >= peak - 1e-12
    max_recovery = 0
    i = 0
    n = len(eq)
    while i < n:
        if is_new_high[i]:
            i += 1
            continue
        trough_i = i
        while i < n and not is_new_high[i]:
            if eq[i] < eq[trough_i]:
                trough_i = i
            i += 1
        if i < n and is_new_high[i]:
            rec_days = int((pd.Timestamp(dt[i]) - pd.Timestamp(dt[trough_i])).days)
            max_recovery = max(max_recovery, rec_days)
        i += 1

    return (max_dd, int(max_under), int(max_recovery))


def _daily_risk_stats(c: pd.DataFrame) -> tuple[float, float, float]:
    """
    returns (DailyVol_Ann, Sharpe0_Ann, Sortino0_Ann)
    """
    if c is None or c.empty or len(c) < 5:
        return (float("nan"), float("nan"), float("nan"))

    eq = pd.to_numeric(c["Equity"], errors="coerce")
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return (float("nan"), float("nan"), float("nan"))

    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    vol = float(sd * np.sqrt(252.0)) if np.isfinite(sd) else float("nan")
    sharpe = float((mu / sd) * np.sqrt(252.0)) if np.isfinite(mu) and np.isfinite(sd) and sd > 0 else float("nan")

    neg = r[r < 0]
    dd_sd = float(neg.std(ddof=1)) if len(neg) >= 2 else float("nan")
    sortino = float((mu / dd_sd) * np.sqrt(252.0)) if np.isfinite(mu) and np.isfinite(dd_sd) and dd_sd > 0 else float("nan")

    return (vol, sharpe, sortino)


def _recent_window(c: pd.DataFrame, years: int) -> pd.DataFrame:
    if c is None or c.empty:
        return pd.DataFrame()
    end = pd.Timestamp(c["Date"].iloc[-1])
    start = end - pd.Timedelta(days=int(years * 365))
    return c[c["Date"] >= start].copy()


def _read_prices_if_any() -> pd.DataFrame:
    # aggregate job에서 raw-data를 안 받는 경우가 있어서 "있으면 쓰고, 없으면 NaN"
    cand = [
        Path("data/raw/prices.parquet"),
        Path("data/raw/prices.csv"),
        Path("data/prices.parquet"),
        Path("data/prices.csv"),
    ]
    p = _find_first_existing(cand)
    if p is None:
        return pd.DataFrame()

    df = _read_any(p)
    if df.empty:
        return pd.DataFrame()

    if "Date" not in df.columns or "Ticker" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["Date"] = _safe_to_dt(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def _qqq_mult_same_period(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if prices is None or prices.empty:
        return float("nan")
    q = prices[prices["Ticker"] == "QQQ"].copy()
    if q.empty:
        return float("nan")
    q = q.sort_values("Date")
    q0 = q[q["Date"] >= start]
    q1 = q[q["Date"] <= end]
    if q0.empty or q1.empty:
        return float("nan")
    p0 = float(q0["Close"].iloc[0])
    p1 = float(q1["Close"].iloc[-1])
    if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(p1) or p1 <= 0:
        return float("nan")
    return float(p1 / p0)


# -----------------------
# BadExit stats
# -----------------------
def _is_badexit_reason(reason: str) -> int:
    r = str(reason or "").strip().upper()
    if r.startswith("REVAL_FAIL"):
        return 1
    if r.startswith("GRACE_END_EXIT"):
        return 1
    return 0


def _badexit_stats(trades: pd.DataFrame) -> dict:
    out = {
        "BadExitRate_Row": float("nan"),
        "BadExitRate_Ticker": float("nan"),
        "BadExitReasonShare_RevalFail": float("nan"),
        "BadExitReasonShare_GraceEnd": float("nan"),
        "BadExitReturnMean": float("nan"),
        "NonBadExitReturnMean": float("nan"),
        "BadExitReturnDiff": float("nan"),
    }
    if trades is None or trades.empty:
        return out
    if "Reason" not in trades.columns:
        return out

    t = trades.copy()
    t["Reason"] = t["Reason"].astype(str)
    t["BadExit"] = t["Reason"].apply(_is_badexit_reason).astype(int)
    out["BadExitRate_Row"] = float(t["BadExit"].mean()) if len(t) else float("nan")

    # ticker-exploded
    if "Tickers" in t.columns:
        rows = []
        for _, r in t.iterrows():
            ticks = [x.strip().upper() for x in str(r.get("Tickers", "")).split(",") if x.strip()]
            for tk in ticks:
                rows.append({"Ticker": tk, "BadExit": int(r["BadExit"]), "Reason": str(r["Reason"])})
        if rows:
            e = pd.DataFrame(rows)
            out["BadExitRate_Ticker"] = float(e["BadExit"].mean())

            be = e[e["BadExit"] == 1]
            if len(be):
                rr = be["Reason"].str.upper().str.strip()
                out["BadExitReasonShare_RevalFail"] = float((rr.str.startswith("REVAL_FAIL")).mean())
                out["BadExitReasonShare_GraceEnd"] = float((rr.str.startswith("GRACE_END_EXIT")).mean())

    # return diff (if available)
    ret_col = None
    for c in ["CycleReturn", "Return", "PnLPct", "PnL"]:
        if c in t.columns:
            ret_col = c
            break
    if ret_col is not None:
        rr = pd.to_numeric(t[ret_col], errors="coerce")
        be = rr[t["BadExit"] == 1].dropna()
        nb = rr[t["BadExit"] == 0].dropna()
        if len(be):
            out["BadExitReturnMean"] = float(be.mean())
        if len(nb):
            out["NonBadExitReturnMean"] = float(nb.mean())
        if np.isfinite(out["BadExitReturnMean"]) and np.isfinite(out["NonBadExitReturnMean"]):
            out["BadExitReturnDiff"] = float(out["BadExitReturnMean"] - out["NonBadExitReturnMean"])

    return out


def _trade_structure_stats(trades: pd.DataFrame, curve: pd.DataFrame) -> dict:
    out = {
        "TradeCount": float("nan"),
        "TradesPerYear": float("nan"),
        "HoldDays_Mean": float("nan"),
        "HoldDays_Median": float("nan"),
        "HoldDays_P90": float("nan"),
    }
    if trades is None or trades.empty:
        return out

    out["TradeCount"] = float(len(trades))

    # trades per year: span from curve if possible
    days = float("nan")
    if curve is not None and not curve.empty:
        days = float((pd.Timestamp(curve["Date"].iloc[-1]) - pd.Timestamp(curve["Date"].iloc[0])).days)
    elif "EntryDate" in trades.columns:
        dt = _safe_to_dt(trades["EntryDate"]).dropna()
        if len(dt) >= 2:
            days = float((dt.max() - dt.min()).days)

    if np.isfinite(days) and days > 0:
        out["TradesPerYear"] = float(len(trades) / (days / 365.0))

    # holding days
    hd_col = None
    for c in ["HoldingDays", "HoldDays", "DaysHeld"]:
        if c in trades.columns:
            hd_col = c
            break
    if hd_col is not None:
        hd = pd.to_numeric(trades[hd_col], errors="coerce").dropna()
        if len(hd):
            out["HoldDays_Mean"] = float(hd.mean())
            out["HoldDays_Median"] = float(hd.median())
            out["HoldDays_P90"] = float(hd.quantile(0.90))

    return out


# -----------------------
# file name helpers
# -----------------------
def _suffix_from_gate_summary_name(p: Path) -> str:
    # gate_summary_<TAG>_gate_<suffix>.csv
    m = re.match(r"gate_summary_(.+)_gate_(.+)\.csv$", p.name)
    if m:
        return m.group(2)
    return p.stem.replace("gate_summary_", "")


def _tag_from_gate_summary_name(p: Path, period: str) -> str:
    m = re.match(r"gate_summary_(.+)_gate_(.+)\.csv$", p.name)
    if m:
        return m.group(1)
    # fallback
    return f"wf_{period}"


def _find_curve_trades(period_dir: Path, tag: str, suffix: str) -> tuple[Path | None, Path | None]:
    # prefer parquet, fallback csv
    curve = _find_first_existing([
        period_dir / f"sim_engine_curve_{tag}_gate_{suffix}.parquet",
        period_dir / f"sim_engine_curve_{tag}_gate_{suffix}.csv",
    ])
    trades = _find_first_existing([
        period_dir / f"sim_engine_trades_{tag}_gate_{suffix}.parquet",
        period_dir / f"sim_engine_trades_{tag}_gate_{suffix}.csv",
    ])
    return curve, trades


# -----------------------
# main
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str, help="root folder with subdirs like 2024H1, 2024H2, ...")
    ap.add_argument("--out", required=True, type=str, help="output summary csv path")
    ap.add_argument("--pattern", default="gate_summary_*.csv", type=str)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    prices = _read_prices_if_any()

    rows: list[dict] = []

    period_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not period_dirs:
        raise FileNotFoundError(f"no period dirs under: {root}")

    for pd_dir in period_dirs:
        period = pd_dir.name

        summ_files = sorted(pd_dir.glob(args.pattern))
        if not summ_files:
            # period dir may contain only curves/trades without gate_summary; skip quietly
            continue

        for sp in summ_files:
            base = _read_any(sp)
            if base.empty:
                continue
            row = base.iloc[0].to_dict()

            suffix = row.get("GateSuffix")
            if not suffix:
                suffix = _suffix_from_gate_summary_name(sp)
            suffix = str(suffix)

            tag = row.get("TAG")
            if not tag:
                tag = _tag_from_gate_summary_name(sp, period)
            tag = str(tag)

            row["WF_PERIOD"] = period
            row["TAG"] = tag
            row["GateSuffix"] = suffix

            # locate curve/trades
            cpath, tpath = _find_curve_trades(pd_dir, tag, suffix)
            curve = _clean_curve(_read_any(cpath) if cpath else pd.DataFrame())
            trades = _read_any(tpath) if tpath else pd.DataFrame()

            # warmup end from first EntryDate
            warmup_end = pd.NaT
            if trades is not None and not trades.empty and "EntryDate" in trades.columns:
                entry = _safe_to_dt(trades["EntryDate"]).dropna()
                if len(entry):
                    warmup_end = entry.min()

            row["WarmupEndDate"] = str(warmup_end.date()) if pd.notna(warmup_end) else ""

            # after-warmup curve
            cur2 = pd.DataFrame()
            if pd.notna(warmup_end) and curve is not None and not curve.empty:
                cur2 = curve[curve["Date"] >= warmup_end].copy()

            # ---- seed multiple 확장 (curve 기준)
            sm_aw, cagr_aw, days_aw = _seed_multiple_and_cagr(cur2)
            row["SeedMultiple_AfterWarmup"] = sm_aw
            row["CAGR_AfterWarmup"] = cagr_aw
            row["BacktestDaysAfterWarmup"] = days_aw

            # recent 5y (after-warmup)
            cur5 = _recent_window(cur2, 5)
            sm5, cagr5, _ = _seed_multiple_and_cagr(cur5)
            row["Recent5Y_SeedMultiple_AfterWarmup"] = sm5
            row["Recent5Y_CAGR_AfterWarmup"] = cagr5

            # QQQ same period -> ExcessSeedMultiple_AfterWarmup
            if cur2 is not None and not cur2.empty:
                start_dt = pd.Timestamp(cur2["Date"].iloc[0])
                end_dt = pd.Timestamp(cur2["Date"].iloc[-1])
                qqq_mult = _qqq_mult_same_period(prices, start_dt, end_dt)
            else:
                qqq_mult = float("nan")

            row["QQQ_SeedMultiple_SamePeriod"] = qqq_mult
            if np.isfinite(sm_aw) and np.isfinite(qqq_mult) and qqq_mult > 0:
                row["ExcessSeedMultiple_AfterWarmup"] = float(sm_aw / qqq_mult)
            else:
                row["ExcessSeedMultiple_AfterWarmup"] = float("nan")

            # ---- 리스크/회복력
            maxdd, maxuw, maxrec = _max_dd_stats(cur2)
            row["MaxDD_AfterWarmup"] = maxdd
            row["MaxUnderwaterDays_AfterWarmup"] = float(maxuw)
            row["MaxDDRecoveryDays_AfterWarmup"] = float(maxrec)

            # ---- 운영/거래 구조
            row.update(_trade_structure_stats(trades, cur2))

            # ---- BadExit 분포
            row.update(_badexit_stats(trades))

            # ---- 일간 통계 (equity curve 기반)
            vol, sharpe, sortino = _daily_risk_stats(cur2)
            row["DailyVol_Ann_AfterWarmup"] = vol
            row["Sharpe0_Ann_AfterWarmup"] = sharpe
            row["Sortino0_Ann_AfterWarmup"] = sortino

            rows.append(row)

    if not rows:
        raise RuntimeError("no rows aggregated (no gate_summary files found?)")

    out = pd.DataFrame(rows)

    # numeric coercion (중요: 문자열/None 때문에 0으로 떨어지는 걸 막음)
    must_numeric = [
        "SeedMultiple_AfterWarmup",
        "CAGR_AfterWarmup",
        "BacktestDaysAfterWarmup",
        "Recent5Y_SeedMultiple_AfterWarmup",
        "Recent5Y_CAGR_AfterWarmup",
        "QQQ_SeedMultiple_SamePeriod",
        "ExcessSeedMultiple_AfterWarmup",
        "MaxDD_AfterWarmup",
        "MaxUnderwaterDays_AfterWarmup",
        "MaxDDRecoveryDays_AfterWarmup",
        "TradeCount",
        "TradesPerYear",
        "HoldDays_Mean",
        "HoldDays_Median",
        "HoldDays_P90",
        "BadExitRate_Row",
        "BadExitRate_Ticker",
        "BadExitReasonShare_RevalFail",
        "BadExitReasonShare_GraceEnd",
        "BadExitReturnMean",
        "NonBadExitReturnMean",
        "BadExitReturnDiff",
        "DailyVol_Ann_AfterWarmup",
        "Sharpe0_Ann_AfterWarmup",
        "Sortino0_Ann_AfterWarmup",
    ]
    for c in must_numeric:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[DONE] wrote: {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()