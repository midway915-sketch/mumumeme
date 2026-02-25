#!/usr/bin/env python3
# scripts/aggregate_walkforward_halfyear.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# helpers
# -------------------------
def _safe_dt(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _to_float(x) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")


def _pp_to_float(tok: str) -> float:
    # "0p10" -> 0.10, "1p0" -> 1.0
    if tok is None:
        return float("nan")
    t = str(tok).strip()
    t = t.replace("p", ".")
    try:
        return float(t)
    except Exception:
        return float("nan")


def _calc_cagr(mult: float, days: float) -> float:
    if not np.isfinite(mult) or mult <= 0:
        return float("nan")
    if not np.isfinite(days) or days <= 0:
        return float("nan")
    years = days / 365.0
    if years <= 0:
        return float("nan")
    return float(mult ** (1.0 / years) - 1.0)


def _read_any(path_parq: Path, path_csv: Path) -> pd.DataFrame:
    if path_parq.exists():
        return pd.read_parquet(path_parq)
    if path_csv.exists():
        return pd.read_csv(path_csv)
    raise FileNotFoundError(f"missing: {path_parq} (or {path_csv})")


def _find_first_existing(paths) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _is_badexit_reason(reason: str) -> int:
    r = str(reason or "").strip().upper()
    if r.startswith("REVAL_FAIL"):
        return 1
    if r.startswith("GRACE_END_EXIT"):
        return 1
    return 0


def _parse_suffix_meta(suffix: str) -> Dict[str, object]:
    """
    suffix example:
      tail_utility_tu1_t0p3_q0p9_rutility_lam1p0_ps0p0_be1p0_k1_w1.0_tp50_tr0p1_capnone
    or gate mode chunk may vary.

    We output keys that analyze expects:
      suffix, cap, ps_min, tail_threshold, utility_quantile, lambda_tail, topk, badexit_max
    plus some extras (gate_mode, trail_stop, tp1_frac).
    """
    s = str(suffix or "").strip()

    out: Dict[str, object] = {
        "suffix": s,
        "cap": "",
        "ps_min": float("nan"),
        "tail_threshold": float("nan"),
        "utility_quantile": float("nan"),
        "lambda_tail": float("nan"),
        "topk": float("nan"),
        "badexit_max": float("nan"),
        "gate_mode": "",
        "trail_stop": float("nan"),
        "tp1_frac": float("nan"),
    }

    # cap
    m = re.search(r"_cap(none|h2|total)$", s)
    if m:
        out["cap"] = m.group(1)

    # gate_mode: take prefix before first "_tu" token if exists, else first chunk
    # e.g. "tail_utility_tu1_..." -> "tail_utility"
    gm = s
    m2 = re.search(r"^(.*?)(?:_tu\d+_)", s)
    if m2:
        gm = m2.group(1)
    else:
        gm = s.split("_")[0] if s else ""
    out["gate_mode"] = gm

    # tail threshold t0p3
    m = re.search(r"(?:^|_)t(\d+p\d+)", s)
    if m:
        out["tail_threshold"] = _pp_to_float(m.group(1))

    # utility quantile q0p9
    m = re.search(r"(?:^|_)q(\d+p\d+)", s)
    if m:
        out["utility_quantile"] = _pp_to_float(m.group(1))

    # lambda lam1p0
    m = re.search(r"(?:^|_)lam(\d+p\d+)", s)
    if m:
        out["lambda_tail"] = _pp_to_float(m.group(1))

    # ps ps0p0
    m = re.search(r"(?:^|_)ps(\d+p\d+)", s)
    if m:
        out["ps_min"] = _pp_to_float(m.group(1))

    # badexit max be1p0
    m = re.search(r"(?:^|_)be(\d+p\d+)", s)
    if m:
        out["badexit_max"] = _pp_to_float(m.group(1))

    # topk k1
    m = re.search(r"(?:^|_)k(\d+)", s)
    if m:
        out["topk"] = _to_float(m.group(1))

    # trail stop tr0p1
    m = re.search(r"(?:^|_)tr(\d+p\d+)", s)
    if m:
        out["trail_stop"] = _pp_to_float(m.group(1))

    # tp1 fraction tp50 -> 0.50 (percent-like)
    m = re.search(r"(?:^|_)tp(\d+)", s)
    if m:
        v = _to_float(m.group(1))
        out["tp1_frac"] = v / 100.0 if np.isfinite(v) else float("nan")

    return out


def _parse_period_and_suffix_from_filename(name: str) -> Tuple[str, str]:
    """
    Expect:
      sim_engine_trades_wf_2024H1_gate_<suffix>.csv
      sim_engine_curve_wf_2024H1_gate_<suffix>.parquet
    """
    m = re.search(r"(?:trades|curve)_(wf_\d{4}H[12])_gate_(.+)\.(?:csv|parquet)$", name)
    if not m:
        return ("", "")
    tag = m.group(1)          # wf_2024H1
    period = tag.replace("wf_", "")
    suffix = m.group(2)
    return (period, suffix)


def _load_trades(trades_path: Path) -> pd.DataFrame:
    if trades_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(trades_path)
    else:
        df = pd.read_csv(trades_path)

    if df.empty:
        return df

    # normalize
    for c in ["EntryDate", "ExitDate"]:
        if c in df.columns:
            df[c] = _safe_dt(df[c])

    if "Tickers" in df.columns:
        df["Tickers"] = df["Tickers"].astype(str)

    if "Reason" in df.columns:
        df["Reason"] = df["Reason"].astype(str)
    else:
        df["Reason"] = ""

    # ensure returns column
    if "CycleReturn" in df.columns:
        df["CycleReturn"] = pd.to_numeric(df["CycleReturn"], errors="coerce")
    elif "Return" in df.columns:
        df["CycleReturn"] = pd.to_numeric(df["Return"], errors="coerce")
    else:
        df["CycleReturn"] = np.nan

    # holding days
    if "HoldingDays" in df.columns:
        df["HoldingDays"] = pd.to_numeric(df["HoldingDays"], errors="coerce")
    else:
        df["HoldingDays"] = np.nan

    df["BadExit"] = df["Reason"].apply(_is_badexit_reason).astype(int)

    return df


def _load_curve(curve_path: Path) -> pd.DataFrame:
    if curve_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(curve_path)
    else:
        df = pd.read_csv(curve_path)

    if df.empty:
        return df

    if "Date" in df.columns:
        df["Date"] = _safe_dt(df["Date"])
    if "Equity" in df.columns:
        df["Equity"] = pd.to_numeric(df["Equity"], errors="coerce")

    # normalize InCycle flag
    if "InCycle" in df.columns:
        inc = pd.to_numeric(df["InCycle"], errors="coerce").fillna(0).astype(int)
    elif "InPosition" in df.columns:
        inc = pd.to_numeric(df["InPosition"], errors="coerce").fillna(0).astype(int)
    else:
        inc = pd.Series([0] * len(df), index=df.index, dtype=int)
    df["InCycle"] = (inc > 0).astype(int)

    df = df.dropna(subset=["Date", "Equity"]).sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return df


def _equity_stats_after_warmup(curve: pd.DataFrame, warmup_end: pd.Timestamp) -> Dict[str, float]:
    """
    curve: Date/Equity/InCycle
    warmup_end: earliest entry date (from trades)
    """
    out: Dict[str, float] = {
        "WarmupEndDate": "",
        "BacktestDaysAfterWarmup": float("nan"),
        "SeedMultiple_AfterWarmup": float("nan"),
        "CAGR_AfterWarmup": float("nan"),
        "ActiveDaysAfterWarmup": float("nan"),
        "IdleDaysAfterWarmup": float("nan"),
        "IdlePctAfterWarmup": float("nan"),
        "Recent5Y_SeedMultiple_AfterWarmup": float("nan"),
        "Recent5Y_CAGR_AfterWarmup": float("nan"),
        "MaxDD_AfterWarmup": float("nan"),
        "MaxUnderwaterDays_AfterWarmup": float("nan"),
        "MaxDDRecoveryDays_AfterWarmup": float("nan"),
        "DailyVol_AfterWarmup": float("nan"),
        "Sharpe0_AfterWarmup": float("nan"),
        "Sortino0_AfterWarmup": float("nan"),
    }

    if curve is None or curve.empty or "Date" not in curve.columns or "Equity" not in curve.columns:
        return out

    if pd.isna(warmup_end):
        return out

    c = curve[curve["Date"] >= warmup_end].copy()
    if c.empty:
        out["WarmupEndDate"] = str(pd.Timestamp(warmup_end).date())
        return out

    c = c.dropna(subset=["Date", "Equity"]).sort_values("Date").reset_index(drop=True)
    start_date = pd.Timestamp(c["Date"].iloc[0])
    end_date = pd.Timestamp(c["Date"].iloc[-1])
    days = float((end_date - start_date).days)

    start_eq = float(c["Equity"].iloc[0])
    end_eq = float(c["Equity"].iloc[-1])
    mult = (end_eq / start_eq) if (np.isfinite(start_eq) and start_eq > 0 and np.isfinite(end_eq) and end_eq > 0) else float("nan")

    cagr = _calc_cagr(mult, days)

    # active/idle
    total_days = int(c["Date"].nunique())
    active_days = int((c["InCycle"] > 0).sum()) if "InCycle" in c.columns else 0
    idle_days = int(max(0, total_days - active_days))
    idle_pct = float(idle_days / total_days) if total_days > 0 else float("nan")

    out["WarmupEndDate"] = str(pd.Timestamp(warmup_end).date())
    out["BacktestDaysAfterWarmup"] = days
    out["SeedMultiple_AfterWarmup"] = float(mult) if np.isfinite(mult) else float("nan")
    out["CAGR_AfterWarmup"] = float(cagr) if np.isfinite(cagr) else float("nan")
    out["ActiveDaysAfterWarmup"] = float(active_days)
    out["IdleDaysAfterWarmup"] = float(idle_days)
    out["IdlePctAfterWarmup"] = float(idle_pct) if np.isfinite(idle_pct) else float("nan")

    # recent 5y
    if np.isfinite(days) and days > 0:
        window_start = end_date - pd.Timedelta(days=int(round(365.25 * 5)))
        c5 = c[c["Date"] >= window_start].copy()
        if len(c5) >= 2:
            s5 = float(c5["Equity"].iloc[0])
            e5 = float(c5["Equity"].iloc[-1])
            d5 = float((pd.Timestamp(c5["Date"].iloc[-1]) - pd.Timestamp(c5["Date"].iloc[0])).days)
            m5 = (e5 / s5) if (np.isfinite(s5) and s5 > 0 and np.isfinite(e5) and e5 > 0) else float("nan")
            out["Recent5Y_SeedMultiple_AfterWarmup"] = float(m5) if np.isfinite(m5) else float("nan")
            out["Recent5Y_CAGR_AfterWarmup"] = float(_calc_cagr(m5, d5)) if np.isfinite(m5) and np.isfinite(d5) else float("nan")

    # drawdown stats
    eq = c["Equity"].astype(float).to_numpy()
    run_max = np.maximum.accumulate(eq)
    dd = (eq / run_max) - 1.0
    maxdd = float(np.nanmin(dd)) if dd.size else float("nan")
    out["MaxDD_AfterWarmup"] = maxdd

    # underwater / recovery days
    # underwater: days since last high
    underwater = eq < run_max
    max_underwater = 0
    cur = 0
    for u in underwater:
        if u:
            cur += 1
            max_underwater = max(max_underwater, cur)
        else:
            cur = 0
    out["MaxUnderwaterDays_AfterWarmup"] = float(max_underwater)

    # recovery days from max drawdown trough to next new high
    if dd.size:
        trough_idx = int(np.nanargmin(dd))
        trough_peak = run_max[trough_idx]
        # find first index after trough where equity >= previous peak at trough
        rec_idx = None
        for j in range(trough_idx + 1, len(eq)):
            if eq[j] >= trough_peak:
                rec_idx = j
                break
        if rec_idx is None:
            out["MaxDDRecoveryDays_AfterWarmup"] = float("nan")
        else:
            out["MaxDDRecoveryDays_AfterWarmup"] = float(rec_idx - trough_idx)

    # daily vol/sharpe/sortino (0 rf)
    if len(eq) >= 3:
        rets = (eq[1:] / eq[:-1]) - 1.0
        rets = rets[np.isfinite(rets)]
        if rets.size >= 2:
            vol = float(np.std(rets, ddof=1))
            out["DailyVol_AfterWarmup"] = vol
            mu = float(np.mean(rets))
            if vol > 0:
                out["Sharpe0_AfterWarmup"] = float(mu / vol)
            neg = rets[rets < 0]
            if neg.size >= 2:
                down = float(np.std(neg, ddof=1))
                if down > 0:
                    out["Sortino0_AfterWarmup"] = float(mu / down)

    return out


def _badexit_stats(trades: pd.DataFrame) -> Dict[str, float]:
    out = {
        "TradeCount": float("nan"),
        "TradesPerYear": float("nan"),
        "HoldDays_Mean": float("nan"),
        "HoldDays_Median": float("nan"),
        "HoldDays_P90": float("nan"),
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

    n = len(trades)
    out["TradeCount"] = float(n)

    # trades per year (based on entry date span)
    if "EntryDate" in trades.columns:
        e = _safe_dt(trades["EntryDate"]).dropna()
        if len(e) >= 2:
            days = float((e.max() - e.min()).days)
            if np.isfinite(days) and days > 0:
                out["TradesPerYear"] = float(n / (days / 365.0))

    # holding days stats
    hd = pd.to_numeric(trades.get("HoldingDays", pd.Series([], dtype=float)), errors="coerce").dropna()
    if len(hd):
        out["HoldDays_Mean"] = float(hd.mean())
        out["HoldDays_Median"] = float(hd.median())
        out["HoldDays_P90"] = float(hd.quantile(0.9))

    # badexit row rate
    be = pd.to_numeric(trades.get("BadExit", pd.Series([0] * n)), errors="coerce").fillna(0).astype(int)
    out["BadExitRate_Row"] = float(be.mean()) if n > 0 else float("nan")

    # ticker-expanded rate
    if "Tickers" in trades.columns:
        rows = []
        for _, r in trades.iterrows():
            tickers = [t.strip().upper() for t in str(r.get("Tickers", "")).split(",") if t.strip()]
            b = int(r.get("BadExit", 0) or 0)
            for _t in tickers:
                rows.append(b)
        if rows:
            out["BadExitRate_Ticker"] = float(np.mean(rows))

    # reason shares among badexit only
    if "Reason" in trades.columns:
        bad = trades[be == 1].copy()
        if len(bad) > 0:
            reason = bad["Reason"].astype(str).str.upper()
            rf = float((reason.str.startswith("REVAL_FAIL")).mean())
            ge = float((reason.str.startswith("GRACE_END_EXIT")).mean())
            out["BadExitReasonShare_RevalFail"] = rf
            out["BadExitReasonShare_GraceEnd"] = ge

    # badexit vs nonbadexit return
    ret = pd.to_numeric(trades.get("CycleReturn", np.nan), errors="coerce")
    bad_ret = ret[be == 1].dropna()
    ok_ret = ret[be == 0].dropna()
    if len(bad_ret):
        out["BadExitReturnMean"] = float(bad_ret.mean())
    if len(ok_ret):
        out["NonBadExitReturnMean"] = float(ok_ret.mean())
    if np.isfinite(out["BadExitReturnMean"]) and np.isfinite(out["NonBadExitReturnMean"]):
        out["BadExitReturnDiff"] = float(out["BadExitReturnMean"] - out["NonBadExitReturnMean"])

    return out


def _qqq_mult_same_period(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if prices is None or prices.empty:
        return float("nan")
    q = prices[prices["Ticker"] == "QQQ"].copy()
    if q.empty:
        return float("nan")
    q["Date"] = _safe_dt(q["Date"])
    q["Close"] = pd.to_numeric(q["Close"], errors="coerce")
    q = q.dropna(subset=["Date", "Close"]).sort_values("Date")
    s = q[q["Date"] >= start]
    e = q[q["Date"] <= end]
    if s.empty or e.empty:
        return float("nan")
    p0 = float(s["Close"].iloc[0])
    p1 = float(e["Close"].iloc[-1])
    if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(p1) or p1 <= 0:
        return float("nan")
    return float(p1 / p0)


def _load_prices(raw_dir: Path) -> pd.DataFrame:
    parq = raw_dir / "prices.parquet"
    csv = raw_dir / "prices.csv"
    df = _read_any(parq, csv)
    if not {"Date", "Ticker", "Close"}.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()
    df["Date"] = _safe_dt(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


# -------------------------
# main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="e.g. data/signals/walkforward2")
    ap.add_argument("--out", type=str, required=True, help="e.g. data/signals/walkforward2/_summary_walkforward2.csv")
    ap.add_argument("--raw-prices-dir", type=str, default="data/raw", help="for QQQ mult reference")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    prices = _load_prices(Path(args.raw_prices_dir))

    rows = []
    # each period folder
    period_dirs = sorted([p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}H[12]", p.name)])
    if not period_dirs:
        raise FileNotFoundError(f"no period dirs found under {root}")

    for pdir in period_dirs:
        period = pdir.name

        # trades files
        trades_files = sorted(list(pdir.glob("sim_engine_trades_*_gate_*.parquet")) + list(pdir.glob("sim_engine_trades_*_gate_*.csv")))
        # curve files
        curve_files = sorted(list(pdir.glob("sim_engine_curve_*_gate_*.parquet")) + list(pdir.glob("sim_engine_curve_*_gate_*.csv")))

        # map suffix -> path
        trades_map: Dict[str, Path] = {}
        for tf in trades_files:
            per2, suf = _parse_period_and_suffix_from_filename(tf.name)
            if per2 != period or not suf:
                continue
            # prefer parquet over csv
            if suf not in trades_map or (trades_map[suf].suffix.lower() != ".parquet" and tf.suffix.lower() == ".parquet"):
                trades_map[suf] = tf

        curve_map: Dict[str, Path] = {}
        for cf in curve_files:
            per2, suf = _parse_period_and_suffix_from_filename(cf.name)
            if per2 != period or not suf:
                continue
            if suf not in curve_map or (curve_map[suf].suffix.lower() != ".parquet" and cf.suffix.lower() == ".parquet"):
                curve_map[suf] = cf

        suffixes = sorted(set(list(trades_map.keys()) + list(curve_map.keys())))
        for suffix in suffixes:
            r: Dict[str, object] = {}

            r["period"] = period
            r.update(_parse_suffix_meta(suffix))

            # file refs (analyze expects curve_file)
            r["trades_file"] = str(trades_map.get(suffix, ""))
            r["curve_file"] = str(curve_map.get(suffix, ""))

            trades = None
            if suffix in trades_map:
                try:
                    trades = _load_trades(trades_map[suffix])
                except Exception:
                    trades = None

            curve = None
            if suffix in curve_map:
                try:
                    curve = _load_curve(curve_map[suffix])
                except Exception:
                    curve = None

            # warmup end from trades
            warmup_end = pd.NaT
            if trades is not None and not trades.empty and "EntryDate" in trades.columns:
                e = trades["EntryDate"].dropna()
                if len(e):
                    warmup_end = pd.Timestamp(e.min())

            # equity stats
            eq_stats = _equity_stats_after_warmup(curve, warmup_end) if curve is not None else _equity_stats_after_warmup(pd.DataFrame(), warmup_end)
            for k, v in eq_stats.items():
                r[k] = v

            # add QQQ reference + excess seed multiple
            # (same period as AfterWarmup window)
            if curve is not None and not curve.empty and isinstance(warmup_end, pd.Timestamp) and not pd.isna(warmup_end):
                c2 = curve[curve["Date"] >= warmup_end].copy()
                if not c2.empty:
                    start_date = pd.Timestamp(c2["Date"].iloc[0])
                    end_date = pd.Timestamp(c2["Date"].iloc[-1])
                    qqq_mult = _qqq_mult_same_period(prices, start_date, end_date)
                    r["QQQ_SeedMultiple_SamePeriod"] = qqq_mult
                    sm = _to_float(r.get("SeedMultiple_AfterWarmup"))
                    if np.isfinite(sm) and np.isfinite(qqq_mult) and qqq_mult > 0:
                        r["ExcessSeedMultiple_AfterWarmup"] = float(sm / qqq_mult)
                    else:
                        r["ExcessSeedMultiple_AfterWarmup"] = float("nan")
            else:
                r["QQQ_SeedMultiple_SamePeriod"] = float("nan")
                r["ExcessSeedMultiple_AfterWarmup"] = float("nan")

            # badexit + trade structure
            be_stats = _badexit_stats(trades) if trades is not None else _badexit_stats(pd.DataFrame())
            for k, v in be_stats.items():
                r[k] = v

            rows.append(r)

    out = pd.DataFrame(rows)

    # numeric coercion (avoid "0" strings + make analyze stable)
    num_cols = [
        "ps_min", "tail_threshold", "utility_quantile", "lambda_tail", "topk", "badexit_max",
        "trail_stop", "tp1_frac",
        "BacktestDaysAfterWarmup", "SeedMultiple_AfterWarmup", "CAGR_AfterWarmup",
        "Recent5Y_SeedMultiple_AfterWarmup", "Recent5Y_CAGR_AfterWarmup",
        "QQQ_SeedMultiple_SamePeriod", "ExcessSeedMultiple_AfterWarmup",
        "ActiveDaysAfterWarmup", "IdleDaysAfterWarmup", "IdlePctAfterWarmup",
        "MaxDD_AfterWarmup", "MaxUnderwaterDays_AfterWarmup", "MaxDDRecoveryDays_AfterWarmup",
        "DailyVol_AfterWarmup", "Sharpe0_AfterWarmup", "Sortino0_AfterWarmup",
        "TradeCount", "TradesPerYear",
        "HoldDays_Mean", "HoldDays_Median", "HoldDays_P90",
        "BadExitRate_Row", "BadExitRate_Ticker",
        "BadExitReasonShare_RevalFail", "BadExitReasonShare_GraceEnd",
        "BadExitReturnMean", "NonBadExitReturnMean", "BadExitReturnDiff",
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[DONE] wrote summary: {out_path} rows={len(out)}")
    print("[INFO] columns:", ", ".join(list(out.columns)[:50]) + (" ..." if len(out.columns) > 50 else ""))


if __name__ == "__main__":
    main()