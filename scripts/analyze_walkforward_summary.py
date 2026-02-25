#!/usr/bin/env python3
# scripts/analyze_walkforward_summary.py
from __future__ import annotations

import argparse
from pathlib import Path
import math
import re
import numpy as np
import pandas as pd


# -----------------------
# IO helpers
# -----------------------
def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _safe_to_dt(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.tz_localize(None)


def _to_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _calc_cagr(mult: float, days: float) -> float:
    if not np.isfinite(mult) or mult <= 0:
        return float("nan")
    if not np.isfinite(days) or days <= 0:
        return float("nan")
    years = days / 365.0
    if years <= 0:
        return float("nan")
    return float(mult ** (1.0 / years) - 1.0)


def _prefer_parquet_dedupe(df: pd.DataFrame, subset_cols: list[str]) -> pd.DataFrame:
    """
    같은 config+period에 curve_file이 csv/parquet 2줄씩 들어오는 케이스 방지.
    parquet 우선으로 한 줄만 남김.
    """
    if df.empty:
        return df
    if "curve_file" not in df.columns:
        return df.drop_duplicates(subset=subset_cols, keep="first")

    tmp = df.copy()
    tmp["__is_parq__"] = tmp["curve_file"].astype(str).str.lower().str.endswith(".parquet").astype(int)
    tmp = tmp.sort_values(subset_cols + ["__is_parq__"], ascending=[True] * len(subset_cols) + [False])
    tmp = tmp.drop_duplicates(subset=subset_cols, keep="first").drop(columns=["__is_parq__"])
    return tmp


# -----------------------
# Curve stitching + stats
# -----------------------
def _clean_curve(curve: pd.DataFrame) -> pd.DataFrame:
    if curve is None or curve.empty:
        return pd.DataFrame()

    c = curve.copy()
    if "Date" not in c.columns or "Equity" not in c.columns:
        return pd.DataFrame()

    c["Date"] = _safe_to_dt(c["Date"])
    c["Equity"] = _to_num(c["Equity"])

    if "InCycle" in c.columns:
        inc = _to_num(c["InCycle"]).fillna(0).astype(int)
        c["InCycle"] = (inc > 0).astype(int)
    elif "InPosition" in c.columns:
        inc = _to_num(c["InPosition"]).fillna(0).astype(int)
        c["InCycle"] = (inc > 0).astype(int)
    else:
        c["InCycle"] = 0

    c = c.dropna(subset=["Date", "Equity"]).sort_values("Date").reset_index(drop=True)
    c = c.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return c


def _stitch_curves(curve_paths: list[Path]) -> pd.DataFrame:
    """
    period별로 equity가 리셋되는 curve들을 "연속 수익률"로 이어붙임.
    - 각 period curve를 (Equity / 첫 Equity)로 정규화
    - 누적 multiplier를 곱해서 전체 연속 Equity를 만듦
    """
    out_parts = []
    cum_mult = 1.0

    for p in curve_paths:
        c = _clean_curve(_read_any(p))
        if c.empty:
            continue

        e0 = float(c["Equity"].iloc[0])
        if not np.isfinite(e0) or e0 <= 0:
            continue

        c = c.copy()
        c["EquityNorm"] = c["Equity"] / e0  # period multiplier
        c["EquityStitched"] = c["EquityNorm"] * cum_mult

        # 다음 period 시작을 위해 마지막 multiplier 반영
        last_mult = float(c["EquityNorm"].iloc[-1])
        if np.isfinite(last_mult) and last_mult > 0:
            cum_mult *= last_mult

        out_parts.append(c[["Date", "EquityStitched", "InCycle"]])

    if not out_parts:
        return pd.DataFrame()

    allc = pd.concat(out_parts, ignore_index=True)
    allc = allc.dropna(subset=["Date", "EquityStitched"]).sort_values("Date").reset_index(drop=True)
    allc = allc.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    allc = allc.rename(columns={"EquityStitched": "Equity"})
    return allc


def _max_drawdown_stats(curve: pd.DataFrame) -> dict:
    """
    MaxDD, MaxUnderwaterDays, MaxDDRecoveryDays
    - drawdown = Equity / peak - 1
    - underwater days = drawdown < 0 인 연속 구간 최대 길이
    - recovery days = 직전 peak 회복까지 걸린 최대 일수
    """
    if curve is None or curve.empty:
        return {
            "MaxDD": float("nan"),
            "MaxUnderwaterDays": float("nan"),
            "MaxDDRecoveryDays": float("nan"),
        }

    c = curve[["Date", "Equity"]].copy()
    c["Date"] = _safe_to_dt(c["Date"])
    c["Equity"] = _to_num(c["Equity"])
    c = c.dropna(subset=["Date", "Equity"]).sort_values("Date").reset_index(drop=True)
    if len(c) < 2:
        return {
            "MaxDD": float("nan"),
            "MaxUnderwaterDays": float("nan"),
            "MaxDDRecoveryDays": float("nan"),
        }

    eq = c["Equity"].astype(float).values
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    max_dd = float(np.nanmin(dd))

    # underwater run lengths
    uw = dd < 0
    max_uw = 0
    cur = 0
    for v in uw:
        if v:
            cur += 1
            max_uw = max(max_uw, cur)
        else:
            cur = 0

    # recovery days: peak 갱신 시점부터 다음에 그 peak 이상 회복하는데 걸린 days
    # 단순 구현: peak index를 기록하고, 그 이후 equity가 peak 이상 되는 첫 index까지 거리
    max_rec = 0
    last_peak_i = 0
    last_peak_val = eq[0]
    for i in range(1, len(eq)):
        if eq[i] >= last_peak_val:
            # recovered (or new peak)
            rec = i - last_peak_i
            max_rec = max(max_rec, rec)
            last_peak_i = i
            last_peak_val = eq[i]
        # else: still underwater

    return {
        "MaxDD": max_dd,
        "MaxUnderwaterDays": float(max_uw),
        "MaxDDRecoveryDays": float(max_rec),
    }


def _daily_risk_stats(curve: pd.DataFrame) -> dict:
    """
    DailyVol, Sharpe0, Sortino0 (risk-free=0)
    - returns = pct_change of Equity
    """
    if curve is None or curve.empty or len(curve) < 3:
        return {"DailyVol": float("nan"), "Sharpe0": float("nan"), "Sortino0": float("nan")}

    c = curve[["Date", "Equity"]].copy()
    c["Equity"] = _to_num(c["Equity"])
    c = c.dropna(subset=["Equity"]).sort_values("Date").reset_index(drop=True)

    r = c["Equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 3:
        return {"DailyVol": float("nan"), "Sharpe0": float("nan"), "Sortino0": float("nan")}

    mu = float(r.mean())
    sig = float(r.std(ddof=1))
    daily_vol = sig

    sharpe = float(mu / sig) if np.isfinite(sig) and sig > 0 else float("nan")

    downside = r[r < 0]
    dsig = float(downside.std(ddof=1)) if len(downside) >= 2 else float("nan")
    sortino = float(mu / dsig) if np.isfinite(dsig) and dsig > 0 else float("nan")

    return {"DailyVol": daily_vol, "Sharpe0": sharpe, "Sortino0": sortino}


# -----------------------
# Trades + BadExit stats
# -----------------------
def _is_badexit_reason(reason: str) -> bool:
    r = str(reason or "").strip().upper()
    return r.startswith("REVAL_FAIL") or r.startswith("GRACE_END_EXIT")


def _infer_trades_path_from_curve(curve_path: Path) -> Path:
    s = str(curve_path)
    s = s.replace("sim_engine_curve_", "sim_engine_trades_")
    # curve는 csv/parquet 둘 다 있을 수 있으니 trades도 같은 확장자로 우선 시도
    return Path(s)


def _read_trades_any(trades_path: Path) -> pd.DataFrame:
    if trades_path.exists():
        return _read_any(trades_path)

    # 확장자 바꿔서도 시도
    if trades_path.suffix.lower() == ".parquet":
        alt = trades_path.with_suffix(".csv")
    else:
        alt = trades_path.with_suffix(".parquet")
    if alt.exists():
        return _read_any(alt)

    return pd.DataFrame()


def _badexit_stats_from_trades(trades: pd.DataFrame) -> dict:
    """
    - BadExitRate_Row: trades row 기준
    - BadExitRate_Ticker: Tickters 분해 기준
    - BadExitReasonShare_RevalFail / _GraceEnd
    - BadExitReturnMean / NonBadExitReturnMean / BadExitReturnDiff (가능하면)
    """
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
    t["IsBadExit"] = t["Reason"].apply(_is_badexit_reason).astype(int)

    out["BadExitRate_Row"] = float(t["IsBadExit"].mean()) if len(t) else float("nan")

    # reason share among badexits
    bad = t[t["IsBadExit"] == 1]
    if len(bad):
        rr = bad["Reason"].str.upper()
        out["BadExitReasonShare_RevalFail"] = float((rr.str.startswith("REVAL_FAIL")).mean())
        out["BadExitReasonShare_GraceEnd"] = float((rr.str.startswith("GRACE_END_EXIT")).mean())

    # ticker exploded rate
    if "Tickers" in t.columns:
        rows = []
        for _, r in t.iterrows():
            ticks = [x.strip().upper() for x in str(r.get("Tickers", "")).split(",") if x.strip()]
            for tk in ticks:
                rows.append((tk, int(r["IsBadExit"])))
        if rows:
            x = pd.DataFrame(rows, columns=["Ticker", "IsBadExit"])
            # Date 정보가 없어서 단순 평균(티커-엔트리 단위)로 계산
            out["BadExitRate_Ticker"] = float(x["IsBadExit"].mean())

    # return gap
    ret_col = None
    for c in ["CycleReturn", "Return", "PnL", "PnLPct", "NetReturn"]:
        if c in t.columns:
            ret_col = c
            break

    if ret_col is not None:
        r = _to_num(t[ret_col])
        t["_ret_"] = r
        bad_ret = t.loc[t["IsBadExit"] == 1, "_ret_"].dropna()
        ok_ret = t.loc[t["IsBadExit"] == 0, "_ret_"].dropna()
        if len(bad_ret):
            out["BadExitReturnMean"] = float(bad_ret.mean())
        if len(ok_ret):
            out["NonBadExitReturnMean"] = float(ok_ret.mean())
        if np.isfinite(out["BadExitReturnMean"]) and np.isfinite(out["NonBadExitReturnMean"]):
            out["BadExitReturnDiff"] = float(out["BadExitReturnMean"] - out["NonBadExitReturnMean"])

    return out


def _trade_structure_stats(trades: pd.DataFrame, total_days: float) -> dict:
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
    if np.isfinite(total_days) and total_days > 0:
        out["TradesPerYear"] = float(len(trades) / (total_days / 365.0))

    hold_col = None
    for c in ["HoldingDays", "HoldDays", "DaysHeld"]:
        if c in trades.columns:
            hold_col = c
            break
    if hold_col is not None:
        h = _to_num(trades[hold_col]).dropna()
        if len(h):
            out["HoldDays_Mean"] = float(h.mean())
            out["HoldDays_Median"] = float(h.median())
            out["HoldDays_P90"] = float(h.quantile(0.90))
    else:
        # ExitDate/EntryDate 있으면 계산
        if "EntryDate" in trades.columns and "ExitDate" in trades.columns:
            ed = _safe_to_dt(trades["EntryDate"])
            xd = _safe_to_dt(trades["ExitDate"])
            dd = (xd - ed).dt.days
            dd = _to_num(dd).dropna()
            if len(dd):
                out["HoldDays_Mean"] = float(dd.mean())
                out["HoldDays_Median"] = float(dd.median())
                out["HoldDays_P90"] = float(dd.quantile(0.90))

    return out


# -----------------------
# Main
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    sp = Path(args.summary)
    df = pd.read_csv(sp)

    # column normalize: strip + BOM-safe
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    required = [
        "period",
        "curve_file",
        "suffix",
        "cap",
        "ps_min",
        "tail_threshold",
        "utility_quantile",
        "lambda_tail",
        "topk",
        "badexit_max",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] summary missing columns: {missing}")

    # types
    df["period"] = df["period"].astype(str).str.strip()
    df["curve_file"] = df["curve_file"].astype(str).str.strip()
    df["suffix"] = df["suffix"].astype(str).str.strip()
    df["cap"] = df["cap"].astype(str).str.strip()

    # numeric keys
    for c in ["ps_min", "tail_threshold", "utility_quantile", "lambda_tail", "topk", "badexit_max"]:
        df[c] = _to_num(df[c])

    key_cols = ["period", "suffix", "cap", "ps_min", "tail_threshold", "utility_quantile", "lambda_tail", "topk", "badexit_max"]

    # parquet 우선 dedupe (period+config 단위로 한 줄만)
    df = _prefer_parquet_dedupe(df, subset_cols=key_cols)

    # config 단위로 "전체기간 합산 성적표"
    cfg_cols = ["suffix", "cap", "ps_min", "tail_threshold", "utility_quantile", "lambda_tail", "topk", "badexit_max"]

    out_rows = []

    for cfg, g in df.groupby(cfg_cols, dropna=False):
        g = g.sort_values("period")
        curve_paths = [Path(x) for x in g["curve_file"].tolist()]
        curve_paths = [p for p in curve_paths if p.exists()]

        stitched = _stitch_curves(curve_paths)
        if stitched.empty:
            # 그래도 키는 남기고 NaN으로
            row = dict(zip(cfg_cols, cfg))
            row.update({
                "PeriodCount": int(g["period"].nunique()),
                "TotalDays": float("nan"),
                "SeedMultiple_Total": float("nan"),
                "CAGR_Total": float("nan"),
                "StartDate": "",
                "EndDate": "",
            })
            # badexit placeholders
            row.update(_badexit_stats_from_trades(pd.DataFrame()))
            row.update({
                "MaxDD_Total": float("nan"),
                "MaxUnderwaterDays_Total": float("nan"),
                "MaxDDRecoveryDays_Total": float("nan"),
                "DailyVol_Total": float("nan"),
                "Sharpe0_Total": float("nan"),
                "Sortino0_Total": float("nan"),
            })
            row.update(_trade_structure_stats(pd.DataFrame(), float("nan")))
            out_rows.append(row)
            continue

        start_date = pd.Timestamp(stitched["Date"].iloc[0])
        end_date = pd.Timestamp(stitched["Date"].iloc[-1])
        total_days = float((end_date - start_date).days)

        e0 = float(stitched["Equity"].iloc[0])
        e1 = float(stitched["Equity"].iloc[-1])
        mult = float(e1 / e0) if (np.isfinite(e0) and e0 > 0 and np.isfinite(e1) and e1 > 0) else float("nan")
        cagr = _calc_cagr(mult, total_days)

        dd_stats = _max_drawdown_stats(stitched)
        risk_stats = _daily_risk_stats(stitched)

        # trades aggregate
        trades_list = []
        for p in curve_paths:
            tp = _infer_trades_path_from_curve(p)
            tdf = _read_trades_any(tp)
            if tdf is not None and not tdf.empty:
                trades_list.append(tdf)
        trades_all = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()

        bad_stats = _badexit_stats_from_trades(trades_all)
        trade_stats = _trade_structure_stats(trades_all, total_days)

        row = dict(zip(cfg_cols, cfg))
        row.update({
            "PeriodCount": int(g["period"].nunique()),
            "TotalDays": total_days,
            "SeedMultiple_Total": mult,
            "CAGR_Total": cagr,
            "StartDate": str(start_date.date()),
            "EndDate": str(end_date.date()),
        })
        row.update({
            "MaxDD_Total": dd_stats["MaxDD"],
            "MaxUnderwaterDays_Total": dd_stats["MaxUnderwaterDays"],
            "MaxDDRecoveryDays_Total": dd_stats["MaxDDRecoveryDays"],
        })
        row.update({
            "DailyVol_Total": risk_stats["DailyVol"],
            "Sharpe0_Total": risk_stats["Sharpe0"],
            "Sortino0_Total": risk_stats["Sortino0"],
        })
        row.update(bad_stats)
        row.update(trade_stats)

        out_rows.append(row)

    out = pd.DataFrame(out_rows)

    # 정렬: Total SeedMultiple -> CAGR
    sort_cols = []
    if "SeedMultiple_Total" in out.columns:
        sort_cols.append("SeedMultiple_Total")
    if "CAGR_Total" in out.columns:
        sort_cols.append("CAGR_Total")
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[DONE] wrote analysis: {out_path} rows={len(out)}")

    if len(out):
        best = out.iloc[0].to_dict()
        print("=" * 60)
        print("[BEST] (overall stitched)")
        print(f"suffix={best.get('suffix')} cap={best.get('cap')} ps={best.get('ps_min')} tail={best.get('tail_threshold')} q={best.get('utility_quantile')} lam={best.get('lambda_tail')} topk={best.get('topk')} be={best.get('badexit_max')}")
        print(f"SeedMultiple_Total={best.get('SeedMultiple_Total')} CAGR_Total={best.get('CAGR_Total')} Days={best.get('TotalDays')} Periods={best.get('PeriodCount')}")
        print(f"MaxDD_Total={best.get('MaxDD_Total')} UnderwaterDays={best.get('MaxUnderwaterDays_Total')} RecoveryDays={best.get('MaxDDRecoveryDays_Total')}")
        print(f"BadExitRate_Row={best.get('BadExitRate_Row')} BadExitRate_Ticker={best.get('BadExitRate_Ticker')}")
        print(f"BadExitReturnDiff={best.get('BadExitReturnDiff')}")
        print("=" * 60)


if __name__ == "__main__":
    main()