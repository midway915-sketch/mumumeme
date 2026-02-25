#!/usr/bin/env python3
# scripts/analyze_walkforward_summary.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


REQ_COLS = [
    "period", "curve_file", "suffix", "cap",
    "ps_min", "tail_threshold", "utility_quantile", "lambda_tail", "topk", "badexit_max",
]


def _calc_cagr(mult: float, days: float) -> float:
    if not np.isfinite(mult) or mult <= 0:
        return float("nan")
    if not np.isfinite(days) or days <= 0:
        return float("nan")
    years = days / 365.0
    if years <= 0:
        return float("nan")
    return float(mult ** (1.0 / years) - 1.0)


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise SystemExit(f"[ERROR] summary missing columns: {miss}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--topn", type=int, default=200)
    args = ap.parse_args()

    s = Path(args.summary)
    if not s.exists():
        raise FileNotFoundError(f"summary not found: {s}")

    df = pd.read_csv(s)
    _require_cols(df, REQ_COLS)

    # numeric coercion
    num_cols = [
        "ps_min", "tail_threshold", "utility_quantile", "lambda_tail", "topk", "badexit_max",
        "BacktestDaysAfterWarmup", "SeedMultiple_AfterWarmup", "CAGR_AfterWarmup",
        "QQQ_SeedMultiple_SamePeriod", "ExcessSeedMultiple_AfterWarmup",
        "Recent5Y_SeedMultiple_AfterWarmup", "Recent5Y_CAGR_AfterWarmup",
        "IdlePctAfterWarmup", "MaxDD_AfterWarmup", "MaxUnderwaterDays_AfterWarmup", "MaxDDRecoveryDays_AfterWarmup",
        "DailyVol_AfterWarmup", "Sharpe0_AfterWarmup", "Sortino0_AfterWarmup",
        "TradeCount", "TradesPerYear",
        "HoldDays_Mean", "HoldDays_Median", "HoldDays_P90",
        "BadExitRate_Row", "BadExitRate_Ticker",
        "BadExitReasonShare_RevalFail", "BadExitReasonShare_GraceEnd",
        "BadExitReturnMean", "NonBadExitReturnMean", "BadExitReturnDiff",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # group keys = suffix-level config
    keys = ["suffix", "cap", "ps_min", "tail_threshold", "utility_quantile", "lambda_tail", "topk", "badexit_max"]
    g = df.groupby(keys, dropna=False)

    rows = []
    for k, part in g:
        part = part.copy()

        # total days & total mult
        days = part["BacktestDaysAfterWarmup"].dropna().astype(float)
        mults = part["SeedMultiple_AfterWarmup"].dropna().astype(float)

        # 곱셈으로 전기간 seed multiple (compounded)
        total_mult = float(np.prod(mults.values)) if len(mults) else float("nan")
        total_days = float(days.sum()) if len(days) else float("nan")
        total_cagr = _calc_cagr(total_mult, total_days) if np.isfinite(total_mult) and np.isfinite(total_days) else float("nan")

        # QQQ also compounded if present
        qqq_mults = part["QQQ_SeedMultiple_SamePeriod"].dropna().astype(float) if "QQQ_SeedMultiple_SamePeriod" in part.columns else pd.Series([], dtype=float)
        qqq_total_mult = float(np.prod(qqq_mults.values)) if len(qqq_mults) else float("nan")
        excess_mult = float(total_mult / qqq_total_mult) if (np.isfinite(total_mult) and np.isfinite(qqq_total_mult) and qqq_total_mult > 0) else float("nan")
        excess_cagr = _calc_cagr(excess_mult, total_days) if np.isfinite(excess_mult) and np.isfinite(total_days) else float("nan")

        # risk aggregation (max-ish)
        maxdd = float(part["MaxDD_AfterWarmup"].min()) if "MaxDD_AfterWarmup" in part.columns and part["MaxDD_AfterWarmup"].notna().any() else float("nan")
        maxuw = float(part["MaxUnderwaterDays_AfterWarmup"].max()) if "MaxUnderwaterDays_AfterWarmup" in part.columns and part["MaxUnderwaterDays_AfterWarmup"].notna().any() else float("nan")
        rec = float(part["MaxDDRecoveryDays_AfterWarmup"].max()) if "MaxDDRecoveryDays_AfterWarmup" in part.columns and part["MaxDDRecoveryDays_AfterWarmup"].notna().any() else float("nan")

        # operational aggregation
        trade_count = float(part["TradeCount"].sum()) if "TradeCount" in part.columns and part["TradeCount"].notna().any() else float("nan")
        # TradesPerYear: recompute from total_days if possible
        trades_per_year = float(trade_count / (total_days / 365.0)) if (np.isfinite(trade_count) and np.isfinite(total_days) and total_days > 0) else float("nan")

        # holds: weighted by trade count where possible
        def _wavg(col: str) -> float:
            if col not in part.columns:
                return float("nan")
            x = part[col].astype(float)
            w = part["TradeCount"].astype(float) if "TradeCount" in part.columns else None
            if w is None or w.isna().all() or x.isna().all():
                return float(x.mean()) if x.notna().any() else float("nan")
            m = x.notna() & w.notna() & (w > 0)
            if not m.any():
                return float("nan")
            return float(np.average(x[m], weights=w[m]))

        hold_mean = _wavg("HoldDays_Mean")
        hold_med = _wavg("HoldDays_Median")
        hold_p90 = _wavg("HoldDays_P90")

        # badexit aggregation (weighted)
        bad_row = _wavg("BadExitRate_Row")
        bad_tic = _wavg("BadExitRate_Ticker")
        rf = _wavg("BadExitReasonShare_RevalFail")
        ge = _wavg("BadExitReasonShare_GraceEnd")
        bad_ret = _wavg("BadExitReturnMean")
        ok_ret = _wavg("NonBadExitReturnMean")
        bad_diff = bad_ret - ok_ret if np.isfinite(bad_ret) and np.isfinite(ok_ret) else float("nan")

        idle = _wavg("IdlePctAfterWarmup")

        # (optional) daily stats: 평균으로만 (기간 합산 정확히 하려면 curve를 다시 붙여야 해서 여기선 보수적으로 mean)
        daily_vol = _wavg("DailyVol_AfterWarmup")
        sharpe = _wavg("Sharpe0_AfterWarmup")
        sortino = _wavg("Sortino0_AfterWarmup")

        rows.append({
            "suffix": k[0],
            "cap": k[1],
            "ps_min": k[2],
            "tail_threshold": k[3],
            "utility_quantile": k[4],
            "lambda_tail": k[5],
            "topk": k[6],
            "badexit_max": k[7],

            # 전체기간 핵심
            "TotalPeriods": int(part["period"].nunique()),
            "TotalDaysAfterWarmup": total_days,
            "SeedMultiple_Total": total_mult,
            "CAGR_Total": total_cagr,

            "QQQ_SeedMultiple_Total": qqq_total_mult,
            "ExcessSeedMultiple_Total": excess_mult,
            "ExcessCAGR_Total": excess_cagr,

            # 리스크/회복력
            "MaxDD_Total": maxdd,
            "MaxUnderwaterDays_Total": maxuw,
            "MaxDDRecoveryDays_Total": rec,

            # 운영/구조
            "TradeCount_Total": trade_count,
            "TradesPerYear_Total": trades_per_year,
            "HoldDays_Mean_Total": hold_mean,
            "HoldDays_Median_Total": hold_med,
            "HoldDays_P90_Total": hold_p90,
            "IdlePct_Total": idle,

            # badexit
            "BadExitRate_Row_Total": bad_row,
            "BadExitRate_Ticker_Total": bad_tic,
            "BadExitReasonShare_RevalFail_Total": rf,
            "BadExitReasonShare_GraceEnd_Total": ge,
            "BadExitReturnMean_Total": bad_ret,
            "NonBadExitReturnMean_Total": ok_ret,
            "BadExitReturnDiff_Total": bad_diff,

            # daily(옵션)
            "DailyVol_Total": daily_vol,
            "Sharpe0_Total": sharpe,
            "Sortino0_Total": sortino,
        })

    out = pd.DataFrame(rows)

    # ranking score: 너 취향대로 바꿔도 됨 (일단 ExcessCAGR/MaxDD/BadExit로 간단 가중)
    # - MaxDD는 음수(예:-0.35)니까 절댓값을 페널티로
    if len(out):
        dd_pen = out["MaxDD_Total"].abs().fillna(0.0)
        be_pen = out["BadExitRate_Row_Total"].fillna(0.0)
        out["Score"] = out["ExcessCAGR_Total"].fillna(0.0) / (1.0 + 1.5 * dd_pen + 0.5 * be_pen)

        out = out.sort_values(["Score", "ExcessCAGR_Total", "CAGR_Total"], ascending=[False, False, False])
        out = out.head(int(args.topn)).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[DONE] wrote analysis: {out_path} rows={len(out)}")
    if len(out):
        best = out.iloc[0].to_dict()
        print("=" * 70)
        print("[BEST] suffix/cap:", best.get("suffix"), best.get("cap"))
        print("Score=", best.get("Score"))
        print("SeedMultiple_Total=", best.get("SeedMultiple_Total"), "CAGR_Total=", best.get("CAGR_Total"))
        print("ExcessSeedMultiple_Total=", best.get("ExcessSeedMultiple_Total"), "ExcessCAGR_Total=", best.get("ExcessCAGR_Total"))
        print("MaxDD_Total=", best.get("MaxDD_Total"), "BadExitRate_Row_Total=", best.get("BadExitRate_Row_Total"))
        print("=" * 70)


if __name__ == "__main__":
    main()