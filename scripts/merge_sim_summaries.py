# scripts/merge_sim_summaries.py
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import pyarrow.parquet as pq


SIGNALS_DIR = Path("data/signals")


def read_parquet_any(path: str) -> pd.DataFrame:
    # pandas.read_parquet이 환경에 따라 애매하게 깨질 때가 있어서 pyarrow로 고정
    tbl = pq.read_table(path)
    return tbl.to_pandas()


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))


def busday_gap(prev_exit: pd.Timestamp, next_entry: pd.Timestamp) -> int:
    # (prev_exit 다음날) ~ (next_entry 전날) 사이의 "평일" 갯수 근사
    # US holiday까지 완벽하진 않지만, 대략적인 skipped day로 충분히 쓸 수 있음
    start = (prev_exit + pd.Timedelta(days=1)).date()
    end = next_entry.date()  # end는 미포함
    if start >= end:
        return 0
    return int(np.busday_count(start, end))


def summarize_one_method(
    tag: str,
    method: str,
    curve: pd.DataFrame | None,
    trades: pd.DataFrame | None,
    max_days: int,
    recent_years: int,
) -> dict:
    out: dict = {"label": method, "tag": tag}

    # ---- trades 기반 (사이클/승률/레버리지 등)
    if trades is not None and not trades.empty:
        t = trades[trades["Method"] == method].copy()
        t = t.sort_values("EntryDate")
        cycle_cnt = int(len(t))
        out["Cycle_Count_Closed"] = cycle_cnt
        out["Success_Rate_Closed"] = float((t["CycleReturn"] > 0).mean()) if cycle_cnt > 0 else 0.0
        out["Avg_LeveragePct_Closed"] = float(t["LeveragePctMax"].mean()) if cycle_cnt > 0 else float("nan")
        out["Max_LeveragePct_Closed"] = float(t["LeveragePctMax"].max()) if cycle_cnt > 0 else float("nan")

        # Skipped days(근사): 사이클 간 갭의 평일 수 합
        skipped = 0
        if cycle_cnt >= 2:
            prev_exits = t["ExitDate"].iloc[:-1].tolist()
            next_entries = t["EntryDate"].iloc[1:].tolist()
            for pe, ne in zip(prev_exits, next_entries):
                skipped += busday_gap(pd.to_datetime(pe), pd.to_datetime(ne))
        out["Skipped_Days"] = int(skipped)

    else:
        out["Cycle_Count_Closed"] = 0
        out["Success_Rate_Closed"] = 0.0
        out["Avg_LeveragePct_Closed"] = float("nan")
        out["Max_LeveragePct_Closed"] = float("nan")
        out["Skipped_Days"] = 0

    # ---- curve 기반 (시드배수/최근10년/드로우다운/홀딩 등)
    if curve is not None and not curve.empty:
        c = curve[curve["Method"] == method].copy()
        c = c.sort_values("Date")

        last_date = pd.to_datetime(c["Date"].iloc[-1]).normalize()
        out["last_date"] = str(last_date.date())

        eq = c["Equity"].to_numpy(dtype=float)
        final_eq = float(eq[-1])
        start_eq = float(eq[0])
        out["Final_Equity"] = final_eq
        out["Seed_Multiple_All"] = float(final_eq / start_eq) if start_eq != 0 else float("nan")
        out["Max_Drawdown"] = max_drawdown(eq)

        # 최대 홀딩일
        out["max_holding_days"] = int(pd.to_numeric(c["HoldingDays"], errors="coerce").max())

        # max_days 초과분(=연장 최대치)
        if max_days > 0:
            hd = pd.to_numeric(c["HoldingDays"], errors="coerce").fillna(0).to_numpy(dtype=float)
            exceed = np.maximum(hd - max_days, 0)
            out["max_extend_days_over_maxday"] = int(np.max(exceed))
        else:
            out["max_extend_days_over_maxday"] = int(0)

        # 최근 N년 배수
        start_date = last_date - pd.DateOffset(years=recent_years)
        out["recent10y_start_date"] = str(start_date.date())

        # start_date 이상 첫 번째 시점의 equity를 기준으로
        idx = int(np.searchsorted(c["Date"].to_numpy(), np.datetime64(start_date)))
        if idx >= len(c):
            idx = 0
        base_eq = float(c["Equity"].iloc[idx])
        out["recent10y_seed_multiple"] = float(final_eq / base_eq) if base_eq != 0 else float("nan")

        # Entered_Days(근사): 닫힌 사이클 수 + (마지막에 포지션 살아있으면 1)
        open_pos = float(c["PosValue"].iloc[-1]) > 0
        out["Entered_Days"] = int(out["Cycle_Count_Closed"] + (1 if open_pos else 0))

    else:
        # curve가 없으면 최소한 trades 기반으로만이라도 값 채움
        out["last_date"] = ""
        out["recent10y_start_date"] = ""
        out["Final_Equity"] = float("nan")
        out["Seed_Multiple_All"] = float("nan")
        out["Max_Drawdown"] = float("nan")
        out["max_holding_days"] = int(0)
        out["max_extend_days_over_maxday"] = int(0)
        out["recent10y_seed_multiple"] = float("nan")
        out["Entered_Days"] = int(out["Cycle_Count_Closed"])

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--recent-years", type=int, default=10)
    args = ap.parse_args()

    tag = args.tag
    max_days = int(args.max_days)
    recent_years = int(args.recent_years)

    # parquet 찾기 (파일명이 method가 2번 붙든 말든 tag만 포함하면 OK)
    curve_paths = sorted(glob.glob(f"{SIGNALS_DIR}/sim_engine_curve_*{tag}*.parquet"))
    trade_paths = sorted(glob.glob(f"{SIGNALS_DIR}/sim_engine_trades_*{tag}*.parquet"))

    if not curve_paths and not trade_paths:
        raise SystemExit(f"[ERROR] No curve/trades parquet found for tag={tag} in {SIGNALS_DIR}")

    curve_df = pd.DataFrame()
    trades_df = pd.DataFrame()

    if curve_paths:
        curve_df = pd.concat([read_parquet_any(p) for p in curve_paths], ignore_index=True)
    if trade_paths:
        trades_df = pd.concat([read_parquet_any(p) for p in trade_paths], ignore_index=True)

    # method 목록
    methods = set()
    if not curve_df.empty and "Method" in curve_df.columns:
        methods |= set(curve_df["Method"].dropna().astype(str).unique().tolist())
    if not trades_df.empty and "Method" in trades_df.columns:
        methods |= set(trades_df["Method"].dropna().astype(str).unique().tolist())

    if not methods:
        raise SystemExit(f"[ERROR] No Method column found in curve/trades parquet for tag={tag}")

    rows = []
    for m in sorted(methods):
        rows.append(
            summarize_one_method(
                tag=tag,
                method=m,
                curve=curve_df if not curve_df.empty else None,
                trades=trades_df if not trades_df.empty else None,
                max_days=max_days,
                recent_years=recent_years,
            )
        )

    out = pd.DataFrame(rows)

    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) "summary" 이름으로도 저장 (워크플로 호환)
    summary_path = SIGNALS_DIR / f"sim_engine_summary_{tag}_GATES_ALL.csv"
    out.to_csv(summary_path, index=False)

    # 2) gate_summary도 같이 저장 (Artifacts로 받기 편한 최종표)
    gate_path = SIGNALS_DIR / f"gate_summary_{tag}.csv"
    out.to_csv(gate_path, index=False)

    print(f"[DONE] wrote:\n  - {summary_path}\n  - {gate_path}\nrows={len(out)}")


if __name__ == "__main__":
    main()
