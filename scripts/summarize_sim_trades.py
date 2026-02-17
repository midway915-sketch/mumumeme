# scripts/summarize_sim_trades.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np


def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")


def _seed_multiple_from_equity(df: pd.DataFrame) -> float | None:
    # 가장 확실: Equity/SeedMultiple/FinalEquity 같은 컬럼이 있으면 그걸 사용
    for col in ["SeedMultiple", "seed_multiple", "FinalSeedMultiple", "final_seed_multiple"]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v):
                return float(v.iloc[-1])
    # Equity curve가 있으면 initial 대비
    for col in ["Equity", "equity", "TotalEquity", "total_equity"]:
        if col in df.columns:
            eq = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(eq) >= 2:
                return float(eq.iloc[-1] / eq.iloc[0]) if eq.iloc[0] != 0 else None
    return None


def _holding_days(df: pd.DataFrame) -> pd.Series:
    if "HoldingDays" in df.columns:
        return pd.to_numeric(df["HoldingDays"], errors="coerce")
    if "holding_days" in df.columns:
        return pd.to_numeric(df["holding_days"], errors="coerce")
    # Entry/Exit date로 계산
    entry_cols = [c for c in df.columns if c.lower() in ("entrydate", "entry_date", "entry")]
    exit_cols = [c for c in df.columns if c.lower() in ("exitdate", "exit_date", "exit")]
    if entry_cols and exit_cols:
        e = _to_dt(df[entry_cols[0]])
        x = _to_dt(df[exit_cols[0]])
        d = (x - e).dt.days
        return pd.to_numeric(d, errors="coerce")
    return pd.Series([np.nan] * len(df))


def _cycle_return(df: pd.DataFrame) -> pd.Series:
    for col in ["CycleReturn", "cycle_return", "Return", "ret", "PnL_pct", "pnl_pct"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([np.nan] * len(df))


def _is_win(df: pd.DataFrame) -> pd.Series:
    for col in ["Win", "win", "IsWin", "is_win", "Success", "success"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            # 0/1 형태면 그대로, True/False면 1/0 처리
            return (s.fillna(0) > 0).astype(int)
    r = _cycle_return(df)
    return (r > 0).astype(int)


def _max_leverage_pct(df: pd.DataFrame) -> float | None:
    for col in ["MaxLeveragePct", "max_leverage_pct", "LeveragePct", "leverage_pct"]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v):
                return float(v.max())
    return None


def _recent10y_seed_multiple(df: pd.DataFrame) -> float | None:
    # Date가 있으면 마지막 10년 컷
    date_col = None
    for c in ["Date", "date", "TradeDate", "trade_date", "ExitDate", "exit_date"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        return None

    d = _to_dt(df[date_col])
    if d.isna().all():
        return None

    last = d.max()
    start = last - pd.Timedelta(days=365 * 10)
    sub = df.loc[d >= start].copy()
    if sub.empty:
        return None

    sm = _seed_multiple_from_equity(sub)
    return sm


def parse_tag_suffix_from_filename(path: Path) -> tuple[str | None, str | None]:
    # 파일명 예:
    # sim_engine_trades_pt10_h40_sl10_ex30_tail_utility_t0p30_q0p75_rutility_....parquet
    name = path.name
    m = re.search(r"sim_engine_trades_(pt\d+_h\d+_sl\d+_ex\d+)_", name)
    tag = m.group(1) if m else None

    # gate suffix는 "gate_" 뒤에 붙는 picks 규칙을 우리가 강제하고 있으니
    # 파일명에 "gate_"가 없을 수 있음 -> 우리가 호출 때 suffix를 알고 있음(인자로 받는 게 더 안정적)
    return tag, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-path", required=True, type=str)
    ap.add_argument("--tag", required=True, type=str)
    ap.add_argument("--suffix", required=True, type=str)
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)
    ap.add_argument("--out-dir", default="data/signals", type=str)
    args = ap.parse_args()

    p = Path(args.trades_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing trades parquet: {p}")

    df = pd.read_parquet(p)

    hold = _holding_days(df)
    max_hold = float(pd.to_numeric(hold, errors="coerce").max()) if len(hold) else np.nan
    if not np.isfinite(max_hold):
        max_hold = np.nan

    # max_days 초과분(연장 길이) 최대
    max_extend = np.nan
    if np.isfinite(max_hold):
        max_extend = float(max(0.0, max_hold - float(args.max_days)))

    # cycle count / success rate
    wins = _is_win(df)
    cycle_cnt = int(pd.to_numeric(wins, errors="coerce").dropna().shape[0])
    win_cnt = int(pd.to_numeric(wins, errors="coerce").sum()) if cycle_cnt > 0 else 0
    success_rate = (win_cnt / cycle_cnt) if cycle_cnt > 0 else 0.0

    seed_mult = _seed_multiple_from_equity(df)
    recent10y = _recent10y_seed_multiple(df)

    max_lev = _max_leverage_pct(df)

    out = {
        "TAG": args.tag,
        "GateSuffix": args.suffix,
        "ProfitTarget": args.profit_target,
        "MaxHoldingDays": args.max_days,
        "StopLevel": args.stop_level,
        "MaxExtendDaysParam": args.max_extend_days,
        "Recent10Y_SeedMultiple": recent10y if recent10y is not None else np.nan,
        "SeedMultiple": seed_mult if seed_mult is not None else np.nan,
        "MaxHoldingDaysObserved": max_hold,
        "MaxExtendDaysObserved": max_extend,
        "CycleCount": cycle_cnt,
        "SuccessRate": success_rate,
        "MaxLeveragePct": max_lev if max_lev is not None else np.nan,
        "TradesFile": str(p),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gate_summary_{args.tag}_gate_{args.suffix}.csv"
    pd.DataFrame([out]).to_csv(out_path, index=False)
    print(f"[DONE] wrote gate summary: {out_path}")


if __name__ == "__main__":
    main()