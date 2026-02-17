# scripts/summarize_sim_trades.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def _to_dt(x):
    return pd.to_datetime(x, errors="coerce")


def _force_series(x, n: int) -> pd.Series:
    """Always return a Series of length n (broadcast scalar if needed)."""
    if isinstance(x, pd.Series):
        if len(x) == n:
            return x
        # length mismatch -> align by broadcasting last value
        if len(x) == 0:
            return pd.Series([np.nan] * n)
        return pd.Series([x.iloc[-1]] * n)
    # scalar / list / ndarray
    if np.isscalar(x):
        return pd.Series([x] * n)
    try:
        s = pd.Series(x)
        if len(s) == n:
            return s
        if len(s) == 0:
            return pd.Series([np.nan] * n)
        return pd.Series([s.iloc[-1]] * n)
    except Exception:
        return pd.Series([np.nan] * n)


def _safe_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    n = len(df)
    if col not in df.columns:
        return pd.Series([np.nan] * n)

    x = df[col]

    # df[col] can be Series, scalar-like, or even weird object -> force Series
    s = _force_series(x, n)
    s = pd.to_numeric(s, errors="coerce")
    return s


def _seed_multiple_from_equity(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None

    for col in ["SeedMultiple", "seed_multiple", "FinalSeedMultiple", "final_seed_multiple"]:
        if col in df.columns:
            v = _safe_numeric_series(df, col).dropna()
            if len(v):
                return float(v.iloc[-1])

    for col in ["Equity", "equity", "TotalEquity", "total_equity"]:
        if col in df.columns:
            eq = _safe_numeric_series(df, col).dropna()
            if len(eq) >= 2 and float(eq.iloc[0]) != 0:
                return float(eq.iloc[-1] / eq.iloc[0])

    return None


def _holding_days(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    for col in ["HoldingDays", "holding_days"]:
        if col in df.columns:
            return _safe_numeric_series(df, col)

    # compute from Entry/Exit
    entry_col = None
    exit_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ("entrydate", "entry_date", "entry"):
            entry_col = c
        if cl in ("exitdate", "exit_date", "exit"):
            exit_col = c

    if entry_col and exit_col:
        e = _to_dt(df[entry_col])
        x = _to_dt(df[exit_col])
        d = (x - e).dt.days
        return pd.to_numeric(_force_series(d, len(df)), errors="coerce")

    return pd.Series([np.nan] * len(df))


def _cycle_return(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    for col in ["CycleReturn", "cycle_return", "Return", "ret", "PnL_pct", "pnl_pct"]:
        if col in df.columns:
            return _safe_numeric_series(df, col)

    return pd.Series([np.nan] * len(df))


def _is_win(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype=int)

    n = len(df)

    for col in ["Win", "win", "IsWin", "is_win", "Success", "success"]:
        if col in df.columns:
            s = _safe_numeric_series(df, col)
            # ✅ 핵심: 여기서 어떤 경우든 Series이므로 fillna 안전
            return (s.fillna(0) > 0).astype(int)

    r = _cycle_return(df)
    r = pd.to_numeric(_force_series(r, n), errors="coerce")
    return (r.fillna(0) > 0).astype(int)


def _max_leverage_pct(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None

    for col in ["MaxLeveragePct", "max_leverage_pct", "LeveragePct", "leverage_pct"]:
        if col in df.columns:
            v = _safe_numeric_series(df, col).dropna()
            if len(v):
                return float(v.max())
    return None


def _recent10y_seed_multiple(df: pd.DataFrame) -> float | None:
    if df is None or len(df) == 0:
        return None

    date_col = None
    for c in ["Date", "date", "TradeDate", "trade_date", "ExitDate", "exit_date"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        return None

    d = _to_dt(df[date_col])
    if isinstance(d, pd.Series) and d.isna().all():
        return None

    last = pd.to_datetime(d, errors="coerce").max()
    if pd.isna(last):
        return None

    start = last - pd.Timedelta(days=365 * 10)
    sub = df.loc[pd.to_datetime(d, errors="coerce") >= start].copy()
    if sub.empty:
        return None

    return _seed_multiple_from_equity(sub)


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

    # ultra defensive
    if isinstance(df, pd.Series):
        df = df.to_frame()
    elif not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    hold = _holding_days(df)
    max_hold = pd.to_numeric(hold, errors="coerce").max()
    max_hold = float(max_hold) if np.isfinite(max_hold) else np.nan

    max_extend = np.nan
    if np.isfinite(max_hold):
        max_extend = float(max(0.0, max_hold - float(args.max_days)))

    wins = _is_win(df)
    wins = _force_series(wins, len(df))
    cycle_cnt = int(len(wins))
    win_cnt = int(pd.to_numeric(wins, errors="coerce").fillna(0).sum()) if cycle_cnt > 0 else 0
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