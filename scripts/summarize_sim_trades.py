# scripts/summarize_sim_trades.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def _to_dt(x):
    return pd.to_datetime(x, errors="coerce")


def _force_series(x, n: int) -> pd.Series:
    if isinstance(x, pd.Series):
        if len(x) == n:
            return x
        if len(x) == 0:
            return pd.Series([np.nan] * n)
        return pd.Series([x.iloc[-1]] * n)
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
    s = _force_series(df[col], n)
    return pd.to_numeric(s, errors="coerce")


def _infer_curve_path(trades_path: Path) -> Path | None:
    # sim_engine_trades_XXX.parquet -> sim_engine_curve_XXX.parquet
    name = trades_path.name
    if "sim_engine_trades_" not in name:
        return None
    curve_name = name.replace("sim_engine_trades_", "sim_engine_curve_")
    p = trades_path.with_name(curve_name)
    return p if p.exists() else None


def _pick_equity_col(curve: pd.DataFrame) -> str | None:
    # 가능한 equity 컬럼 후보들
    candidates = [
        "Equity", "equity", "TotalEquity", "total_equity",
        "AccountValue", "account_value", "NAV", "nav",
        "Seed", "seed", "CashPlusValue", "cash_plus_value",
    ]
    for c in candidates:
        if c in curve.columns:
            return c
    return None


def _seed_multiple_from_curve(curve: pd.DataFrame) -> float | None:
    if curve is None or len(curve) == 0:
        return None
    eq_col = _pick_equity_col(curve)
    if eq_col is None:
        return None
    eq = pd.to_numeric(curve[eq_col], errors="coerce").dropna()
    if len(eq) < 2:
        return None
    if float(eq.iloc[0]) == 0:
        return None
    return float(eq.iloc[-1] / eq.iloc[0])


def _recent10y_seed_multiple_from_curve(curve: pd.DataFrame) -> float | None:
    if curve is None or len(curve) == 0:
        return None

    date_col = None
    for c in ["Date", "date", "Datetime", "datetime", "Time", "time"]:
        if c in curve.columns:
            date_col = c
            break
    if date_col is None:
        return None

    d = _to_dt(curve[date_col])
    if isinstance(d, pd.Series) and d.isna().all():
        return None

    last = pd.to_datetime(d, errors="coerce").max()
    if pd.isna(last):
        return None

    start = last - pd.Timedelta(days=365 * 10)
    sub = curve.loc[pd.to_datetime(d, errors="coerce") >= start].copy()
    if len(sub) < 2:
        return None

    return _seed_multiple_from_curve(sub)


def _max_drawdown_from_curve(curve: pd.DataFrame) -> float | None:
    if curve is None or len(curve) == 0:
        return None
    eq_col = _pick_equity_col(curve)
    if eq_col is None:
        return None
    eq = pd.to_numeric(curve[eq_col], errors="coerce").dropna()
    if len(eq) < 2:
        return None
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.min())


def _holding_days(trades: pd.DataFrame) -> pd.Series:
    if trades is None or len(trades) == 0:
        return pd.Series(dtype=float)

    for col in ["HoldingDays", "holding_days"]:
        if col in trades.columns:
            return _safe_numeric_series(trades, col)

    # fallback: entry/exit date diff
    entry_col = None
    exit_col = None
    for c in trades.columns:
        cl = c.lower()
        if cl in ("entrydate", "entry_date", "entry"):
            entry_col = c
        if cl in ("exitdate", "exit_date", "exit"):
            exit_col = c
    if entry_col and exit_col:
        e = _to_dt(trades[entry_col])
        x = _to_dt(trades[exit_col])
        d = (x - e).dt.days
        return pd.to_numeric(_force_series(d, len(trades)), errors="coerce")

    return pd.Series([np.nan] * len(trades))


def _cycle_return(trades: pd.DataFrame) -> pd.Series:
    if trades is None or len(trades) == 0:
        return pd.Series(dtype=float)

    for col in ["CycleReturn", "cycle_return", "Return", "ret", "PnL_pct", "pnl_pct"]:
        if col in trades.columns:
            return _safe_numeric_series(trades, col)

    return pd.Series([np.nan] * len(trades))


def _is_win(trades: pd.DataFrame) -> pd.Series:
    if trades is None or len(trades) == 0:
        return pd.Series(dtype=int)

    for col in ["Win", "win", "IsWin", "is_win", "Success", "success"]:
        if col in trades.columns:
            s = _safe_numeric_series(trades, col)
            return (s.fillna(0) > 0).astype(int)

    r = _cycle_return(trades)
    r = pd.to_numeric(_force_series(r, len(trades)), errors="coerce")
    return (r.fillna(0) > 0).astype(int)


def _max_leverage_pct(trades: pd.DataFrame) -> float | None:
    if trades is None or len(trades) == 0:
        return None
    for col in ["MaxLeveragePct", "max_leverage_pct", "LeveragePct", "leverage_pct",
                "Max_LeveragePct_Closed", "max_leverage_pct_closed"]:
        if col in trades.columns:
            v = pd.to_numeric(trades[col], errors="coerce").dropna()
            if len(v):
                return float(v.max())
    return None


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

    trades_path = Path(args.trades_path)
    if not trades_path.exists():
        raise FileNotFoundError(f"Missing trades parquet: {trades_path}")

    trades = pd.read_parquet(trades_path)
    if isinstance(trades, pd.Series):
        trades = trades.to_frame()
    elif not isinstance(trades, pd.DataFrame):
        trades = pd.DataFrame(trades)

    curve_path = _infer_curve_path(trades_path)
    curve = None
    if curve_path is not None and curve_path.exists():
        curve = pd.read_parquet(curve_path)
        if isinstance(curve, pd.Series):
            curve = curve.to_frame()
        elif not isinstance(curve, pd.DataFrame):
            curve = pd.DataFrame(curve)

    # ---- compute metrics from trades ----
    hold = _holding_days(trades)
    max_hold = pd.to_numeric(hold, errors="coerce").max()
    max_hold = float(max_hold) if np.isfinite(max_hold) else np.nan

    max_extend = np.nan
    if np.isfinite(max_hold):
        max_extend = float(max(0.0, max_hold - float(args.max_days)))

    wins = _is_win(trades)
    wins = _force_series(wins, len(trades))
    cycle_cnt = int(len(wins))
    win_cnt = int(pd.to_numeric(wins, errors="coerce").fillna(0).sum()) if cycle_cnt > 0 else 0
    success_rate = (win_cnt / cycle_cnt) if cycle_cnt > 0 else 0.0

    max_lev = _max_leverage_pct(trades)

    # ---- compute equity-based metrics from curve (this is the key fix) ----
    seed_mult = _seed_multiple_from_curve(curve) if curve is not None else None
    recent10y = _recent10y_seed_multiple_from_curve(curve) if curve is not None else None
    mdd = _max_drawdown_from_curve(curve) if curve is not None else None

    out = {
        "TAG": args.tag,
        "GateSuffix": args.suffix,
        "ProfitTarget": args.profit_target,
        "MaxHoldingDays": args.max_days,
        "StopLevel": args.stop_level,
        "MaxExtendDaysParam": args.max_extend_days,
        "Recent10Y_SeedMultiple": recent10y if recent10y is not None else np.nan,
        "SeedMultiple": seed_mult if seed_mult is not None else np.nan,
        "MaxDrawdown": mdd if mdd is not None else np.nan,
        "MaxHoldingDaysObserved": max_hold,
        "MaxExtendDaysObserved": max_extend,
        "CycleCount": cycle_cnt,
        "SuccessRate": success_rate,
        "MaxLeveragePct": max_lev if max_lev is not None else np.nan,
        "TradesFile": str(trades_path),
        "CurveFile": str(curve_path) if curve_path is not None else "",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gate_summary_{args.tag}_gate_{args.suffix}.csv"
    pd.DataFrame([out]).to_csv(out_path, index=False)
    print(f"[DONE] wrote gate summary: {out_path}")


if __name__ == "__main__":
    main()