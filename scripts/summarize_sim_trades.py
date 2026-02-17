# scripts/summarize_sim_trades.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _to_dt(x):
    return pd.to_datetime(x, errors="coerce")


def _read_parquet_if_exists(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or len(df) == 0 or col not in df.columns:
        return pd.Series([np.nan] * (0 if df is None else len(df)))
    return pd.to_numeric(df[col], errors="coerce")


def _curve_seed_multiple(curve: pd.DataFrame) -> float | None:
    if curve is None or len(curve) == 0:
        return None
    if "SeedMultiple" in curve.columns:
        s = pd.to_numeric(curve["SeedMultiple"], errors="coerce").dropna()
        if len(s):
            return float(s.iloc[-1])
    if "Equity" in curve.columns and len(curve) >= 2:
        eq = pd.to_numeric(curve["Equity"], errors="coerce").dropna()
        if len(eq) >= 2 and float(eq.iloc[0]) != 0:
            return float(eq.iloc[-1] / eq.iloc[0])
    return None


def _curve_recent10y_seed_multiple(curve: pd.DataFrame) -> float | None:
    if curve is None or len(curve) == 0:
        return None
    if "Date" not in curve.columns:
        return None
    d = _to_dt(curve["Date"]).dropna()
    if len(d) == 0:
        return None

    last = d.max()
    start = last - pd.Timedelta(days=365 * 10)

    sub = curve.loc[d >= start].copy()
    if sub.empty:
        return None

    # use SeedMultiple if exists
    if "SeedMultiple" in sub.columns:
        s = pd.to_numeric(sub["SeedMultiple"], errors="coerce").dropna()
        if len(s):
            # If seed multiple is normalized from initial, we need ratio end/start within 10y window:
            # safer to use Equity ratio if Equity exists
            if "Equity" in sub.columns:
                eq = pd.to_numeric(sub["Equity"], errors="coerce").dropna()
                if len(eq) >= 2 and float(eq.iloc[0]) != 0:
                    return float(eq.iloc[-1] / eq.iloc[0])
            # fallback: ratio within window using SeedMultiple
            if float(s.iloc[0]) != 0:
                return float(s.iloc[-1] / s.iloc[0])

    if "Equity" in sub.columns:
        eq = pd.to_numeric(sub["Equity"], errors="coerce").dropna()
        if len(eq) >= 2 and float(eq.iloc[0]) != 0:
            return float(eq.iloc[-1] / eq.iloc[0])

    return None


def _max_extend_over_maxdays(trades: pd.DataFrame, max_days: int) -> float | None:
    if trades is None or len(trades) == 0:
        return None
    if "HoldingDays" not in trades.columns:
        return None
    hd = pd.to_numeric(trades["HoldingDays"], errors="coerce")
    if hd.dropna().empty:
        return None
    mx = float(hd.max())
    return float(max(0.0, mx - float(max_days)))


def _closed_mask(trades: pd.DataFrame) -> pd.Series:
    if trades is None or len(trades) == 0:
        return pd.Series([], dtype=bool)
    if "ExitDate" in trades.columns:
        x = _to_dt(trades["ExitDate"])
        return x.notna()
    # if no ExitDate column, assume all closed
    return pd.Series([True] * len(trades))


def _success_rate(trades: pd.DataFrame, closed_mask: pd.Series) -> float:
    if trades is None or len(trades) == 0:
        return 0.0
    if closed_mask is None or len(closed_mask) != len(trades):
        closed_mask = pd.Series([True] * len(trades))

    t = trades.loc[closed_mask].copy()
    if t.empty:
        return 0.0

    # prefer Win column if exists else CycleReturn > 0
    if "Win" in t.columns:
        w = pd.to_numeric(t["Win"], errors="coerce").fillna(0)
        return float((w > 0).mean())
    if "CycleReturn" in t.columns:
        r = pd.to_numeric(t["CycleReturn"], errors="coerce").fillna(0)
        return float((r > 0).mean())
    return 0.0


def _max_leverage_closed(trades: pd.DataFrame, closed_mask: pd.Series) -> float | None:
    if trades is None or len(trades) == 0:
        return None
    if "MaxLeveragePct" not in trades.columns:
        return None
    if closed_mask is None or len(closed_mask) != len(trades):
        closed_mask = pd.Series([True] * len(trades))

    t = trades.loc[closed_mask].copy()
    if t.empty:
        return None

    lev = pd.to_numeric(t["MaxLeveragePct"], errors="coerce").dropna()
    if lev.empty:
        return None
    return float(lev.max())


def _lev_adjust(mult: float | None, max_lev: float | None, k: float) -> float | None:
    if mult is None or not np.isfinite(mult):
        return None
    lev = 0.0 if (max_lev is None or not np.isfinite(max_lev)) else float(max_lev)
    denom = (1.0 + lev) ** float(k)
    if denom <= 0:
        return None
    return float(mult / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-path", required=True, type=str)
    ap.add_argument("--curve-path", default="", type=str)
    ap.add_argument("--tag", required=True, type=str)
    ap.add_argument("--suffix", required=True, type=str)
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)
    ap.add_argument("--max-leverage-pct", default=1.0, type=float)  # cap param (100% -> 1.0)
    ap.add_argument("--lev-penalty-k", default=1.0, type=float)     # k in / (1+lev)^k
    ap.add_argument("--out-dir", default="data/signals", type=str)
    args = ap.parse_args()

    trades_p = Path(args.trades_path)
    if not trades_p.exists():
        raise FileNotFoundError(f"Missing trades parquet: {trades_p}")

    trades = pd.read_parquet(trades_p)
    if not isinstance(trades, pd.DataFrame):
        trades = pd.DataFrame(trades)

    curve = None
    if args.curve_path:
        curve = _read_parquet_if_exists(args.curve_path)

    closed = _closed_mask(trades)
    cycle_cnt = int(len(trades))
    closed_cnt = int(closed.sum()) if len(closed) else 0

    success_rate = _success_rate(trades, closed)
    max_hold = float(pd.to_numeric(trades.get("HoldingDays", pd.Series([], dtype=float)), errors="coerce").max()) if ("HoldingDays" in trades.columns and len(trades)) else np.nan
    if not np.isfinite(max_hold):
        max_hold = np.nan

    max_extend_over = _max_extend_over_maxdays(trades, int(args.max_days))
    max_lev_closed = _max_leverage_closed(trades, closed)

    seed_mult = _curve_seed_multiple(curve) if curve is not None else None
    recent10y = _curve_recent10y_seed_multiple(curve) if curve is not None else None

    adj_seed = _lev_adjust(seed_mult, max_lev_closed, args.lev_penalty_k)
    adj_10y = _lev_adjust(recent10y, max_lev_closed, args.lev_penalty_k)

    out = {
        "TAG": args.tag,
        "GateSuffix": args.suffix,

        "ProfitTarget": args.profit_target,
        "MaxHoldingDays": args.max_days,
        "StopLevel": args.stop_level,
        "MaxExtendDaysParam": args.max_extend_days,

        "LeverageCapParam": args.max_leverage_pct,
        "LevPenaltyK": args.lev_penalty_k,

        "SeedMultiple": seed_mult if seed_mult is not None else np.nan,
        "Recent10Y_SeedMultiple": recent10y if recent10y is not None else np.nan,

        "MaxHoldingDaysObserved": max_hold,
        "Max_Extend_Over_MaxDays": max_extend_over if max_extend_over is not None else np.nan,

        "CycleCount": cycle_cnt,
        "ClosedCycleCount": closed_cnt,
        "SuccessRate": success_rate,

        "Max_LeveragePct_Closed": max_lev_closed if max_lev_closed is not None else np.nan,

        # leverage-adjusted
        "Adj_SeedMultiple": adj_seed if adj_seed is not None else np.nan,
        "Adj_Recent10Y_SeedMultiple": adj_10y if adj_10y is not None else np.nan,

        "TradesFile": str(trades_p),
        "CurveFile": str(args.curve_path) if args.curve_path else "",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gate_summary_{args.tag}_gate_{args.suffix}.csv"
    pd.DataFrame([out]).to_csv(out_path, index=False)
    print(f"[DONE] wrote gate summary: {out_path}")


if __name__ == "__main__":
    main()