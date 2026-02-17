# scripts/summarize_sim_trades.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def _to_dt(x):
    return pd.to_datetime(x, errors="coerce").dt.tz_localize(None)


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    if p.suffix.lower() in (".parquet",):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _col_first(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _seed_multiple_from_curve(curve: pd.DataFrame, initial_seed: float | None = None) -> tuple[float | None, float | None]:
    """
    Returns (seed_multiple, recent10y_seed_multiple)
    """
    if curve is None or len(curve) == 0:
        return (None, None)

    date_col = _col_first(curve, ["Date", "date"])
    eq_col = _col_first(curve, ["Equity", "equity", "TotalEquity", "total_equity"])
    sm_col = _col_first(curve, ["SeedMultiple", "seed_multiple"])

    if date_col is None or (eq_col is None and sm_col is None):
        return (None, None)

    d = _to_dt(curve[date_col])
    if d.isna().all():
        return (None, None)

    curve = curve.copy()
    curve["_Date"] = d
    curve = curve.dropna(subset=["_Date"]).sort_values("_Date").reset_index(drop=True)

    if len(curve) == 0:
        return (None, None)

    # seed multiple overall
    seed_mult = None
    if sm_col is not None:
        v = pd.to_numeric(curve[sm_col], errors="coerce").dropna()
        if len(v):
            seed_mult = float(v.iloc[-1])
    if seed_mult is None and eq_col is not None:
        eq = pd.to_numeric(curve[eq_col], errors="coerce").dropna()
        if len(eq) >= 2:
            base = float(eq.iloc[0])
            if base != 0:
                seed_mult = float(eq.iloc[-1] / base)
            elif initial_seed is not None and initial_seed != 0:
                seed_mult = float(eq.iloc[-1] / float(initial_seed))
        elif len(eq) == 1 and initial_seed is not None and initial_seed != 0:
            seed_mult = float(eq.iloc[-1] / float(initial_seed))

    # recent 10y
    last = curve["_Date"].max()
    start = last - pd.Timedelta(days=365 * 10)
    sub = curve.loc[curve["_Date"] >= start].copy()
    recent10y = None
    if len(sub) >= 2:
        if sm_col is not None:
            vv = pd.to_numeric(sub[sm_col], errors="coerce").dropna()
            if len(vv):
                recent10y = float(vv.iloc[-1] / vv.iloc[0]) if float(vv.iloc[0]) != 0 else float(vv.iloc[-1])
        if recent10y is None and eq_col is not None:
            eqs = pd.to_numeric(sub[eq_col], errors="coerce").dropna()
            if len(eqs) >= 2 and float(eqs.iloc[0]) != 0:
                recent10y = float(eqs.iloc[-1] / eqs.iloc[0])

    return (seed_mult, recent10y)


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df))
    return pd.to_numeric(df[col], errors="coerce")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-path", required=True, type=str)
    ap.add_argument("--curve-path", default="", type=str, help="Optional curve parquet/csv with Date, Equity (preferred).")

    ap.add_argument("--tag", required=True, type=str)
    ap.add_argument("--suffix", required=True, type=str)

    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--initial-seed", default=np.nan, type=float)
    ap.add_argument("--out-dir", default="data/signals", type=str)
    args = ap.parse_args()

    trades = _read_any(args.trades_path)
    if not isinstance(trades, pd.DataFrame):
        trades = pd.DataFrame(trades)

    # --- Cycle metrics
    hold_col = _col_first(trades, ["HoldingDays", "holding_days"])
    if hold_col is None:
        max_hold = np.nan
    else:
        hd = pd.to_numeric(trades[hold_col], errors="coerce")
        max_hold = float(hd.max()) if np.isfinite(hd.max()) else np.nan

    max_extend_obs = np.nan
    if np.isfinite(max_hold):
        max_extend_obs = float(max(0.0, max_hold - float(args.max_days)))

    # wins / success rate
    win_col = _col_first(trades, ["Win", "win", "IsWin", "is_win", "Success", "success"])
    if win_col is not None:
        wins = pd.to_numeric(trades[win_col], errors="coerce").fillna(0).to_numpy()
        wins = (wins > 0).astype(int)
    else:
        ret_col = _col_first(trades, ["CycleReturn", "cycle_return", "Return", "ret", "PnL_pct", "pnl_pct"])
        if ret_col is not None:
            r = pd.to_numeric(trades[ret_col], errors="coerce").fillna(0).to_numpy()
            wins = (r > 0).astype(int)
        else:
            wins = np.array([], dtype=int)

    cycle_cnt = int(len(trades))
    win_cnt = int(wins.sum()) if cycle_cnt > 0 else 0
    success_rate = (win_cnt / cycle_cnt) if cycle_cnt > 0 else 0.0

    # max leverage
    lev_col = _col_first(trades, ["MaxLeveragePct", "max_leverage_pct", "LeveragePct", "leverage_pct"])
    max_lev = np.nan
    if lev_col is not None:
        lv = pd.to_numeric(trades[lev_col], errors="coerce")
        if lv.notna().any():
            max_lev = float(lv.max())

    # --- SeedMultiple: prefer curve
    seed_mult = None
    recent10y = None

    if args.curve_path:
        try:
            curve = _read_any(args.curve_path)
            init = None if not np.isfinite(args.initial_seed) else float(args.initial_seed)
            seed_mult, recent10y = _seed_multiple_from_curve(curve, initial_seed=init)
        except Exception as e:
            print(f"[WARN] curve read/compute failed: {e}")

    # fallback: try derive from trades itself if any equity-like columns exist
    if seed_mult is None:
        eq_col = _col_first(trades, ["Equity", "equity", "TotalEquity", "total_equity"])
        sm_col = _col_first(trades, ["SeedMultiple", "seed_multiple", "FinalSeedMultiple", "final_seed_multiple"])
        if sm_col is not None:
            v = pd.to_numeric(trades[sm_col], errors="coerce").dropna()
            if len(v):
                seed_mult = float(v.iloc[-1])
        elif eq_col is not None:
            eq = pd.to_numeric(trades[eq_col], errors="coerce").dropna()
            if len(eq) >= 2 and float(eq.iloc[0]) != 0:
                seed_mult = float(eq.iloc[-1] / eq.iloc[0])

    out = {
        "TAG": args.tag,
        "GateSuffix": args.suffix,
        "ProfitTarget": args.profit_target,
        "MaxHoldingDays": args.max_days,
        "StopLevel": args.stop_level,
        "MaxExtendDaysParam": args.max_extend_days,
        "SeedMultiple": seed_mult if seed_mult is not None else np.nan,
        "Recent10Y_SeedMultiple": recent10y if recent10y is not None else np.nan,
        "MaxHoldingDaysObserved": max_hold,
        "MaxExtendDaysObserved": max_extend_obs,
        "CycleCount": cycle_cnt,
        "SuccessRate": success_rate,
        "MaxLeveragePct": max_lev,
        "TradesFile": str(Path(args.trades_path)),
        "CurveFile": str(Path(args.curve_path)) if args.curve_path else "",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gate_summary_{args.tag}_gate_{args.suffix}.csv"
    pd.DataFrame([out]).to_csv(out_path, index=False)
    print(f"[DONE] wrote gate summary: {out_path}")


if __name__ == "__main__":
    main()