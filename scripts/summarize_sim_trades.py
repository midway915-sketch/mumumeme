#!/usr/bin/env python3
# scripts/summarize_sim_trades.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _to_dt(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.tz_localize(None)


def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _infer_curve_from_trades(trades_path: Path) -> Path:
    name = trades_path.name.replace("sim_engine_trades_", "sim_engine_curve_")
    return trades_path.with_name(name)


def _safe_num(s: pd.Series, default=np.nan) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust: curve/trades에서 Date가 index로 들어오는 케이스 방어.
    - Date 컬럼이 없으면 DatetimeIndex(또는 변환 가능한 index)에서 뽑아서 Date로 만든다.
    """
    out = df.copy()

    if "Date" not in out.columns:
        if not isinstance(out.index, pd.RangeIndex):
            idx = out.index
            if isinstance(idx, pd.DatetimeIndex):
                out = out.reset_index()
                if "index" in out.columns and "Date" not in out.columns:
                    out = out.rename(columns={"index": "Date"})
            else:
                try:
                    tmp = pd.to_datetime(idx, errors="coerce")
                    if tmp.notna().any():
                        out = out.reset_index()
                        if "index" in out.columns and "Date" not in out.columns:
                            out = out.rename(columns={"index": "Date"})
                except Exception:
                    pass

    # common aliases (defensive)
    colmap = {c.lower(): c for c in out.columns}
    if "date" in colmap and "Date" not in out.columns:
        out = out.rename(columns={colmap["date"]: "Date"})
    if "datetime" in colmap and "Date" not in out.columns:
        out = out.rename(columns={colmap["datetime"]: "Date"})
    if "timestamp" in colmap and "Date" not in out.columns:
        out = out.rename(columns={colmap["timestamp"]: "Date"})

    return out


def _seed_multiple_from_curve(curve: pd.DataFrame) -> float | None:
    if curve is None or curve.empty:
        return None

    c = _ensure_date_column(curve)

    # prefer explicit column
    for col in ["SeedMultiple", "seed_multiple"]:
        if col in c.columns:
            v = _safe_num(c[col]).dropna()
            if len(v):
                return float(v.iloc[-1])

    # fallback: Equity(last)/Equity(first) by date ordering if possible
    if "Equity" in c.columns:
        eq = _safe_num(c["Equity"]).dropna()
        if len(eq) >= 2:
            # if Date exists, sort by Date for safety
            if "Date" in c.columns:
                d = _to_dt(c["Date"])
                tmp = c.copy()
                tmp["_d"] = d
                tmp = tmp.dropna(subset=["_d"]).sort_values("_d")
                eq2 = _safe_num(tmp["Equity"]).dropna()
                if len(eq2) >= 2 and float(eq2.iloc[0]) != 0:
                    return float(eq2.iloc[-1] / eq2.iloc[0])
            # else keep row order
            if float(eq.iloc[0]) != 0:
                return float(eq.iloc[-1] / eq.iloc[0])

    return None


def _recent10y_seed_multiple_from_curve(curve: pd.DataFrame) -> float | None:
    if curve is None or curve.empty:
        return None

    c = _ensure_date_column(curve)
    if "Date" not in c.columns:
        return None

    d = _to_dt(c["Date"])
    if d.isna().all():
        return None

    last = d.max()
    start = last - pd.Timedelta(days=365 * 10)

    sub = c.loc[d >= start].copy()
    if sub.empty:
        return None

    # sort by Date for deterministic first/last
    sub["_d"] = _to_dt(sub["Date"])
    sub = sub.dropna(subset=["_d"]).sort_values("_d")
    if sub.empty:
        return None

    for col in ["SeedMultiple", "seed_multiple"]:
        if col in sub.columns:
            v = _safe_num(sub[col]).dropna()
            if len(v):
                first = float(v.iloc[0])
                lastv = float(v.iloc[-1])
                if first != 0:
                    return float(lastv / first)
                return float(lastv)

    if "Equity" in sub.columns:
        eq = _safe_num(sub["Equity"]).dropna()
        if len(eq) >= 2 and float(eq.iloc[0]) != 0:
            return float(eq.iloc[-1] / eq.iloc[0])

    return None


def _cycle_stats(trades: pd.DataFrame) -> tuple[int, float, float | None, float | None]:
    if trades is None or trades.empty:
        return 0, 0.0, None, None

    t = trades.copy()
    cycle_cnt = int(len(t))

    # success
    if "Win" in t.columns:
        wins = (_safe_num(t["Win"], 0.0) > 0).astype(int)
    elif "CycleReturn" in t.columns:
        wins = (_safe_num(t["CycleReturn"], 0.0) > 0).astype(int)
    else:
        wins = pd.Series([0] * cycle_cnt)

    success_rate = float(wins.sum() / cycle_cnt) if cycle_cnt > 0 else 0.0

    # max holding observed
    max_hold = None
    if "HoldingDays" in t.columns:
        mh = _safe_num(t["HoldingDays"]).max()
        max_hold = float(mh) if np.isfinite(mh) else None

    # max leverage pct (cycle max among cycles)
    max_lev = None
    for c in ["MaxLeveragePct", "max_leverage_pct", "LeveragePct", "leverage_pct"]:
        if c in t.columns:
            mv = _safe_num(t[c]).max()
            max_lev = float(mv) if np.isfinite(mv) else None
            break

    return cycle_cnt, success_rate, max_hold, max_lev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-path", required=True, type=str)
    ap.add_argument("--curve-path", default="", type=str)  # optional
    ap.add_argument("--tag", required=True, type=str)
    ap.add_argument("--suffix", required=True, type=str)
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)
    ap.add_argument("--out-dir", default="data/signals", type=str)
    args = ap.parse_args()

    trades_path = Path(args.trades_path)
    trades = _read_any(trades_path)

    if args.curve_path:
        curve_path = Path(args.curve_path)
    else:
        curve_path = _infer_curve_from_trades(trades_path)

    curve = None
    if curve_path.exists():
        curve = _read_any(curve_path)

    cycle_cnt, success_rate, max_hold_obs, max_lev = _cycle_stats(trades)

    seed_mult = _seed_multiple_from_curve(curve) if isinstance(curve, pd.DataFrame) else None
    recent10y = _recent10y_seed_multiple_from_curve(curve) if isinstance(curve, pd.DataFrame) else None

    max_extend_obs = None
    if max_hold_obs is not None:
        max_extend_obs = float(max(0.0, max_hold_obs - float(args.max_days)))

    out = {
        "TAG": args.tag,
        "GateSuffix": args.suffix,
        "ProfitTarget": float(args.profit_target),
        "MaxHoldingDays": int(args.max_days),
        "StopLevel": float(args.stop_level),
        "MaxExtendDaysParam": int(args.max_extend_days),

        "SeedMultiple": seed_mult if seed_mult is not None else np.nan,
        "Recent10Y_SeedMultiple": recent10y if recent10y is not None else np.nan,

        "MaxHoldingDaysObserved": max_hold_obs if max_hold_obs is not None else np.nan,
        "MaxExtendDaysObserved": max_extend_obs if max_extend_obs is not None else np.nan,

        "CycleCount": int(cycle_cnt),
        "SuccessRate": float(success_rate),
        "MaxLeveragePct": max_lev if max_lev is not None else np.nan,

        "TradesFile": str(trades_path),
        "CurveFile": str(curve_path) if curve_path.exists() else "",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gate_summary_{args.tag}_gate_{args.suffix}.csv"
    pd.DataFrame([out]).to_csv(out_path, index=False)
    print(f"[DONE] wrote gate summary: {out_path}")


if __name__ == "__main__":
    main()