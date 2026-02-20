# scripts/summarize_sim_trades.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def _to_dt(x):
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


def _seed_multiple_from_curve(curve: pd.DataFrame) -> float | None:
    if curve is None or curve.empty:
        return None

    for c in ["SeedMultiple", "seed_multiple"]:
        if c in curve.columns:
            v = _safe_num(curve[c]).dropna()
            if len(v):
                return float(v.iloc[-1])

    if "Equity" in curve.columns:
        eq = _safe_num(curve["Equity"]).dropna()
        if len(eq) >= 2 and float(eq.iloc[0]) != 0:
            return float(eq.iloc[-1] / eq.iloc[0])

    return None


def _recent10y_seed_multiple_from_curve(curve: pd.DataFrame) -> float | None:
    if curve is None or curve.empty or "Date" not in curve.columns:
        return None

    d = _to_dt(curve["Date"])
    if d.isna().all():
        return None

    last = d.max()
    start = last - pd.Timedelta(days=365 * 10)
    sub = curve.loc[d >= start].copy()
    if sub.empty:
        return None

    for c in ["SeedMultiple", "seed_multiple"]:
        if c in sub.columns:
            v = _safe_num(sub[c]).dropna()
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


def _pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _cycle_stats(trades: pd.DataFrame) -> dict:
    """
    trades(사이클 단위 테이블)에서 요약 통계 산출.

    호환:
      - trailing entry: TrailingEntries(신규) 또는 TrailEntryCount(구버전)
      - peak cycle return: CycleMaxReturn(신규) 또는 PeakCycleReturn(구버전)
    """
    if trades is None or trades.empty:
        return {
            "CycleCount": 0,
            "SuccessRate": 0.0,
            "MaxHoldingDaysObserved": np.nan,
            "MaxLeveragePct": np.nan,
            "TrailEntryCountTotal": 0,
            "TrailEntryCountPerCycleAvg": 0.0,
            "MaxCyclePeakReturn": np.nan,
        }

    cycle_cnt = int(len(trades))

    # success
    if "Win" in trades.columns:
        wins = (_safe_num(trades["Win"], 0.0) > 0).astype(int)
    elif "CycleReturn" in trades.columns:
        wins = (_safe_num(trades["CycleReturn"], 0.0) > 0).astype(int)
    else:
        wins = pd.Series([0] * cycle_cnt)

    success_rate = float(wins.sum() / cycle_cnt) if cycle_cnt > 0 else 0.0

    # max holding observed
    max_hold_obs = np.nan
    if "HoldingDays" in trades.columns:
        mh = _safe_num(trades["HoldingDays"]).max()
        max_hold_obs = float(mh) if np.isfinite(mh) else np.nan

    # max leverage pct
    max_lev = np.nan
    lev_col = _pick_first_existing_col(trades, ["MaxLeveragePct", "max_leverage_pct", "LeveragePct", "leverage_pct"])
    if lev_col:
        mv = _safe_num(trades[lev_col]).max()
        max_lev = float(mv) if np.isfinite(mv) else np.nan

    # ✅ trailing entry stats (신규/구버전 둘 다)
    trail_col = _pick_first_existing_col(trades, ["TrailingEntries", "TrailEntryCount"])
    trail_total = 0
    trail_avg = 0.0
    if trail_col:
        te = _safe_num(trades[trail_col], 0.0).fillna(0).astype(int)
        trail_total = int(te.sum())
        trail_avg = float(te.mean()) if cycle_cnt > 0 else 0.0

    # ✅ peak cycle return (신규/구버전 둘 다)
    peak_col = _pick_first_existing_col(trades, ["CycleMaxReturn", "PeakCycleReturn"])
    max_peak_ret = np.nan
    if peak_col:
        mpr = _safe_num(trades[peak_col]).max()
        max_peak_ret = float(mpr) if np.isfinite(mpr) else np.nan

    return {
        "CycleCount": cycle_cnt,
        "SuccessRate": float(success_rate),
        "MaxHoldingDaysObserved": max_hold_obs,
        "MaxLeveragePct": max_lev,
        "TrailEntryCountTotal": int(trail_total),
        "TrailEntryCountPerCycleAvg": float(trail_avg),
        "MaxCyclePeakReturn": max_peak_ret,
    }


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

    st = _cycle_stats(trades)

    seed_mult = _seed_multiple_from_curve(curve) if isinstance(curve, pd.DataFrame) else None
    recent10y = _recent10y_seed_multiple_from_curve(curve) if isinstance(curve, pd.DataFrame) else None

    max_extend_obs = np.nan
    if np.isfinite(st["MaxHoldingDaysObserved"]):
        max_extend_obs = float(max(0.0, float(st["MaxHoldingDaysObserved"]) - float(args.max_days)))

    out = {
        "TAG": args.tag,
        "GateSuffix": args.suffix,
        "ProfitTarget": float(args.profit_target),
        "MaxHoldingDays": int(args.max_days),
        "StopLevel": float(args.stop_level),
        "MaxExtendDaysParam": int(args.max_extend_days),

        "SeedMultiple": seed_mult if seed_mult is not None else np.nan,
        "Recent10Y_SeedMultiple": recent10y if recent10y is not None else np.nan,

        "MaxHoldingDaysObserved": st["MaxHoldingDaysObserved"],
        "MaxExtendDaysObserved": max_extend_obs,

        "CycleCount": int(st["CycleCount"]),
        "SuccessRate": float(st["SuccessRate"]),
        "MaxLeveragePct": st["MaxLeveragePct"],

        # ✅ NEW
        "TrailEntryCountTotal": int(st["TrailEntryCountTotal"]),
        "TrailEntryCountPerCycleAvg": float(st["TrailEntryCountPerCycleAvg"]),
        "MaxCyclePeakReturn": st["MaxCyclePeakReturn"],

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