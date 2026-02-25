#!/usr/bin/env python3
# scripts/analyze_walkforward_summary.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _coerce_num(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _safe(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    p = Path(args.summary)
    if not p.exists():
        raise FileNotFoundError(f"summary not found: {p}")

    df = pd.read_csv(p)
    if df.empty:
        raise RuntimeError("summary is empty")

    # 핵심 numeric들 강제 변환 (여기서 실패하면 기존 코드가 0으로 만들기 쉬움)
    num_cols = [
        "SeedMultiple_AfterWarmup",
        "ExcessSeedMultiple_AfterWarmup",
        "CAGR_AfterWarmup",
        "Recent5Y_SeedMultiple_AfterWarmup",
        "Recent5Y_CAGR_AfterWarmup",
        "MaxDD_AfterWarmup",
        "MaxUnderwaterDays_AfterWarmup",
        "MaxDDRecoveryDays_AfterWarmup",
        "BadExitRate_Row",
        "BadExitRate_Ticker",
        "BadExitReturnDiff",
        "TradesPerYear",
        "HoldDays_Median",
        "DailyVol_Ann_AfterWarmup",
        "Sharpe0_Ann_AfterWarmup",
        "Sortino0_Ann_AfterWarmup",
    ]
    _coerce_num(df, num_cols)

    # -----------------------
    # Score 설계 (0으로만 찍히는 문제 방지)
    # - seed multiple 위주
    # - drawdown / badexit / vol 패널티
    # - excess seed multiple 있으면 가점
    # -----------------------
    sm = df["SeedMultiple_AfterWarmup"] if "SeedMultiple_AfterWarmup" in df.columns else np.nan
    exsm = df["ExcessSeedMultiple_AfterWarmup"] if "ExcessSeedMultiple_AfterWarmup" in df.columns else np.nan
    maxdd = df["MaxDD_AfterWarmup"] if "MaxDD_AfterWarmup" in df.columns else np.nan
    bad = df["BadExitRate_Ticker"] if "BadExitRate_Ticker" in df.columns else np.nan
    vol = df["DailyVol_Ann_AfterWarmup"] if "DailyVol_Ann_AfterWarmup" in df.columns else np.nan

    # log seed multiple (seed multiple이 1.0 근처면 점수 변화가 너무 작아서 log가 유리)
    log_sm = np.log(np.clip(sm.astype(float), 1e-9, None))

    # excess seed multiple 가점 (있으면)
    ex_bonus = np.log(np.clip(exsm.astype(float), 1e-9, None)) if "ExcessSeedMultiple_AfterWarmup" in df.columns else 0.0

    # penalty terms (NaN은 0 패널티로)
    dd_pen = np.nan_to_num(maxdd.astype(float), nan=0.0) * 1.5
    bad_pen = np.nan_to_num(bad.astype(float), nan=0.0) * 0.8
    vol_pen = np.nan_to_num(vol.astype(float), nan=0.0) * 0.25

    # final score
    df["Score"] = (np.nan_to_num(log_sm, nan=-1e9) + 0.35 * np.nan_to_num(ex_bonus, nan=0.0)) - dd_pen - bad_pen - vol_pen

    # 보조 지표: “안정성 점수” 같은 것도 같이
    df["StabilityScore"] = (
        np.nan_to_num(df.get("Sharpe0_Ann_AfterWarmup", np.nan), nan=0.0)
        + 0.5 * np.nan_to_num(df.get("Sortino0_Ann_AfterWarmup", np.nan), nan=0.0)
        - 1.2 * np.nan_to_num(df.get("MaxDD_AfterWarmup", np.nan), nan=0.0)
        - 0.6 * np.nan_to_num(df.get("BadExitRate_Ticker", np.nan), nan=0.0)
    )

    # 보기 좋은 정렬
    sort_cols = ["Score"]
    if "SeedMultiple_AfterWarmup" in df.columns:
        sort_cols.append("SeedMultiple_AfterWarmup")
    if "StabilityScore" in df.columns:
        sort_cols.append("StabilityScore")

    df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[DONE] wrote: {out_path} rows={len(df)}")

    # quick sanity print
    top = df.iloc[0].to_dict()
    print("=" * 60)
    print("[TOP]")
    print(f"WF_PERIOD={top.get('WF_PERIOD')} TAG={top.get('TAG')} GateSuffix={top.get('GateSuffix')}")
    print(f"Score={top.get('Score')}")
    print(f"SeedMultiple_AfterWarmup={top.get('SeedMultiple_AfterWarmup')}")
    print(f"ExcessSeedMultiple_AfterWarmup={top.get('ExcessSeedMultiple_AfterWarmup')}")
    print(f"MaxDD_AfterWarmup={top.get('MaxDD_AfterWarmup')} BadExitRate_Ticker={top.get('BadExitRate_Ticker')}")
    print("=" * 60)


if __name__ == "__main__":
    main()