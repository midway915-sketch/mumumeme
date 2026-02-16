# scripts/build_strategy_labels.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEATURE_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"

PRICES_PARQUET = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"
FEATURES_PARQUET = FEATURE_DIR / "features.parquet"
FEATURES_CSV = FEATURE_DIR / "features.csv"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_tag(pt: float, max_days: int, sl: float, max_ext: int) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{max_days}_sl{sl_tag}_ex{max_ext}"


def load_prices() -> pd.DataFrame:
    if PRICES_PARQUET.exists():
        df = pd.read_parquet(PRICES_PARQUET)
    elif PRICES_CSV.exists():
        df = pd.read_csv(PRICES_CSV)
    else:
        raise FileNotFoundError("No prices found. Run scripts/fetch_prices.py first.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def load_feature_tickers() -> list[str]:
    if FEATURES_PARQUET.exists():
        f = pd.read_parquet(FEATURES_PARQUET, columns=["Ticker"])
    elif FEATURES_CSV.exists():
        f = pd.read_csv(FEATURES_CSV, usecols=["Ticker"])
    else:
        raise FileNotFoundError("features not found. Run scripts/build_features.py first.")
    f["Ticker"] = f["Ticker"].astype(str).str.upper().str.strip()
    return sorted(f["Ticker"].unique().tolist())


def simulate_cycle(
    i: int,
    dates: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    profit_target: float,
    max_days: int,
    stop_level: float,
    unit: float,
    max_extend_days: int,
) -> dict:
    """
    단일 티커/단일 사이클 시뮬(네 엔진 핵심 로직 반영)
    - entry: close[i]
    - 일반구간: PT 익절 or (max_days 도달 시) 조건부 손절/extend
    - extend구간: avg*(1+stop_level) 회복손절이면 종료, 아니면 full DCA
    - extend가 max_extend_days 초과면 강제 종료(실패 간주)
    반환: CycleReturn, ExitDate, CycleDays, ExtendDays, MinCycleRet, MaxCycleRet, ExtendedFlag, ForcedExitFlag
    """
    n = len(close)
    if i >= n or not np.isfinite(close[i]) or close[i] <= 0:
        return {
            "CycleReturn": np.nan,
            "ExitDate": pd.NaT,
            "CycleDays": 0,
            "ExtendDays": 0,
            "MinCycleRet": np.nan,
            "MaxCycleRet": np.nan,
            "ExtendedFlag": 0,
            "ForcedExitFlag": 0,
        }

    invested = unit
    shares = unit / close[i]
    holding_day = 1
    extending = False
    extend_days = 0
    extended_flag = 0
    forced_flag = 0

    min_ret = 0.0
    max_ret = 0.0

    for j in range(i + 1, n):
        holding_day += 1

        avg = invested / shares
        mtm = (shares * close[j] - invested) / invested
        if np.isfinite(mtm):
            min_ret = min(min_ret, mtm)
            max_ret = max(max_ret, mtm)

        if not extending:
            # 익절
            if high[j] >= avg * (1.0 + profit_target):
                exit_price = avg * (1.0 + profit_target)
                proceeds = shares * exit_price
                cycle_ret = (proceeds - invested) / invested
                return {
                    "CycleReturn": float(cycle_ret),
                    "ExitDate": dates[j],
                    "CycleDays": int(holding_day),
                    "ExtendDays": int(extend_days),
                    "MinCycleRet": float(min_ret),
                    "MaxCycleRet": float(max_ret),
                    "ExtendedFlag": int(extended_flag),
                    "ForcedExitFlag": int(forced_flag),
                }

            # max_days 도달 분기
            if holding_day >= max_days:
                current_ret = (close[j] - avg) / avg
                if current_ret >= stop_level:
                    # 종가 청산
                    exit_price = close[j]
                    proceeds = shares * exit_price
                    cycle_ret = (proceeds - invested) / invested
                    return {
                        "CycleReturn": float(cycle_ret),
                        "ExitDate": dates[j],
                        "CycleDays": int(holding_day),
                        "ExtendDays": int(extend_days),
                        "MinCycleRet": float(min_ret),
                        "MaxCycleRet": float(max_ret),
                        "ExtendedFlag": int(extended_flag),
                        "ForcedExitFlag": int(forced_flag),
                    }
                else:
                    # extend 시작
                    extending = True
                    extended_flag = 1

            # 일반구간 DCA
            avg = invested / shares
            cp = close[j]
            add = 0.0
            if cp <= avg:
                add = unit
            elif cp <= avg * 1.05:
                add = unit / 2.0

            if add > 0 and cp > 0 and np.isfinite(cp):
                invested += add
                shares += add / cp

        # extend 구간
        if extending:
            extend_days += 1
            avg = invested / shares

            # extend 너무 길면 강제 종료(실패 간주)
            if extend_days > max_extend_days:
                forced_flag = 1
                exit_price = close[j]
                proceeds = shares * exit_price
                cycle_ret = (proceeds - invested) / invested
                return {
                    "CycleReturn": float(cycle_ret),
                    "ExitDate": dates[j],
                    "CycleDays": int(holding_day),
                    "ExtendDays": int(extend_days),
                    "MinCycleRet": float(min_ret),
                    "MaxCycleRet": float(max_ret),
                    "ExtendedFlag": int(extended_flag),
                    "ForcedExitFlag": int(forced_flag),
                }

            # 회복손절
            if high[j] >= avg * (1.0 + stop_level):
                exit_price = avg * (1.0 + stop_level)
                proceeds = shares * exit_price
                cycle_ret = (proceeds - invested) / invested
                return {
                    "CycleReturn": float(cycle_ret),
                    "ExitDate": dates[j],
                    "CycleDays": int(holding_day),
                    "ExtendDays": int(extend_days),
                    "MinCycleRet": float(min_ret),
                    "MaxCycleRet": float(max_ret),
                    "ExtendedFlag": int(extended_flag),
                    "ForcedExitFlag": int(forced_flag),
                }

            # full DCA
            cp = close[j]
            if cp > 0 and np.isfinite(cp):
                invested += unit
                shares += unit / cp

    # 데이터 끝 → 미완료
    return {
        "CycleReturn": np.nan,
        "ExitDate": pd.NaT,
        "CycleDays": int(holding_day),
        "ExtendDays": int(extend_days),
        "MinCycleRet": float(min_ret),
        "MaxCycleRet": float(max_ret),
        "ExtendedFlag": int(extended_flag),
        "ForcedExitFlag": int(forced_flag),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--unit", type=float, default=1.0)
    ap.add_argument("--max-extend-days", type=int, default=30, help="extend가 이 일수 넘으면 강제 종료(실패 간주)")
    args = ap.parse_args()

    prices = load_prices()
    tickers = load_feature_tickers()

    tag = fmt_tag(args.profit_target, args.max_days, args.stop_level, args.max_extend_days)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    out_path = LABEL_DIR / f"strategy_labels_{tag}.parquet"
    out_csv = LABEL_DIR / f"strategy_labels_{tag}.csv"
    meta_path = LABEL_DIR / f"strategy_labels_{tag}_meta.json"

    rows = []
    for t in tickers:
        d = prices[prices["Ticker"] == t].copy()
        if d.empty:
            continue
        d = d.dropna(subset=["Close", "High", "Low"]).sort_values("Date").reset_index(drop=True)
        if len(d) < args.max_days + 10:
            continue

        dates = d["Date"].to_numpy()
        close = d["Close"].astype(float).to_numpy()
        high = d["High"].astype(float).to_numpy()
        low = d["Low"].astype(float).to_numpy()

        for i in range(len(d)):
            r = simulate_cycle(
                i=i,
                dates=dates,
                close=close,
                high=high,
                low=low,
                profit_target=args.profit_target,
                max_days=args.max_days,
                stop_level=args.stop_level,
                unit=args.unit,
                max_extend_days=args.max_extend_days,
            )
            rows.append(
                {
                    "Date": dates[i],
                    "Ticker": t,
                    **r,
                }
            )

    lab = pd.DataFrame(rows).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # 미완료 제거(학습용)
    before = len(lab)
    lab = lab.dropna(subset=["CycleReturn", "ExitDate"]).reset_index(drop=True)
    after = len(lab)
    print(f"[INFO] drop unfinished: {before} -> {after}")

    saved_to = ""
    try:
        lab.to_parquet(out_path, index=False)
        saved_to = f"parquet:{out_path}"
    except Exception as e:
        print(f"[WARN] parquet save failed ({e}), saving csv: {out_csv}")
        lab.to_csv(out_csv, index=False)
        saved_to = f"csv:{out_csv}"

    meta = {
        "updated_at_utc": now_utc_iso(),
        "saved_to": saved_to,
        "tag": tag,
        "profit_target": args.profit_target,
        "max_days": args.max_days,
        "stop_level": args.stop_level,
        "unit": args.unit,
        "max_extend_days": args.max_extend_days,
        "rows": int(len(lab)),
        "min_date": str(lab["Date"].min().date()) if len(lab) else None,
        "max_date": str(lab["Date"].max().date()) if len(lab) else None,
        "tickers": tickers,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] {saved_to} rows={len(lab)}")
    print(lab.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
