# scripts/simulate_single_position_engine.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
SIGNALS_DIR = DATA_DIR / "signals"

PRICES_PARQUET = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"


def read_prices() -> pd.DataFrame:
    if PRICES_PARQUET.exists():
        prices = pd.read_parquet(PRICES_PARQUET)
    elif PRICES_CSV.exists():
        prices = pd.read_csv(PRICES_CSV)
    else:
        raise FileNotFoundError("Missing raw prices: data/raw/prices.parquet (or .csv). Run fetch_prices.py first.")

    prices = prices.copy()
    if "Date" not in prices.columns or "Ticker" not in prices.columns:
        raise ValueError(f"prices must have Date, Ticker. cols={list(prices.columns)}")

    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()

    # 기본 OHLCV 기대
    for c in ["Open", "High", "Low", "Close"]:
        if c not in prices.columns:
            raise ValueError(f"prices missing column: {c}")

    prices = prices.dropna(subset=["Date", "Ticker", "Close", "High"])
    prices = prices.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return prices


def read_picks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing picks file: {path}")

    picks = pd.read_csv(path)
    if "Date" not in picks.columns or "Ticker" not in picks.columns:
        raise ValueError(f"picks must have Date, Ticker. cols={list(picks.columns)}")

    picks = picks.copy()
    picks["Date"] = pd.to_datetime(picks["Date"], errors="coerce")  # ✅ 핵심 fix
    picks["Ticker"] = picks["Ticker"].astype(str).str.upper().str.strip()
    if "Skipped" not in picks.columns:
        picks["Skipped"] = 0

    picks = picks.dropna(subset=["Date", "Ticker"])
    picks = picks.sort_values(["Date"]).reset_index(drop=True)
    return picks


def safe_to_parquet(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return f"parquet:{path}"
    except Exception as e:
        # parquet 엔진(pyarrow) 없으면 csv로라도 저장
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return f"csv:{csv_path} (parquet failed: {e})"


def derive_tag_and_label(picks: pd.DataFrame) -> tuple[str, str]:
    tag = "custom"
    label = "BASE"
    if "TAG" in picks.columns and picks["TAG"].notna().any():
        tag = str(picks["TAG"].dropna().iloc[0])
    if "label" in picks.columns and picks["label"].notna().any():
        label = str(picks["label"].dropna().iloc[0])
    return tag, label


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-position DCA engine (infinite extend allowed).")
    ap.add_argument("--picks-path", required=True, type=str)

    # 전략 파라미터(워크플로우 입력으로 들어오는 값들)
    ap.add_argument("--profit-target", required=True, type=float)   # e.g. 0.10
    ap.add_argument("--max-days", required=True, type=int)          # e.g. 40
    ap.add_argument("--stop-level", required=True, type=float)      # e.g. -0.10
    ap.add_argument("--max-extend-days", required=True, type=int)   # 현재 엔진에서는 "제한 없음"이 기본. 기록/라벨용.

    ap.add_argument("--initial-seed", type=float, default=40_000_000.0)
    args = ap.parse_args()

    picks_path = Path(args.picks_path)

    prices = read_prices()
    picks = read_picks(picks_path)

    tag, label = derive_tag_and_label(picks)

    # ✅ 게이트별로 파일명 분리 (덮어쓰기 방지)
    trades_out = SIGNALS_DIR / f"sim_engine_trades_{tag}_gate_{label}.parquet"
    curve_out = SIGNALS_DIR / f"sim_engine_curve_{tag}_gate_{label}.parquet"

    # 날짜 단위로 1개 종목 pick (Skipped=0인 것만)
    picks_day = (
        picks[picks["Skipped"].fillna(0).astype(int) == 0]
        .groupby("Date", as_index=False)
        .head(1)
        .set_index("Date")
    )

    grouped = prices.groupby("Date", sort=True)

    seed = float(args.initial_seed)

    in_pos = False
    extending = False

    entry_date = None
    entry_seed = None
    ticker = None

    shares = 0.0
    invested = 0.0
    holding_day = 0

    # 레버리지(대출) 최대치 기록: seed가 음수로 내려간 최저점 기준
    min_seed_during_cycle = seed

    trades = []
    curve = []

    # “청산 당일 재진입 불가” 구현용
    sold_today = False

    for date, day in grouped:
        date = pd.to_datetime(date).to_pydatetime()
        sold_today = False

        day = day.set_index("Ticker")

        # 오늘 pick이 있으면 후보 티커
        pick_ticker = None
        if pd.Timestamp(date) in picks_day.index:
            pick_ticker = str(picks_day.loc[pd.Timestamp(date), "Ticker"]).upper().strip()

        # -------------------------
        # 1) 진입 전
        # -------------------------
        if (not in_pos) and (not sold_today):
            if pick_ticker is not None and pick_ticker in day.index:
                # 진입
                in_pos = True
                extending = False
                ticker = pick_ticker
                entry_date = pd.Timestamp(date)
                entry_seed = seed

                holding_day = 1

                # ✅ “진입 시 시드 기준”으로 daily unit 재계산
                cycle_unit = float(entry_seed) / float(args.max_days)

                # ✅ 진입일 매수도 daily unit
                px = float(day.loc[ticker, "Close"])
                buy = cycle_unit
                shares += buy / px
                invested += buy
                seed -= buy

                min_seed_during_cycle = min(min_seed_during_cycle, seed)

            # curve 기록(포지션 없으면 value=0)
            pos_value = 0.0
            equity = seed + pos_value
            curve.append({"Date": pd.Timestamp(date), "Equity": equity, "Cash": seed, "PositionValue": pos_value})
            continue

        # -------------------------
        # 2) 보유 중
        # -------------------------
        pos_value = 0.0
        if in_pos and ticker in day.index:
            close_px = float(day.loc[ticker, "Close"])
            high_px = float(day.loc[ticker, "High"])

            pos_value = shares * close_px
            avg_px = invested / shares if shares > 0 else close_px

            # ✅ 매일 매수 (진입 시 seed 기준 unit 고정, 연장에도 동일)
            cycle_unit = float(entry_seed) / float(args.max_days)
            buy = cycle_unit
            shares += buy / close_px
            invested += buy
            seed -= buy
            min_seed_during_cycle = min(min_seed_during_cycle, seed)

            holding_day += 1

            # ---- 일반 구간: 수익 실현 ----
            if not extending:
                if high_px >= avg_px * (1.0 + args.profit_target):
                    sell_px = avg_px * (1.0 + args.profit_target)
                    proceeds = shares * sell_px
                    ret = (proceeds - invested) / invested if invested > 0 else 0.0

                    # 레버리지% = (최대 대출액 / 진입시 시드) * 100
                    max_borrow = max(0.0, -min_seed_during_cycle)
                    lev_pct = (max_borrow / entry_seed * 100.0) if entry_seed and entry_seed > 0 else np.nan

                    seed += proceeds
                    trades.append({
                        "EntryDate": entry_date,
                        "ExitDate": pd.Timestamp(date),
                        "Ticker": ticker,
                        "HoldingDays": holding_day,
                        "CycleReturn": ret,
                        "Proceeds": proceeds,
                        "Invested": invested,
                        "MaxLeveragePct": lev_pct,
                        "ExitType": "PT",
                    })

                    # reset
                    in_pos = False
                    extending = False
                    ticker = None
                    entry_date = None
                    entry_seed = None
                    shares = 0.0
                    invested = 0.0
                    holding_day = 0
                    min_seed_during_cycle = seed
                    sold_today = True

            # ---- max_days 도달 시 분기 ----
            if in_pos and (not extending) and holding_day >= args.max_days:
                cur_ret = (close_px - avg_px) / avg_px if avg_px != 0 else 0.0

                # 손실이 stop_level 이상(덜 나쁨)이면 종가 청산
                if cur_ret >= args.stop_level:
                    sell_px = close_px
                    proceeds = shares * sell_px
                    ret = (proceeds - invested) / invested if invested > 0 else 0.0

                    max_borrow = max(0.0, -min_seed_during_cycle)
                    lev_pct = (max_borrow / entry_seed * 100.0) if entry_seed and entry_seed > 0 else np.nan

                    seed += proceeds
                    trades.append({
                        "EntryDate": entry_date,
                        "ExitDate": pd.Timestamp(date),
                        "Ticker": ticker,
                        "HoldingDays": holding_day,
                        "CycleReturn": ret,
                        "Proceeds": proceeds,
                        "Invested": invested,
                        "MaxLeveragePct": lev_pct,
                        "ExitType": "MAXDAYS_CLOSE",
                    })

                    in_pos = False
                    extending = False
                    ticker = None
                    entry_date = None
                    entry_seed = None
                    shares = 0.0
                    invested = 0.0
                    holding_day = 0
                    min_seed_during_cycle = seed
                    sold_today = True
                else:
                    extending = True

            # ---- 연장 구간: stop_level 만큼 회복하면 손절가로 청산 ----
            if in_pos and extending:
                # “손절가”는 평균단가*(1+stop_level) (stop_level이 음수면 avg보다 아래)
                trigger_px = avg_px * (1.0 + args.stop_level)
                if high_px >= trigger_px:
                    sell_px = trigger_px
                    proceeds = shares * sell_px
                    ret = (proceeds - invested) / invested if invested > 0 else 0.0

                    max_borrow = max(0.0, -min_seed_during_cycle)
                    lev_pct = (max_borrow / entry_seed * 100.0) if entry_seed and entry_seed > 0 else np.nan

                    seed += proceeds
                    trades.append({
                        "EntryDate": entry_date,
                        "ExitDate": pd.Timestamp(date),
                        "Ticker": ticker,
                        "HoldingDays": holding_day,
                        "CycleReturn": ret,
                        "Proceeds": proceeds,
                        "Invested": invested,
                        "MaxLeveragePct": lev_pct,
                        "ExitType": "EXT_STOP",
                    })

                    in_pos = False
                    extending = False
                    ticker = None
                    entry_date = None
                    entry_seed = None
                    shares = 0.0
                    invested = 0.0
                    holding_day = 0
                    min_seed_during_cycle = seed
                    sold_today = True

        equity = seed + pos_value
        curve.append({"Date": pd.Timestamp(date), "Equity": equity, "Cash": seed, "PositionValue": pos_value})

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve)

    saved_trades = safe_to_parquet(trades_df, trades_out)
    saved_curve = safe_to_parquet(curve_df, curve_out)

    print("=" * 60)
    print(f"[DONE] tag={tag} label={label}")
    print(f"[DONE] trades_saved={saved_trades}")
    print(f"[DONE] curve_saved={saved_curve}")
    print(f"[DONE] trades_rows={len(trades_df)} curve_rows={len(curve_df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()