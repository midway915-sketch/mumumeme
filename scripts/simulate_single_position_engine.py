# scripts/simulate_single_position_engine.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
SIGNALS_DIR = DATA_DIR / "signals"
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"


def read_prices() -> pd.DataFrame:
    if PRICES_PARQ.exists():
        df = pd.read_parquet(PRICES_PARQ)
    elif PRICES_CSV.exists():
        df = pd.read_csv(PRICES_CSV)
    else:
        raise FileNotFoundError(f"Missing prices: {PRICES_PARQ} (or {PRICES_CSV})")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    need = {"Date", "Ticker", "Open", "High", "Low", "Close"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"prices missing columns: {miss}")
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return df


def read_picks(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing picks file: {path}")
    df = pd.read_csv(p)
    df["Date"] = pd.to_datetime(df["Date"])
    if "Skipped" not in df.columns:
        df["Skipped"] = df[df.columns[1]].isna().astype(int)  # best-effort
    return df.sort_values("Date").reset_index(drop=True)


def make_tag(profit_target: float, max_days: int, stop_level: float, max_extend_days: int) -> str:
    pt = int(round(profit_target * 100))
    sl = int(round(abs(stop_level) * 100))
    return f"pt{pt}_h{int(max_days)}_sl{sl}_ex{int(max_extend_days)}"


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--initial-seed", type=float, default=40000000)

    ap.add_argument("--method", type=str, default="custom")
    ap.add_argument("--pick-col", type=str, default="pick_custom")
    ap.add_argument("--picks-path", type=str, required=True)

    ap.add_argument("--label", type=str, default="custom")
    ap.add_argument("--out-suffix", type=str, default="custom")
    ap.add_argument("--variant", type=str, default="BASE")

    # A/B 옵션 (extend에만 적용)
    ap.add_argument("--extend-lev-cap", type=float, default=None)
    ap.add_argument("--extend-min-buy-frac", type=float, default=0.0)
    ap.add_argument("--extend-buy-every", type=int, default=1)

    args = ap.parse_args()

    profit_target = float(args.profit_target)
    max_days = int(args.max_days)
    stop_level = float(args.stop_level)
    max_extend_days = int(args.max_extend_days)  # 현재 엔진에서는 강제 종료는 안 하지만 tag/라벨용 유지

    tag = make_tag(profit_target, max_days, stop_level, max_extend_days)

    prices = read_prices()
    picks = read_picks(args.picks_path)

    # 날짜별로 빠르게 접근
    day_groups = {d: g.set_index("Ticker") for d, g in prices.groupby("Date", sort=False)}

    seed = float(args.initial_seed)

    in_position = False
    extending = False
    holding_day = 0

    picked_ticker = None
    total_shares = 0.0
    total_invested = 0.0
    cycle_start_seed = 0.0
    cycle_unit = 0.0

    # 레버 기록(진입시 시드 대비 대출 최대치)
    cycle_min_seed = seed
    cycle_max_leverage_pct = 0.0

    # 재진입 금지(청산 당일)
    sold_today = False

    curve_rows = []
    trade_rows = []

    # picks의 날짜 범위만 돌리되, 가격이 있는 날에 대해서만 실행
    sim_dates = sorted(set(picks["Date"].unique()).intersection(set(day_groups.keys())))

    for date in sim_dates:
        sold_today = False

        day_prices = day_groups.get(date)
        if day_prices is None or day_prices.empty:
            continue

        # 오늘 pick
        today = picks[picks["Date"] == date]
        pick = None
        skipped = 1
        if len(today) > 0:
            skipped = int(today["Skipped"].iloc[0]) if "Skipped" in today.columns else 0
            if skipped == 0 and args.pick_col in today.columns:
                pick = today[args.pick_col].iloc[0]
                if isinstance(pick, float) and np.isnan(pick):
                    pick = None
        if isinstance(pick, str):
            pick = pick.upper().strip()

        # ====== 보유 중 처리 ======
        if in_position:
            if picked_ticker not in day_prices.index:
                # 가격 없으면 그냥 스킵(홀딩일은 증가시키지 않음: 데이터 누락은 엄격 처리 성격)
                equity = seed
                curve_rows.append(
                    dict(Date=date, Equity=equity, Method=args.label, Variant=args.variant,
                         InPosition=1, Ticker=picked_ticker, HoldingDays=holding_day)
                )
                continue

            row = day_prices.loc[picked_ticker]
            close_price = float(row["Close"])
            high_price = float(row["High"])

            holding_day += 1

            # 레버 업데이트
            cycle_min_seed = min(cycle_min_seed, seed)
            if cycle_start_seed > 0 and cycle_min_seed < 0:
                cycle_max_leverage_pct = max(cycle_max_leverage_pct, (-cycle_min_seed) / cycle_start_seed * 100.0)

            avg_price = total_invested / total_shares if total_shares > 0 else close_price

            # ---- max_days 도달 시 분기 ----
            if (holding_day >= max_days) and (not extending):
                current_return = (close_price - avg_price) / avg_price
                if current_return >= stop_level:
                    # 종가 청산
                    sell_price = close_price
                    proceeds = total_shares * sell_price
                    cycle_return = (proceeds - total_invested) / total_invested if total_invested > 0 else 0.0

                    seed += proceeds

                    trade_rows.append(
                        dict(
                            EntryDate=entry_date,
                            ExitDate=date,
                            Ticker=picked_ticker,
                            Invested=total_invested,
                            Proceeds=proceeds,
                            CycleReturn=cycle_return,
                            HoldingDays=holding_day,
                            LeveragePctMax=cycle_max_leverage_pct,
                            Method=args.label,
                            Variant=args.variant,
                        )
                    )

                    # 리셋
                    in_position = False
                    extending = False
                    sold_today = True
                    picked_ticker = None
                    total_shares = 0.0
                    total_invested = 0.0
                    holding_day = 0
                    cycle_start_seed = 0.0
                    cycle_unit = 0.0
                    cycle_min_seed = seed
                    cycle_max_leverage_pct = 0.0
                else:
                    extending = True

            # ---- extending: 회복하면 손해보고라도(avg*(1+stop_level)) 청산 ----
            if in_position and extending:
                trigger = avg_price * (1.0 + stop_level)
                if high_price >= trigger:
                    sell_price = trigger
                    proceeds = total_shares * sell_price
                    cycle_return = (proceeds - total_invested) / total_invested if total_invested > 0 else 0.0

                    seed += proceeds

                    trade_rows.append(
                        dict(
                            EntryDate=entry_date,
                            ExitDate=date,
                            Ticker=picked_ticker,
                            Invested=total_invested,
                            Proceeds=proceeds,
                            CycleReturn=cycle_return,
                            HoldingDays=holding_day,
                            LeveragePctMax=cycle_max_leverage_pct,
                            Method=args.label,
                            Variant=args.variant,
                        )
                    )

                    # 리셋
                    in_position = False
                    extending = False
                    sold_today = True
                    picked_ticker = None
                    total_shares = 0.0
                    total_invested = 0.0
                    holding_day = 0
                    cycle_start_seed = 0.0
                    cycle_unit = 0.0
                    cycle_min_seed = seed
                    cycle_max_leverage_pct = 0.0

            # ---- 매수 로직(하루 1회) ----
            if in_position:
                daily_buy_done = False

                # extend면 옵션 적용, 아니면 기본 규칙(기존: 가격에 따라 full/half/0)
                invest = 0.0
                if extending:
                    every = max(1, int(args.extend_buy_every))
                    allow_today = (holding_day % every == 0)

                    invest = cycle_unit

                    # 레버 기반 soft brake (extend에만)
                    if args.extend_lev_cap is not None and float(args.extend_lev_cap) > 0:
                        lev_cap = float(args.extend_lev_cap)
                        min_frac = float(args.extend_min_buy_frac)
                        lev = 0.0
                        if cycle_start_seed > 0 and seed < 0:
                            lev = (-seed) / cycle_start_seed  # 0.. (ratio)
                        scale = 1.0 - (lev / lev_cap)
                        if scale < min_frac:
                            scale = min_frac
                        if scale < 0.0:
                            scale = 0.0
                        invest = cycle_unit * scale

                    if allow_today and invest > 0:
                        shares = invest / close_price
                        total_shares += shares
                        total_invested += invest
                        seed -= invest
                        daily_buy_done = True
                else:
                    # 일반 구간: 기존 규칙
                    if close_price <= avg_price:
                        invest = cycle_unit
                    elif close_price <= avg_price * 1.05:
                        invest = cycle_unit / 2.0
                    else:
                        invest = 0.0

                    if invest > 0 and not daily_buy_done:
                        shares = invest / close_price
                        total_shares += shares
                        total_invested += invest
                        seed -= invest
                        daily_buy_done = True

            # curve 기록
            current_value = total_shares * close_price if in_position else 0.0
            equity = seed + current_value
            curve_rows.append(
                dict(Date=date, Equity=equity, Method=args.label, Variant=args.variant,
                     InPosition=int(in_position), Ticker=picked_ticker, HoldingDays=holding_day)
            )
            continue  # 보유 중이면 오늘 진입은 없음

        # ====== 미보유(진입 전) ======
        # 청산 당일 재진입 불가 (sold_today True면 skip)
        if (not in_position) and (not sold_today):
            if pick is not None and skipped == 0 and pick in day_prices.index:
                # 진입
                row = day_prices.loc[pick]
                entry_price = float(row["Close"])

                cycle_start_seed = seed  # 진입 시점 기준
                cycle_unit = cycle_start_seed / max_days if max_days > 0 else 0.0

                invest = cycle_unit
                shares = invest / entry_price

                total_shares = shares
                total_invested = invest
                seed -= invest

                in_position = True
                extending = False
                holding_day = 1
                picked_ticker = pick
                entry_date = date

                cycle_min_seed = seed
                cycle_max_leverage_pct = 0.0
            else:
                # 아무 것도 안 함
                pass

        # curve 기록(미보유)
        equity = seed
        curve_rows.append(
            dict(Date=date, Equity=equity, Method=args.label, Variant=args.variant,
                 InPosition=0, Ticker=None, HoldingDays=0)
        )

    curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)

    # 저장
    out_curve = SIGNALS_DIR / f"sim_engine_curve_{tag}_{args.label}_{args.variant}.parquet"
    out_trades = SIGNALS_DIR / f"sim_engine_trades_{tag}_{args.label}_{args.variant}.parquet"

    save_parquet(curve, out_curve)
    save_parquet(trades, out_trades)

    print("=" * 60)
    print("[DONE] simulate_single_position_engine")
    print("curve:", out_curve, "rows=", len(curve))
    print("trades:", out_trades, "rows=", len(trades))
    if len(trades) > 0:
        print("final equity:", float(curve["Equity"].iloc[-1]))
        print("seed multiple(all):", float(curve["Equity"].iloc[-1] / float(args.initial_seed)))
        print("max leverage pct(closed):", float(pd.to_numeric(trades["LeveragePctMax"], errors="coerce").max()))
    print("=" * 60)


if __name__ == "__main__":
    main()