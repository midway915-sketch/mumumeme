# scripts/simulate_single_position_engine.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
LABEL_DIR = DATA_DIR / "labels"
SIGNAL_DIR = DATA_DIR / "signals"

PRICES_PARQUET = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

METHOD_TO_PICKCOL = {
    "ret_only": "pick_ret_only",
    "p_only": "pick_p_only",
    "gate_topk_then_ret": "pick_gate_ret",
    "utility": "pick_utility",
    "gate_topk_then_utility": "pick_gate_utility",
    "blend_rank": "pick_blend",
}

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
    # 필요한 컬럼 보장
    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            raise ValueError(f"prices missing column: {c}")
    df = df.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last").reset_index(drop=True)
    return df

def load_picks(tag: str) -> pd.DataFrame:
    p = SIGNAL_DIR / f"picks_{tag}_ks.csv"
    if not p.exists():
        raise FileNotFoundError(f"picks file not found: {p}")
    df = pd.read_csv(p)
    df["Date"] = pd.to_datetime(df["Date"])
    if "Skipped" not in df.columns:
        df["Skipped"] = 0
    return df.sort_values("Date").reset_index(drop=True)

def safe_save(df: pd.DataFrame, path_parquet: Path, path_csv: Path) -> str:
    path_parquet.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path_parquet, index=False)
        return str(path_parquet)
    except Exception:
        df.to_csv(path_csv, index=False)
        return str(path_csv)

@dataclass
class TradeState:
    entry_date: pd.Timestamp
    ticker: str
    cycle_unit: float
    entry_seed: float
    shares: float = 0.0
    invested: float = 0.0
    holding_days: int = 0          # “가격 데이터가 있는 날” 기준
    extending: bool = False
    min_seed_in_cycle: float = 0.0 # 사이클 중 seed 최저치(=대출 최대치 측정용)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", type=float, required=True)   # 예: 0.10
    ap.add_argument("--max-days", type=int, required=True)          # 예: 40
    ap.add_argument("--stop-level", type=float, required=True)      # 예: -0.10
    ap.add_argument("--max-extend-days", type=int, default=30)      # 태그용(실제로는 무시)
    ap.add_argument("--method", type=str, default="gate_topk_then_utility",
                    help="ret_only|p_only|gate_topk_then_ret|utility|gate_topk_then_utility|blend_rank|all")
    ap.add_argument("--initial-seed", type=float, default=40_000_000)
    ap.add_argument("--daily-buy", type=float, default=None,
                    help="매일 매수금액. 기본값: initial_seed/max_days")
    ap.add_argument("--shadow-on-skip", action="store_true",
                    help="Skipped인 날에도 ret_only로 진입했다고 가정(기회비용)")
    args = ap.parse_args()

    pt = float(args.profit_target)
    max_days = int(args.max_days)
    sl = float(args.stop_level)

    # 태그는 기존 파일명 규칙과 맞추기 위해 유지(실제 extend 제한은 사용 안 함)
    tag = fmt_tag(pt, max_days, sl, int(args.max_extend_days))

    picks = load_picks(tag)
    prices = load_prices()

    # daily buy amount
    # daily_buy는 이제 "고정 매수금액 강제" 용도(옵션)로만 사용
    fixed_daily_buy = float(args.daily_buy) if args.daily_buy is not None else None

    # 빠른 조회용 (Date, Ticker) -> row
    px = prices.set_index(["Date", "Ticker"], drop=False)

    def get_px(date: pd.Timestamp, ticker: str) -> Optional[pd.Series]:
        key = (date, ticker)
        if key in px.index:
            return px.loc[key]
        return None

    # methods
    if args.method == "all":
        methods = list(METHOD_TO_PICKCOL.items())
    else:
        if args.method not in METHOD_TO_PICKCOL:
            raise ValueError(f"Unknown method: {args.method}")
        methods = [(args.method, METHOD_TO_PICKCOL[args.method])]

    # run per method
    summaries: List[dict] = []
    for method_key, pick_col in methods:
        seed = float(args.initial_seed)   # 현금(대출 가능)
        peak_equity = seed
        max_dd = 0.0

        state: Optional[TradeState] = None
        just_closed_date: Optional[pd.Timestamp] = None

        trades: List[dict] = []
        curve: List[dict] = []

        skipped_days = 0
        no_reentry_blocked = 0
        missing_price_days = 0
        entered = 0
        closed = 0

        for _, r in picks.iterrows():
            date = r["Date"]
            skipped = int(r.get("Skipped", 0))

            # 현재 포지션 가치/총자산 계산용
            def mark_equity(curr_date: pd.Timestamp) -> Tuple[float, float]:
                if state is None:
                    return seed, 0.0
                row = get_px(curr_date, state.ticker)
                if row is None:
                    # 가격 없으면 그냥 마지막 체결가로 마크 못 하니까, 현금만
                    return seed, 0.0
                value = float(state.shares) * float(row["Close"])
                return seed + value, value

            # 1) 포지션 보유중이면: 매도 트리거 체크 -> 아니면 매일 매수
            if state is not None:
                row = get_px(date, state.ticker)
                if row is None:
                    # 해당 날짜에 가격 데이터가 없으면 아무 것도 못 함
                    missing_price_days += 1
                    eq, val = mark_equity(date)
                    curve.append({
                        "Date": date, "Method": method_key, "Action": "HOLD_NO_PX",
                        "Ticker": state.ticker, "Seed": seed, "PosValue": val, "Equity": eq,
                        "HoldingDays": state.holding_days, "Extending": int(state.extending),
                    })
                    peak_equity = max(peak_equity, eq)
                    dd = (eq - peak_equity) / peak_equity if peak_equity > 0 else 0.0
                    max_dd = min(max_dd, dd)
                    continue

                # holding day 증가(가격 데이터가 있는 날만 카운트)
                state.holding_days += 1

                avg_price = (state.invested / state.shares) if state.shares > 0 else np.nan
                sold = False
                sell_price = None
                sell_reason = None

                high = float(row["High"])
                close = float(row["Close"])

                # (A) 익절: extending이 아닐 때만
                if not state.extending:
                    target = avg_price * (1.0 + pt)
                    if high >= target:
                        sold = True
                        sell_price = target
                        sell_reason = "TP"

                # (B) max_days 도달 시 분기(익절 안 된 경우)
                if (not sold) and (not state.extending) and (state.holding_days >= max_days):
                    current_ret = (close - avg_price) / avg_price
                    if current_ret >= sl:
                        sold = True
                        sell_price = close
                        sell_reason = "TIME_EXIT"
                    else:
                        state.extending = True  # 연장 진입(무제한)

                # (C) 연장 구간 회복 매도(무제한)
                if (not sold) and state.extending:
                    recover_px = avg_price * (1.0 + sl)  # sl이 -0.10이면 0.90*avg_price
                    if high >= recover_px:
                        sold = True
                        sell_price = recover_px
                        sell_reason = "RECOVER_EXIT"

                if sold:
                    proceeds = float(state.shares) * float(sell_price)
                    equity_before = seed + float(state.shares) * close  # 청산 직전 mark(대략)
                    seed += proceeds

                    cycle_ret = (proceeds - state.invested) / state.invested if state.invested > 0 else 0.0

                    min_seed = float(state.min_seed_in_cycle)
                    max_loan = max(0.0, -min_seed)
                    lev_pct = (max_loan / state.entry_seed) * 100.0 if state.entry_seed != 0 else np.nan

                    trades.append({
                        "Method": method_key,
                        "EntryDate": state.entry_date,
                        "ExitDate": date,
                        "Ticker": state.ticker,
                        "SellReason": sell_reason,
                        "HoldingDays": int(state.holding_days),
                        "ExtendingExit": int(state.extending),
                        "CycleUnit": state.cycle_unit,
                        "Invested": state.invested,
                        "Proceeds": proceeds,
                        "CycleReturn": cycle_ret,
                        "EntrySeed": state.entry_seed,
                        "MinSeedInCycle": min_seed,
                        "MaxLoan": max_loan,
                        "LeveragePctMax": lev_pct,
                        "SeedAfter": seed,
                        "EquityBeforeApprox": equity_before,
                    })

                    eq = seed
                    peak_equity = max(peak_equity, eq)
                    dd = (eq - peak_equity) / peak_equity if peak_equity > 0 else 0.0
                    max_dd = min(max_dd, dd)

                    curve.append({
                        "Date": date, "Method": method_key, "Action": f"SELL_{sell_reason}",
                        "Ticker": state.ticker, "Seed": seed, "PosValue": 0.0, "Equity": seed,
                        "HoldingDays": int(state.holding_days), "Extending": int(state.extending),
                        "SellPrice": float(sell_price),
                    })

                    state = None
                    just_closed_date = date
                    closed += 1
                    continue  # ✅ 당일 재진입 금지(여기서 바로 다음 날짜로)

                # 매도 안 됐으면: ✅ 매일 동일 금액 매수(조건 없음)
                buy_amt = state.cycle_unit   # ✅ 연장 포함 항상 동일
                buy_px = close
                add_sh = buy_amt / buy_px
                
                state.shares += add_sh
                state.invested += buy_amt
                seed -= buy_amt
                state.min_seed_in_cycle = min(state.min_seed_in_cycle, seed)

                eq, val = mark_equity(date)
                peak_equity = max(peak_equity, eq)
                dd = (eq - peak_equity) / peak_equity if peak_equity > 0 else 0.0
                max_dd = min(max_dd, dd)

                curve.append({
                    "Date": date, "Method": method_key, "Action": "HOLD_BUY",
                    "Ticker": state.ticker, "Seed": seed, "PosValue": val, "Equity": eq,
                    "HoldingDays": int(state.holding_days), "Extending": int(state.extending),
                    "AvgPrice": (state.invested / state.shares) if state.shares > 0 else np.nan,
                    "BuyAmt": buy_amt,
                    "BuyPx": buy_px,
                })
                continue

            # 2) 포지션 없으면: 당일 재진입 금지 체크
            if just_closed_date is not None and date == just_closed_date:
                no_reentry_blocked += 1
                curve.append({
                    "Date": date, "Method": method_key, "Action": "NO_REENTRY_SAME_DAY",
                    "Ticker": np.nan, "Seed": seed, "PosValue": 0.0, "Equity": seed
                })
                continue

            # 3) 스킵 처리
            if skipped == 1 and not args.shadow_on_skip:
                skipped_days += 1
                curve.append({
                    "Date": date, "Method": method_key, "Action": "SKIP",
                    "Ticker": np.nan, "Seed": seed, "PosValue": 0.0, "Equity": seed
                })
                peak_equity = max(peak_equity, seed)
                dd = (seed - peak_equity) / peak_equity if peak_equity > 0 else 0.0
                max_dd = min(max_dd, dd)
                continue

            # 4) 진입 티커 결정
            ticker_pick = r.get(pick_col, None)
            if skipped == 1 and args.shadow_on_skip:
                ticker_pick = r.get("pick_ret_only", ticker_pick)

            if ticker_pick is None or (isinstance(ticker_pick, float) and np.isnan(ticker_pick)) or str(ticker_pick).strip() == "":
                curve.append({
                    "Date": date, "Method": method_key, "Action": "NO_PICK",
                    "Ticker": np.nan, "Seed": seed, "PosValue": 0.0, "Equity": seed
                })
                continue

            ticker = str(ticker_pick).upper().strip()
            row = get_px(date, ticker)
            if row is None:
                missing_price_days += 1
                curve.append({
                    "Date": date, "Method": method_key, "Action": "ENTER_FAIL_NO_PX",
                    "Ticker": ticker, "Seed": seed, "PosValue": 0.0, "Equity": seed
                })
                continue

            # ✅ 진입: 진입일부터 매일 daily_buy로 매수(진입일도 매수함)
            entry_seed = seed  # ✅ 진입 "시점"의 seed(첫 매수 전에)
            buy_px = float(row["Close"])
            
            # ✅ 사이클 단위: 진입 시점 seed 기준으로 재계산
            # 옵션(--daily-buy)이 있으면 그걸로 고정, 없으면 entry_seed/max_days
            if fixed_daily_buy is not None:
                cycle_unit = fixed_daily_buy
            else:
                # entry_seed가 0 이하로 들어오면 daily_buy가 음수/0이 되니까 안전장치
                # (원하면 abs(entry_seed)/max_days로 바꿔도 됨)
                base = entry_seed if entry_seed > 0 else float(args.initial_seed)
                cycle_unit = base / max_days
            
            buy_amt = cycle_unit
            sh = buy_amt / buy_px
            seed -= buy_amt
            
            state = TradeState(
                entry_date=date,
                ticker=ticker,
                cycle_unit=cycle_unit,   # ✅ 사이클 고정 단위
                entry_seed=entry_seed,   # ✅ 레버리지% 기준 seed
                shares=sh,
                invested=buy_amt,
                holding_days=1,
                extending=False,
                min_seed_in_cycle=min(seed, seed),
            )

            eq = seed + state.shares * buy_px
            peak_equity = max(peak_equity, eq)
            dd = (eq - peak_equity) / peak_equity if peak_equity > 0 else 0.0
            max_dd = min(max_dd, dd)

            curve.append({
                "Date": date, "Method": method_key, "Action": "ENTER_BUY",
                "Ticker": ticker, "Seed": seed, "PosValue": state.shares * buy_px, "Equity": eq,
                "HoldingDays": 1, "Extending": 0,
                "BuyAmt": buy_amt, "BuyPx": buy_px,
                "AvgPrice": (state.invested / state.shares),
            })
            entered += 1

        # 종료 시점에 포지션 열려있으면 평가
        final_equity = seed
        open_note = None
        if state is not None:
            last_date = picks["Date"].max()
            row = get_px(last_date, state.ticker)
            if row is not None:
                final_equity = seed + float(state.shares) * float(row["Close"])
            else:
                final_equity = seed
            open_note = "OPEN_AT_END"

            # 오픈 트레이드도 별도 기록(수익률은 unrealized)
            row2 = row
            mkt_close = float(row2["Close"]) if row2 is not None else np.nan
            mkt_value = float(state.shares) * mkt_close if np.isfinite(mkt_close) else np.nan
            unreal_ret = (mkt_value - state.invested) / state.invested if (np.isfinite(mkt_value) and state.invested > 0) else np.nan
            min_seed = float(state.min_seed_in_cycle)
            max_loan = max(0.0, -min_seed)
            lev_pct = (max_loan / state.entry_seed) * 100.0 if state.entry_seed > 0 else np.nan

            trades.append({
                "Method": method_key,
                "EntryDate": state.entry_date,
                "ExitDate": pd.NaT,
                "Ticker": state.ticker,
                "SellReason": "OPEN_AT_END",
                "HoldingDays": int(state.holding_days),
                "ExtendingExit": int(state.extending),
                "CycleUnit": state.cycle_unit,
                "Invested": state.invested,
                "Proceeds": np.nan,
                "CycleReturn": unreal_ret,
                "EntrySeed": state.entry_seed,
                "MinSeedInCycle": min_seed,
                "MaxLoan": max_loan,
                "LeveragePctMax": lev_pct,
                "SeedAfter": seed,
                "EquityBeforeApprox": final_equity,
                "NOTE": "UNREALIZED",
            })

        trades_df = pd.DataFrame(trades)
        curve_df = pd.DataFrame(curve)

        closed_trades = trades_df[trades_df["SellReason"].ne("OPEN_AT_END")].copy()
        n_closed = int(len(closed_trades))
        win_rate = float((closed_trades["CycleReturn"] > 0).mean()) if n_closed > 0 else 0.0
        avg_ret = float(closed_trades["CycleReturn"].mean()) if n_closed > 0 else 0.0
        avg_hold = float(closed_trades["HoldingDays"].mean()) if n_closed > 0 else 0.0
        avg_lev = float(closed_trades["LeveragePctMax"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if n_closed > 0 else 0.0
        max_lev = float(closed_trades["LeveragePctMax"].replace([np.inf, -np.inf], np.nan).dropna().max()) if n_closed > 0 else 0.0
        avg_cycle_unit = float(closed_trades["CycleUnit"].mean()) if n_closed > 0 else 0.0
        max_cycle_unit = float(closed_trades["CycleUnit"].max()) if n_closed > 0 else 0.0

        summary = {
            "method": method_key,
            "tag": tag,
            "initial_seed": float(args.initial_seed),
            "cycle_unit_mode": ("fixed" if fixed_daily_buy is not None else "entry_seed_div_max_days"),
            "fixed_daily_buy": (float(fixed_daily_buy) if fixed_daily_buy is not None else None),
            "avg_cycle_unit_closed": avg_cycle_unit,
            "max_cycle_unit_closed": max_cycle_unit,
            "final_equity": float(final_equity),
            "total_return": float(final_equity / float(args.initial_seed) - 1.0) if args.initial_seed != 0 else np.nan,
            "max_drawdown": float(max_dd),
            "trades_closed": n_closed,
            "trades_total_including_open": int(len(trades_df)),
            "win_rate_closed": win_rate,
            "avg_cycle_return_closed": avg_ret,
            "avg_holding_days_closed": avg_hold,
            "avg_leverage_pct_closed": avg_lev,
            "max_leverage_pct_closed": max_lev,
            "entered_days": int(entered),
            "closed_trades_count": int(closed),
            "skipped_days": int(skipped_days),
            "blocked_same_day_reentry": int(no_reentry_blocked),
            "missing_price_days": int(missing_price_days),
            "shadow_on_skip": bool(args.shadow_on_skip),
            "open_note": open_note,
        }

        # save
        out_trades_parq = SIGNAL_DIR / f"sim_engine_trades_{tag}_{method_key}.parquet"
        out_trades_csv = SIGNAL_DIR / f"sim_engine_trades_{tag}_{method_key}.csv"
        out_curve_parq = SIGNAL_DIR / f"sim_engine_curve_{tag}_{method_key}.parquet"
        out_curve_csv = SIGNAL_DIR / f"sim_engine_curve_{tag}_{method_key}.csv"
        out_summary_json = SIGNAL_DIR / f"sim_engine_summary_{tag}_{method_key}.json"

        saved_trades = safe_save(trades_df, out_trades_parq, out_trades_csv)
        saved_curve = safe_save(curve_df, out_curve_parq, out_curve_csv)
        SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
        out_summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

        print(f"[DONE] {method_key} trades -> {saved_trades}")
        print(f"[DONE] {method_key} curve  -> {saved_curve}")
        print(f"[DONE] {method_key} summary-> {out_summary_json}")
        print(f"[SUMMARY] final_equity={summary['final_equity']:.2f}  "
              f"trades_closed={summary['trades_closed']}  maxDD={summary['max_drawdown']:.4f}  "
              f"maxLev%={summary['max_leverage_pct_closed']:.2f}")

        summaries.append(summary)

    # combined summary
    summ_df = pd.DataFrame(summaries)
    comb_csv = SIGNAL_DIR / f"sim_engine_summary_{tag}_ALL.csv"
    summ_df.sort_values("final_equity", ascending=False).to_csv(comb_csv, index=False)
    print("\n=== ENGINE SINGLE-POSITION SUMMARY (sorted by final_equity) ===")
    print(summ_df.sort_values("final_equity", ascending=False).to_string(index=False))
    print(f"\n[COMBINED] {comb_csv}")

if __name__ == "__main__":
    main()
