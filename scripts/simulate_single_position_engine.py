# scripts/simulate_single_position_engine.py
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
SIGNAL_DIR = DATA_DIR / "signals"

PRICES_PARQUET = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

DEFAULT_PICKS_KS_TEMPLATE = "picks_{tag}_ks.csv"  # 기존 multi-method picks
# gate picks는 predict_gate.py가: picks_{tag}_gate_{suffix}.csv 형태로 저장

METHOD_TO_PICKCOL = {
    "ret_only": "pick_ret_only",
    "p_only": "pick_p_only",
    "gate_topk_then_ret": "pick_gate_ret",
    "utility": "pick_utility",
    "gate_topk_then_utility": "pick_gate_utility",
    "blend_rank": "pick_blend",
    "custom": "pick_custom",
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_tag(pt: float, max_days: int, sl: float, max_ext: int) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{max_days}_sl{sl_tag}_ex{max_ext}"


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    needed = ["Date", "Ticker", "Open", "High", "Low", "Close"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"prices missing columns: {miss}")
    df = df.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last").reset_index(drop=True)
    return df


def load_prices() -> pd.DataFrame:
    if PRICES_PARQUET.exists():
        df = pd.read_parquet(PRICES_PARQUET)
        return normalize_prices(df)
    if PRICES_CSV.exists():
        df = pd.read_csv(PRICES_CSV)
        return normalize_prices(df)
    raise FileNotFoundError("No prices found. Run scripts/fetch_prices.py first.")


def load_picks(picks_path: Optional[str], tag: str) -> pd.DataFrame:
    if picks_path:
        p = Path(picks_path)
        if not p.exists():
            raise FileNotFoundError(f"picks-path not found: {p}")
        df = pd.read_csv(p)
    else:
        p = SIGNAL_DIR / DEFAULT_PICKS_KS_TEMPLATE.format(tag=tag)
        if not p.exists():
            raise FileNotFoundError(f"default picks not found: {p}")
        df = pd.read_csv(p)

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if "Skipped" not in df.columns:
        df["Skipped"] = 0
    return df.sort_values("Date").reset_index(drop=True)


def safe_save(df: pd.DataFrame, parq: Path, csv: Path) -> str:
    parq.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parq, index=False)
        return str(parq)
    except Exception as e:
        # parquet 엔진이 없으면 CSV로라도 남김
        df.to_csv(csv, index=False)
        return str(csv)


@dataclass
class TradeState:
    entry_date: pd.Timestamp
    ticker: str
    cycle_unit: float
    entry_seed: float
    shares: float = 0.0
    invested: float = 0.0
    holding_days: int = 0
    extending: bool = False
    min_seed_in_cycle: float = 0.0


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, default=30)  # tag용(로직엔 미사용)

    ap.add_argument("--method", type=str, default="custom",
                    help="ret_only|p_only|gate_topk_then_ret|utility|gate_topk_then_utility|blend_rank|custom")
    ap.add_argument("--pick-col", type=str, default=None,
                    help="picks 파일에서 사용할 컬럼명. 지정하면 method 매핑 대신 이걸 씀.")
    ap.add_argument("--picks-path", type=str, default=None,
                    help="기본 picks_{tag}_ks.csv 대신 이 파일을 읽음")

    ap.add_argument("--initial-seed", type=float, default=40_000_000)
    ap.add_argument("--daily-buy", type=float, default=None,
                    help="매일 매수금액 고정 강제. 없으면 사이클 진입 시점 seed/max_days로 자동")
    ap.add_argument("--shadow-on-skip", action="store_true")

    ap.add_argument("--label", type=str, default=None,
                    help="summary에서 method 이름 대신 표시할 라벨(조합명 등)")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="출력 파일명 구분자. 예: none_t025_q075")

    args = ap.parse_args()

    pt = float(args.profit_target)
    max_days = int(args.max_days)
    sl = float(args.stop_level)
    tag = fmt_tag(pt, max_days, sl, int(args.max_extend_days))

    prices = load_prices()
    picks = load_picks(args.picks_path, tag)

    # 빠른 조회: (Date, Ticker) -> row
    px = prices.set_index(["Date", "Ticker"], drop=False)

    def get_px(date: pd.Timestamp, ticker: str) -> Optional[pd.Series]:
        key = (date, ticker)
        if key in px.index:
            return px.loc[key]
        return None

    fixed_daily_buy = float(args.daily_buy) if args.daily_buy is not None else None

    # pick column 결정
    if args.pick_col:
        pick_col = args.pick_col
    else:
        if args.method not in METHOD_TO_PICKCOL:
            raise ValueError(f"Unknown method: {args.method}")
        pick_col = METHOD_TO_PICKCOL[args.method]

    method_key = args.label.strip() if args.label else args.method

    # state
    seed = float(args.initial_seed)
    peak_equity = seed
    max_dd = 0.0

    state: Optional[TradeState] = None
    just_closed_date: Optional[pd.Timestamp] = None

    trades: List[dict] = []
    curve: List[dict] = []

    skipped_days = 0
    blocked_same_day = 0
    missing_price_days = 0
    entered = 0
    closed = 0

    def mark_equity(curr_date: pd.Timestamp) -> Tuple[float, float]:
        if state is None:
            return seed, 0.0
        row = get_px(curr_date, state.ticker)
        if row is None:
            return seed, 0.0
        val = float(state.shares) * float(row["Close"])
        return seed + val, val

    for _, r in picks.iterrows():
        date = r["Date"]
        skipped = int(r.get("Skipped", 0))

        # 포지션 보유 중
        if state is not None:
            row = get_px(date, state.ticker)
            if row is None:
                missing_price_days += 1
                eq, val = mark_equity(date)
                curve.append({
                    "Date": date, "Method": method_key, "Action": "HOLD_NO_PX",
                    "Ticker": state.ticker, "Seed": seed, "PosValue": val, "Equity": eq,
                    "HoldingDays": int(state.holding_days), "Extending": int(state.extending),
                })
                peak_equity = max(peak_equity, eq)
                dd = (eq - peak_equity) / peak_equity if peak_equity > 0 else 0.0
                max_dd = min(max_dd, dd)
                continue

            state.holding_days += 1
            high = float(row["High"])
            close = float(row["Close"])

            avg_price = (state.invested / state.shares) if state.shares > 0 else np.nan

            sold = False
            sell_price = None
            sell_reason = None

            # (A) 익절: extending 아닐 때만
            if not state.extending:
                tp_px = avg_price * (1.0 + pt)
                if high >= tp_px:
                    sold = True
                    sell_price = tp_px
                    sell_reason = "TP"

            # (B) max_days 도달 시 분기
            if (not sold) and (not state.extending) and (state.holding_days >= max_days):
                cur_ret = (close - avg_price) / avg_price
                if cur_ret >= sl:
                    sold = True
                    sell_price = close
                    sell_reason = "TIME_EXIT"
                else:
                    state.extending = True  # 연장 무제한

            # (C) 연장 회복 매도: -10%까지 회복하면 매도
            if (not sold) and state.extending:
                recover_px = avg_price * (1.0 + sl)  # sl=-0.10 -> 0.9*avg
                if high >= recover_px:
                    sold = True
                    sell_price = recover_px
                    sell_reason = "RECOVER_EXIT"

            if sold:
                proceeds = float(state.shares) * float(sell_price)
                seed += proceeds

                cycle_ret = (proceeds - state.invested) / state.invested if state.invested > 0 else 0.0

                min_seed = float(state.min_seed_in_cycle)
                max_loan = max(0.0, -min_seed)
                lev_pct = (max_loan / state.entry_seed) * 100.0 if state.entry_seed > 0 else np.nan

                trades.append({
                    "Method": method_key,
                    "EntryDate": state.entry_date,
                    "ExitDate": date,
                    "Ticker": state.ticker,
                    "SellReason": sell_reason,
                    "HoldingDays": int(state.holding_days),
                    "ExtendingExit": int(state.extending),
                    "CycleUnit": float(state.cycle_unit),
                    "Invested": float(state.invested),
                    "Proceeds": float(proceeds),
                    "CycleReturn": float(cycle_ret),
                    "EntrySeed": float(state.entry_seed),
                    "MinSeedInCycle": float(min_seed),
                    "MaxLoan": float(max_loan),
                    "LeveragePctMax": float(lev_pct) if np.isfinite(lev_pct) else np.nan,
                    "SeedAfter": float(seed),
                })

                curve.append({
                    "Date": date, "Method": method_key, "Action": f"SELL_{sell_reason}",
                    "Ticker": state.ticker, "Seed": seed, "PosValue": 0.0, "Equity": seed,
                    "HoldingDays": int(state.holding_days), "Extending": int(state.extending),
                    "SellPrice": float(sell_price),
                })

                eq = seed
                peak_equity = max(peak_equity, eq)
                dd = (eq - peak_equity) / peak_equity if peak_equity > 0 else 0.0
                max_dd = min(max_dd, dd)

                state = None
                just_closed_date = date
                closed += 1
                continue  # ✅ 당일 재진입 금지

            # ✅ 매도 안 했으면: 매일 매수(연장 포함 동일)
            buy_amt = float(state.cycle_unit)
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
                "AvgPrice": float(state.invested / state.shares) if state.shares > 0 else np.nan,
                "BuyAmt": buy_amt,
                "BuyPx": buy_px,
            })
            continue

        # 포지션 없음: 당일 재진입 금지 체크
        if just_closed_date is not None and date == just_closed_date:
            blocked_same_day += 1
            curve.append({
                "Date": date, "Method": method_key, "Action": "NO_REENTRY_SAME_DAY",
                "Ticker": "", "Seed": seed, "PosValue": 0.0, "Equity": seed
            })
            continue

        # skip
        if skipped == 1 and not args.shadow_on_skip:
            skipped_days += 1
            curve.append({
                "Date": date, "Method": method_key, "Action": "SKIP",
                "Ticker": "", "Seed": seed, "PosValue": 0.0, "Equity": seed
            })
            peak_equity = max(peak_equity, seed)
            dd = (seed - peak_equity) / peak_equity if peak_equity > 0 else 0.0
            max_dd = min(max_dd, dd)
            continue

        # pick ticker
        ticker_pick = r.get(pick_col, None)
        if ticker_pick is None or (isinstance(ticker_pick, float) and np.isnan(ticker_pick)) or str(ticker_pick).strip() == "":
            curve.append({
                "Date": date, "Method": method_key, "Action": "NO_PICK",
                "Ticker": "", "Seed": seed, "PosValue": 0.0, "Equity": seed
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

        # ✅ 진입: 진입 시점 seed로 cycle_unit 재계산 (옵션 있으면 고정)
        entry_seed = seed
        if fixed_daily_buy is not None:
            cycle_unit = fixed_daily_buy
            cycle_unit_mode = "fixed"
        else:
            base = entry_seed if entry_seed > 0 else float(args.initial_seed)
            cycle_unit = base / max_days
            cycle_unit_mode = "entry_seed_div_max_days"

        buy_px = float(row["Close"])
        buy_amt = float(cycle_unit)
        sh = buy_amt / buy_px
        seed -= buy_amt

        state = TradeState(
            entry_date=date,
            ticker=ticker,
            cycle_unit=float(cycle_unit),
            entry_seed=float(entry_seed),
            shares=float(sh),
            invested=float(buy_amt),
            holding_days=1,
            extending=False,
            min_seed_in_cycle=float(seed),
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
            "AvgPrice": float(state.invested / state.shares),
            "CycleUnitMode": cycle_unit_mode,
        })
        entered += 1

    # 종료 시점 평가
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

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve)

    closed_trades = trades_df.copy()
    n_closed = int(len(closed_trades))
    win_rate = float((closed_trades["CycleReturn"] > 0).mean()) if n_closed > 0 else 0.0
    avg_ret = float(closed_trades["CycleReturn"].mean()) if n_closed > 0 else 0.0
    avg_hold = float(closed_trades["HoldingDays"].mean()) if n_closed > 0 else 0.0

    avg_lev = float(closed_trades["LeveragePctMax"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if n_closed > 0 else 0.0
    max_lev = float(closed_trades["LeveragePctMax"].replace([np.inf, -np.inf], np.nan).dropna().max()) if n_closed > 0 else 0.0

    avg_cycle_unit = float(closed_trades["CycleUnit"].mean()) if n_closed > 0 else 0.0
    max_cycle_unit = float(closed_trades["CycleUnit"].max()) if n_closed > 0 else 0.0

    summary = {
        "updated_at_utc": now_utc_iso(),
        "label": method_key,
        "tag": tag,
        "pick_col": pick_col,
        "picks_path": args.picks_path,
        "initial_seed": float(args.initial_seed),
        "final_equity": float(final_equity),
        "seed_multiple": float(final_equity / float(args.initial_seed)) if args.initial_seed != 0 else np.nan,
        "total_return": float(final_equity / float(args.initial_seed) - 1.0) if args.initial_seed != 0 else np.nan,
        "max_drawdown": float(max_dd),
        "trades_closed": n_closed,
        "win_rate_closed": win_rate,
        "avg_cycle_return_closed": avg_ret,
        "avg_holding_days_closed": avg_hold,
        "avg_leverage_pct_closed": avg_lev,
        "max_leverage_pct_closed": max_lev,
        "cycle_unit_mode": ("fixed" if fixed_daily_buy is not None else "entry_seed_div_max_days"),
        "fixed_daily_buy": float(fixed_daily_buy) if fixed_daily_buy is not None else None,
        "avg_cycle_unit_closed": avg_cycle_unit,
        "max_cycle_unit_closed": max_cycle_unit,
        "entered_days": int(entered),
        "closed_trades_count": int(closed),
        "skipped_days": int(skipped_days),
        "blocked_same_day_reentry": int(blocked_same_day),
        "missing_price_days": int(missing_price_days),
        "shadow_on_skip": bool(args.shadow_on_skip),
        "open_note": open_note,
    }

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.out_suffix}" if str(args.out_suffix).strip() else ""

    out_trades_parq = SIGNAL_DIR / f"sim_engine_trades_{tag}_{method_key}{suffix}.parquet"
    out_trades_csv = SIGNAL_DIR / f"sim_engine_trades_{tag}_{method_key}{suffix}.csv"
    out_curve_parq = SIGNAL_DIR / f"sim_engine_curve_{tag}_{method_key}{suffix}.parquet"
    out_curve_csv = SIGNAL_DIR / f"sim_engine_curve_{tag}_{method_key}{suffix}.csv"
    out_summary_json = SIGNAL_DIR / f"sim_engine_summary_{tag}_{method_key}{suffix}.json"

    saved_trades = safe_save(trades_df, out_trades_parq, out_trades_csv)
    saved_curve = safe_save(curve_df, out_curve_parq, out_curve_csv)
    out_summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"[DONE] trades -> {saved_trades}")
    print(f"[DONE] curve  -> {saved_curve}")
    print(f"[DONE] summary-> {out_summary_json}")
    print(f"[SUMMARY] final_equity={summary['final_equity']:.2f}  seed_multiple={summary['seed_multiple']:.4f}  "
          f"maxDD={summary['max_drawdown']:.4f}  maxLev%={summary['max_leverage_pct_closed']:.2f}")


if __name__ == "__main__":
    main()
