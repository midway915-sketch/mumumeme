# scripts/simulate_single_position_engine.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd


def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _pick_default_out_tag(picks_path: str, tag: str | None, suffix: str | None) -> tuple[str, str]:
    if tag and suffix:
        return tag, suffix
    name = Path(picks_path).name
    m = re.search(r"picks_(pt\d+_h\d+_sl\d+_ex\d+)_gate_(.+)\.(csv|parquet)$", name)
    if m:
        inferred_tag = m.group(1)
        inferred_suffix = m.group(2)
        return (tag or inferred_tag), (suffix or inferred_suffix)
    inferred_tag = tag or "run"
    inferred_suffix = suffix or Path(picks_path).stem.replace("picks_", "")
    return inferred_tag, inferred_suffix


@dataclass
class Position:
    in_pos: bool = False
    ticker: str | None = None

    seed: float = 0.0
    shares: float = 0.0
    invested: float = 0.0

    entry_seed: float = 0.0
    unit_base: float = 0.0
    entry_date: pd.Timestamp | None = None
    holding_days: int = 0
    extending: bool = False

    tau_class: int = 1  # 0=FAST,1=MID,2=SLOW (default MID)

    max_leverage_pct: float = 0.0
    max_equity: float = 0.0
    max_dd: float = 0.0

    def avg_price(self) -> float:
        return (self.invested / self.shares) if (self.shares > 0 and self.invested > 0) else np.nan

    def current_value(self, px_close: float | None) -> float:
        if not self.in_pos or self.shares <= 0 or px_close is None or not np.isfinite(px_close):
            return 0.0
        return float(self.shares) * float(px_close)

    def equity(self, px_close: float | None) -> float:
        return float(self.seed) + self.current_value(px_close)

    def update_drawdown(self, px_close: float | None) -> None:
        eq = self.equity(px_close)
        if eq > self.max_equity:
            self.max_equity = eq
        if self.max_equity > 0:
            dd = (eq - self.max_equity) / self.max_equity
            if dd < self.max_dd:
                self.max_dd = dd


def clamp_invest_by_leverage(seed: float, entry_seed: float, desired: float, max_leverage_pct: float) -> float:
    if desired <= 0:
        return 0.0
    if not np.isfinite(entry_seed) or entry_seed <= 0:
        return float(min(desired, max(seed, 0.0)))
    borrow_limit = float(entry_seed) * float(max_leverage_pct)
    floor_seed = -borrow_limit
    room = float(seed) - float(floor_seed)  # seed + borrow_limit
    if room <= 0:
        return 0.0
    return float(min(desired, room))


def update_max_leverage_pct(pos: Position, entry_seed: float, max_leverage_pct_cap: float) -> None:
    if entry_seed <= 0:
        return
    lev = max(0.0, -pos.seed) / entry_seed
    if lev > pos.max_leverage_pct:
        pos.max_leverage_pct = float(lev)
    if pos.max_leverage_pct > max_leverage_pct_cap + 1e-9:
        pos.max_leverage_pct = float(max_leverage_pct_cap)


def segment_multiplier(holding_day: int, max_days: int, tau_class: int) -> float:
    """
    3 segments based on max_days:
      A: 1 .. floor(0.5*max_days)
      B: next .. floor(0.75*max_days)
      C: rest .. max_days+

    Multipliers:
      FAST: A 1.2, B 0.8, C 0.4
      MID : A 1.0, B 1.0, C 0.8
      SLOW: A 0.6, B 0.8, C 1.2
    """
    md = max(int(max_days), 1)
    a_end = max(1, int(np.floor(0.5 * md)))
    b_end = max(a_end + 1, int(np.floor(0.75 * md)))

    d = int(holding_day)
    if d <= a_end:
        seg = "A"
    elif d <= b_end:
        seg = "B"
    else:
        seg = "C"

    if int(tau_class) == 0:  # FAST
        return {"A": 1.2, "B": 0.8, "C": 0.4}[seg]
    if int(tau_class) == 2:  # SLOW
        return {"A": 0.6, "B": 0.8, "C": 1.2}[seg]
    return {"A": 1.0, "B": 1.0, "C": 0.8}[seg]


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-position engine with leverage cap + tau-based buy sizing.")
    ap.add_argument("--picks-path", required=True, type=str)
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--initial-seed", default=40_000_000, type=float)
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)  # kept for compatibility
    ap.add_argument("--max-leverage-pct", default=1.0, type=float)  # 1.0 => 100%

    ap.add_argument("--extend-buy-mult", default=0.5, type=float, help="extending daily buy = unit_base * extend-buy-mult (then capped)")
    ap.add_argument("--extend-stop-buy-near-cap", default=0.90, type=float, help="stop buying if leverage >= this*cap while extending")

    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--suffix", default="", type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag, suffix = _pick_default_out_tag(args.picks_path, args.tag or None, args.suffix or None)

    # picks
    if args.picks_path.lower().endswith(".parquet"):
        picks = pd.read_parquet(args.picks_path)
    else:
        picks = pd.read_csv(args.picks_path)

    if "Date" not in picks.columns or "Ticker" not in picks.columns:
        raise ValueError(f"picks must have Date/Ticker. cols={list(picks.columns)[:50]}")

    picks = picks.copy()
    picks["Date"] = _norm_date(picks["Date"])
    picks["Ticker"] = picks["Ticker"].astype(str).str.upper().str.strip()
    picks = picks.dropna(subset=["Date", "Ticker"]).sort_values(["Date"]).reset_index(drop=True)
    picks = picks.drop_duplicates(["Date"], keep="last")

    # optional tau in picks
    has_tau = "TauClassPred" in picks.columns
    if has_tau:
        picks["TauClassPred"] = pd.to_numeric(picks["TauClassPred"], errors="coerce").fillna(1).astype(int)
    else:
        picks["TauClassPred"] = 1

    picks_by_date = {d: (t, int(tc)) for d, t, tc in zip(picks["Date"].tolist(), picks["Ticker"].tolist(), picks["TauClassPred"].tolist())}

    # prices
    prices = read_table(args.prices_parq, args.prices_csv).copy()
    for c in ["Date", "Ticker", "Open", "High", "Low", "Close"]:
        if c not in prices.columns:
            raise ValueError(f"prices missing {c}")
    prices["Date"] = _norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices = prices.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    grouped = prices.groupby("Date", sort=True)

    pos = Position(seed=float(args.initial_seed), max_equity=float(args.initial_seed), max_dd=0.0)
    cooldown_today = False
    trades = []
    curve = []

    def close_cycle(exit_date: pd.Timestamp, exit_price: float, reason: str) -> None:
        nonlocal cooldown_today, pos, trades
        proceeds = pos.shares * exit_price
        cycle_return = (proceeds - pos.invested) / pos.invested if pos.invested > 0 else np.nan
        win = int(cycle_return > 0) if np.isfinite(cycle_return) else 0

        trades.append({
            "EntryDate": pos.entry_date,
            "ExitDate": exit_date,
            "Ticker": pos.ticker,
            "TauClass": int(pos.tau_class),
            "EntrySeed": pos.entry_seed,
            "ProfitTarget": args.profit_target,
            "MaxDays": args.max_days,
            "StopLevel": args.stop_level,
            "MaxExtendDaysParam": args.max_extend_days,
            "MaxLeveragePctCap": args.max_leverage_pct,
            "MaxLeveragePct": pos.max_leverage_pct,
            "Invested": pos.invested,
            "Shares": pos.shares,
            "ExitPrice": exit_price,
            "Proceeds": proceeds,
            "CycleReturn": cycle_return,
            "Win": win,
            "HoldingDays": pos.holding_days,
            "Extending": int(pos.extending),
            "Reason": reason,
            "MaxDrawdown": pos.max_dd,
        })

        pos.seed += proceeds

        # reset position state (keep portfolio dd tracking)
        pos.in_pos = False
        pos.ticker = None
        pos.shares = 0.0
        pos.invested = 0.0
        pos.entry_seed = 0.0
        pos.unit_base = 0.0
        pos.entry_date = None
        pos.holding_days = 0
        pos.extending = False
        pos.tau_class = 1
        pos.max_leverage_pct = 0.0

        cooldown_today = True

    for date, day_df in grouped:
        day_df = day_df.set_index("Ticker", drop=False)
        cooldown_today = False

        # holding logic
        if pos.in_pos:
            if pos.ticker not in day_df.index:
                pos.update_drawdown(None)
            else:
                row = day_df.loc[pos.ticker]
                close_px = float(row["Close"])
                high_px = float(row["High"])

                pos.holding_days += 1
                avg = pos.avg_price()

                # ---- exits
                if not pos.extending:
                    # TP intraday
                    if np.isfinite(avg) and np.isfinite(high_px) and high_px >= avg * (1.0 + args.profit_target):
                        exit_px = avg * (1.0 + args.profit_target)
                        close_cycle(date, float(exit_px), reason="TP")
                    else:
                        # max_days reached => close or extend
                        if pos.in_pos and pos.holding_days >= int(args.max_days):
                            cur_ret = (close_px - avg) / avg if np.isfinite(avg) and avg != 0 else -np.inf
                            if cur_ret >= float(args.stop_level):
                                close_cycle(date, float(close_px), reason="MAXDAY_CLOSE")
                            else:
                                pos.extending = True

                # ---- buys
                if pos.in_pos:
                    lev_now = (max(0.0, -pos.seed) / pos.entry_seed) if pos.entry_seed > 0 else 0.0

                    if pos.extending:
                        # exit on recovery to stop-level threshold from avg
                        if np.isfinite(avg) and np.isfinite(high_px) and high_px >= avg * (1.0 + args.stop_level):
                            exit_px = avg * (1.0 + args.stop_level)
                            close_cycle(date, float(exit_px), reason="EXT_RECOVERY")
                        else:
                            # buy only if not near leverage cap
                            if lev_now >= float(args.extend_stop_buy_near_cap) * float(args.max_leverage_pct):
                                # stop buying
                                pass
                            else:
                                desired = float(pos.unit_base) * float(args.extend_buy_mult)
                                invest = clamp_invest_by_leverage(pos.seed, pos.entry_seed, desired, args.max_leverage_pct)
                                if invest > 0 and np.isfinite(close_px) and close_px > 0:
                                    pos.seed -= invest
                                    pos.invested += invest
                                    pos.shares += invest / close_px
                                    update_max_leverage_pct(pos, pos.entry_seed, args.max_leverage_pct)

                    else:
                        # normal zone: tau-based unit sizing by segment
                        mult = segment_multiplier(pos.holding_days, int(args.max_days), int(pos.tau_class))
                        unit_today = float(pos.unit_base) * float(mult)

                        desired = 0.0
                        if np.isfinite(avg) and np.isfinite(close_px) and close_px > 0:
                            if close_px <= avg:
                                desired = unit_today
                            elif close_px <= avg * 1.05:
                                desired = unit_today / 2.0
                            else:
                                desired = 0.0

                        invest = clamp_invest_by_leverage(pos.seed, pos.entry_seed, desired, args.max_leverage_pct)
                        if invest > 0:
                            pos.seed -= invest
                            pos.invested += invest
                            pos.shares += invest / close_px
                            update_max_leverage_pct(pos, pos.entry_seed, args.max_leverage_pct)

                # dd update
                if pos.in_pos:
                    pos.update_drawdown(close_px)
                else:
                    pos.update_drawdown(None)

        # entry logic (no re-entry on sell day)
        if (not pos.in_pos) and (not cooldown_today):
            pick_info = picks_by_date.get(date, None)
            if pick_info is not None:
                pick, tau_cls = pick_info
                if pick in day_df.index:
                    row = day_df.loc[pick]
                    close_px = float(row["Close"])
                    if np.isfinite(close_px) and close_px > 0:
                        S0 = float(pos.seed)  # entry seed
                        unit_base = (S0 / float(args.max_days)) if args.max_days > 0 else 0.0

                        # day1 buy uses tau segment for holding_day=1
                        mult = segment_multiplier(1, int(args.max_days), int(tau_cls))
                        unit_today = float(unit_base) * float(mult)

                        invest = clamp_invest_by_leverage(pos.seed, S0, unit_today, args.max_leverage_pct)
                        if invest > 0:
                            pos.in_pos = True
                            pos.ticker = pick
                            pos.tau_class = int(tau_cls)
                            pos.entry_seed = S0
                            pos.unit_base = unit_base
                            pos.entry_date = date
                            pos.holding_days = 1
                            pos.extending = False
                            pos.max_leverage_pct = 0.0

                            pos.seed -= invest
                            pos.invested = invest
                            pos.shares = invest / close_px
                            update_max_leverage_pct(pos, pos.entry_seed, args.max_leverage_pct)

                            pos.update_drawdown(close_px)
                        else:
                            pos.update_drawdown(None)
                    else:
                        pos.update_drawdown(None)
                else:
                    pos.update_drawdown(None)
            else:
                pos.update_drawdown(None)

        # curve record
        px = None
        if pos.in_pos and pos.ticker is not None and pos.ticker in day_df.index:
            px = float(day_df.loc[pos.ticker]["Close"])

        eq = pos.equity(px)
        curve.append({
            "Date": date,
            "Equity": eq,
            "Seed": pos.seed,
            "Ticker": pos.ticker if pos.in_pos else "",
            "InPosition": int(pos.in_pos),
            "TauClass": int(pos.tau_class) if pos.in_pos else np.nan,
            "Shares": pos.shares if pos.in_pos else 0.0,
            "Invested": pos.invested if pos.in_pos else 0.0,
            "PositionValue": pos.current_value(px),
            "AvgPrice": pos.avg_price() if pos.in_pos else np.nan,
            "HoldingDays": pos.holding_days if pos.in_pos else 0,
            "Extending": int(pos.extending) if pos.in_pos else 0,
            "MaxLeveragePctCycle": pos.max_leverage_pct if pos.in_pos else 0.0,
            "MaxDrawdownPortfolio": pos.max_dd,
        })

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve)
    if not curve_df.empty:
        curve_df["SeedMultiple"] = curve_df["Equity"] / float(args.initial_seed)

    trades_path = out_dir / f"sim_engine_trades_{tag}_gate_{suffix}.parquet"
    curve_path = out_dir / f"sim_engine_curve_{tag}_gate_{suffix}.parquet"
    trades_df.to_parquet(trades_path, index=False)
    curve_df.to_parquet(curve_path, index=False)

    print(f"[DONE] wrote trades: {trades_path} rows={len(trades_df)}")
    print(f"[DONE] wrote curve : {curve_path} rows={len(curve_df)}")
    if not curve_df.empty:
        final_mult = float(curve_df["SeedMultiple"].iloc[-1])
        print(f"[INFO] final SeedMultiple={final_mult:.4f} maxDD={float(pos.max_dd):.4f}")


if __name__ == "__main__":
    main()