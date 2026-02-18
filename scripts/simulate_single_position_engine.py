# scripts/simulate_single_position_engine.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd


# -----------------------------
# IO helpers
# -----------------------------
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
    """
    Try to infer tag/suffix from picks filename. Fallback to passed values.
    """
    if tag and suffix:
        return tag, suffix

    name = Path(picks_path).name
    # e.g. picks_pt10_h40_sl10_ex30_gate_none_t0p20_q0p75_rutility.csv
    m = re.search(r"picks_(pt\d+_h\d+_sl\d+_ex\d+)_gate_(.+)\.(csv|parquet)$", name)
    if m:
        inferred_tag = m.group(1)
        inferred_suffix = m.group(2)
        return (tag or inferred_tag), (suffix or inferred_suffix)

    inferred_tag = tag or "run"
    inferred_suffix = suffix or Path(picks_path).stem.replace("picks_", "")
    return inferred_tag, inferred_suffix


# -----------------------------
# Strategy state
# -----------------------------
@dataclass
class Position:
    in_pos: bool = False
    ticker: str | None = None

    # cash balance (can be negative)
    seed: float = 0.0

    # position
    shares: float = 0.0
    invested: float = 0.0

    # cycle params
    entry_seed: float = 0.0          # S0 = seed at entry time
    unit: float = 0.0               # daily buy amount = entry_seed / max_days
    entry_date: pd.Timestamp | None = None
    holding_days: int = 0
    extending: bool = False

    # partial take-profit + trailing
    tp1_taken: bool = False
    tp1_date: pd.Timestamp | None = None
    peak_high: float = np.nan       # used after tp1_taken

    # tracking
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


# -----------------------------
# Leverage cap
# -----------------------------
def clamp_invest_by_leverage(seed: float, entry_seed: float, desired: float, max_leverage_pct: float) -> float:
    """
    Enforce: seed_after >= - entry_seed * max_leverage_pct

    desired : amount we'd like to spend today (>=0)
    returns : allowed invest (>=0), possibly reduced to satisfy the cap.
    """
    if desired <= 0:
        return 0.0

    if not np.isfinite(entry_seed) or entry_seed <= 0:
        # only allow spending existing cash (no borrowing) if entry_seed invalid
        return float(min(desired, max(seed, 0.0)))

    borrow_limit = float(entry_seed) * float(max_leverage_pct)  # e.g. 1.0 => 100%
    floor_seed = -borrow_limit

    # seed_after = seed - invest >= floor_seed  => invest <= seed - floor_seed
    room = float(seed) - float(floor_seed)
    if room <= 0:
        return 0.0
    return float(min(desired, room))


def update_max_leverage_pct(pos: Position, entry_seed: float, cap: float) -> None:
    if entry_seed <= 0:
        return
    lev = max(0.0, -pos.seed) / entry_seed
    if lev > pos.max_leverage_pct:
        pos.max_leverage_pct = float(lev)
    if pos.max_leverage_pct > cap + 1e-9:
        pos.max_leverage_pct = float(cap)


# -----------------------------
# Main simulation
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Single-position engine with leverage cap + partial TP + trailing.")
    ap.add_argument("--picks-path", required=True, type=str, help="CSV/Parquet with columns Date, Ticker (one pick per date).")
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--initial-seed", default=40_000_000, type=float)
    ap.add_argument("--profit-target", required=True, type=float)   # TP1 trigger e.g. 0.10
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)   # kept for tag/label compatibility (NO hard limit)

    ap.add_argument("--max-leverage-pct", default=1.0, type=float)  # 1.0 => 100%

    # partial TP + trailing params
    ap.add_argument("--tp1-sell-frac", default=0.5, type=float, help="fraction to sell at TP1 (0~1). default 0.5")
    ap.add_argument("--trail-dd", default=0.10, type=float, help="trailing drawdown after TP1. e.g. 0.10 => -10% from peak_high")

    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--suffix", default="", type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    args = ap.parse_args()

    if not (0.0 < args.tp1_sell_frac < 1.0):
        raise ValueError("--tp1-sell-frac must be between 0 and 1 (exclusive).")
    if args.trail_dd < 0:
        raise ValueError("--trail-dd must be >= 0")

    picks_path = args.picks_path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag, suffix = _pick_default_out_tag(picks_path, args.tag or None, args.suffix or None)

    # load picks
    if picks_path.lower().endswith(".parquet"):
        picks = pd.read_parquet(picks_path)
    else:
        picks = pd.read_csv(picks_path)

    if "Date" not in picks.columns or "Ticker" not in picks.columns:
        raise ValueError(f"picks file must have Date/Ticker. cols={list(picks.columns)[:50]}")

    picks = picks.copy()
    picks["Date"] = _norm_date(picks["Date"])
    picks["Ticker"] = picks["Ticker"].astype(str).str.upper().str.strip()
    picks = picks.dropna(subset=["Date", "Ticker"]).sort_values(["Date"]).reset_index(drop=True)
    picks = picks.drop_duplicates(["Date"], keep="last")  # one pick per date

    # load prices
    prices = read_table(args.prices_parq, args.prices_csv).copy()
    if "Date" not in prices.columns or "Ticker" not in prices.columns:
        raise ValueError(f"prices must have Date/Ticker. cols={list(prices.columns)[:50]}")
    prices["Date"] = _norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    for c in ["Open", "High", "Low", "Close"]:
        if c not in prices.columns:
            raise ValueError(f"prices missing {c}")
    prices = prices.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    picks_by_date = dict(zip(picks["Date"].tolist(), picks["Ticker"].tolist()))
    grouped = prices.groupby("Date", sort=True)

    pos = Position(seed=float(args.initial_seed), max_equity=float(args.initial_seed), max_dd=0.0)
    cooldown_today = False

    trades = []
    curve = []

    def record_trade_row(exit_date: pd.Timestamp, exit_price: float, reason: str) -> None:
        proceeds = pos.shares * exit_price
        cycle_return = (proceeds - pos.invested) / pos.invested if pos.invested > 0 else np.nan

        trades.append({
            "EntryDate": pos.entry_date,
            "ExitDate": exit_date,
            "Ticker": pos.ticker,
            "EntrySeed": pos.entry_seed,
            "ProfitTarget": args.profit_target,
            "TP1SellFrac": args.tp1_sell_frac,
            "TrailDD": args.trail_dd,
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
            "HoldingDays": pos.holding_days,
            "Extending": int(pos.extending),
            "TP1Taken": int(pos.tp1_taken),
            "TP1Date": pos.tp1_date,
            "Reason": reason,
            "MaxDrawdown": pos.max_dd,
        })

    def close_cycle(exit_date: pd.Timestamp, exit_price: float, reason: str) -> None:
        nonlocal cooldown_today
        record_trade_row(exit_date, exit_price, reason)

        proceeds = pos.shares * exit_price
        pos.seed += proceeds

        # reset position state
        pos.in_pos = False
        pos.ticker = None
        pos.shares = 0.0
        pos.invested = 0.0
        pos.entry_seed = 0.0
        pos.unit = 0.0
        pos.entry_date = None
        pos.holding_days = 0
        pos.extending = False

        pos.tp1_taken = False
        pos.tp1_date = None
        pos.peak_high = np.nan

        pos.max_leverage_pct = 0.0
        cooldown_today = True  # no re-entry on the same day

    def partial_take_profit(date: pd.Timestamp, tp_price: float) -> None:
        """
        Sell fraction of shares at tp_price. Reduce shares and invested proportionally.
        """
        frac = float(args.tp1_sell_frac)
        sell_shares = pos.shares * frac
        if sell_shares <= 0:
            return

        proceeds = sell_shares * tp_price

        # reduce cost basis proportionally
        pos.shares -= sell_shares
        pos.invested *= (1.0 - frac)

        pos.seed += proceeds

        pos.tp1_taken = True
        pos.tp1_date = date
        pos.peak_high = float(tp_price)  # initialize peak

        # note: we DO NOT end the cycle; still in position
        # update dd with close handled later

    # simulate across all dates
    for date, day_df in grouped:
        day_df = day_df.set_index("Ticker", drop=False)
        cooldown_today = False

        # ---- If in position: update, exits, buys
        if pos.in_pos:
            if pos.ticker not in day_df.index:
                pos.update_drawdown(None)
            else:
                row = day_df.loc[pos.ticker]
                close_px = float(row["Close"])
                high_px = float(row["High"])
                low_px = float(row["Low"])

                pos.holding_days += 1
                avg = pos.avg_price()

                # 1) TP1 partial sell (only once, only in non-extending zone)
                if (not pos.extending) and (not pos.tp1_taken):
                    if np.isfinite(avg) and np.isfinite(high_px) and high_px >= avg * (1.0 + args.profit_target):
                        tp_px = avg * (1.0 + args.profit_target)
                        partial_take_profit(date, float(tp_px))

                # 2) Trailing exit after TP1 (only while not extending)
                if (not pos.extending) and pos.tp1_taken and pos.shares > 0:
                    # update peak_high using today's high
                    if np.isfinite(high_px):
                        if not np.isfinite(pos.peak_high):
                            pos.peak_high = float(high_px)
                        else:
                            pos.peak_high = float(max(pos.peak_high, high_px))

                    trail_px = None
                    if np.isfinite(pos.peak_high) and args.trail_dd > 0:
                        trail_px = float(pos.peak_high * (1.0 - float(args.trail_dd)))

                    # if low breaches trailing price -> exit remaining
                    if trail_px is not None and np.isfinite(low_px) and low_px <= trail_px:
                        close_cycle(date, float(trail_px), reason="TRAIL_EXIT")

                # 3) max_days decision (only if still in pos, non-extending)
                if pos.in_pos and (not pos.extending) and (pos.holding_days >= int(args.max_days)):
                    avg2 = pos.avg_price()
                    cur_ret = (close_px - avg2) / avg2 if np.isfinite(avg2) and avg2 != 0 else -np.inf
                    if cur_ret >= float(args.stop_level):
                        close_cycle(date, float(close_px), reason="MAXDAY_CLOSE")
                    else:
                        pos.extending = True

                # 4) extending zone logic
                if pos.in_pos and pos.extending:
                    avg3 = pos.avg_price()

                    # exit if rebound hits stop-level threshold from avg
                    if np.isfinite(avg3) and np.isfinite(high_px) and high_px >= avg3 * (1.0 + args.stop_level):
                        exit_px = avg3 * (1.0 + args.stop_level)
                        close_cycle(date, float(exit_px), reason="EXT_RECOVERY")
                    else:
                        # DCA every day in extending: desired = unit (capped)
                        desired = float(pos.unit)
                        invest = clamp_invest_by_leverage(pos.seed, pos.entry_seed, desired, args.max_leverage_pct)
                        if invest > 0 and np.isfinite(close_px) and close_px > 0:
                            pos.seed -= invest
                            pos.invested += invest
                            pos.shares += invest / close_px
                            update_max_leverage_pct(pos, pos.entry_seed, args.max_leverage_pct)

                # 5) normal zone DCA (only if still in pos and not extending)
                if pos.in_pos and (not pos.extending):
                    avg4 = pos.avg_price()
                    if np.isfinite(avg4) and np.isfinite(close_px) and close_px > 0:
                        if close_px <= avg4:
                            desired = float(pos.unit)
                        elif close_px <= avg4 * 1.05:
                            desired = float(pos.unit) / 2.0
                        else:
                            desired = 0.0

                        invest = clamp_invest_by_leverage(pos.seed, pos.entry_seed, desired, args.max_leverage_pct)
                        if invest > 0:
                            pos.seed -= invest
                            pos.invested += invest
                            pos.shares += invest / close_px
                            update_max_leverage_pct(pos, pos.entry_seed, args.max_leverage_pct)

                # update drawdown tracking
                if pos.in_pos:
                    pos.update_drawdown(close_px)
                else:
                    pos.update_drawdown(None)

        # ---- Entry (only if flat and not sell-day)
        if (not pos.in_pos) and (not cooldown_today):
            pick = picks_by_date.get(date, None)
            if pick is not None and pick in day_df.index:
                row = day_df.loc[pick]
                close_px = float(row["Close"])
                if np.isfinite(close_px) and close_px > 0:
                    S0 = float(pos.seed)
                    unit = (S0 / float(args.max_days)) if args.max_days > 0 else 0.0

                    desired = float(unit)
                    invest = clamp_invest_by_leverage(pos.seed, S0, desired, args.max_leverage_pct)

                    if invest > 0:
                        pos.in_pos = True
                        pos.ticker = pick
                        pos.entry_seed = S0
                        pos.unit = unit
                        pos.entry_date = date
                        pos.holding_days = 1
                        pos.extending = False

                        pos.tp1_taken = False
                        pos.tp1_date = None
                        pos.peak_high = np.nan

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

        # ---- Record daily curve
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
            "Shares": pos.shares if pos.in_pos else 0.0,
            "Invested": pos.invested if pos.in_pos else 0.0,
            "PositionValue": pos.current_value(px),
            "AvgPrice": pos.avg_price() if pos.in_pos else np.nan,
            "HoldingDays": pos.holding_days if pos.in_pos else 0,
            "Extending": int(pos.extending) if pos.in_pos else 0,
            "TP1Taken": int(pos.tp1_taken) if pos.in_pos else 0,
            "PeakHigh": pos.peak_high if (pos.in_pos and pos.tp1_taken) else np.nan,
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