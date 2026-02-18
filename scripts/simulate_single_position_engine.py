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

    # cash balance (can be negative; leverage cap controls it)
    seed: float = 0.0

    # position
    shares: float = 0.0
    invested: float = 0.0

    # cycle params
    entry_seed: float = 0.0          # S0 = seed at entry time
    unit_base: float = 0.0           # base daily buy = entry_seed / max_days
    entry_date: pd.Timestamp | None = None
    holding_days: int = 0
    extending: bool = False

    # tau budget control (only for pre-max_days DCA schedule)
    tau_class_pred: int | None = None
    tau_budget_total: float = 0.0      # total budget for days 1..max_days (== entry_seed ideally)
    tau_budget_used: float = 0.0       # cum spend within days 1..max_days

    # tracking
    max_leverage_pct: float = 0.0    # max(-seed)/entry_seed observed in this cycle
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
# Leverage cap: core
# -----------------------------
def clamp_invest_by_leverage(seed: float, entry_seed: float, desired: float, max_leverage_pct: float) -> float:
    """
    Enforce: seed_after >= - entry_seed * max_leverage_pct

    desired : amount we'd like to spend today (>=0)
    returns : allowed invest (>=0), possibly reduced to satisfy the cap.
    """
    if desired <= 0:
        return 0.0

    # If entry_seed <= 0, cap is meaningless; safest: disallow borrowing beyond 0 (no negative).
    if not np.isfinite(entry_seed) or entry_seed <= 0:
        return float(min(desired, max(seed, 0.0)))

    borrow_limit = float(entry_seed) * float(max_leverage_pct)  # 1.0 => 100%
    floor_seed = -borrow_limit

    # seed_after = seed - invest >= floor_seed  => invest <= seed - floor_seed = seed + borrow_limit
    room = float(seed) - float(floor_seed)
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


# -----------------------------
# Tau-based buy schedule
# -----------------------------
def infer_tau_class_from_row(row: pd.Series) -> int | None:
    """
    Prefer TauClassPred if exists.
    Else infer from probabilities if exist.
    Returns 0/1/2/3 or None.
    """
    if row is None:
        return None

    if "TauClassPred" in row.index:
        try:
            v = int(pd.to_numeric(row["TauClassPred"], errors="coerce"))
            if v in (0, 1, 2, 3):
                return v
        except Exception:
            pass

    # Infer from probs if present (TauP10, TauP20, TauPH)
    # We choose the earliest horizon that has high probability
    def g(name: str) -> float:
        if name not in row.index:
            return np.nan
        try:
            return float(pd.to_numeric(row[name], errors="coerce"))
        except Exception:
            return np.nan

    p10 = g("TauP10")
    p20 = g("TauP20")
    pH  = g("TauPH")

    # heuristic thresholds (can be tuned later)
    if np.isfinite(p10) and p10 >= 0.45:
        return 0
    if np.isfinite(p20) and p20 >= 0.55:
        return 1
    if np.isfinite(pH) and pH >= 0.60:
        return 2
    if np.isfinite(pH):
        return 3

    return None


def tau_multiplier(day: int, max_days: int, tau_class: int | None) -> float:
    """
    Profit-max default schedule (balanced budget later).
    - Fast (0): front-load
    - Mid  (1): flat
    - Slow (2/3): back-load
    """
    if tau_class is None:
        return 1.0

    d = int(day)
    H = int(max_days)

    # split points
    # 40 -> early=10, mid=20; 30 -> early=8, mid=15; 50 -> early=12, mid=25
    early = max(1, int(round(H * 0.25)))
    mid = max(early + 1, int(round(H * 0.50)))

    if tau_class == 0:
        # fast: buy more early, less later
        if d <= early:
            return 1.40
        elif d <= mid:
            return 1.05
        else:
            return 0.85

    if tau_class == 1:
        # mid: mostly flat
        if d <= early:
            return 1.10
        elif d <= mid:
            return 1.00
        else:
            return 0.95

    # slow / very slow
    if d <= early:
        return 0.70
    elif d <= mid:
        return 0.90
    else:
        return 1.30


def desired_buy_amount(pos: Position, day_close: float, profit_target: float, max_days: int, stop_level: float) -> float:
    """
    Determine desired buy today (pre-leverage-cap), based on:
      - extending mode => always unit_base (as per your definition)
      - normal mode => original DCA logic (<=avg => full, <=avg*1.05 => half)
      - tau schedule adjusts ONLY within day<=max_days, and keeps total budget ~entry_seed
    """
    if not pos.in_pos:
        return 0.0

    # extending: keep it fixed = unit_base (your current logic)
    if pos.extending:
        return float(pos.unit_base)

    avg = pos.avg_price()
    if not np.isfinite(avg) or avg <= 0 or not np.isfinite(day_close) or day_close <= 0:
        return 0.0

    # original DCA rule -> base desired
    if day_close <= avg:
        base = float(pos.unit_base)
    elif day_close <= avg * 1.05:
        base = float(pos.unit_base) / 2.0
    else:
        base = 0.0

    if base <= 0:
        return 0.0

    # tau schedule: apply only during day 1..max_days
    if pos.holding_days <= int(max_days):
        mult = tau_multiplier(pos.holding_days, max_days, pos.tau_class_pred)
        scaled = base * float(mult)

        # budget constraint: total spend within days<=max_days should not exceed entry_seed (tau_budget_total)
        # remaining budget:
        remaining = float(max(0.0, pos.tau_budget_total - pos.tau_budget_used))
        return float(min(scaled, remaining))

    return float(base)


# -----------------------------
# Main simulation
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Single-position engine with leverage cap + tau-based buy schedule.")
    ap.add_argument("--picks-path", required=True, type=str, help="CSV/Parquet with columns Date, Ticker (one pick per date).")
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--initial-seed", default=40_000_000, type=float)
    ap.add_argument("--profit-target", required=True, type=float)   # e.g. 0.10
    ap.add_argument("--max-days", required=True, type=int)          # e.g. 40
    ap.add_argument("--stop-level", required=True, type=float)      # e.g. -0.10
    ap.add_argument("--max-extend-days", required=True, type=int)   # kept for tag/label compatibility (NO hard limit)
    ap.add_argument("--max-leverage-pct", default=1.0, type=float)  # 1.0 => 100%

    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--suffix", default="", type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    args = ap.parse_args()

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

    # ensure 1 pick per date (top-1 selection already done upstream)
    picks = picks.drop_duplicates(["Date"], keep="last")

    # picks lookup by date -> row (to get tau columns too)
    picks_by_date = {}
    for _, r in picks.iterrows():
        picks_by_date[pd.Timestamp(r["Date"])] = r

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
            "EntrySeed": pos.entry_seed,
            "ProfitTarget": args.profit_target,
            "MaxDays": args.max_days,
            "StopLevel": args.stop_level,
            "MaxExtendDaysParam": args.max_extend_days,
            "MaxLeveragePctCap": args.max_leverage_pct,
            "MaxLeveragePct": pos.max_leverage_pct,
            "TauClassPred": pos.tau_class_pred if pos.tau_class_pred is not None else -1,
            "Invested": pos.invested,
            "Shares": pos.shares,
            "ExitPrice": exit_price,
            "Proceeds": proceeds,
            "CycleReturn": cycle_return,
            "HoldingDays": pos.holding_days,
            "Extending": int(pos.extending),
            "Reason": reason,
            "MaxDrawdown": pos.max_dd,
        })

        pos.seed += proceeds

        # reset position state (keep portfolio-level max_equity/max_dd rolling)
        pos.in_pos = False
        pos.ticker = None
        pos.shares = 0.0
        pos.invested = 0.0
        pos.entry_seed = 0.0
        pos.unit_base = 0.0
        pos.entry_date = None
        pos.holding_days = 0
        pos.extending = False
        pos.max_leverage_pct = 0.0

        pos.tau_class_pred = None
        pos.tau_budget_total = 0.0
        pos.tau_budget_used = 0.0

        cooldown_today = True  # no re-entry on the same day

    for date, day_df in grouped:
        day_df = day_df.set_index("Ticker", drop=False)
        cooldown_today = False

        # ---- holding position
        if pos.in_pos:
            if pos.ticker not in day_df.index:
                pos.update_drawdown(None)
            else:
                row = day_df.loc[pos.ticker]
                close_px = float(row["Close"])
                high_px = float(row["High"])

                pos.holding_days += 1
                avg = pos.avg_price()

                # exits
                if not pos.extending:
                    # TP intraday
                    if np.isfinite(avg) and np.isfinite(high_px) and high_px >= avg * (1.0 + args.profit_target):
                        exit_px = avg * (1.0 + args.profit_target)
                        close_cycle(date, float(exit_px), reason="TP")
                    else:
                        # max_days decision
                        if pos.in_pos and pos.holding_days >= int(args.max_days):
                            cur_ret = (close_px - avg) / avg if np.isfinite(avg) and avg != 0 else -np.inf
                            if cur_ret >= float(args.stop_level):
                                close_cycle(date, float(close_px), reason="MAXDAY_CLOSE")
                            else:
                                pos.extending = True

                # buys
                if pos.in_pos and pos.extending:
                    # exit if recovery to stop-level threshold
                    if np.isfinite(avg) and np.isfinite(high_px) and high_px >= avg * (1.0 + args.stop_level):
                        exit_px = avg * (1.0 + args.stop_level)
                        close_cycle(date, float(exit_px), reason="EXT_RECOVERY")
                    else:
                        desired = float(pos.unit_base)
                        invest = clamp_invest_by_leverage(pos.seed, pos.entry_seed, desired, args.max_leverage_pct)
                        if invest > 0 and np.isfinite(close_px) and close_px > 0:
                            pos.seed -= invest
                            pos.invested += invest
                            pos.shares += invest / close_px
                            update_max_leverage_pct(pos, pos.entry_seed, args.max_leverage_pct)
                else:
                    if pos.in_pos:
                        desired = desired_buy_amount(
                            pos=pos,
                            day_close=float(close_px),
                            profit_target=float(args.profit_target),
                            max_days=int(args.max_days),
                            stop_level=float(args.stop_level),
                        )
                        invest = clamp_invest_by_leverage(pos.seed, pos.entry_seed, desired, args.max_leverage_pct)
                        if invest > 0 and np.isfinite(close_px) and close_px > 0:
                            pos.seed -= invest
                            pos.invested += invest
                            pos.shares += invest / close_px
                            update_max_leverage_pct(pos, pos.entry_seed, args.max_leverage_pct)

                            # tau budget accounting only within day<=max_days and not extending
                            if (not pos.extending) and pos.holding_days <= int(args.max_days):
                                pos.tau_budget_used += float(invest)

                # update dd
                if pos.in_pos:
                    pos.update_drawdown(close_px)
                else:
                    pos.update_drawdown(None)

        # ---- entry
        if (not pos.in_pos) and (not cooldown_today):
            pick_row = picks_by_date.get(pd.Timestamp(date), None)
            if pick_row is not None:
                pick = str(pick_row["Ticker"]).upper().strip()
                if pick in day_df.index:
                    row = day_df.loc[pick]
                    close_px = float(row["Close"])
                    if np.isfinite(close_px) and close_px > 0:
                        # S0 = current seed at entry
                        S0 = float(pos.seed)
                        unit_base = (S0 / float(args.max_days)) if args.max_days > 0 else 0.0

                        # infer tau class from pick_row (if present)
                        tau_cls = infer_tau_class_from_row(pick_row)

                        # first day desired amount: base * tau_multiplier (budgeted)
                        mult = tau_multiplier(1, int(args.max_days), tau_cls)
                        desired = float(unit_base) * float(mult)

                        # budget total (for pre-max_days window) is entry_seed (S0).
                        # if S0 <= 0, budget is 0 and entry will fail anyway.
                        budget_total = max(0.0, float(S0))
                        desired = min(desired, budget_total)

                        invest = clamp_invest_by_leverage(pos.seed, S0, desired, args.max_leverage_pct)

                        if invest > 0:
                            pos.in_pos = True
                            pos.ticker = pick
                            pos.entry_seed = S0
                            pos.unit_base = unit_base
                            pos.entry_date = date
                            pos.holding_days = 1
                            pos.extending = False
                            pos.max_leverage_pct = 0.0

                            pos.tau_class_pred = tau_cls
                            pos.tau_budget_total = float(budget_total)
                            pos.tau_budget_used = float(invest)  # day1 used

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

        # ---- curve record
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
            "TauClassPred": pos.tau_class_pred if pos.in_pos and pos.tau_class_pred is not None else -1,
            "TauBudgetTotal": pos.tau_budget_total if pos.in_pos else 0.0,
            "TauBudgetUsed": pos.tau_budget_used if pos.in_pos else 0.0,
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