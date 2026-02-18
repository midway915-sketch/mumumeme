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


# -----------------------------
# Leverage cap: core (portfolio-level cap based on cycle entry seed S0)
# -----------------------------
def clamp_invest_by_leverage(seed: float, entry_seed: float, desired: float, max_leverage_pct: float) -> float:
    """
    Enforce: seed_after >= - entry_seed * max_leverage_pct
    """
    if desired <= 0:
        return 0.0
    if not np.isfinite(entry_seed) or entry_seed <= 0:
        # no borrowing allowed if entry_seed invalid/non-positive
        return float(min(desired, max(seed, 0.0)))

    borrow_limit = float(entry_seed) * float(max_leverage_pct)
    floor_seed = -borrow_limit
    room = float(seed) - float(floor_seed)  # seed + borrow_limit
    if room <= 0:
        return 0.0
    return float(min(desired, room))


# -----------------------------
# Position + portfolio state
# -----------------------------
@dataclass
class Leg:
    active: bool = False
    ticker: str = ""
    weight: float = 0.0

    shares: float = 0.0
    invested: float = 0.0

    entry_date: pd.Timestamp | None = None
    holding_days: int = 0
    extending: bool = False

    # 2-step profit / trailing
    tp1_done: bool = False
    peak_price: float = 0.0  # for trailing after tp1

    def avg_price(self) -> float:
        return (self.invested / self.shares) if (self.shares > 0 and self.invested > 0) else np.nan


@dataclass
class Portfolio:
    seed: float
    entry_seed: float = 0.0       # S0 at cycle entry
    unit_total: float = 0.0       # daily budget = entry_seed / max_days
    in_cycle: bool = False
    cycle_start_date: pd.Timestamp | None = None

    # tracking
    max_leverage_pct: float = 0.0
    max_equity: float = 0.0
    max_dd: float = 0.0

    def equity(self, legs: list[Leg], prices_today: dict[str, float]) -> float:
        v = 0.0
        for leg in legs:
            if leg.active and leg.ticker in prices_today:
                v += leg.shares * float(prices_today[leg.ticker])
        return float(self.seed) + float(v)

    def update_drawdown(self, legs: list[Leg], prices_today: dict[str, float]) -> None:
        eq = self.equity(legs, prices_today)
        if eq > self.max_equity:
            self.max_equity = eq
        if self.max_equity > 0:
            dd = (eq - self.max_equity) / self.max_equity
            if dd < self.max_dd:
                self.max_dd = dd

    def update_max_leverage(self, max_leverage_cap: float) -> None:
        if self.entry_seed <= 0:
            return
        lev = max(0.0, -float(self.seed)) / float(self.entry_seed)
        if lev > self.max_leverage_pct:
            self.max_leverage_pct = float(lev)
        if self.max_leverage_pct > max_leverage_cap + 1e-9:
            self.max_leverage_pct = float(max_leverage_cap)


# -----------------------------
# Main simulation
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-leg (Top-1/Top-2) engine with 2-step take-profit + trailing, leverage cap on ALL buys.")
    ap.add_argument("--picks-path", required=True, type=str, help="CSV/Parquet with columns Date,Ticker and optional Weight.")
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--initial-seed", default=40_000_000, type=float)
    ap.add_argument("--profit-target", required=True, type=float)   # PT
    ap.add_argument("--max-days", required=True, type=int)          # H
    ap.add_argument("--stop-level", required=True, type=float)      # SL (negative)
    ap.add_argument("--max-extend-days", required=True, type=int)   # kept for tag/label compatibility (NO hard limit)
    ap.add_argument("--max-leverage-pct", default=1.0, type=float)  # 1.0 => 100%

    # NEW: 2-step TP + trailing
    ap.add_argument("--tp1-frac", default=0.5, type=float, help="fraction to sell at PT (e.g. 0.5)")
    ap.add_argument("--trail-stop", default=0.10, type=float, help="trailing stop drawdown from peak AFTER tp1 (e.g. 0.10)")
    ap.add_argument("--enable-trailing", default="true", type=str, help="true/false")

    # optional (won't break if passed)
    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--suffix", default="", type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    args = ap.parse_args()

    enable_trailing = str(args.enable_trailing).strip().lower() in ("1", "true", "yes", "y", "on")

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
    if "Weight" not in picks.columns:
        picks["Weight"] = 1.0
    picks["Weight"] = pd.to_numeric(picks["Weight"], errors="coerce").fillna(0.0).astype(float)
    picks = picks.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Weight"], ascending=[True, False]).reset_index(drop=True)

    # normalize per-day weights (for Top-2 split)
    def _norm_w(g: pd.DataFrame) -> pd.DataFrame:
        w = g["Weight"].clip(lower=0.0)
        s = float(w.sum())
        if s <= 0:
            g["Weight"] = 0.0
        else:
            g["Weight"] = (w / s).astype(float)
        return g

    picks = picks.groupby("Date", group_keys=False).apply(_norm_w)
    # keep at most 2 legs per day (Top-1/Top-2 compare)
    picks = picks.groupby("Date", group_keys=False).head(2).reset_index(drop=True)

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

    # per date list of picks
    picks_by_date = {}
    for d, g in picks.groupby("Date", sort=True):
        rows = []
        for _, r in g.iterrows():
            rows.append((str(r["Ticker"]), float(r["Weight"])))
        picks_by_date[d] = rows

    grouped = prices.groupby("Date", sort=True)

    port = Portfolio(seed=float(args.initial_seed), max_equity=float(args.initial_seed), max_dd=0.0)
    legs = [Leg(), Leg()]  # up to 2 legs
    cooldown_today = False

    trades = []
    curve = []

    def any_active() -> bool:
        return any(l.active for l in legs)

    def reset_cycle_state_only() -> None:
        port.in_cycle = False
        port.entry_seed = 0.0
        port.unit_total = 0.0
        port.cycle_start_date = None
        port.max_leverage_pct = 0.0

    def close_leg(idx: int, exit_date: pd.Timestamp, exit_price: float, reason: str) -> None:
        nonlocal cooldown_today
        leg = legs[idx]
        if not leg.active:
            return
        proceeds = leg.shares * float(exit_price)
        cycle_return = (proceeds - leg.invested) / leg.invested if leg.invested > 0 else np.nan
        win = int(cycle_return > 0) if np.isfinite(cycle_return) else 0

        trades.append({
            "EntryDate": leg.entry_date,
            "ExitDate": exit_date,
            "Ticker": leg.ticker,
            "Weight": leg.weight,
            "EntrySeed": port.entry_seed,
            "ProfitTarget": args.profit_target,
            "MaxDays": args.max_days,
            "StopLevel": args.stop_level,
            "MaxExtendDaysParam": args.max_extend_days,
            "MaxLeveragePctCap": args.max_leverage_pct,
            "MaxLeveragePct": port.max_leverage_pct,
            "Invested": leg.invested,
            "Shares": leg.shares,
            "ExitPrice": float(exit_price),
            "Proceeds": proceeds,
            "CycleReturn": cycle_return,
            "Win": win,
            "HoldingDays": leg.holding_days,
            "Extending": int(leg.extending),
            "TP1_Done": int(leg.tp1_done),
            "Reason": reason,
            "MaxDrawdownPortfolio": port.max_dd,
        })

        port.seed += proceeds

        # reset leg
        legs[idx] = Leg()
        cooldown_today = True  # no new entries on sell day

    def partial_sell(idx: int, date: pd.Timestamp, sell_price: float, frac: float, reason: str) -> None:
        nonlocal cooldown_today
        leg = legs[idx]
        if not leg.active or leg.shares <= 0:
            return
        frac = float(max(0.0, min(1.0, frac)))
        if frac <= 0:
            return

        sell_shares = leg.shares * frac
        proceeds = sell_shares * float(sell_price)

        # proportional cost basis reduction
        leg.shares -= sell_shares
        leg.invested *= (1.0 - frac)
        port.seed += proceeds

        # record as a trade event row (optional)
        trades.append({
            "EntryDate": leg.entry_date,
            "ExitDate": date,
            "Ticker": leg.ticker,
            "Weight": leg.weight,
            "EntrySeed": port.entry_seed,
            "ProfitTarget": args.profit_target,
            "MaxDays": args.max_days,
            "StopLevel": args.stop_level,
            "MaxExtendDaysParam": args.max_extend_days,
            "MaxLeveragePctCap": args.max_leverage_pct,
            "MaxLeveragePct": port.max_leverage_pct,
            "Invested": proceeds * 0.0,  # keep simple
            "Shares": sell_shares,
            "ExitPrice": float(sell_price),
            "Proceeds": proceeds,
            "CycleReturn": np.nan,
            "Win": np.nan,
            "HoldingDays": leg.holding_days,
            "Extending": int(leg.extending),
            "TP1_Done": int(leg.tp1_done),
            "Reason": reason,
            "MaxDrawdownPortfolio": port.max_dd,
        })

        cooldown_today = True  # treat partial sell day as no re-entry too (simple + safe)

    # simulate
    for date, day_df in grouped:
        day_df = day_df.set_index("Ticker", drop=False)
        cooldown_today = False

        # build price lookup for active tickers
        prices_today = {}
        highs_today = {}
        lows_today = {}
        for t in day_df.index:
            try:
                prices_today[t] = float(day_df.loc[t]["Close"])
                highs_today[t] = float(day_df.loc[t]["High"])
                lows_today[t] = float(day_df.loc[t]["Low"])
            except Exception:
                pass

        # --------------------------
        # 1) Manage existing legs
        # --------------------------
        if any_active():
            for i in range(2):
                leg = legs[i]
                if not leg.active:
                    continue
                if leg.ticker not in day_df.index:
                    continue

                close_px = prices_today.get(leg.ticker, np.nan)
                high_px = highs_today.get(leg.ticker, np.nan)

                leg.holding_days += 1
                avg = leg.avg_price()

                # --- 2-step TP: if not tp1_done and hit PT -> partial sell at PT price
                if (not leg.tp1_done) and np.isfinite(avg) and np.isfinite(high_px) and high_px >= avg * (1.0 + args.profit_target):
                    tp_px = avg * (1.0 + args.profit_target)
                    # sell tp1-frac
                    frac = float(args.tp1_frac)
                    if frac >= 1.0:
                        # full exit
                        close_leg(i, date, float(tp_px), reason="TP_FULL")
                    else:
                        partial_sell(i, date, float(tp_px), frac=frac, reason="TP1_PARTIAL")
                        # if leg still active (shares remain)
                        if legs[i].active and legs[i].shares > 0:
                            legs[i].tp1_done = True
                            # set peak for trailing as current high (or tp price)
                            legs[i].peak_price = float(max(tp_px, high_px))
                    # after sell/partial sell, continue to next leg
                    continue

                # --- trailing AFTER tp1_done
                if enable_trailing and leg.active and leg.tp1_done and leg.shares > 0 and np.isfinite(high_px) and high_px > 0:
                    # update peak
                    if leg.peak_price <= 0:
                        leg.peak_price = float(high_px)
                    else:
                        leg.peak_price = float(max(leg.peak_price, high_px))

                    trail_pct = float(max(0.0, args.trail_stop))
                    if trail_pct > 0 and np.isfinite(close_px) and close_px > 0:
                        trail_floor = leg.peak_price * (1.0 - trail_pct)
                        # use Close as trigger (simple), could use Low for stricter
                        if close_px <= trail_floor:
                            close_leg(i, date, float(close_px), reason="TRAIL_STOP")
                            continue

                # --- extend decision at max_days (only if not already extending and not tp1_done)
                if leg.active and (not leg.extending) and leg.holding_days >= int(args.max_days):
                    if np.isfinite(avg) and np.isfinite(close_px) and avg > 0:
                        cur_ret = (close_px - avg) / avg
                    else:
                        cur_ret = -np.inf

                    if cur_ret >= float(args.stop_level):
                        close_leg(i, date, float(close_px), reason="MAXDAY_CLOSE")
                        continue
                    else:
                        legs[i].extending = True

                # --- DCA rules (normal vs extending), leverage cap applies at PORTFOLIO level
                if leg.active and np.isfinite(close_px) and close_px > 0:
                    # budget for this leg = unit_total * leg.weight
                    desired_base = float(port.unit_total) * float(leg.weight)

                    if leg.extending:
                        desired = desired_base  # daily in extending
                    else:
                        if np.isfinite(avg) and avg > 0:
                            if close_px <= avg:
                                desired = desired_base
                            elif close_px <= avg * 1.05:
                                desired = desired_base / 2.0
                            else:
                                desired = 0.0
                        else:
                            desired = 0.0

                    invest = clamp_invest_by_leverage(port.seed, port.entry_seed, desired, args.max_leverage_pct)
                    if invest > 0:
                        port.seed -= invest
                        leg.invested += invest
                        leg.shares += invest / float(close_px)
                        port.update_max_leverage(args.max_leverage_pct)

        # if all legs closed, end cycle
        if port.in_cycle and (not any_active()):
            reset_cycle_state_only()

        # update portfolio DD
        port.update_drawdown(legs, prices_today)

        # --------------------------
        # 2) Entry (only if not in cycle, and not sell day)
        # --------------------------
        if (not port.in_cycle) and (not cooldown_today):
            daily_picks = picks_by_date.get(date, [])
            # keep only tickers that exist in price today
            daily_picks = [(t, w) for (t, w) in daily_picks if t in day_df.index and w > 0]
            if daily_picks:
                # cycle entry
                port.in_cycle = True
                port.cycle_start_date = date
                port.entry_seed = float(port.seed)  # S0 = seed at entry
                port.unit_total = (port.entry_seed / float(args.max_days)) if args.max_days > 0 else 0.0
                port.max_leverage_pct = 0.0

                # open up to 2 legs
                for i in range(2):
                    if i >= len(daily_picks):
                        break
                    tkr, w = daily_picks[i]
                    close_px = prices_today.get(tkr, np.nan)
                    if not (np.isfinite(close_px) and close_px > 0):
                        continue

                    # first-day buy desired for this leg = unit_total * w
                    desired = float(port.unit_total) * float(w)
                    invest = clamp_invest_by_leverage(port.seed, port.entry_seed, desired, args.max_leverage_pct)
                    if invest <= 0:
                        continue

                    leg = Leg(
                        active=True,
                        ticker=str(tkr),
                        weight=float(w),
                        shares=float(invest / close_px),
                        invested=float(invest),
                        entry_date=date,
                        holding_days=1,
                        extending=False,
                        tp1_done=False,
                        peak_price=float(close_px),
                    )
                    legs[i] = leg
                    port.seed -= invest
                    port.update_max_leverage(args.max_leverage_pct)

                # if nothing opened (e.g. leverage cap / bad price), cancel cycle
                if not any_active():
                    reset_cycle_state_only()

        # --------------------------
        # 3) record curve
        # --------------------------
        eq = port.equity(legs, prices_today)
        curve.append({
            "Date": date,
            "Equity": eq,
            "Seed": float(port.seed),
            "InCycle": int(port.in_cycle),
            "Tickers": ",".join([l.ticker for l in legs if l.active]),
            "Leg1_Ticker": legs[0].ticker if legs[0].active else "",
            "Leg2_Ticker": legs[1].ticker if legs[1].active else "",
            "Leg1_Shares": float(legs[0].shares) if legs[0].active else 0.0,
            "Leg2_Shares": float(legs[1].shares) if legs[1].active else 0.0,
            "Leg1_Invested": float(legs[0].invested) if legs[0].active else 0.0,
            "Leg2_Invested": float(legs[1].invested) if legs[1].active else 0.0,
            "MaxLeveragePctCycle": float(port.max_leverage_pct) if port.in_cycle else 0.0,
            "MaxDrawdownPortfolio": float(port.max_dd),
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
        print(f"[INFO] final SeedMultiple={final_mult:.4f} maxDD={float(port.max_dd):.4f}")


if __name__ == "__main__":
    main()