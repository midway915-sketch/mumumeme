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


def _force_columns_from_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    If Date/Ticker are in index (including MultiIndex), reset_index().
    """
    out = df
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    else:
        if getattr(out.index, "name", None) in ("Date", "Ticker"):
            out = out.reset_index()
    return out


def _canonicalize_date_ticker_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee columns 'Date' and 'Ticker' exist by:
    - resetting index if needed
    - renaming common variants (date/Datetime/index/level_0 etc.)
    - renaming ticker variants (ticker/symbol/level_1 etc.)
    """
    out = _force_columns_from_index(df).copy()

    # ---- Date
    if "Date" not in out.columns:
        for c in ["date", "Datetime", "datetime", "DATE", "index", "level_0"]:
            if c in out.columns:
                out = out.rename(columns={c: "Date"})
                break

    # ---- Ticker
    if "Ticker" not in out.columns:
        for c in ["ticker", "TICKER", "Symbol", "symbol", "SYMBOL", "level_1"]:
            if c in out.columns:
                out = out.rename(columns={c: "Ticker"})
                break

    if "Date" not in out.columns or "Ticker" not in out.columns:
        raise ValueError(
            "[sim_engine] Cannot find required columns Date/Ticker.\n"
            f"Columns={list(out.columns)[:80]}\n"
            f"IndexType={type(out.index)} IndexName={getattr(out.index,'name',None)}"
        )
    return out


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


def _coerce_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(float)


# -----------------------------
# Strategy state
# -----------------------------
@dataclass
class Leg:
    ticker: str
    weight: float

    shares: float = 0.0
    invested: float = 0.0  # cost basis (remaining)

    tp1_done: bool = False
    peak_after_tp1: float = 0.0  # for trailing stop on remaining shares

    def avg_price(self) -> float:
        return (self.invested / self.shares) if (self.shares > 0 and self.invested > 0) else np.nan

    def value(self, px_close: float | None) -> float:
        if self.shares <= 0 or px_close is None or not np.isfinite(px_close):
            return 0.0
        return float(self.shares) * float(px_close)


@dataclass
class Position:
    in_pos: bool = False

    # cash balance (can be negative)
    seed: float = 0.0

    # cycle params
    entry_seed: float = 0.0          # S0 = seed at entry time
    unit: float = 0.0               # daily buy amount = entry_seed / max_days
    entry_date: pd.Timestamp | None = None
    holding_days: int = 0
    extending: bool = False

    legs: list[Leg] = None

    # tracking
    max_leverage_pct: float = 0.0    # max(-seed)/entry_seed observed in this cycle
    max_equity: float = 0.0
    max_dd: float = 0.0

    def total_invested(self) -> float:
        if not self.legs:
            return 0.0
        return float(sum(l.invested for l in self.legs))

    def total_value(self, prices_today: dict[str, float]) -> float:
        if not self.legs:
            return 0.0
        v = 0.0
        for l in self.legs:
            v += l.value(prices_today.get(l.ticker))
        return float(v)

    def equity(self, prices_today: dict[str, float]) -> float:
        return float(self.seed) + self.total_value(prices_today)

    def update_drawdown(self, prices_today: dict[str, float]) -> None:
        eq = self.equity(prices_today)
        if eq > self.max_equity:
            self.max_equity = eq
        if self.max_equity > 0:
            dd = (eq - self.max_equity) / self.max_equity
            if dd < self.max_dd:
                self.max_dd = float(dd)


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

    # If entry_seed <= 0, safest: disallow borrowing beyond 0 (no negative).
    if not np.isfinite(entry_seed) or entry_seed <= 0:
        return float(min(desired, max(seed, 0.0)))

    borrow_limit = float(entry_seed) * float(max_leverage_pct)  # 1.0 => 100%
    floor_seed = -borrow_limit

    # seed_after = seed - invest >= floor_seed  => invest <= seed - floor_seed = seed + borrow_limit
    room = float(seed) - float(floor_seed)
    if room <= 0:
        return 0.0
    return float(min(desired, room))


def update_max_leverage_pct(pos: Position, max_leverage_pct_cap: float) -> None:
    if not np.isfinite(pos.entry_seed) or pos.entry_seed <= 0:
        return
    lev = max(0.0, -pos.seed) / float(pos.entry_seed)
    if lev > pos.max_leverage_pct:
        pos.max_leverage_pct = float(lev)
    if pos.max_leverage_pct > float(max_leverage_pct_cap) + 1e-9:
        pos.max_leverage_pct = float(max_leverage_pct_cap)


# -----------------------------
# Execution helpers
# -----------------------------
def buy_leg(pos: Position, leg: Leg, invest: float, px: float) -> None:
    if invest <= 0 or not np.isfinite(px) or px <= 0:
        return
    pos.seed -= float(invest)
    leg.invested += float(invest)
    leg.shares += float(invest) / float(px)


def sell_leg_fraction(pos: Position, leg: Leg, frac: float, px: float) -> float:
    """
    Sell fraction of current shares. Reduce invested proportionally (keeps avg consistent).
    Return proceeds.
    """
    if frac <= 0 or frac > 1 or leg.shares <= 0 or not np.isfinite(px) or px <= 0:
        return 0.0
    sell_shares = leg.shares * float(frac)
    proceeds = sell_shares * float(px)

    # reduce cost basis proportionally
    leg.shares -= sell_shares
    leg.invested *= (1.0 - float(frac))

    pos.seed += float(proceeds)
    return float(proceeds)


# -----------------------------
# Main simulation
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Single-position engine (Top-1/Top-K picks) with leverage cap and optional 2-step TP+trailing.")
    ap.add_argument("--picks-path", required=True, type=str, help="CSV/Parquet with columns Date,Ticker (and optional Weight,RankInDay).")
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--initial-seed", default=40_000_000, type=float)
    ap.add_argument("--profit-target", required=True, type=float)   # e.g. 0.10
    ap.add_argument("--max-days", required=True, type=int)          # e.g. 40
    ap.add_argument("--stop-level", required=True, type=float)      # e.g. -0.10
    ap.add_argument("--max-extend-days", required=True, type=int)   # kept for tag/label compatibility (NO hard limit)
    ap.add_argument("--max-leverage-pct", default=1.0, type=float)  # 1.0 => 100%

    # 2-step TP + trailing
    ap.add_argument("--enable-trailing", default="false", type=str)
    ap.add_argument("--tp1-frac", default=0.50, type=float)         # fraction to sell at PT hit
    ap.add_argument("--trail-stop", default=0.10, type=float)       # trailing drawdown from peak (e.g. 0.10 => -10%)

    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--suffix", default="", type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    args = ap.parse_args()

    enable_trailing = str(args.enable_trailing).strip().lower() in ("1", "true", "yes", "y", "on")
    tp1_frac = float(args.tp1_frac)
    trail_stop = float(args.trail_stop)

    picks_path = args.picks_path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag, suffix = _pick_default_out_tag(picks_path, args.tag or None, args.suffix or None)

    # ---- load picks
    if picks_path.lower().endswith(".parquet"):
        picks = pd.read_parquet(picks_path)
    else:
        picks = pd.read_csv(picks_path)

    picks = _canonicalize_date_ticker_columns(picks)

    picks = picks.copy()
    picks["Date"] = _norm_date(picks["Date"])
    picks["Ticker"] = picks["Ticker"].astype(str).str.upper().str.strip()
    picks = picks.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # optional Weight/RankInDay
    if "Weight" in picks.columns:
        picks["Weight"] = _coerce_num(picks["Weight"], 0.0)
    else:
        picks["Weight"] = 1.0

    if "RankInDay" in picks.columns:
        picks["RankInDay"] = _coerce_num(picks["RankInDay"], 1.0).astype(int)
    else:
        picks["RankInDay"] = picks.groupby("Date").cumcount() + 1

    # normalize weights per day (safe even if top1)
    picks["Weight"] = picks["Weight"].clip(lower=0.0)
    wsum = picks.groupby("Date")["Weight"].transform(lambda s: float(s.sum()) if len(s) else 0.0)
    picks["Weight"] = np.where(wsum > 0, picks["Weight"] / wsum, 0.0)

    # ---- load prices
    prices = read_table(args.prices_parq, args.prices_csv)
    prices = _canonicalize_date_ticker_columns(prices)

    prices = prices.copy()
    prices["Date"] = _norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()

    for c in ["Open", "High", "Low", "Close"]:
        if c not in prices.columns:
            raise ValueError(f"prices missing {c}")

    prices = prices.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # per day group
    prices_by_date = {d: g.set_index("Ticker", drop=False) for d, g in prices.groupby("Date", sort=True)}
    pick_dates = picks["Date"].dropna().unique().tolist()
    picks_by_date = {d: picks[picks["Date"] == d].copy() for d in pick_dates}

    pos = Position(seed=float(args.initial_seed), max_equity=float(args.initial_seed), max_dd=0.0, legs=[])
    cooldown_today = False

    trades = []
    curve = []

    def close_cycle(exit_date: pd.Timestamp, px_map: dict[str, float], reason: str) -> None:
        nonlocal cooldown_today, pos, trades
        proceeds_total = 0.0
        invested_total = 0.0

        # close each leg at Close price
        for leg in pos.legs:
            px = px_map.get(leg.ticker)
            if px is None or not np.isfinite(px) or px <= 0:
                continue
            proceeds = leg.shares * float(px)
            proceeds_total += proceeds
            invested_total += float(leg.invested)
            pos.seed += float(proceeds)

        cycle_return = (proceeds_total - invested_total) / invested_total if invested_total > 0 else np.nan
        win = int(cycle_return > 0) if np.isfinite(cycle_return) else 0

        trades.append({
            "EntryDate": pos.entry_date,
            "ExitDate": exit_date,
            "Tickers": ",".join([l.ticker for l in pos.legs]),
            "Weights": ",".join([f"{l.weight:.4f}" for l in pos.legs]),
            "EntrySeed": pos.entry_seed,
            "ProfitTarget": args.profit_target,
            "MaxDays": args.max_days,
            "StopLevel": args.stop_level,
            "MaxExtendDaysParam": args.max_extend_days,
            "MaxLeveragePctCap": args.max_leverage_pct,
            "MaxLeveragePct": pos.max_leverage_pct,
            "EnableTrailing": int(enable_trailing),
            "TP1Frac": tp1_frac,
            "TrailStop": trail_stop,
            "Invested": invested_total,
            "Proceeds": proceeds_total,
            "CycleReturn": cycle_return,
            "HoldingDays": pos.holding_days,
            "Extending": int(pos.extending),
            "Reason": reason,
            "Win": win,
            "MaxDrawdown": pos.max_dd,
        })

        # reset
        pos.in_pos = False
        pos.entry_seed = 0.0
        pos.unit = 0.0
        pos.entry_date = None
        pos.holding_days = 0
        pos.extending = False
        pos.legs = []
        pos.max_leverage_pct = 0.0

        cooldown_today = True  # no re-entry on same day

    # ---- simulate across all price dates
    all_dates = sorted(prices_by_date.keys())

    for date in all_dates:
        day_df = prices_by_date[date]
        cooldown_today = False
        px_close = {}
        px_high = {}
        for t, row in day_df.iterrows():
            try:
                px_close[str(t)] = float(row["Close"])
                px_high[str(t)] = float(row["High"])
            except Exception:
                pass

        # -------------------------
        # If in position: advance & manage
        # -------------------------
        if pos.in_pos:
            pos.holding_days += 1

            # compute avg across legs (weighted by invested)
            inv_total = pos.total_invested()
            val_total = pos.total_value(px_close)
            avg_port = np.nan
            if inv_total > 0 and val_total >= 0:
                # portfolio avg proxy: invested / shares doesn't exist across multi legs, use weighted avg price:
                # sum(invested) / sum(shares*?) not possible w/out per-leg avg, so use value-based exit per-leg.
                pass

            # ---- 2-step TP + trailing (per-leg)
            if enable_trailing and not pos.extending:
                for leg in pos.legs:
                    if leg.ticker not in day_df.index:
                        continue
                    high_px = px_high.get(leg.ticker)
                    close_px = px_close.get(leg.ticker)
                    if high_px is None or close_px is None:
                        continue

                    avg = leg.avg_price()
                    if not leg.tp1_done:
                        # TP1 trigger
                        if np.isfinite(avg) and high_px >= avg * (1.0 + float(args.profit_target)):
                            tp_px = avg * (1.0 + float(args.profit_target))
                            sell_leg_fraction(pos, leg, tp1_frac, float(tp_px))
                            leg.tp1_done = True
                            leg.peak_after_tp1 = float(max(leg.peak_after_tp1, close_px, tp_px))
                    else:
                        # update peak and apply trailing stop on remaining
                        leg.peak_after_tp1 = float(max(leg.peak_after_tp1, close_px))
                        trail_px = leg.peak_after_tp1 * (1.0 - float(trail_stop))
                        if close_px <= trail_px and leg.shares > 0:
                            # sell ALL remaining at close (simple)
                            sell_leg_fraction(pos, leg, 1.0, float(close_px))

                # if all legs sold out via trailing, close cycle
                if all(l.shares <= 0 for l in pos.legs):
                    close_cycle(date, px_close, reason="TRAIL_EXIT")

            # ---- max_days / extending decision (cycle-level)
            if pos.in_pos and not pos.extending:
                # at max_days, if portfolio is not recovered to stop threshold, extend
                # define "recovered" as each remaining leg close >= avg*(1+stop_level) (conservative)
                if pos.holding_days >= int(args.max_days):
                    all_ok = True
                    for leg in pos.legs:
                        if leg.shares <= 0:
                            continue
                        close_px = px_close.get(leg.ticker)
                        if close_px is None:
                            all_ok = False
                            break
                        avg = leg.avg_price()
                        if not (np.isfinite(avg) and close_px >= avg * (1.0 + float(args.stop_level))):
                            all_ok = False
                            break
                    if all_ok:
                        close_cycle(date, px_close, reason="MAXDAY_CLOSE")
                    else:
                        pos.extending = True

            # ---- extending logic: exit on recovery to stop-level, else DCA
            if pos.in_pos and pos.extending:
                # if all remaining legs have intraday high >= avg*(1+stop_level) => exit at that level (approx)
                all_recovered = True
                for leg in pos.legs:
                    if leg.shares <= 0:
                        continue
                    high_px = px_high.get(leg.ticker)
                    if high_px is None:
                        all_recovered = False
                        break
                    avg = leg.avg_price()
                    if not (np.isfinite(avg) and high_px >= avg * (1.0 + float(args.stop_level))):
                        all_recovered = False
                        break
                if all_recovered:
                    close_cycle(date, px_close, reason="EXT_RECOVERY")
                else:
                    # DCA desired total = unit (fixed), allocate by original weights across active legs
                    desired_total = float(pos.unit)
                    invest_total = clamp_invest_by_leverage(pos.seed, pos.entry_seed, desired_total, args.max_leverage_pct)
                    if invest_total > 0:
                        # allocate by weights among legs that still exist
                        wsum = sum(max(0.0, l.weight) for l in pos.legs) or 1.0
                        for leg in pos.legs:
                            if leg.ticker not in day_df.index:
                                continue
                            close_px = px_close.get(leg.ticker)
                            if close_px is None or close_px <= 0:
                                continue
                            alloc = invest_total * (max(0.0, leg.weight) / wsum)
                            buy_leg(pos, leg, alloc, float(close_px))
                        update_max_leverage_pct(pos, args.max_leverage_pct)

            # ---- normal zone DCA: if close <= avg -> unit, elif <= avg*1.05 -> unit/2
            if pos.in_pos and not pos.extending:
                # allocate total buy based on portfolio condition:
                # If ANY remaining leg close <= its avg => unit, else if ANY close <= avg*1.05 => unit/2, else 0
                desire = 0.0
                any_below_avg = False
                any_near = False
                for leg in pos.legs:
                    if leg.shares <= 0:
                        continue
                    close_px = px_close.get(leg.ticker)
                    if close_px is None or close_px <= 0:
                        continue
                    avg = leg.avg_price()
                    if np.isfinite(avg):
                        if close_px <= avg:
                            any_below_avg = True
                        elif close_px <= avg * 1.05:
                            any_near = True
                if any_below_avg:
                    desire = float(pos.unit)
                elif any_near:
                    desire = float(pos.unit) / 2.0

                invest_total = clamp_invest_by_leverage(pos.seed, pos.entry_seed, desire, args.max_leverage_pct)
                if invest_total > 0:
                    wsum = sum(max(0.0, l.weight) for l in pos.legs) or 1.0
                    for leg in pos.legs:
                        if leg.ticker not in day_df.index:
                            continue
                        close_px = px_close.get(leg.ticker)
                        if close_px is None or close_px <= 0:
                            continue
                        alloc = invest_total * (max(0.0, leg.weight) / wsum)
                        buy_leg(pos, leg, alloc, float(close_px))
                    update_max_leverage_pct(pos, args.max_leverage_pct)

            # ---- update drawdown
            if pos.in_pos:
                pos.update_drawdown(px_close)
            else:
                pos.update_drawdown({})

        # -------------------------
        # If NOT in position: entry (but not on sell-day)
        # -------------------------
        if (not pos.in_pos) and (not cooldown_today):
            day_picks = picks_by_date.get(date, None)
            if day_picks is not None and len(day_picks) > 0:
                # entry legs are whatever picks provides that day (Top-1 or Top-2)
                legs = []
                for _, r in day_picks.iterrows():
                    t = str(r["Ticker"]).upper().strip()
                    w = float(r.get("Weight", 0.0))
                    if t in day_df.index and w > 0:
                        legs.append(Leg(ticker=t, weight=w))

                if legs:
                    S0 = float(pos.seed)
                    unit = (S0 / float(args.max_days)) if args.max_days > 0 else 0.0

                    desired_total = float(unit)
                    invest_total = clamp_invest_by_leverage(pos.seed, S0, desired_total, args.max_leverage_pct)

                    if invest_total > 0:
                        pos.in_pos = True
                        pos.entry_seed = S0
                        pos.unit = unit
                        pos.entry_date = date
                        pos.holding_days = 1
                        pos.extending = False
                        pos.legs = legs
                        pos.max_leverage_pct = 0.0

                        wsum = sum(l.weight for l in legs) or 1.0
                        for leg in legs:
                            close_px = px_close.get(leg.ticker)
                            if close_px is None or close_px <= 0:
                                continue
                            alloc = invest_total * (leg.weight / wsum)
                            buy_leg(pos, leg, alloc, float(close_px))

                        update_max_leverage_pct(pos, args.max_leverage_pct)
                        pos.update_drawdown(px_close)
                    else:
                        pos.update_drawdown({})
                else:
                    pos.update_drawdown({})
            else:
                pos.update_drawdown({})

        # -------------------------
        # Record daily curve
        # -------------------------
        eq = pos.equity(px_close)
        curve.append({
            "Date": date,
            "Equity": eq,
            "Seed": pos.seed,
            "InPosition": int(pos.in_pos),
            "Tickers": ",".join([l.ticker for l in pos.legs]) if pos.in_pos else "",
            "Weights": ",".join([f"{l.weight:.4f}" for l in pos.legs]) if pos.in_pos else "",
            "HoldingDays": pos.holding_days if pos.in_pos else 0,
            "Extending": int(pos.extending) if pos.in_pos else 0,
            "MaxLeveragePctCycle": pos.max_leverage_pct if pos.in_pos else 0.0,
            "MaxDrawdownPortfolio": pos.max_dd,
            "PositionValue": pos.total_value(px_close) if pos.in_pos else 0.0,
            "Invested": pos.total_invested() if pos.in_pos else 0.0,
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