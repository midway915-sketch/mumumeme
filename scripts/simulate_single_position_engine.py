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


def clamp_invest_by_leverage(seed: float, entry_seed: float, desired: float, max_leverage_pct: float) -> float:
    """
    Enforce: seed_after >= - entry_seed * max_leverage_pct
    """
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


@dataclass
class Leg:
    ticker: str
    shares: float = 0.0
    invested: float = 0.0
    holding_days: int = 0
    extending: bool = False

    # 2-stage TP state
    tp1_done: bool = False
    peak_after_tp1: float = 0.0  # track peak High after TP1

    def avg_price(self) -> float:
        return (self.invested / self.shares) if (self.shares > 0 and self.invested > 0) else np.nan


@dataclass
class Portfolio:
    seed: float
    entry_seed: float = 0.0
    unit: float = 0.0
    in_cycle: bool = False
    entry_date: pd.Timestamp | None = None

    # per-cycle tracking
    max_leverage_pct: float = 0.0

    # portfolio dd
    max_equity: float = 0.0
    max_dd: float = 0.0

    legs: dict[str, Leg] = None  # ticker -> Leg

    def __post_init__(self):
        if self.legs is None:
            self.legs = {}

    def value(self, prices_by_ticker: dict[str, float]) -> float:
        v = 0.0
        for t, leg in self.legs.items():
            px = prices_by_ticker.get(t)
            if px is None or not np.isfinite(px):
                continue
            v += leg.shares * float(px)
        return float(v)

    def equity(self, prices_by_ticker: dict[str, float]) -> float:
        return float(self.seed) + self.value(prices_by_ticker)

    def update_dd(self, prices_by_ticker: dict[str, float]) -> None:
        eq = self.equity(prices_by_ticker)
        if eq > self.max_equity:
            self.max_equity = eq
        if self.max_equity > 0:
            dd = (eq - self.max_equity) / self.max_equity
            if dd < self.max_dd:
                self.max_dd = dd


def parse_weights(s: str, topk: int) -> list[float]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    if not parts:
        # default: top1=1.0
        return [1.0] + [0.0] * (topk - 1)
    w = [float(x) for x in parts]
    if len(w) < topk:
        w = w + [0.0] * (topk - len(w))
    w = w[:topk]
    sm = sum(max(0.0, x) for x in w)
    if sm <= 0:
        return [1.0] + [0.0] * (topk - 1)
    w = [max(0.0, x) / sm for x in w]
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description="Max-2 legs engine (Top-1 / Top-2 split) with leverage cap and optional 2-stage TP.")
    ap.add_argument("--picks-path", required=True, type=str, help="CSV/Parquet with Date,Ticker,Rank (Rank optional for topk=1).")
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--initial-seed", default=40_000_000, type=float)
    ap.add_argument("--profit-target", required=True, type=float)   # PT (e.g. 0.10)
    ap.add_argument("--max-days", required=True, type=int)          # H (e.g. 40)
    ap.add_argument("--stop-level", required=True, type=float)      # SL (e.g. -0.10)
    ap.add_argument("--max-extend-days", required=True, type=int)   # for tags/labels only (NO hard limit)
    ap.add_argument("--max-leverage-pct", default=1.0, type=float)  # 1.0 => 100%

    # NEW: topk/weights
    ap.add_argument("--topk", default=1, type=int, help="1=Top-1 baseline, 2=Top-2 split (max 2 supported)")
    ap.add_argument("--topk-weights", default="0.7,0.3", type=str, help="weights for Rank1,Rank2 (only used if topk=2)")

    # NEW: 2-stage TP (optional)
    ap.add_argument("--tp1-frac", default=0.0, type=float, help="at +PT, sell this fraction (e.g. 0.5). 0 disables partial TP.")
    ap.add_argument("--trail-stop", default=0.0, type=float, help="after TP1, trailing stop from peak (e.g. 0.10). 0 disables trail.")

    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--suffix", default="", type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    args = ap.parse_args()

    if int(args.topk) not in (1, 2):
        raise ValueError("--topk must be 1 or 2")
    topk = int(args.topk)
    weights = parse_weights(args.topk_weights, topk=topk)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag, suffix = _pick_default_out_tag(args.picks_path, args.tag or None, args.suffix or None)

    # load picks
    if args.picks_path.lower().endswith(".parquet"):
        picks = pd.read_parquet(args.picks_path)
    else:
        picks = pd.read_csv(args.picks_path)

    if "Date" not in picks.columns or "Ticker" not in picks.columns:
        raise ValueError(f"picks must have Date,Ticker. cols={list(picks.columns)[:50]}")

    picks = picks.copy()
    picks["Date"] = _norm_date(picks["Date"])
    picks["Ticker"] = picks["Ticker"].astype(str).str.upper().str.strip()
    if "Rank" not in picks.columns:
        picks["Rank"] = 1
    picks["Rank"] = pd.to_numeric(picks["Rank"], errors="coerce").fillna(1).astype(int)
    picks = picks.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Rank"]).reset_index(drop=True)

    # keep up to topk per day
    picks = picks[picks["Rank"].between(1, topk)]
    # if duplicated (Date,Rank) keep last
    picks = picks.drop_duplicates(["Date", "Rank"], keep="last")

    # map date -> list of (rank, ticker)
    picks_by_date: dict[pd.Timestamp, list[tuple[int, str]]] = {}
    for d, g in picks.groupby("Date"):
        rows = [(int(r), str(t)) for r, t in zip(g["Rank"].tolist(), g["Ticker"].tolist())]
        rows = sorted(rows, key=lambda x: x[0])
        picks_by_date[d] = rows

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

    port = Portfolio(seed=float(args.initial_seed), max_equity=float(args.initial_seed), max_dd=0.0)

    trades = []
    curve = []
    cooldown_today = False  # no re-entry on sell-day

    def update_max_leverage(entry_seed: float) -> None:
        if entry_seed <= 0:
            return
        lev = max(0.0, -port.seed) / entry_seed
        if lev > port.max_leverage_pct:
            port.max_leverage_pct = float(lev)
        if port.max_leverage_pct > float(args.max_leverage_pct) + 1e-9:
            port.max_leverage_pct = float(args.max_leverage_pct)

    def close_leg(date: pd.Timestamp, leg: Leg, exit_px: float, reason: str, max_dd_port: float) -> None:
        nonlocal cooldown_today
        proceeds = leg.shares * exit_px
        cycle_return = (proceeds - leg.invested) / leg.invested if leg.invested > 0 else np.nan
        win = int(cycle_return > 0) if np.isfinite(cycle_return) else 0

        trades.append({
            "EntryDate": port.entry_date,
            "ExitDate": date,
            "Ticker": leg.ticker,
            "EntrySeed": port.entry_seed,
            "ProfitTarget": args.profit_target,
            "MaxDays": args.max_days,
            "StopLevel": args.stop_level,
            "MaxExtendDaysParam": args.max_extend_days,
            "MaxLeveragePctCap": args.max_leverage_pct,
            "MaxLeveragePct": port.max_leverage_pct,
            "Invested": leg.invested,
            "Shares": leg.shares,
            "ExitPrice": exit_px,
            "Proceeds": proceeds,
            "CycleReturn": cycle_return,
            "HoldingDays": leg.holding_days,
            "Extending": int(leg.extending),
            "Reason": reason,
            "Win": win,
            "MaxDrawdownPortfolio": max_dd_port,
        })

        port.seed += proceeds
        cooldown_today = True

        # remove leg
        if leg.ticker in port.legs:
            del port.legs[leg.ticker]

    def reset_cycle_if_done():
        if len(port.legs) == 0 and port.in_cycle:
            port.in_cycle = False
            port.entry_seed = 0.0
            port.unit = 0.0
            port.entry_date = None
            port.max_leverage_pct = 0.0

    # simulate
    for date, day_df in grouped:
        day_df = day_df.set_index("Ticker", drop=False)
        cooldown_today = False

        # Build price dict for current holdings (for equity curve)
        px_close_map: dict[str, float] = {}
        for t in list(port.legs.keys()):
            if t in day_df.index:
                px_close_map[t] = float(day_df.loc[t]["Close"])

        # 1) Manage existing legs
        for t in list(port.legs.keys()):
            leg = port.legs[t]
            if t not in day_df.index:
                continue

            row = day_df.loc[t]
            close_px = float(row["Close"])
            high_px = float(row["High"])
            low_px = float(row["Low"])

            leg.holding_days += 1
            avg = leg.avg_price()

            # ---------- NORMAL ZONE (before extending)
            if not leg.extending:
                # (A) partial TP + trailing
                if args.tp1_frac > 0 and not leg.tp1_done:
                    if np.isfinite(avg) and high_px >= avg * (1.0 + args.profit_target):
                        # sell tp1_frac at PT price
                        sell_px = avg * (1.0 + args.profit_target)
                        frac = float(min(max(args.tp1_frac, 0.0), 1.0))
                        sell_shares = leg.shares * frac
                        if sell_shares > 0:
                            proceeds = sell_shares * sell_px
                            # reduce leg
                            leg.shares -= sell_shares
                            leg.invested *= (1.0 - frac)  # approximate proportional cost basis
                            port.seed += proceeds
                            cooldown_today = True
                            update_max_leverage(port.entry_seed)

                            leg.tp1_done = True
                            leg.peak_after_tp1 = max(leg.peak_after_tp1, high_px)

                # (B) trailing stop after TP1
                if leg.tp1_done and args.trail_stop > 0:
                    leg.peak_after_tp1 = max(leg.peak_after_tp1, high_px)
                    trail_px = leg.peak_after_tp1 * (1.0 - float(args.trail_stop))
                    # intraday breach => exit at trail_px
                    if np.isfinite(trail_px) and low_px <= trail_px and leg.shares > 0:
                        close_leg(date, leg, float(trail_px), reason="TRAIL_STOP", max_dd_port=port.max_dd)
                        continue

                # (C) full TP if partial TP disabled
                if (args.tp1_frac <= 0) and np.isfinite(avg) and high_px >= avg * (1.0 + args.profit_target):
                    exit_px = avg * (1.0 + args.profit_target)
                    close_leg(date, leg, float(exit_px), reason="TP_FULL", max_dd_port=port.max_dd)
                    continue

                # (D) max_days branch
                if leg.ticker in port.legs and leg.holding_days >= int(args.max_days):
                    cur_ret = (close_px - avg) / avg if np.isfinite(avg) and avg != 0 else -np.inf
                    if cur_ret >= float(args.stop_level):
                        close_leg(date, leg, float(close_px), reason="MAXDAY_CLOSE", max_dd_port=port.max_dd)
                        continue
                    else:
                        leg.extending = True

            # ---------- EXTENDING ZONE
            if leg.ticker in port.legs and leg.extending:
                # exit on recovery to stop level threshold
                if np.isfinite(avg) and high_px >= avg * (1.0 + args.stop_level):
                    exit_px = avg * (1.0 + args.stop_level)
                    close_leg(date, leg, float(exit_px), reason="EXT_RECOVERY", max_dd_port=port.max_dd)
                    continue

                # DCA every day in extending (using full unit * weight split later)
                # (handled in daily buy section below)

            # ---------- Drawdown update later (portfolio)

        # 2) If in cycle, do DAILY BUYS (split by weights) for legs that still exist
        if port.in_cycle and len(port.legs) > 0:
            # for each rank/leg, compute desired based on zone
            # daily buy only once per day per leg
            for i, (t, leg) in enumerate(list(port.legs.items())):
                if t not in day_df.index:
                    continue
                row = day_df.loc[t]
                close_px = float(row["Close"])
                if not np.isfinite(close_px) or close_px <= 0:
                    continue

                avg = leg.avg_price()
                w = weights[i] if i < len(weights) else 0.0
                if w <= 0:
                    continue

                # after TP1 (partial take), stop DCA for that leg (let it run)
                if leg.tp1_done:
                    continue

                # desired allocation for this leg today
                if leg.extending:
                    desired = float(port.unit) * w
                else:
                    if np.isfinite(avg) and close_px <= avg:
                        desired = float(port.unit) * w
                    elif np.isfinite(avg) and close_px <= avg * 1.05:
                        desired = float(port.unit) * w / 2.0
                    else:
                        desired = 0.0

                invest = clamp_invest_by_leverage(port.seed, port.entry_seed, desired, float(args.max_leverage_pct))
                if invest > 0:
                    port.seed -= invest
                    leg.invested += invest
                    leg.shares += invest / close_px
                    update_max_leverage(port.entry_seed)

        # 3) Entry if not in cycle and not sell-day
        if (not port.in_cycle) and (not cooldown_today):
            picks_today = picks_by_date.get(date, [])
            # keep tickers that exist in today's price
            picks_today = [(r, t) for (r, t) in picks_today if t in day_df.index]
            if len(picks_today) > 0:
                # use up to topk and weights
                picks_today = picks_today[:topk]

                S0 = float(port.seed)
                unit = (S0 / float(args.max_days)) if args.max_days > 0 else 0.0

                # first-day buys split by weights
                bought_any = False
                legs = {}
                for idx, (rank, ticker) in enumerate(picks_today):
                    w = weights[idx] if idx < len(weights) else 0.0
                    if w <= 0:
                        continue
                    close_px = float(day_df.loc[ticker]["Close"])
                    desired = float(unit) * w
                    invest = clamp_invest_by_leverage(port.seed, S0, desired, float(args.max_leverage_pct))
                    if invest > 0:
                        port.seed -= invest
                        update_max_leverage(S0)
                        leg = Leg(ticker=ticker, shares=invest / close_px, invested=invest, holding_days=1)
                        legs[ticker] = leg
                        bought_any = True

                if bought_any:
                    port.in_cycle = True
                    port.entry_seed = S0
                    port.unit = float(unit)
                    port.entry_date = date
                    port.legs = legs

        # 4) Update portfolio DD + record curve
        # Build close map for holdings
        px_close_map = {}
        for t in list(port.legs.keys()):
            if t in day_df.index:
                px_close_map[t] = float(day_df.loc[t]["Close"])
        port.update_dd(px_close_map)

        curve.append({
            "Date": date,
            "Equity": port.equity(px_close_map),
            "Seed": port.seed,
            "InCycle": int(port.in_cycle),
            "NumLegs": int(len(port.legs)),
            "Tickers": ",".join(sorted(list(port.legs.keys()))),
            "PositionValue": port.value(px_close_map),
            "EntrySeed": port.entry_seed if port.in_cycle else np.nan,
            "Unit": port.unit if port.in_cycle else np.nan,
            "MaxLeveragePctCycle": port.max_leverage_pct if port.in_cycle else 0.0,
            "MaxDrawdownPortfolio": port.max_dd,
        })

        # 5) cycle reset if no legs left
        reset_cycle_if_done()

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