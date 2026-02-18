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


def _infer_tag_suffix(picks_path: str, tag: str | None, suffix: str | None) -> tuple[str, str]:
    if tag and suffix:
        return tag, suffix

    name = Path(picks_path).name
    m = re.search(r"picks_(pt\d+_h\d+_sl\d+_ex\d+)_gate_(.+)\.csv$", name)
    if m:
        inferred_tag = m.group(1)
        inferred_suffix = m.group(2)
        return (tag or inferred_tag), (suffix or inferred_suffix)

    inferred_tag = tag or "run"
    inferred_suffix = suffix or Path(picks_path).stem.replace("picks_", "")
    return inferred_tag, inferred_suffix


# -----------------------------
# Leverage cap: core
# -----------------------------
def clamp_invest_by_leverage(seed: float, entry_seed: float, desired: float, max_leverage_pct: float) -> float:
    """
    Enforce: seed_after >= - entry_seed * max_leverage_pct
    """
    if desired <= 0:
        return 0.0

    if not np.isfinite(entry_seed) or entry_seed <= 0:
        # no borrowing if entry_seed invalid/<=0
        return float(min(desired, max(seed, 0.0)))

    borrow_limit = float(entry_seed) * float(max_leverage_pct)  # 1.0 => 100%
    floor_seed = -borrow_limit
    room = float(seed) - float(floor_seed)  # seed + borrow_limit
    if room <= 0:
        return 0.0
    return float(min(desired, room))


# -----------------------------
# Position state (multi-leg)
# -----------------------------
@dataclass
class Leg:
    ticker: str
    weight: float

    shares: float = 0.0
    invested: float = 0.0  # cost basis (cash spent)

    tp1_done: bool = False
    max_price: float = 0.0  # for trailing


def _avg_price(leg: Leg) -> float:
    if leg.shares > 0 and leg.invested > 0:
        return leg.invested / leg.shares
    return np.nan


def _leg_value(leg: Leg, close_px: float) -> float:
    if leg.shares <= 0 or not np.isfinite(close_px):
        return 0.0
    return float(leg.shares) * float(close_px)


# -----------------------------
# Main simulation
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Single-cycle engine (Top-1/Top-2) with leverage cap on ALL buys.")
    ap.add_argument("--picks-path", required=True, type=str)
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--initial-seed", default=40_000_000, type=float)

    ap.add_argument("--profit-target", required=True, type=float)   # e.g. 0.10
    ap.add_argument("--max-days", required=True, type=int)          # e.g. 40
    ap.add_argument("--stop-level", required=True, type=float)      # e.g. -0.10
    ap.add_argument("--max-extend-days", required=True, type=int)   # param-only (no hard limit)
    ap.add_argument("--max-leverage-pct", default=1.0, type=float)  # 1.0 => 100%

    # TopK entry config
    ap.add_argument("--topk", default=1, type=int)
    ap.add_argument("--weights", default="1.0", type=str, help="comma weights, e.g. '0.7,0.3' for topk=2")

    # 2-step take profit + trailing (optional)
    ap.add_argument("--enable-trailing", default="true", type=str)  # true/false
    ap.add_argument("--tp1-frac", default=0.50, type=float)         # sell this fraction at +PT
    ap.add_argument("--trail-stop", default=0.10, type=float)       # 0.10 => -10% from max after TP1

    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--suffix", default="", type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    args = ap.parse_args()

    enable_trailing = str(args.enable_trailing).lower() in ("1", "true", "yes", "y")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag, suffix = _infer_tag_suffix(args.picks_path, args.tag or None, args.suffix or None)

    topk = int(args.topk)
    if topk <= 0:
        raise ValueError("--topk must be >= 1")

    weights = [float(x.strip()) for x in str(args.weights).split(",") if x.strip()]
    if len(weights) != topk:
        raise ValueError(f"--weights must have exactly topk numbers. topk={topk} weights={weights}")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(f"Invalid weights: {weights}")
    # normalize weights to sum=1
    s = sum(weights)
    weights = [w / s for w in weights]

    # ---- load picks
    picks = pd.read_csv(args.picks_path)
    # Defensive column normalize
    cols = {c: c.strip() for c in picks.columns}
    picks = picks.rename(columns=cols)

    if "Date" not in picks.columns:
        # try fallback
        for cand in ["date", "DATE", "Datetime", "datetime"]:
            if cand in picks.columns:
                picks = picks.rename(columns={cand: "Date"})
                break
    if "Ticker" not in picks.columns:
        for cand in ["ticker", "TICKER"]:
            if cand in picks.columns:
                picks = picks.rename(columns={cand: "Ticker"})
                break

    if "Date" not in picks.columns or "Ticker" not in picks.columns:
        raise ValueError(f"picks file must have Date/Ticker. cols={list(picks.columns)}")

    picks = picks.copy()
    picks["Date"] = _norm_date(picks["Date"])
    picks["Ticker"] = picks["Ticker"].astype(str).str.upper().str.strip()
    if "Rank" in picks.columns:
        picks["Rank"] = pd.to_numeric(picks["Rank"], errors="coerce")
    else:
        picks["Rank"] = np.nan

    picks = picks.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Rank", "Ticker"]).reset_index(drop=True)

    # topk rows per date (if picks already topk from predict_gate, this keeps it)
    picks = picks.groupby("Date", group_keys=False).head(topk).reset_index(drop=True)

    # ---- load prices
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

    # picks map: Date -> list[tickers in rank order]
    picks_map: dict[pd.Timestamp, list[str]] = {}
    for d, g in picks.groupby("Date"):
        picks_map[pd.Timestamp(d)] = g.sort_values(["Rank", "Ticker"])["Ticker"].tolist()[:topk]

    # ---- portfolio state
    seed = float(args.initial_seed)
    entry_seed = 0.0
    unit = 0.0
    in_cycle = False
    entry_date = None
    holding_days = 0
    cooldown_today = False

    max_leverage_pct_obs = 0.0  # max(-seed)/entry_seed within cycle
    max_equity = float(seed)
    max_dd = 0.0

    legs: list[Leg] = []

    trades = []
    curve = []

    def equity(day_df: pd.DataFrame | None) -> float:
        nonlocal seed, legs
        val = 0.0
        if day_df is not None and len(legs) > 0:
            for leg in legs:
                if leg.ticker in day_df.index:
                    val += _leg_value(leg, float(day_df.loc[leg.ticker]["Close"]))
        return float(seed) + float(val)

    def update_dd(eq: float) -> None:
        nonlocal max_equity, max_dd
        if eq > max_equity:
            max_equity = eq
        if max_equity > 0:
            dd = (eq - max_equity) / max_equity
            if dd < max_dd:
                max_dd = dd

    def update_max_lev() -> None:
        nonlocal seed, entry_seed, max_leverage_pct_obs
        if entry_seed <= 0:
            return
        lev = max(0.0, -seed) / entry_seed
        if lev > max_leverage_pct_obs:
            max_leverage_pct_obs = float(lev)
        if max_leverage_pct_obs > float(args.max_leverage_pct) + 1e-9:
            max_leverage_pct_obs = float(args.max_leverage_pct)

    def cycle_invested_total() -> float:
        return float(sum(l.invested for l in legs))

    def cycle_value_total(day_df: pd.DataFrame) -> float:
        tot = 0.0
        for l in legs:
            if l.ticker in day_df.index:
                tot += _leg_value(l, float(day_df.loc[l.ticker]["Close"]))
        return float(tot)

    def close_cycle(exit_date: pd.Timestamp, reason: str, day_df: pd.DataFrame) -> None:
        nonlocal seed, in_cycle, legs, holding_days, entry_date, entry_seed, unit, cooldown_today

        invested_tot = cycle_invested_total()
        value_tot = cycle_value_total(day_df)

        cycle_ret = (value_tot - invested_tot) / invested_tot if invested_tot > 0 else np.nan
        win = int(cycle_ret > 0) if np.isfinite(cycle_ret) else 0

        # liquidate all legs at Close (for summary consistency)
        proceeds = 0.0
        for l in legs:
            if l.ticker in day_df.index and l.shares > 0:
                px = float(day_df.loc[l.ticker]["Close"])
                proceeds += float(l.shares) * px

        seed += proceeds

        trades.append({
            "EntryDate": entry_date,
            "ExitDate": exit_date,
            "Tickers": ",".join([l.ticker for l in legs]),
            "EntrySeed": entry_seed,
            "ProfitTarget": float(args.profit_target),
            "MaxDays": int(args.max_days),
            "StopLevel": float(args.stop_level),
            "MaxExtendDaysParam": int(args.max_extend_days),
            "MaxLeveragePctCap": float(args.max_leverage_pct),
            "MaxLeveragePct": float(max_leverage_pct_obs),
            "Invested": float(invested_tot),
            "Proceeds": float(proceeds),
            "CycleReturn": float(cycle_ret) if np.isfinite(cycle_ret) else np.nan,
            "HoldingDays": int(holding_days),
            "Win": int(win),
            "Reason": reason,
            "MaxDrawdown": float(max_dd),
            "TopK": int(topk),
            "Weights": ",".join([f"{w:.4f}" for w in weights]),
            "EnableTrailing": int(enable_trailing),
            "TP1_Frac": float(args.tp1_frac),
            "TrailStop": float(args.trail_stop),
        })

        # reset
        in_cycle = False
        legs = []
        holding_days = 0
        entry_date = None
        entry_seed = 0.0
        unit = 0.0
        cooldown_today = True

    # ---- run simulation
    for date, day_df in grouped:
        day_df = day_df.set_index("Ticker", drop=False)

        cooldown_today = False

        # update holding day and manage open legs
        if in_cycle:
            holding_days += 1

            # --- manage each leg
            for leg in list(legs):
                if leg.ticker not in day_df.index:
                    continue

                row = day_df.loc[leg.ticker]
                close_px = float(row["Close"])
                high_px = float(row["High"])
                low_px = float(row["Low"])

                avg = _avg_price(leg)

                # update trailing max
                if enable_trailing and leg.tp1_done:
                    if np.isfinite(high_px) and high_px > leg.max_price:
                        leg.max_price = float(high_px)

                # 2-step take profit: TP1
                if (not leg.tp1_done) and np.isfinite(avg) and np.isfinite(high_px) and high_px >= avg * (1.0 + args.profit_target):
                    # sell tp1_frac at target price
                    sell_frac = float(args.tp1_frac)
                    sell_frac = max(0.0, min(1.0, sell_frac))
                    if sell_frac > 0 and leg.shares > 0:
                        sell_shares = leg.shares * sell_frac
                        exit_px = avg * (1.0 + args.profit_target)
                        proceeds = sell_shares * exit_px

                        # reduce basis proportionally
                        leg.shares -= sell_shares
                        leg.invested *= (1.0 - sell_frac)

                        seed += proceeds
                        update_max_lev()

                        leg.tp1_done = True
                        leg.max_price = float(high_px) if np.isfinite(high_px) else float(exit_px)

                # trailing stop after TP1 (sell remainder)
                if enable_trailing and leg.tp1_done and leg.shares > 0 and leg.max_price > 0 and np.isfinite(low_px):
                    stop_px = leg.max_price * (1.0 - float(args.trail_stop))
                    if low_px <= stop_px:
                        proceeds = leg.shares * stop_px
                        seed += proceeds
                        leg.shares = 0.0
                        leg.invested = 0.0
                        update_max_lev()

                # remove empty legs
                if leg.shares <= 0:
                    legs = [x for x in legs if x.shares > 0]

            # if all legs closed, end cycle (sell-day cooldown)
            if in_cycle and len(legs) == 0:
                close_cycle(date, reason="ALL_LEGS_CLOSED", day_df=day_df)

            # --- DCA / extend logic (shared holding_days, per-leg sizing by weight)
            if in_cycle and len(legs) > 0:
                # decide extend mode by portfolio condition at max_days
                extending = holding_days >= int(args.max_days)

                for idx, leg in enumerate(legs):
                    if leg.ticker not in day_df.index:
                        continue
                    row = day_df.loc[leg.ticker]
                    close_px = float(row["Close"])
                    high_px = float(row["High"])

                    avg = _avg_price(leg)
                    if not (np.isfinite(close_px) and close_px > 0 and np.isfinite(avg) and avg > 0):
                        continue

                    # extending exit: rebound to stop-level threshold from avg
                    if extending:
                        if np.isfinite(high_px) and high_px >= avg * (1.0 + args.stop_level):
                            exit_px = avg * (1.0 + args.stop_level)
                            proceeds = leg.shares * exit_px
                            seed += proceeds
                            leg.shares = 0.0
                            leg.invested = 0.0
                            update_max_lev()
                            continue

                        desired = float(unit) * float(leg.weight)
                        invest = clamp_invest_by_leverage(seed, entry_seed, desired, float(args.max_leverage_pct))
                        if invest > 0:
                            seed -= invest
                            leg.invested += invest
                            leg.shares += invest / close_px
                            update_max_lev()

                    else:
                        # normal zone DCA rule
                        if close_px <= avg:
                            desired = float(unit) * float(leg.weight)
                        elif close_px <= avg * 1.05:
                            desired = float(unit) * float(leg.weight) / 2.0
                        else:
                            desired = 0.0

                        invest = clamp_invest_by_leverage(seed, entry_seed, desired, float(args.max_leverage_pct))
                        if invest > 0:
                            seed -= invest
                            leg.invested += invest
                            leg.shares += invest / close_px
                            update_max_lev()

                # cleanup empty legs again
                legs = [x for x in legs if x.shares > 0]
                if in_cycle and len(legs) == 0:
                    close_cycle(date, reason="ALL_LEGS_CLOSED", day_df=day_df)

        # entry (not on sell-day)
        if (not in_cycle) and (not cooldown_today):
            picks_today = picks_map.get(pd.Timestamp(date), [])
            picks_today = [t for t in picks_today if t in day_df.index]

            if len(picks_today) >= 1:
                # entry seed (S0) = current seed
                entry_seed = float(seed)
                unit = (entry_seed / float(args.max_days)) if args.max_days > 0 else 0.0

                # build legs with weights aligned to picks count
                w = weights[: len(picks_today)]
                # normalize weights to sum to 1 for actual picks_today count
                sw = sum(w)
                w = [x / sw for x in w]

                # initial buy: per-leg desired = unit * weight
                new_legs: list[Leg] = []
                for t, wt in zip(picks_today, w):
                    close_px = float(day_df.loc[t]["Close"])
                    desired = float(unit) * float(wt)
                    invest = clamp_invest_by_leverage(seed, entry_seed, desired, float(args.max_leverage_pct))
                    if invest > 0 and np.isfinite(close_px) and close_px > 0:
                        seed -= invest
                        leg = Leg(ticker=t, weight=float(wt))
                        leg.invested = invest
                        leg.shares = invest / close_px
                        new_legs.append(leg)

                if len(new_legs) > 0:
                    legs = new_legs
                    in_cycle = True
                    entry_date = date
                    holding_days = 1
                    update_max_lev()

        # record daily curve
        eq = equity(day_df if isinstance(day_df, pd.DataFrame) else None)
        update_dd(eq)

        tickers_str = ",".join([l.ticker for l in legs]) if in_cycle else ""
        invested_tot = cycle_invested_total() if in_cycle else 0.0
        value_tot = cycle_value_total(day_df) if in_cycle else 0.0

        curve.append({
            "Date": date,
            "Equity": float(eq),
            "Seed": float(seed),
            "InCycle": int(in_cycle),
            "Tickers": tickers_str,
            "HoldingDays": int(holding_days) if in_cycle else 0,
            "Invested": float(invested_tot),
            "PositionValue": float(value_tot),
            "EntrySeed": float(entry_seed) if in_cycle else np.nan,
            "Unit": float(unit) if in_cycle else np.nan,
            "MaxLeveragePctCycle": float(max_leverage_pct_obs) if in_cycle else 0.0,
            "MaxDrawdownPortfolio": float(max_dd),
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
        print(f"[INFO] final SeedMultiple={final_mult:.4f} maxDD={float(max_dd):.4f} maxLev={float(max_leverage_pct_obs):.4f}")


if __name__ == "__main__":
    main()