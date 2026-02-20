# scripts/simulate_single_position_engine.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TAU_CLASS_TO_H = {0: 20, 1: 40, 2: 60}


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


def clamp_invest_by_leverage(seed: float, entry_seed: float, desired: float, max_leverage_pct: float) -> float:
    """
    Enforce: seed_after >= - entry_seed * max_leverage_pct
    desired : amount we'd like to spend today (>=0)
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
    weight: float

    shares: float = 0.0
    invested: float = 0.0

    tp1_done: bool = False
    peak: float = 0.0  # for trailing after TP1

    def avg_price(self) -> float:
        return (self.invested / self.shares) if (self.shares > 0 and self.invested > 0) else np.nan

    def value(self, close_px: float) -> float:
        if not np.isfinite(close_px) or close_px <= 0:
            return 0.0
        return float(self.shares) * float(close_px)


@dataclass
class CycleState:
    in_cycle: bool = False
    entry_date: pd.Timestamp | None = None

    seed: float = 0.0         # cash, can go negative
    entry_seed: float = 0.0   # S0 at entry

    H: int = 0
    unit: float = 0.0
    holding_days: int = 0

    pending_reval: bool = False
    ret_H: float = np.nan
    extending: bool = False
    grace_days_total: int = 0
    grace_days_left: int = 0
    recovery_threshold: float = np.nan
    reval_strength: str = ""

    max_leverage_pct: float = 0.0
    max_equity: float = 0.0
    max_dd: float = 0.0

    legs: list[Leg] = None  # type: ignore

    # ✅ NEW: cycle-level peak tracking + trailing entry count
    peak_equity: float = 0.0
    peak_seed_multiple: float = 1.0
    trail_entry_count: int = 0  # TP1 발생 횟수 = trailing 진입 횟수

    def equity(self, prices: dict[str, float]) -> float:
        v = 0.0
        if self.legs:
            for leg in self.legs:
                px = prices.get(leg.ticker, np.nan)
                if np.isfinite(px):
                    v += leg.value(float(px))
        return float(self.seed) + float(v)

    def update_dd(self, prices: dict[str, float]) -> None:
        eq = self.equity(prices)

        # peak equity / seed multiple tracking
        if eq > self.peak_equity:
            self.peak_equity = float(eq)
            if self.entry_seed > 0:
                self.peak_seed_multiple = float(eq / self.entry_seed)

        if eq > self.max_equity:
            self.max_equity = eq
        if self.max_equity > 0:
            dd = (eq - self.max_equity) / self.max_equity
            if dd < self.max_dd:
                self.max_dd = dd

    def update_lev(self, max_cap: float) -> None:
        if self.entry_seed <= 0:
            return
        lev = max(0.0, -self.seed) / self.entry_seed
        if lev > self.max_leverage_pct:
            self.max_leverage_pct = float(lev)
        if self.max_leverage_pct > max_cap + 1e-9:
            self.max_leverage_pct = float(max_cap)


def parse_weights(weights: str, topk: int) -> list[float]:
    parts = [p.strip() for p in str(weights).split(",") if p.strip()]
    ws = [float(p) for p in parts]
    if len(ws) != topk:
        raise ValueError(f"--weights must have {topk} numbers (got {len(ws)}): {weights}")
    s = sum(ws)
    if s <= 0:
        raise ValueError("--weights sum must be > 0")
    return [w / s for w in ws]


def compute_cycle_return_today(st: CycleState, day_close: dict[str, float]) -> float:
    rets = []
    for leg in st.legs:
        if leg.shares <= 0:
            continue
        px = day_close.get(leg.ticker, np.nan)
        avg = leg.avg_price()
        if np.isfinite(px) and np.isfinite(avg) and avg > 0:
            rets.append((px - avg) / avg)
    return float(np.mean(rets)) if rets else float("nan")


def load_feat_map(features_parq: str, features_csv: str) -> pd.DataFrame:
    p = Path(features_parq)
    c = Path(features_csv)
    if not p.exists() and not c.exists():
        return pd.DataFrame()
    df = read_table(str(p), str(c)).copy()
    if "Date" not in df.columns or "Ticker" not in df.columns:
        return pd.DataFrame()
    df["Date"] = _norm_date(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    keep = [x for x in ["Date", "Ticker", "p_success", "p_tail", "tau_H", "tau_class"] if x in df.columns]
    df = df[keep].dropna(subset=["Date", "Ticker"]).drop_duplicates(subset=["Date", "Ticker"]).reset_index(drop=True)
    df = df.set_index(["Date", "Ticker"], drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Single-cycle engine with TopK (1~2), TP1 partial, trailing stop, leverage cap on ALL buys.\n"
            "Added metrics:\n"
            " - TrailEntryCount (TP1 count) per cycle\n"
            " - PeakSeedMultiple / PeakCycleReturn per cycle\n"
        )
    )

    ap.add_argument("--picks-path", required=True, type=str)
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--features-scored-parq", default="data/features/features_scored.parquet", type=str)
    ap.add_argument("--features-scored-csv", default="data/features/features_scored.csv", type=str)

    ap.add_argument("--initial-seed", default=40_000_000, type=float)

    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--max-leverage-pct", default=1.0, type=float)

    ap.add_argument("--enable-trailing", default="true", type=str)
    ap.add_argument("--tp1-frac", default=0.50, type=float)
    ap.add_argument("--trail-stop", default=0.10, type=float)

    # compat only
    ap.add_argument("--tp1-trail-unlimited", default="true", type=str)

    ap.add_argument("--topk", default=1, type=int)
    ap.add_argument("--weights", default="1.0", type=str)

    ap.add_argument("--reval-ps-strong", default=0.70, type=float)
    ap.add_argument("--reval-pt-strong", default=0.20, type=float)
    ap.add_argument("--reval-ps-pass", default=0.60, type=float)
    ap.add_argument("--reval-pt-pass", default=0.35, type=float)

    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--suffix", default="", type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    args = ap.parse_args()

    enable_trailing = str(args.enable_trailing).lower() in ("1", "true", "yes", "y")
    _ = str(args.tp1_trail_unlimited).lower() in ("1", "true", "yes", "y")  # compat only

    topk = int(args.topk)
    if topk < 1 or topk > 2:
        raise ValueError("--topk should be 1 or 2")
    weights = parse_weights(args.weights, topk=topk)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_map = load_feat_map(args.features_scored_parq, args.features_scored_csv)

    picks_path = Path(args.picks_path)
    if not picks_path.exists():
        raise FileNotFoundError(f"Missing picks file: {picks_path}")

    picks = pd.read_csv(picks_path) if picks_path.suffix.lower() != ".parquet" else pd.read_parquet(picks_path)
    if "Date" not in picks.columns or "Ticker" not in picks.columns:
        raise ValueError(f"picks must have Date,Ticker. cols={list(picks.columns)[:50]}")

    picks = picks.copy()
    picks["Date"] = _norm_date(picks["Date"])
    picks["Ticker"] = picks["Ticker"].astype(str).str.upper().str.strip()
    picks = picks.dropna(subset=["Date", "Ticker"]).sort_values(["Date"]).reset_index(drop=True)
    picks = picks.groupby("Date", group_keys=False).head(topk).reset_index(drop=True)

    prices = read_table(args.prices_parq, args.prices_csv).copy()
    if "Date" not in prices.columns or "Ticker" not in prices.columns:
        raise ValueError("prices must have Date,Ticker")
    prices["Date"] = _norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    for c in ["Open", "High", "Low", "Close"]:
        if c not in prices.columns:
            raise ValueError(f"prices missing {c}")
    prices = prices.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    picks_by_date: dict[pd.Timestamp, list[str]] = {}
    for d, g in picks.groupby("Date"):
        picks_by_date[d] = g["Ticker"].tolist()

    grouped = prices.groupby("Date", sort=True)
    all_dates = list(grouped.groups.keys())
    last_date = all_dates[-1] if all_dates else None

    st = CycleState(
        seed=float(args.initial_seed),
        max_equity=float(args.initial_seed),
        max_dd=0.0,
        legs=[],
        peak_equity=float(args.initial_seed),
        peak_seed_multiple=1.0,
        trail_entry_count=0,
    )

    cooldown_today = False
    trades = []
    curve = []

    def liquidate_all_legs(day_prices_close: dict[str, float]) -> tuple[float, float]:
        proceeds = 0.0
        invested_total = 0.0
        for leg in st.legs:
            px = float(day_prices_close.get(leg.ticker, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue
            proceeds += leg.shares * px
            invested_total += leg.invested
        return proceeds, invested_total

    def close_cycle(exit_date: pd.Timestamp, day_prices_close: dict[str, float], reason: str) -> None:
        nonlocal cooldown_today, st, trades
        proceeds, invested_total = liquidate_all_legs(day_prices_close)

        cycle_return = (proceeds - invested_total) / invested_total if invested_total > 0 else np.nan
        win = int(cycle_return > 0) if np.isfinite(cycle_return) else 0

        peak_seed_mult = float(st.peak_seed_multiple) if np.isfinite(st.peak_seed_multiple) else np.nan
        peak_cycle_ret = float(peak_seed_mult - 1.0) if np.isfinite(peak_seed_mult) else np.nan

        trades.append({
            "CycleType": "MAIN",
            "EntryDate": st.entry_date,
            "ExitDate": exit_date,
            "Tickers": ",".join([l.ticker for l in st.legs]),
            "Weights": ",".join([f"{l.weight:.4f}" for l in st.legs]),
            "EntrySeed": st.entry_seed,
            "H": int(st.H),
            "ProfitTarget": args.profit_target,
            "TP1_Frac": float(args.tp1_frac),
            "TrailStop": float(args.trail_stop) if enable_trailing else np.nan,
            "MaxDaysInput": int(args.max_days),
            "StopLevelInput": float(args.stop_level),
            "MaxExtendDaysInput": int(args.max_extend_days),
            "GraceDays": int(st.grace_days_total),
            "RevalStrength": st.reval_strength,
            "RecoveryThreshold": float(st.recovery_threshold) if np.isfinite(st.recovery_threshold) else np.nan,
            "MaxLeveragePctCap": args.max_leverage_pct,
            "MaxLeveragePct": st.max_leverage_pct,
            "Invested": invested_total,
            "Proceeds": proceeds,
            "CycleReturn": cycle_return,
            "HoldingDays": st.holding_days,
            "Extending": int(st.extending),
            "Reason": reason,
            "MaxDrawdown": st.max_dd,
            "Win": win,

            # ✅ NEW metrics
            "TrailEntryCount": int(st.trail_entry_count),
            "PeakSeedMultiple": peak_seed_mult,
            "PeakCycleReturn": peak_cycle_ret,
        })

        st.seed += proceeds

        # reset cycle
        st.in_cycle = False
        st.entry_date = None
        st.entry_seed = 0.0
        st.H = 0
        st.unit = 0.0
        st.holding_days = 0

        st.pending_reval = False
        st.ret_H = np.nan
        st.extending = False
        st.grace_days_total = 0
        st.grace_days_left = 0
        st.recovery_threshold = np.nan
        st.reval_strength = ""

        st.max_leverage_pct = 0.0
        st.legs = []

        # ✅ reset peak tracking for next cycle (will be set on entry)
        st.peak_equity = float(st.seed)
        st.peak_seed_multiple = 1.0
        st.trail_entry_count = 0

        cooldown_today = True

    def reval_strength_for_day(date: pd.Timestamp) -> tuple[str, float, float]:
        if feat_map.empty:
            return "FAIL", 0.0, 1.0

        ps = []
        pt = []
        for leg in st.legs:
            try:
                row = feat_map.loc[(date, leg.ticker)]
            except Exception:
                continue
            if "p_success" in row.index:
                ps.append(float(row["p_success"]))
            if "p_tail" in row.index:
                pt.append(float(row["p_tail"]))

        if not ps or not pt:
            return "FAIL", float(np.mean(ps)) if ps else 0.0, float(np.mean(pt)) if pt else 1.0

        ps_m = float(np.mean(ps))
        pt_m = float(np.mean(pt))

        if ps_m >= float(args.reval_ps_strong) and pt_m <= float(args.reval_pt_strong):
            return "STRONG", ps_m, pt_m
        if ps_m >= float(args.reval_ps_pass) and pt_m <= float(args.reval_pt_pass):
            return "WEAK", ps_m, pt_m
        return "FAIL", ps_m, pt_m

    # simulate
    for date, day_df in grouped:
        day_df = day_df.set_index("Ticker", drop=False)
        cooldown_today = False

        day_prices_close = {}
        day_prices_high = {}
        day_prices_low = {}

        for t in day_df.index:
            r = day_df.loc[t]
            day_prices_close[t] = float(r["Close"])
            day_prices_high[t] = float(r["High"])
            day_prices_low[t] = float(r["Low"])

        # ----- in cycle: update
        if st.in_cycle:
            st.holding_days += 1

            if st.pending_reval:
                strength, ps_m, pt_m = reval_strength_for_day(date)
                st.reval_strength = strength

                if strength == "FAIL":
                    close_cycle(date, day_prices_close, reason=f"REVAL_FAIL(ps={ps_m:.3f},pt={pt_m:.3f})")
                else:
                    st.extending = True
                    st.pending_reval = False

                    st.grace_days_total = int(min(max(1, st.H // 4), 15))
                    st.grace_days_left = int(st.grace_days_total)

                    base = float(st.ret_H) + 0.05 if np.isfinite(st.ret_H) else -0.05
                    if strength == "STRONG":
                        st.recovery_threshold = float(max(base, -0.05))
                    else:
                        st.recovery_threshold = float(min(base, -0.05))

            # if cycle got closed by reval, skip rest
            if st.in_cycle:
                # 1) TP1 + trailing per leg (priority)
                if enable_trailing:
                    for leg in st.legs:
                        if leg.ticker not in day_df.index:
                            continue
                        high_px = day_prices_high[leg.ticker]
                        low_px = day_prices_low[leg.ticker]
                        avg = leg.avg_price()

                        # TP1 => trailing 진입 카운트
                        if (not leg.tp1_done) and np.isfinite(avg) and high_px >= avg * (1.0 + float(args.profit_target)):
                            tp_px = avg * (1.0 + float(args.profit_target))
                            sell_shares = leg.shares * float(args.tp1_frac)
                            sell_shares = float(min(leg.shares, max(0.0, sell_shares)))
                            proceeds = sell_shares * tp_px

                            leg.shares -= sell_shares
                            leg.invested *= (leg.shares / (leg.shares + sell_shares)) if (leg.shares + sell_shares) > 0 else 0.0

                            st.seed += proceeds
                            leg.tp1_done = True
                            leg.peak = float(max(high_px, tp_px))

                            # ✅ NEW: trailing entry count
                            st.trail_entry_count += 1

                            st.update_lev(float(args.max_leverage_pct))

                        # trailing after TP1
                        if leg.tp1_done and leg.shares > 0:
                            if high_px > leg.peak:
                                leg.peak = float(high_px)
                            stop_px = leg.peak * (1.0 - float(args.trail_stop))
                            if np.isfinite(low_px) and low_px <= stop_px:
                                proceeds = leg.shares * stop_px
                                st.seed += proceeds
                                leg.shares = 0.0
                                leg.invested = 0.0

                # 2) if all legs emptied -> close
                if st.in_cycle and all((leg.shares <= 0 for leg in st.legs)):
                    close_cycle(date, day_prices_close, reason="TRAIL_EXIT_ALL")

            # 3) H reached: set pending_reval (but only if no TP1 triggered)
            if st.in_cycle and (not st.extending) and (not st.pending_reval):
                any_tp1 = any((leg.tp1_done for leg in st.legs))
                if (not any_tp1) and (st.H > 0) and (st.holding_days >= st.H):
                    st.ret_H = compute_cycle_return_today(st, day_prices_close)
                    st.pending_reval = True

            # 4) Grace mode
            if st.in_cycle and st.extending:
                any_tp1 = any((leg.tp1_done for leg in st.legs))
                if not any_tp1:
                    st.grace_days_left = max(0, int(st.grace_days_left) - 1)

                    thr = float(st.recovery_threshold) if np.isfinite(st.recovery_threshold) else -0.05
                    hit = False
                    for leg in st.legs:
                        if leg.shares <= 0:
                            continue
                        if leg.tp1_done:
                            continue
                        px = day_prices_close.get(leg.ticker, np.nan)
                        avg = leg.avg_price()
                        if np.isfinite(px) and np.isfinite(avg) and avg > 0:
                            r = (px - avg) / avg
                            if r >= thr:
                                hit = True
                                break

                    if hit:
                        close_cycle(date, day_prices_close, reason="GRACE_RECOVERY_EXIT")
                    else:
                        if st.in_cycle and st.grace_days_left <= 0:
                            close_cycle(date, day_prices_close, reason="GRACE_END_EXIT")

                if st.in_cycle:
                    st.update_dd(day_prices_close)

            # 5) Normal mode DCA
            if st.in_cycle and (not st.extending) and (not st.pending_reval):
                all_tp1 = all((leg.tp1_done for leg in st.legs))
                if not all_tp1:
                    desired_total = float(st.unit)
                    for leg in st.legs:
                        if leg.ticker not in day_df.index:
                            continue
                        close_px = day_prices_close[leg.ticker]
                        if not np.isfinite(close_px) or close_px <= 0:
                            continue

                        avg = leg.avg_price()
                        desired_leg = desired_total * float(leg.weight)

                        if np.isfinite(avg) and avg > 0:
                            if close_px <= avg:
                                pass
                            elif close_px <= avg * 1.05:
                                desired_leg = desired_leg / 2.0
                            else:
                                desired_leg = 0.0

                        invest = clamp_invest_by_leverage(st.seed, st.entry_seed, desired_leg, float(args.max_leverage_pct))
                        if invest > 0:
                            st.seed -= invest
                            leg.invested += invest
                            leg.shares += invest / close_px
                            st.update_lev(float(args.max_leverage_pct))

                st.update_dd(day_prices_close)

        # ----- entry
        if (not st.in_cycle) and (not cooldown_today):
            picks_today = picks_by_date.get(date, [])
            if picks_today:
                valid = [t for t in picks_today if t in day_df.index and np.isfinite(day_prices_close.get(t, np.nan))]
                if len(valid) >= 1:
                    chosen = valid[:topk]

                    Hs = []
                    if not feat_map.empty:
                        for t in chosen:
                            try:
                                row = feat_map.loc[(date, t)]
                                if "tau_H" in row.index:
                                    hh = int(row["tau_H"])
                                    if hh > 0:
                                        Hs.append(hh)
                            except Exception:
                                pass
                    H_eff = int(max(Hs)) if Hs else int(args.max_days)
                    H_eff = int(max(1, H_eff))

                    S0 = float(st.seed)
                    unit = (S0 / float(H_eff)) if H_eff > 0 else 0.0

                    legs = []
                    for i, t in enumerate(chosen):
                        legs.append(Leg(ticker=t, weight=float(weights[i])))

                    invested_total = 0.0
                    for leg in legs:
                        px = day_prices_close[leg.ticker]
                        desired = float(unit) * float(leg.weight)
                        invest = clamp_invest_by_leverage(st.seed, S0, desired, float(args.max_leverage_pct))
                        if invest > 0:
                            st.seed -= invest
                            leg.invested += invest
                            leg.shares += invest / px
                            invested_total += invest

                    if invested_total > 0:
                        st.in_cycle = True
                        st.entry_date = date
                        st.entry_seed = S0
                        st.H = int(H_eff)
                        st.unit = float(unit)
                        st.holding_days = 1

                        st.pending_reval = False
                        st.ret_H = np.nan
                        st.extending = False
                        st.grace_days_total = 0
                        st.grace_days_left = 0
                        st.recovery_threshold = np.nan
                        st.reval_strength = ""

                        st.max_leverage_pct = 0.0
                        st.legs = legs

                        # ✅ NEW: init peak tracking at entry
                        st.peak_equity = float(st.equity(day_prices_close))
                        st.peak_seed_multiple = float(st.peak_equity / st.entry_seed) if st.entry_seed > 0 else 1.0
                        st.trail_entry_count = 0

                        st.update_lev(float(args.max_leverage_pct))
                        st.update_dd(day_prices_close)

        # ----- record curve
        prices_for_eq = day_prices_close if st.in_cycle else {}
        eq = st.equity(prices_for_eq)

        curve.append({
            "Date": date,
            "Equity": eq,
            "Seed": st.seed,
            "InCycle": int(st.in_cycle),
            "Tickers": ",".join([l.ticker for l in st.legs]) if st.in_cycle else "",
            "HoldingDays": st.holding_days if st.in_cycle else 0,
            "H": int(st.H) if st.in_cycle else 0,
            "PendingReval": int(st.pending_reval),
            "Extending": int(st.extending),
            "GraceLeft": int(st.grace_days_left),
            "RevalStrength": st.reval_strength,
            "RecoveryThreshold": float(st.recovery_threshold) if np.isfinite(st.recovery_threshold) else np.nan,
            "MaxLeveragePctCycle": st.max_leverage_pct if st.in_cycle else 0.0,
            "MaxDrawdownPortfolio": st.max_dd,

            # ✅ NEW: cycle peak snapshot
            "PeakSeedMultipleCycle": float(st.peak_seed_multiple) if st.in_cycle else np.nan,
            "TrailEntryCountCycle": int(st.trail_entry_count) if st.in_cycle else 0,
        })

        # ----- force close on last day
        if last_date is not None and date == last_date and st.in_cycle:
            close_cycle(date, day_prices_close, reason="FINAL_CLOSE")

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve)
    if not curve_df.empty:
        curve_df["SeedMultiple"] = curve_df["Equity"] / float(args.initial_seed)

    tag = args.tag if args.tag else "run"
    suffix = args.suffix if args.suffix else picks_path.stem.replace("picks_", "")

    trades_path = Path(args.out_dir) / f"sim_engine_trades_{tag}_gate_{suffix}.parquet"
    curve_path = Path(args.out_dir) / f"sim_engine_curve_{tag}_gate_{suffix}.parquet"
    trades_df.to_parquet(trades_path, index=False)
    curve_df.to_parquet(curve_path, index=False)

    print(f"[DONE] wrote trades: {trades_path} rows={len(trades_df)}")
    print(f"[DONE] wrote curve : {curve_path} rows={len(curve_df)}")
    if not curve_df.empty:
        print(f"[INFO] final SeedMultiple={float(curve_df['SeedMultiple'].iloc[-1]):.4f} maxDD={float(st.max_dd):.4f}")


if __name__ == "__main__":
    main()