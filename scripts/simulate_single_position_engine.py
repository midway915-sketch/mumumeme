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


def _norm_date_series(s: pd.Series) -> pd.Series:
    # robust datetime normalize (no tz)
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _ensure_col(df: pd.DataFrame, want: str) -> pd.DataFrame:
    """
    Ensure df has column `want` by:
    - if it exists -> ok
    - else if index name matches -> reset_index
    - else if lower/alt names exist -> rename
    """
    if want in df.columns:
        return df

    # if it's index
    if df.index.name == want:
        df = df.reset_index()
        if want in df.columns:
            return df

    # common alternates
    alt = {
        "Date": ["date", "datetime", "Datetime", "TradeDate", "trade_date", "Day"],
        "Ticker": ["ticker", "symbol", "Symbol"],
    }
    for a in alt.get(want, []):
        if a in df.columns:
            df = df.rename(columns={a: want})
            return df

    # sometimes date saved as unnamed first column
    if want == "Date":
        for c in df.columns:
            if str(c).strip().lower() in ("date", "datetime"):
                df = df.rename(columns={c: "Date"})
                return df

    return df


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
# Leverage cap
# -----------------------------
def clamp_invest_by_leverage(seed: float, entry_seed: float, desired: float, max_leverage_pct: float) -> float:
    """
    Enforce: seed_after >= - entry_seed * max_leverage_pct

    desired : intended spend today (>=0)
    returns : allowed spend (>=0)
    """
    if desired <= 0:
        return 0.0
    if not np.isfinite(entry_seed) or entry_seed <= 0:
        # no borrowing if entry_seed invalid
        return float(min(desired, max(seed, 0.0)))

    borrow_limit = float(entry_seed) * float(max_leverage_pct)  # 1.0 => 100%
    floor_seed = -borrow_limit
    room = float(seed) - float(floor_seed)  # seed + borrow_limit
    if room <= 0:
        return 0.0
    return float(min(desired, room))


# -----------------------------
# Position & Bundle (Top-k)
# -----------------------------
@dataclass
class Leg:
    ticker: str
    weight: float

    shares: float = 0.0
    invested: float = 0.0

    holding_days: int = 0
    extending: bool = False

    # 2-step TP + trailing
    tp1_done: bool = False
    trail_on: bool = False
    peak_high_after_tp1: float = np.nan

    def avg_price(self) -> float:
        if self.shares <= 0 or self.invested <= 0:
            return np.nan
        return float(self.invested / self.shares)

    def value(self, close_px: float) -> float:
        if self.shares <= 0 or not np.isfinite(close_px):
            return 0.0
        return float(self.shares) * float(close_px)


@dataclass
class BundleState:
    in_pos: bool = False
    entry_date: pd.Timestamp | None = None

    seed: float = 0.0  # cash (can be negative)
    entry_seed: float = 0.0  # S0
    unit: float = 0.0  # daily base buy amount = entry_seed / max_days

    legs: list[Leg] | None = None
    cooldown_today: bool = False

    # stats
    max_leverage_pct: float = 0.0
    max_equity: float = 0.0
    max_dd: float = 0.0

    def equity(self, close_map: dict[str, float]) -> float:
        v = 0.0
        if self.in_pos and self.legs:
            for leg in self.legs:
                px = close_map.get(leg.ticker, np.nan)
                if np.isfinite(px):
                    v += leg.value(px)
        return float(self.seed) + float(v)

    def update_dd(self, close_map: dict[str, float]) -> None:
        eq = self.equity(close_map)
        if eq > self.max_equity:
            self.max_equity = eq
        if self.max_equity > 0:
            dd = (eq - self.max_equity) / self.max_equity
            if dd < self.max_dd:
                self.max_dd = dd

    def update_max_leverage(self, max_leverage_cap: float) -> None:
        if self.entry_seed <= 0:
            return
        lev = max(0.0, -self.seed) / float(self.entry_seed)
        if lev > self.max_leverage_pct:
            self.max_leverage_pct = float(lev)
        if self.max_leverage_pct > max_leverage_cap + 1e-9:
            self.max_leverage_pct = float(max_leverage_cap)


# -----------------------------
# Picks parsing (Top-k compatible)
# -----------------------------
def load_picks(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df = _ensure_col(df, "Date")
    df = _ensure_col(df, "Ticker")

    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError(f"picks must have Date/Ticker. cols={list(df.columns)[:50]}")

    df = df.copy()
    df["Date"] = _norm_date_series(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # optional cols
    if "RankInDay" in df.columns:
        df["RankInDay"] = pd.to_numeric(df["RankInDay"], errors="coerce").fillna(9999).astype(int)
    else:
        df["RankInDay"] = 1

    if "Weight" in df.columns:
        df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(1.0).astype(float)
    else:
        df["Weight"] = 1.0

    return df


def picks_for_date(df: pd.DataFrame, date: pd.Timestamp, topk: int) -> pd.DataFrame:
    sub = df[df["Date"] == date].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values(["RankInDay", "Ticker"]).head(int(topk)).reset_index(drop=True)

    # normalize weights among selected
    w = pd.to_numeric(sub["Weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    wsum = float(np.sum(w))
    if wsum <= 0:
        sub["Weight"] = 1.0 / len(sub)
    else:
        sub["Weight"] = w / wsum
    return sub


# -----------------------------
# Prices
# -----------------------------
def load_prices(parq: str, csv: str) -> pd.DataFrame:
    prices = read_table(parq, csv).copy()
    prices = _ensure_col(prices, "Date")
    prices = _ensure_col(prices, "Ticker")
    if "Date" not in prices.columns or "Ticker" not in prices.columns:
        raise ValueError(f"prices must have Date/Ticker. cols={list(prices.columns)[:50]}")
    prices["Date"] = _norm_date_series(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    for c in ["Open", "High", "Low", "Close"]:
        if c not in prices.columns:
            raise ValueError(f"prices missing {c}")
    prices = prices.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return prices


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Top-k bundle engine: 2-step TP + trailing, leverage cap applied to ALL buys.")
    ap.add_argument("--picks-path", required=True, type=str)
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--initial-seed", default=40_000_000, type=float)
    ap.add_argument("--profit-target", required=True, type=float)   # ex: 0.10
    ap.add_argument("--max-days", required=True, type=int)          # ex: 40
    ap.add_argument("--stop-level", required=True, type=float)      # ex: -0.10
    ap.add_argument("--max-extend-days", required=True, type=int)   # tag compat (NO hard limit)
    ap.add_argument("--max-leverage-pct", default=1.0, type=float)  # 1.0 => 100%

    # Top-k
    ap.add_argument("--topk", default=1, type=int)                  # 1 or 2
    ap.add_argument("--weights", default="", type=str)              # e.g. "0.7,0.3" for topk=2 (optional override)

    # 2-step TP + trailing
    ap.add_argument("--enable-trailing", default="true", type=str)  # "true"/"false"
    ap.add_argument("--tp1-frac", default=0.50, type=float)         # sell fraction at PT
    ap.add_argument("--trail-stop", default=0.10, type=float)       # trailing stop (e.g. 0.10 => -10% from peak after TP1)

    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--suffix", default="", type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag, suffix = _pick_default_out_tag(args.picks_path, args.tag or None, args.suffix or None)

    enable_trailing = str(args.enable_trailing).strip().lower() in ("1", "true", "yes", "y")
    topk = int(args.topk)
    if topk not in (1, 2):
        raise ValueError("--topk must be 1 or 2")

    # weights override (optional)
    override_w = None
    if args.weights.strip():
        parts = [p.strip() for p in args.weights.split(",") if p.strip()]
        if len(parts) != topk:
            raise ValueError(f"--weights must have {topk} numbers (got {len(parts)}): {args.weights}")
        override_w = np.array([float(x) for x in parts], dtype=float)
        if float(np.sum(override_w)) <= 0:
            raise ValueError("--weights sum must be > 0")
        override_w = override_w / float(np.sum(override_w))

    # load data
    picks = load_picks(args.picks_path)
    prices = load_prices(args.prices_parq, args.prices_csv)

    # group prices by date
    grouped = prices.groupby("Date", sort=True)

    st = BundleState(
        seed=float(args.initial_seed),
        entry_seed=0.0,
        unit=0.0,
        in_pos=False,
        legs=None,
        cooldown_today=False,
        max_leverage_pct=0.0,
        max_equity=float(args.initial_seed),
        max_dd=0.0,
    )

    trades_rows = []
    curve_rows = []

    def close_leg(leg: Leg, exit_price: float, exit_date: pd.Timestamp, reason: str) -> dict:
        proceeds = leg.shares * float(exit_price)
        ret = (proceeds - leg.invested) / leg.invested if leg.invested > 0 else np.nan
        win = int(ret > 0) if np.isfinite(ret) else 0
        row = {
            "EntryDate": st.entry_date,
            "ExitDate": exit_date,
            "Ticker": leg.ticker,
            "TopK": topk,
            "Weight": leg.weight,
            "EntrySeed": st.entry_seed,
            "ProfitTarget": args.profit_target,
            "MaxDays": args.max_days,
            "StopLevel": args.stop_level,
            "MaxExtendDaysParam": args.max_extend_days,
            "EnableTrailing": int(enable_trailing),
            "TP1_Frac": float(args.tp1_frac),
            "TrailStop": float(args.trail_stop),
            "MaxLeveragePctCap": float(args.max_leverage_pct),
            "MaxLeveragePct": float(st.max_leverage_pct),
            "Invested": float(leg.invested),
            "Shares": float(leg.shares),
            "ExitPrice": float(exit_price),
            "Proceeds": float(proceeds),
            "LegReturn": float(ret) if np.isfinite(ret) else np.nan,
            "HoldingDays": int(leg.holding_days),
            "Extending": int(leg.extending),
            "TP1_Done": int(leg.tp1_done),
            "Reason": reason,
            "MaxDrawdownPortfolio": float(st.max_dd),
            "Win": win,
        }
        # update cash
        st.seed += float(proceeds)
        # clear leg
        leg.shares = 0.0
        leg.invested = 0.0
        return row

    def bundle_closed() -> bool:
        if not st.in_pos or not st.legs:
            return True
        for leg in st.legs:
            if leg.shares > 0 and leg.invested > 0:
                return False
        return True

    def close_bundle_if_done(exit_date: pd.Timestamp) -> None:
        if bundle_closed():
            st.in_pos = False
            st.legs = None
            st.entry_date = None
            st.entry_seed = 0.0
            st.unit = 0.0
            st.max_leverage_pct = 0.0
            st.cooldown_today = True  # no re-entry same day

    # simulate day by day
    for date, day_df in grouped:
        st.cooldown_today = False
        day_df = day_df.set_index("Ticker", drop=False)

        # close map for equity curve
        close_map = {t: float(r["Close"]) for t, r in day_df.set_index("Ticker").iterrows()}

        # ---- manage existing bundle
        if st.in_pos and st.legs:
            for leg in st.legs:
                if leg.shares <= 0 or leg.invested <= 0:
                    continue
                if leg.ticker not in day_df.index:
                    continue

                row = day_df.loc[leg.ticker]
                close_px = float(row["Close"])
                high_px = float(row["High"])
                low_px = float(row["Low"])

                leg.holding_days += 1
                avg = leg.avg_price()

                # (A) TP1 (partial) + trailing activation
                if (not leg.tp1_done) and np.isfinite(avg) and np.isfinite(high_px):
                    if high_px >= avg * (1.0 + float(args.profit_target)):
                        # sell tp1_frac at PT price
                        tp_px = avg * (1.0 + float(args.profit_target))
                        sell_frac = float(np.clip(args.tp1_frac, 0.0, 1.0))
                        sell_shares = leg.shares * sell_frac
                        if sell_shares > 0:
                            proceeds = sell_shares * tp_px
                            # reduce position
                            leg.shares -= sell_shares
                            # proportionally reduce invested
                            leg.invested *= (1.0 - sell_frac)
                            st.seed += proceeds
                            st.update_max_leverage(float(args.max_leverage_pct))

                        leg.tp1_done = True
                        if enable_trailing and leg.shares > 0:
                            leg.trail_on = True
                            leg.peak_high_after_tp1 = float(high_px)

                # (B) trailing stop on remaining shares
                if leg.trail_on and leg.shares > 0 and np.isfinite(leg.peak_high_after_tp1):
                    if np.isfinite(high_px) and high_px > leg.peak_high_after_tp1:
                        leg.peak_high_after_tp1 = float(high_px)
                    trail = float(args.trail_stop)
                    stop_px = leg.peak_high_after_tp1 * (1.0 - trail)
                    # intraday hit -> exit remaining at stop_px
                    if np.isfinite(low_px) and low_px <= stop_px:
                        trades_rows.append(close_leg(leg, stop_px, date, reason="TRAIL_STOP"))
                        continue

                # (C) max_days decision (extend or close at close)
                if (not leg.extending) and leg.holding_days >= int(args.max_days):
                    cur_ret = (close_px - avg) / avg if np.isfinite(avg) and avg != 0 else -np.inf
                    if cur_ret >= float(args.stop_level):
                        trades_rows.append(close_leg(leg, close_px, date, reason="MAXDAY_CLOSE"))
                        continue
                    else:
                        leg.extending = True

                # (D) extend recovery exit (stop_level rebound)
                if leg.extending and leg.shares > 0:
                    if np.isfinite(avg) and np.isfinite(high_px) and high_px >= avg * (1.0 + float(args.stop_level)):
                        exit_px = avg * (1.0 + float(args.stop_level))
                        trades_rows.append(close_leg(leg, exit_px, date, reason="EXT_RECOVERY"))
                        continue

                # (E) daily DCA buy logic (ALL buys leverage-capped)
                if leg.shares > 0 and np.isfinite(close_px) and close_px > 0:
                    base_unit = float(st.unit) * float(leg.weight)  # split by weights

                    if leg.extending:
                        desired = base_unit
                    else:
                        if np.isfinite(avg):
                            if close_px <= avg:
                                desired = base_unit
                            elif close_px <= avg * 1.05:
                                desired = base_unit / 2.0
                            else:
                                desired = 0.0
                        else:
                            desired = base_unit

                    invest = clamp_invest_by_leverage(st.seed, st.entry_seed, desired, float(args.max_leverage_pct))
                    if invest > 0:
                        st.seed -= invest
                        leg.invested += invest
                        leg.shares += invest / close_px
                        st.update_max_leverage(float(args.max_leverage_pct))

            # after managing legs, update dd / maybe close bundle
            st.update_dd(close_map)
            close_bundle_if_done(date)
        else:
            # not in position -> still update dd on cash-only
            st.update_dd(close_map)

        # ---- entry (only if not in pos and not cooldown_today)
        if (not st.in_pos) and (not st.cooldown_today):
            sub = picks_for_date(picks, date, topk=topk)
            if not sub.empty:
                tickers = sub["Ticker"].astype(str).str.upper().tolist()
                weights = sub["Weight"].to_numpy(dtype=float)
                if override_w is not None:
                    weights = override_w.copy()

                # filter to tickers present today in prices
                valid = []
                for t, w in zip(tickers, weights):
                    if t in day_df.index:
                        valid.append((t, float(w)))
                if valid:
                    tickers2, weights2 = zip(*valid)
                    weights2 = np.array(weights2, dtype=float)
                    weights2 = weights2 / float(np.sum(weights2))

                    # entry_seed = current seed (S0)
                    S0 = float(st.seed)
                    unit = (S0 / float(args.max_days)) if args.max_days > 0 else 0.0

                    # first-day buy: unit split by weights, leverage-capped at bundle level
                    legs = [Leg(ticker=t, weight=float(w)) for t, w in zip(tickers2, weights2)]

                    total_desired = float(unit)  # first day total desired = unit
                    invest_total = clamp_invest_by_leverage(st.seed, S0, total_desired, float(args.max_leverage_pct))
                    if invest_total > 0:
                        st.in_pos = True
                        st.entry_date = date
                        st.entry_seed = S0
                        st.unit = float(unit)
                        st.legs = legs
                        st.max_leverage_pct = 0.0

                        # allocate invest across legs by weights
                        for leg in st.legs:
                            row = day_df.loc[leg.ticker]
                            close_px = float(row["Close"])
                            leg.holding_days = 1
                            alloc = invest_total * float(leg.weight)
                            if alloc > 0 and np.isfinite(close_px) and close_px > 0:
                                st.seed -= alloc
                                leg.invested += alloc
                                leg.shares += alloc / close_px
                                st.update_max_leverage(float(args.max_leverage_pct))

                        st.update_dd(close_map)

        # ---- curve
        eq = st.equity(close_map)
        curve_rows.append({
            "Date": date,
            "Equity": eq,
            "Seed": float(st.seed),
            "InPosition": int(st.in_pos),
            "TopK": topk,
            "Tickers": ",".join([leg.ticker for leg in st.legs]) if (st.in_pos and st.legs) else "",
            "Invested": float(sum([leg.invested for leg in st.legs]) if (st.in_pos and st.legs) else 0.0),
            "PositionValue": float(sum([leg.value(close_map.get(leg.ticker, np.nan)) for leg in st.legs]) if (st.in_pos and st.legs) else 0.0),
            "HoldingDaysMax": int(max([leg.holding_days for leg in st.legs]) if (st.in_pos and st.legs) else 0),
            "ExtendingAny": int(any([leg.extending for leg in st.legs]) if (st.in_pos and st.legs) else 0),
            "TP1DoneAny": int(any([leg.tp1_done for leg in st.legs]) if (st.in_pos and st.legs) else 0),
            "MaxLeveragePctCycle": float(st.max_leverage_pct if st.in_pos else 0.0),
            "MaxDrawdownPortfolio": float(st.max_dd),
        })

    trades_df = pd.DataFrame(trades_rows)
    curve_df = pd.DataFrame(curve_rows)
    if not curve_df.empty:
        curve_df["SeedMultiple"] = curve_df["Equity"] / float(args.initial_seed)

        # idle days (portfolio not in position)
        curve_df["IdleDay"] = (curve_df["InPosition"] == 0).astype(int)

    trades_path = out_dir / f"sim_engine_trades_{tag}_gate_{suffix}.parquet"
    curve_path = out_dir / f"sim_engine_curve_{tag}_gate_{suffix}.parquet"
    trades_df.to_parquet(trades_path, index=False)
    curve_df.to_parquet(curve_path, index=False)

    print(f"[DONE] wrote trades: {trades_path} rows={len(trades_df)}")
    print(f"[DONE] wrote curve : {curve_path} rows={len(curve_df)}")
    if not curve_df.empty:
        final_mult = float(curve_df["SeedMultiple"].iloc[-1])
        idle = int(curve_df["IdleDay"].sum())
        print(f"[INFO] final SeedMultiple={final_mult:.4f} idle_days={idle} maxDD={float(st.max_dd):.4f}")


if __name__ == "__main__":
    main()