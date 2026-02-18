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

    seed: float = 0.0  # cash (can be negative)

    shares: float = 0.0
    invested: float = 0.0

    entry_seed: float = 0.0     # S0 at entry
    unit: float = 0.0          # base unit = entry_seed / max_days
    entry_date: pd.Timestamp | None = None
    holding_days: int = 0
    extending: bool = False

    # tau
    tau_pred: float = np.nan

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

    borrow_limit = float(entry_seed) * float(max_leverage_pct)  # 1.0 => 100%
    floor_seed = -borrow_limit
    room = float(seed) - float(floor_seed)  # seed + borrow_limit
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


def tau_adjusted_desired(unit: float, holding_day: int, max_days: int, tau_pred: float, tau_gamma: float) -> float:
    """
    τ 기반 매수금 조절 (안정형)
    - tau_pred 짧을수록 초반 매수 더 크게, 후반 더 작게
    - tau_gamma=0이면 비활성 (unit 그대로)
    """
    if unit <= 0:
        return 0.0
    if tau_gamma <= 0:
        return float(unit)

    if not np.isfinite(tau_pred) or tau_pred <= 0:
        return float(unit)

    # 안정적으로 clamp
    tp = float(np.clip(tau_pred, 5.0, float(max_days)))
    d = float(np.clip(holding_day, 1, max_days))

    # ratio = (max_days / tau_pred) : tau가 짧으면 ratio>1 (초반 강화)
    ratio = float(max_days) / tp
    ratio = float(np.clip(ratio, 0.7, 1.8))

    # "초반" 정의: tau_pred 내의 앞 60% 구간을 초반으로 간주
    early_end = max(1.0, 0.6 * tp)

    if d <= early_end:
        mult = (ratio ** tau_gamma)
    else:
        mult = ((1.0 / ratio) ** tau_gamma)

    # 너무 튀지 않게 최종 클램프
    mult = float(np.clip(mult, 0.3, 2.5))
    return float(unit) * mult


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-position engine with leverage cap and τ-based buy sizing.")
    ap.add_argument("--picks-path", required=True, type=str)
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--initial-seed", default=40_000_000, type=float)
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)
    ap.add_argument("--max-leverage-pct", default=1.0, type=float)

    # τ sizing strength (0 disables)
    ap.add_argument("--tau-gamma", default=1.0, type=float, help="0 disables τ sizing, higher = stronger")

    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--suffix", default="", type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag, suffix = _pick_default_out_tag(args.picks_path, args.tag or None, args.suffix or None)

    # load picks
    if args.picks_path.lower().endswith(".parquet"):
        picks = pd.read_parquet(args.picks_path)
    else:
        picks = pd.read_csv(args.picks_path)

    if "Date" not in picks.columns or "Ticker" not in picks.columns:
        raise ValueError(f"picks must have Date/Ticker. cols={list(picks.columns)[:50]}")

    picks = picks.copy()
    picks["Date"] = _norm_date(picks["Date"])
    picks["Ticker"] = picks["Ticker"].astype(str).str.upper().str.strip()
    if "tau_pred" in picks.columns:
        picks["tau_pred"] = pd.to_numeric(picks["tau_pred"], errors="coerce")
    picks = picks.dropna(subset=["Date", "Ticker"]).sort_values(["Date"]).reset_index(drop=True)
    picks = picks.drop_duplicates(["Date"], keep="last")

    # map date -> (ticker, tau_pred)
    if "tau_pred" in picks.columns:
        picks_by_date = {d: (t, float(tp) if np.isfinite(tp) else np.nan) for d, t, tp in zip(picks["Date"], picks["Ticker"], picks["tau_pred"])}
    else:
        picks_by_date = {d: (t, np.nan) for d, t in zip(picks["Date"], picks["Ticker"])}

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
            "TauPred": pos.tau_pred,
            "TauGamma": float(args.tau_gamma),
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
            "HoldingDays": pos.holding_days,
            "Extending": int(pos.extending),
            "Reason": reason,
            "MaxDrawdown": pos.max_dd,
            "Win": win,
        })

        pos.seed += proceeds

        pos.in_pos = False
        pos.ticker = None
        pos.shares = 0.0
        pos.invested = 0.0
        pos.entry_seed = 0.0
        pos.unit = 0.0
        pos.entry_date = None
        pos.holding_days = 0
        pos.extending = False
        pos.tau_pred = np.nan
        pos.max_leverage_pct = 0.0

        cooldown_today = True

    for date, day_df in grouped:
        day_df = day_df.set_index("Ticker", drop=False)
        cooldown_today = False

        # ----- HOLDING LOGIC
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
                    if np.isfinite(avg) and np.isfinite(high_px) and high_px >= avg * (1.0 + args.profit_target):
                        exit_px = avg * (1.0 + args.profit_target)
                        close_cycle(date, float(exit_px), reason="TP")
                    else:
                        if pos.in_pos and pos.holding_days >= int(args.max_days):
                            cur_ret = (close_px - avg) / avg if np.isfinite(avg) and avg != 0 else -np.inf
                            if cur_ret >= float(args.stop_level):
                                close_cycle(date, float(close_px), reason="MAXDAY_CLOSE")
                            else:
                                pos.extending = True

                # buys (after exit checks)
                if pos.in_pos and pos.extending:
                    if np.isfinite(avg) and np.isfinite(high_px) and high_px >= avg * (1.0 + args.stop_level):
                        exit_px = avg * (1.0 + args.stop_level)
                        close_cycle(date, float(exit_px), reason="EXT_RECOVERY")
                    else:
                        desired = tau_adjusted_desired(
                            unit=float(pos.unit),
                            holding_day=int(pos.holding_days),
                            max_days=int(args.max_days),
                            tau_pred=float(pos.tau_pred),
                            tau_gamma=float(args.tau_gamma),
                        )
                        invest = clamp_invest_by_leverage(pos.seed, pos.entry_seed, desired, args.max_leverage_pct)
                        if invest > 0 and close_px > 0:
                            pos.seed -= invest
                            pos.invested += invest
                            pos.shares += invest / close_px
                            update_max_leverage_pct(pos, pos.entry_seed, args.max_leverage_pct)

                if pos.in_pos and (not pos.extending):
                    if np.isfinite(avg) and close_px > 0:
                        if close_px <= avg:
                            base_desired = float(pos.unit)
                        elif close_px <= avg * 1.05:
                            base_desired = float(pos.unit) / 2.0
                        else:
                            base_desired = 0.0

                        desired = tau_adjusted_desired(
                            unit=float(base_desired),
                            holding_day=int(pos.holding_days),
                            max_days=int(args.max_days),
                            tau_pred=float(pos.tau_pred),
                            tau_gamma=float(args.tau_gamma),
                        )
                        invest = clamp_invest_by_leverage(pos.seed, pos.entry_seed, desired, args.max_leverage_pct)
                        if invest > 0:
                            pos.seed -= invest
                            pos.invested += invest
                            pos.shares += invest / close_px
                            update_max_leverage_pct(pos, pos.entry_seed, args.max_leverage_pct)

                if pos.in_pos:
                    pos.update_drawdown(close_px)
                else:
                    pos.update_drawdown(None)

        # ----- ENTRY
        if (not pos.in_pos) and (not cooldown_today):
            pick_info = picks_by_date.get(date, None)
            if pick_info is not None:
                pick, tau_pred = pick_info
                if pick in day_df.index:
                    row = day_df.loc[pick]
                    close_px = float(row["Close"])
                    if np.isfinite(close_px) and close_px > 0:
                        S0 = float(pos.seed)
                        unit = (S0 / float(args.max_days)) if args.max_days > 0 else 0.0

                        # entry buy also uses tau-adjust
                        desired = tau_adjusted_desired(
                            unit=float(unit),
                            holding_day=1,
                            max_days=int(args.max_days),
                            tau_pred=float(tau_pred),
                            tau_gamma=float(args.tau_gamma),
                        )
                        invest = clamp_invest_by_leverage(pos.seed, S0, desired, args.max_leverage_pct)

                        if invest > 0:
                            pos.in_pos = True
                            pos.ticker = pick
                            pos.entry_seed = S0
                            pos.unit = unit
                            pos.entry_date = date
                            pos.holding_days = 1
                            pos.extending = False
                            pos.tau_pred = float(tau_pred) if np.isfinite(tau_pred) else np.nan
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

        # ----- CURVE
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
            "TauPred": pos.tau_pred if pos.in_pos else np.nan,
            "TauGamma": float(args.tau_gamma),
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