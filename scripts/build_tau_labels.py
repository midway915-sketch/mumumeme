# scripts/build_tau_labels.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

FEATURES_PARQ = Path("data/features/features_model.parquet")
FEATURES_CSV = Path("data/features/features_model.csv")
PRICES_PARQ = Path("data/raw/prices.parquet")
PRICES_CSV = Path("data/raw/prices.csv")


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def _to_dt(x):
    return pd.to_datetime(x, errors="coerce")


def _make_tau_cutoffs(max_days: int) -> tuple[int, int, int]:
    """
    Default: (10, 20, max_days)
    but ensure strictly increasing: c1 < c2 < c3
    """
    c3 = int(max_days)
    c1 = min(10, c3)
    c2 = min(20, c3)

    # enforce strict increase
    if c1 >= c3:
        c1 = max(1, c3 - 2)
    if c2 <= c1:
        # try set c2 between c1 and c3
        mid = (c1 + c3) // 2
        c2 = mid if mid > c1 else min(c3 - 1, c1 + 1)
    if c2 >= c3:
        c2 = max(c1 + 1, c3 - 1)

    # final guard
    if not (c1 < c2 < c3):
        # fallback proportional split
        c1 = max(1, int(round(c3 * 0.33)))
        c2 = max(c1 + 1, int(round(c3 * 0.66)))
        c2 = min(c2, c3 - 1)

    return int(c1), int(c2), int(c3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)  # kept for symmetry
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--start-date", default=None, type=str)
    ap.add_argument("--buffer-days", default=0, type=int)
    ap.add_argument("--out-dir", default="data/labels", type=str)
    args = ap.parse_args()

    pt_tag = int(round(args.profit_target * 100))
    sl_tag = int(round(abs(args.stop_level) * 100))
    tag = f"pt{pt_tag}_h{args.max_days}_sl{sl_tag}_ex{args.max_extend_days}"

    feats = read_table(FEATURES_PARQ, FEATURES_CSV).copy()
    prices = read_table(PRICES_PARQ, PRICES_CSV).copy()

    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_model must contain Date and Ticker")

    feats["Date"] = _to_dt(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()

    prices["Date"] = _to_dt(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()

    need_cols = {"Date", "Ticker", "Close", "High"}
    missing = need_cols - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing columns: {missing}")

    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
    prices["High"] = pd.to_numeric(prices["High"], errors="coerce")

    if args.start_date:
        start = pd.to_datetime(args.start_date)
        if args.buffer_days and args.buffer_days > 0:
            start = start - pd.Timedelta(days=int(args.buffer_days))
        feats = feats.loc[feats["Date"] >= start].copy()

    # tau search horizon: max_days + extend
    H = int(args.max_days + args.max_extend_days)
    pt = float(args.profit_target)

    cut1, cut2, cut3 = _make_tau_cutoffs(int(args.max_days))
    # cut3 is essentially max_days (strictly)
    # We'll label success-time within horizon H (but classify <= cut1/2/3 windows)
    # i.e., τ is computed as first day k where High[t+k] >= EntryClose*(1+pt), k in [1..H]

    prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    def compute_tau_for_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date").reset_index(drop=True)
        entry_close = g["Close"].to_numpy(dtype=float)
        future_high = g["High"].to_numpy(dtype=float)

        n = len(g)
        tau = np.full(n, np.nan, dtype=float)

        for i in range(n):
            ec = entry_close[i]
            if not np.isfinite(ec) or ec <= 0:
                continue
            thr = ec * (1.0 + pt)
            jmax = min(n - 1, i + H)
            hit = np.where(future_high[i + 1 : jmax + 1] >= thr)[0]
            if hit.size:
                tau[i] = float(hit[0] + 1)  # days to first hit
        out = g[["Ticker", "Date"]].copy()
        out["TauDays"] = tau
        return out

    tau_df = prices.groupby("Ticker", sort=False, group_keys=False).apply(compute_tau_for_ticker)
    tau_df = tau_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    m = feats.merge(tau_df, on=["Date", "Ticker"], how="left")
    tau_num = pd.to_numeric(m["TauDays"], errors="coerce")

    # Dynamic CDF targets
    m["TauLE1"] = (tau_num <= cut1).astype(int)
    m["TauLE2"] = (tau_num <= cut2).astype(int)
    m["TauLE3"] = (tau_num <= cut3).astype(int)

    # store cutoffs for downstream clarity/debugging
    m["TauCut1"] = cut1
    m["TauCut2"] = cut2
    m["TauCut3"] = cut3
    m["TauHorizon"] = H

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parq = out_dir / f"labels_tau_{tag}.parquet"
    out_csv = out_dir / f"labels_tau_{tag}.csv"

    m = m.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    try:
        m.to_parquet(out_parq, index=False)
    except Exception:
        m.to_csv(out_csv, index=False)

    if not out_csv.exists():
        m.to_csv(out_csv, index=False)

    print(
        f"[DONE] labels_tau built: tag={tag} rows={len(m)} "
        f"horizon={H} cuts=({cut1},{cut2},{cut3}) -> {out_parq} / {out_csv}"
    )


if __name__ == "__main__":
    main()