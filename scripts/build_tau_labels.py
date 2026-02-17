# scripts/build_tau_labels.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEATURE_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"
LABEL_DIR.mkdir(parents=True, exist_ok=True)

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

FEATS_PARQ = FEATURE_DIR / "features_model.parquet"
FEATS_CSV = FEATURE_DIR / "features_model.csv"

OUT_PARQ = LABEL_DIR / "tau_labels.parquet"
OUT_CSV = LABEL_DIR / "tau_labels.csv"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def compute_tau(prices: pd.DataFrame, profit_target: float, horizon: int) -> pd.DataFrame:
    """
    For each (Date, Ticker) treat Close as entry price.
    Tau = first k (1..horizon) such that future High >= entry*(1+profit_target)
    If never hit: Tau = NaN, Censored=1, Success=0
    If hit: Tau = k, Censored=0, Success=1
    """
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    out_rows = []

    for tkr, g in prices.groupby("Ticker", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        highs = g["High"].to_numpy(dtype=float)
        closes = g["Close"].to_numpy(dtype=float)
        dates = g["Date"].to_numpy()

        n = len(g)
        # iterate entry index i and search forward until horizon
        for i in range(n):
            entry = closes[i]
            if not np.isfinite(entry) or entry <= 0:
                continue

            target = entry * (1.0 + profit_target)
            j_end = min(n, i + horizon + 1)

            # find first future index j where High >= target (j>i)
            hit = None
            for j in range(i + 1, j_end):
                if highs[j] >= target:
                    hit = j
                    break

            if hit is None:
                out_rows.append((dates[i], tkr, np.nan, 1, 0))
            else:
                tau = (hit - i)  # days forward count (>=1)
                out_rows.append((dates[i], tkr, int(tau), 0, 1))

    out = pd.DataFrame(out_rows, columns=["Date", "Ticker", "Tau_Days", "Censored", "Success"])
    out["Date"] = pd.to_datetime(out["Date"])
    out["Ticker"] = out["Ticker"].astype(str)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)
    args = ap.parse_args()

    horizon = int(args.max_days + args.max_extend_days)

    prices = read_table(PRICES_PARQ, PRICES_CSV)
    need_cols = {"Date", "Ticker", "High", "Close"}
    missing = need_cols - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing columns: {missing}")

    feats = read_table(FEATS_PARQ, FEATS_CSV)
    feats["Date"] = pd.to_datetime(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()

    tau = compute_tau(prices, profit_target=float(args.profit_target), horizon=horizon)

    merged = feats.merge(tau, on=["Date", "Ticker"], how="left")

    # 성공한 샘플에 tau가 있어야 함
    # (실패는 Tau_Days NaN)
    merged.to_parquet(OUT_PARQ, index=False)
    merged.to_csv(OUT_CSV, index=False)
    print(f"[DONE] wrote {OUT_PARQ} rows={len(merged)} horizon={horizon}")


if __name__ == "__main__":
    main()