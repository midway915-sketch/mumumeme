# scripts/build_tau_labels.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
LBL_DIR = DATA_DIR / "labels"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

OUT_PARQ = LBL_DIR / "labels_tau.parquet"
OUT_CSV = LBL_DIR / "labels_tau.csv"


def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def build_tau_labels(
    prices: pd.DataFrame,
    profit_target: float,
    max_days: int,
    stop_level: float,
    max_extend_days: int,
    k1: int,
    k2: int,
) -> pd.DataFrame:
    """
    TauClass:
      0: no success within max_days (or within extend horizon; you can define)
      1: success within k1 days
      2: success within k2 days
      3: success within max_days
    """
    df = prices.copy()
    df["Date"] = norm_date(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    out_rows = []

    pt = float(profit_target)
    H = int(max_days)

    for t, g in df.groupby("Ticker", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        close = pd.to_numeric(g["Close"], errors="coerce").to_numpy(dtype=float)

        # future max over next H days (shift -1 so "after today")
        fut_max = pd.Series(close).shift(-1).rolling(H, min_periods=H).max().to_numpy()

        # success happens if fut_max >= close*(1+pt)
        success = (fut_max >= close * (1.0 + pt)).astype(int)

        # time-to-success (first day in [1..H] where High/Close hits target)
        # Simple approximation: use Close series next H days (consistent with your existing approach style)
        tau = np.full(len(g), np.nan, dtype=float)

        for i in range(len(g) - H - 1):
            base = close[i]
            target = base * (1.0 + pt)
            window = close[i+1:i+H+1]
            hit = np.where(window >= target)[0]
            if hit.size > 0:
                tau[i] = float(hit[0] + 1)

        # TauClass
        tau_class = np.full(len(g), np.nan, dtype=float)
        for i in range(len(g)):
            if not np.isfinite(tau[i]):
                tau_class[i] = 0.0
            else:
                d = int(tau[i])
                if d <= int(k1):
                    tau_class[i] = 1.0
                elif d <= int(k2):
                    tau_class[i] = 2.0
                elif d <= int(H):
                    tau_class[i] = 3.0
                else:
                    tau_class[i] = 0.0

        tmp = pd.DataFrame({
            "Date": g["Date"].values,
            "Ticker": g["Ticker"].values,
            "Tau": tau,
            "TauClass": tau_class,
        })
        out_rows.append(tmp)

    out = pd.concat(out_rows, ignore_index=True)
    out = out.dropna(subset=["Date", "Ticker", "TauClass"]).copy()
    out["TauClass"] = pd.to_numeric(out["TauClass"], errors="coerce").fillna(0).astype(int)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices-parq", default=str(PRICES_PARQ), type=str)
    ap.add_argument("--prices-csv", default=str(PRICES_CSV), type=str)

    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--k1", default=10, type=int)
    ap.add_argument("--k2", default=20, type=int)
    ap.add_argument("--start-date", default=None, type=str)

    ap.add_argument("--out-parq", default=str(OUT_PARQ), type=str)
    ap.add_argument("--out-csv", default=str(OUT_CSV), type=str)

    args = ap.parse_args()

    prices = read_table(args.prices_parq, args.prices_csv).copy()
    for c in ["Date", "Ticker", "Close"]:
        if c not in prices.columns:
            raise ValueError(f"prices missing required column: {c}")

    prices["Date"] = norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices = prices.dropna(subset=["Date", "Ticker", "Close"]).copy()

    if args.start_date:
        sd = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(sd):
            raise ValueError(f"Invalid --start-date: {args.start_date}")
        prices = prices.loc[prices["Date"] >= sd].copy()

    out = build_tau_labels(
        prices=prices,
        profit_target=float(args.profit_target),
        max_days=int(args.max_days),
        stop_level=float(args.stop_level),
        max_extend_days=int(args.max_extend_days),
        k1=int(args.k1),
        k2=int(args.k2),
    )

    out_p = Path(args.out_parq)
    out_c = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    out.to_parquet(out_p, index=False)
    out.to_csv(out_c, index=False)

    print(f"[DONE] wrote: {out_p} rows={len(out)}")
    if len(out):
        print(f"[INFO] range: {out['Date'].min()}..{out['Date'].max()}")


if __name__ == "__main__":
    main()