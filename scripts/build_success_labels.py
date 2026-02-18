# scripts/build_success_labels.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Training data not found: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def per_ticker(g: pd.DataFrame, pt: float, max_days: int) -> pd.DataFrame:
    g = g.sort_values("Date").copy()
    close = pd.to_numeric(g["Close"], errors="coerce")
    high = pd.to_numeric(g["High"], errors="coerce")

    # next max_days window's max High (starting tomorrow)
    future_max_high = high.shift(-1).rolling(int(max_days), min_periods=int(max_days)).max()
    thr = close * (1.0 + float(pt))

    success = (future_max_high >= thr).astype("float")  # float so NaN handling is easy

    out = pd.DataFrame(
        {
            "Date": g["Date"].values,
            "Ticker": g["Ticker"].values,
            # train_model.py 호환: Success 컬럼
            "Success": success.values,
            # 혹시 기존 코드가 Target을 기대해도 같이 둠
            "Target": success.values,
        }
    )

    # drop tail rows where we don't have full future window
    out = out.dropna(subset=["Success"]).copy()
    out["Success"] = out["Success"].astype(int)
    out["Target"] = out["Target"].astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build labels_model with Success (and Target) from raw prices.")
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)

    ap.add_argument("--start-date", default=None, type=str, help="optional filter: keep Date >= start-date")
    ap.add_argument("--out-parq", default="data/labels/labels_model.parquet", type=str)
    ap.add_argument("--out-csv", default="data/labels/labels_model.csv", type=str)

    args = ap.parse_args()

    df = read_table(args.prices_parq, args.prices_csv).copy()
    need = ["Date", "Ticker", "Close", "High"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"prices missing required columns: {missing}")

    df["Date"] = norm_date(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["Date", "Ticker", "Close", "High"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if args.start_date:
        sd = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(sd):
            raise ValueError(f"Invalid --start-date: {args.start_date}")
        df = df.loc[df["Date"] >= sd].copy()

    outs = []
    for _, g in df.groupby("Ticker", sort=False):
        outs.append(per_ticker(g, pt=float(args.profit_target), max_days=int(args.max_days)))

    labels = pd.concat(outs, ignore_index=True)
    labels = labels.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last").reset_index(drop=True)

    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_csv)
    out_parq.parent.mkdir(parents=True, exist_ok=True)

    labels.to_parquet(out_parq, index=False)
    labels.to_csv(out_csv, index=False)

    print(f"[DONE] wrote: {out_parq} rows={len(labels)}")
    if len(labels):
        dmin = pd.to_datetime(labels["Date"]).min().date()
        dmax = pd.to_datetime(labels["Date"]).max().date()
        print(f"[INFO] range: {dmin}..{dmax}")


if __name__ == "__main__":
    main()