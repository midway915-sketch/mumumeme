# scripts/build_success_labels.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
LBL_DIR = DATA_DIR / "labels"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

OUT_PARQ = LBL_DIR / "labels_success.parquet"
OUT_CSV = LBL_DIR / "labels_success.csv"


def read_prices() -> pd.DataFrame:
    if PRICES_PARQ.exists():
        df = pd.read_parquet(PRICES_PARQ)
    elif PRICES_CSV.exists():
        df = pd.read_csv(PRICES_CSV)
    else:
        raise FileNotFoundError(f"Missing prices: {PRICES_PARQ} (or {PRICES_CSV})")

    df = df.copy()
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("prices must include Date and Ticker columns")

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"prices missing required column: {c}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["Date", "Ticker", "Close", "High"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", required=True, type=float, help="e.g. 0.10")
    ap.add_argument("--max-days", required=True, type=int, help="e.g. 40")
    ap.add_argument("--start-date", default=None, type=str, help="optional YYYY-MM-DD")
    ap.add_argument("--out-parq", default=str(OUT_PARQ), type=str)
    ap.add_argument("--out-csv", default=str(OUT_CSV), type=str)
    args = ap.parse_args()

    LBL_DIR.mkdir(parents=True, exist_ok=True)

    df = read_prices()
    if args.start_date:
        sd = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(sd):
            raise ValueError(f"Invalid --start-date: {args.start_date}")
        df = df.loc[df["Date"] >= sd].copy()

    H = int(args.max_days)
    pt = float(args.profit_target)

    def per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date").copy()
        close = pd.to_numeric(g["Close"], errors="coerce")
        high = pd.to_numeric(g["High"], errors="coerce")

        future_max_high = high.shift(-1).rolling(H, min_periods=H).max()
        thr = close * (1.0 + pt)
        y = (future_max_high >= thr).astype(float)

        return pd.DataFrame({
            "Date": g["Date"].values,
            "Ticker": g["Ticker"].values,
            "Target": y.values,
        })

    out = df.groupby("Ticker", group_keys=False).apply(per_ticker)
    out = out.dropna(subset=["Target"]).copy()
    out["Target"] = pd.to_numeric(out["Target"], errors="coerce").fillna(0).astype(int)

    out = out.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last").reset_index(drop=True)

    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_csv)
    out.to_parquet(out_parq, index=False)
    out.to_csv(out_csv, index=False)

    print(f"[DONE] wrote: {out_parq} rows={len(out)}")
    if len(out):
        print(f"[INFO] range: {out['Date'].min().date()}..{out['Date'].max().date()}")


if __name__ == "__main__":
    main()