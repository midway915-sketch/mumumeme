# scripts/build_tail_labels.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
FEAT_DIR = DATA_DIR / "features"
LBL_DIR = DATA_DIR / "labels"

FEATS_PARQ = FEAT_DIR / "features_model.parquet"
FEATS_CSV  = FEAT_DIR / "features_model.csv"

OUT_PARQ = LBL_DIR / "labels_tail.parquet"
OUT_CSV  = LBL_DIR / "labels_tail.csv"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build tail labels (TailTarget) from features_model.")
    ap.add_argument("--stop-level", required=True, type=float, help="e.g. -0.10 (10% drop threshold)")
    ap.add_argument("--start-date", default=None, type=str, help="optional YYYY-MM-DD")
    ap.add_argument("--out-parq", default=str(OUT_PARQ), type=str)
    ap.add_argument("--out-csv", default=str(OUT_CSV), type=str)
    args = ap.parse_args()

    LBL_DIR.mkdir(parents=True, exist_ok=True)

    df = read_table(FEATS_PARQ, FEATS_CSV).copy()

    # required cols
    required = ["Date", "Ticker", "Close", "Low"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"features_model missing required columns for tail labels: {missing}")

    df["Date"] = norm_date(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    close = pd.to_numeric(df["Close"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")

    if args.start_date:
        sd = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(sd):
            raise ValueError(f"Invalid --start-date: {args.start_date}")
        df = df.loc[df["Date"] >= sd].copy()
        close = pd.to_numeric(df["Close"], errors="coerce")
        low = pd.to_numeric(df["Low"], errors="coerce")

    stop_level = float(args.stop_level)  # negative number expected
    thresh = close * (1.0 + stop_level)

    # TailTarget: did we touch/violate stop threshold intraday?
    tail = (low <= thresh).astype(int)

    out = pd.DataFrame({
        "Date": df["Date"].values,
        "Ticker": df["Ticker"].values,
        "TailTarget": tail.values,
    })

    out = out.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last")

    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_csv)
    out_parq.parent.mkdir(parents=True, exist_ok=True)

    out.to_parquet(out_parq, index=False)
    out.to_csv(out_csv, index=False)

    print(f"[DONE] wrote: {out_parq} rows={len(out)}")
    if len(out):
        dmin = pd.to_datetime(out["Date"]).min().date()
        dmax = pd.to_datetime(out["Date"]).max().date()
        print(f"[INFO] range: {dmin}..{dmax} stop_level={stop_level}")


if __name__ == "__main__":
    main()