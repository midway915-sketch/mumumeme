# scripts/fetch_prices.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
UNIVERSE_CSV = DATA_DIR / "universe.csv"

OUT_PARQUET = RAW_DIR / "prices.parquet"
OUT_CSV_FALLBACK = RAW_DIR / "prices.csv"
META_JSON = RAW_DIR / "prices_meta.json"

DEFAULT_EXTRA_TICKERS = ["SPY", "^VIX"]  # 레짐용


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has columns:
    Date, Ticker, Open, High, Low, Close, AdjClose, Volume
    - If Date is index, reset it.
    - If columns are MultiIndex, flatten.
    - Strip column names, fix common variants.
    """
    if df is None:
        return pd.DataFrame()

    df = df.copy()

    # If columns are MultiIndex, flatten (take first level)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Strip column names
    df.columns = [str(c).strip() for c in df.columns]

    # If Date is not a column but index looks like dates, reset
    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex) or df.index.name in ("Date", "Datetime"):
            idx_name = df.index.name or "Date"
            df = df.reset_index().rename(columns={idx_name: "Date"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "Date"})

    # Ticker name variants
    if "Ticker" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "Ticker"})

    # Adj Close variants
    if "AdjClose" not in df.columns:
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "AdjClose"})
        elif "adjclose" in df.columns:
            df = df.rename(columns={"adjclose": "AdjClose"})
        elif "adj_close" in df.columns:
            df = df.rename(columns={"a
