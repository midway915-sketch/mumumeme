# scripts/build_labels.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEATURE_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"

PRICES_PARQUET = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"
FEATURES_PARQUET = FEATURE_DIR / "features.parquet"
FEATURES_CSV = FEATURE_DIR / "features.csv"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_tag(pt: float, h: int, sl: float) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{h}_sl{sl_tag}"


def load_prices() -> pd.DataFrame:
    if PRICES_PARQUET.exists():
        df = pd.read_parquet(PRICES_PARQUET)
    elif PRICES_CSV.exists():
        df = pd.read_csv(PRICES_CSV)
    else:
        raise FileNotFoundError("No prices found. Run scripts/fetch_prices.py first.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def load_feature_index() -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
    """
    labels는 'features에 존재하는 티커/기간'만 만들면 깔끔함(eligibility 반영).
    """
    if FEATURES_PARQUET.exists():
        f = pd.read_parquet(FEATURES_PARQUET, columns=["Date", "Ticker"])
    elif FEATURES_CSV.exists():
        f = pd.read_csv(FEATURES_CSV, usecols=["Date", "Ticker"])
    else:
        raise FileNotFoundError("features not found. Run scripts/build_features.py first.")

    f["Date"] = pd.to_datetime(f["Date"])
    f["Ticker"] = f["Ticker"].astype(str).str.upper().str.strip()
    tickers = sorted(f["Ticker"].unique().tolist())
    return tickers, f["Date"].min(), f["Date"].max()


def build_labels_for_ticker(
    d: pd.DataFrame,
    profit_target: float,
    holding_days: int,
    stop_level: float,
    rule: str = "profit_before_stop",
) -> pd.DataFrame:
    """
    d: one ticker price DF sorted by Date
    rule:
      - profit_before_stop: PT 먼저 터치하면 1, SL 먼저면 0, 둘 다 없으면 0
      - profit_only: PT 터치만 보면 1/0 (SL 무시)
    """
    d = d.sort_values("Date").reset_index(drop=True).copy()

    # raw는 채우지 않는다: 필요한 컬럼 결측 있으면 라벨 계산 불가
    d = d.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if d.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "Success"])

    close = d["Close"].astype(float).to_numpy()
    high = d["High"].astype(float).to_numpy()
    low = d["Low"].astype(float).to_numpy()

    n = len(d)
    success = np.full(n, np.nan, dtype=float)
    profit_day = np.full(n, np.nan, dtype=float)
    stop_day = np.full(n, np.nan, dtype=float)
    fmax_ret = np.full(n, np.nan, dtype=float)
    fmin_ret = np.full(n, np.nan, dtype=float)

    stop_enabled = stop_level > -0.98  # -0.99 같은 값은 "
