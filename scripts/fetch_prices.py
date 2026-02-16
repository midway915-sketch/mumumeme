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

# build_features.py가 요구하는 시장 티커들
DEFAULT_EXTRA_TICKERS = ["SPY", "^VIX"]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def preview_list(xs: list[str], n: int = 25) -> str:
    xs = list(xs)
    if len(xs) <= n:
        return str(xs)
    return str(xs[:n] + ["..."])


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has columns:
    Date, Ticker, Open, High, Low, Close, AdjClose, Volume
    """
    if df is None:
        return pd.DataFrame()

    df = df.copy()

    # MultiIndex columns -> flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.columns = [str(c).strip() for c in df.columns]

    # Date column normalize
    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex) or df.index.name in ("Date", "Datetime"):
            idx_name = df.index.name or "Date"
            df = df.reset_index().rename(columns={idx_name: "Date"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "Date"})

    # Ticker normalize
    if "Ticker" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "Ticker"})

    # Adj Close normalize
    if "AdjClose" not in df.columns:
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "AdjClose"})
        elif "adjclose" in df.columns:
            df = df.rename(columns={"adjclose": "AdjClose"})
        elif "adj_close" in df.columns:
            df = df.rename(columns={"adj_close": "AdjClose"})

    needed = {"Date", "Ticker"}
    if not needed.issubset(set(df.columns)):
        raise RuntimeError(f"[SCHEMA] Missing {needed - set(df.columns)}. Columns={list(df.columns)[:30]}")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df


def read_universe(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path} (run scripts/universe.py first)")

    uni = pd.read_csv(path)
    if "Ticker" not in uni.columns:
        raise ValueError("universe.csv must contain 'Ticker' column")

    if "Enabled" in uni.columns:
        uni = uni[uni["Enabled"] == True]  # noqa: E712

    tickers = (
        uni["Ticker"].astype(str).str.upper().str.strip().dropna().unique().tolist()
    )
    return tickers


def safe_download_one(
    ticker: str,
    start: str | None,
    end: str | None,
    retries: int = 3,
    sleep_base: float = 1.2,
) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                actions=False,
                threads=False,
            )

            if df is None or df.empty:
                raise ValueError(f"Empty data for {ticker} (start={start}, end={end})")

            df = df.copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"

            expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            missing = [c for c in expected if c not in df.columns]
            if missing:
                df.columns = [str(c).strip() for c in df.columns]
                missing = [c for c in expected if c not in df.columns]
                if missing:
                    raise ValueError(f"{ticker} missing columns: {missing} (cols={list(df.columns)})")

            df = df[expected].rename(columns={"Adj Close": "AdjClose"}).reset_index()
            df.insert(1, "Ticker", ticker)
            return normalize_schema(df)

        except Exception as e:
            last_err = e
            time.sleep(sleep_base * attempt)

    raise RuntimeError(f"Failed to download {ticker} after {retries} retries: {last_err}")


def load_existing_prices() -> pd.DataFrame:
    if OUT_PARQUET.exists():
        try:
            return normalize_schema(pd.read_parquet(OUT_PARQUET))
        except Exception as e:
            print(f"[WARN] read_parquet failed: {e}")

    if OUT_CSV_FALLBACK.exists():
        return normalize_schema(pd.read_csv(OUT_CSV_FALLBACK))

    return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"])


def save_prices(df: pd.DataFrame) -> str:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = normalize_schema(df)

    try:
        df.to_parquet(OUT_PARQUET, index=False)
        return f"parquet:{OUT_PARQUET}"
    except Exception as e:
        print(f"[WARN] Parquet save failed ({e}). Saving CSV fallback: {OUT_CSV_FALLBACK}")
        df.to_csv(OUT_CSV_FALLBACK, index=False)
        return f"csv:{OUT_CSV_FALLBACK}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OHLCV prices from Yahoo Finance (incremental).")
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argumen_
