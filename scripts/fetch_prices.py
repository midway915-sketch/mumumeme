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
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-base", type=float, default=1.2)
    parser.add_argument("--include-extra", action="store_true", help="Also fetch market tickers (SPY, ^VIX)")
    args = parser.parse_args()

    # Universe tickers
    tickers = read_universe(UNIVERSE_CSV)

    # ✅ include-extra는 args 파싱 후 여기서만 적용
    if args.include_extra:
        tickers = sorted(set(tickers) | set(DEFAULT_EXTRA_TICKERS))
    else:
        tickers = sorted(set(tickers))

    print(f"[INFO] tickers={len(tickers)} sample={preview_list(tickers)}")

    if args.reset:
        for p in [OUT_PARQUET, OUT_CSV_FALLBACK, META_JSON]:
            if p.exists():
                p.unlink()
        print("[INFO] reset done")

    existing = pd.DataFrame()
    if not args.force_full:
        existing = load_existing_prices()

    last_dates = {}
    if not existing.empty:
        last_dates = existing.groupby("Ticker")["Date"].max().to_dict()

    downloads = []
    failed = []

    for t in tickers:
        try:
            if args.force_full or existing.empty or t not in last_dates:
                start = args.start
            else:
                ld = pd.to_datetime(last_dates[t])
                start_dt = (ld - pd.Timedelta(days=args.lookback_days)).date()
                start = args.start or str(start_dt)

            df_new = safe_download_one(
                ticker=t,
                start=start,
                end=args.end,
                retries=args.retries,
                sleep_base=args.sleep_base,
            )
            downloads.append(df_new)
            print(f"[OK] {t}: {df_new['Date'].min().date()} -> {df_new['Date'].max().date()} rows={len(df_new)}")

        except Exception as e:
            failed.append((t, str(e)))
            print(f"[FAIL] {t}: {e}")

    if not downloads:
        raise RuntimeError("No data downloaded.")

    new_all = normalize_schema(pd.concat(downloads, ignore_index=True))

    if existing.empty:
        combined = new_all
    else:
        combined = normalize_schema(pd.concat([existing, new_all], ignore_index=True))

    combined = (
        combined.sort_values(["Date", "Ticker"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    saved_to = save_prices(combined)

    downloaded = sorted(set(new_all["Ticker"].unique().tolist()))
    meta = {
        "updated_at_utc": now_utc_iso(),
        "saved_to": saved_to,
        "include_extra": bool(args.include_extra),
        "extra_tickers": DEFAULT_EXTRA_TICKERS if args.include_extra else [],
        "tickers_requested_count": int(len(tickers)),
        "tickers_requested_sample": tickers[:50],
        "tickers_downloaded_count": int(len(downloaded)),
        "tickers_downloaded": downloaded,
        "has_SPY": ("SPY" in set(downloaded)) or ("SPY" in set(combined["Ticker"].unique())),
        "has_VIX": ("^VIX" in set(downloaded)) or ("^VIX" in set(combined["Ticker"].unique())),
        "min_date": str(combined["Date"].min().date()) if not combined.empty else None,
        "max_date": str(combined["Date"].max().date()) if not combined.empty else None,
        "rows": int(len(combined)),
        "failed": [{"ticker": t, "error": err} for t, err in failed],
        "force_full": bool(args.force_full),
        "reset": bool(args.reset),
        "start_arg": args.start,
        "end_arg": args.end,
        "lookback_days": int(args.lookback_days),
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] saved={saved_to} rows={len(combined)} range={meta['min_date']}..{meta['max_date']}")
    print(f"[CHECK] has_SPY={meta['has_SPY']} has_^VIX={meta['has_VIX']}")

    # 너무 많이 실패했으면 fail-fast
    if failed and len(failed) >= max(3, len(tickers) // 2):
        raise RuntimeError(f"Too many ticker download failures: {len(failed)}/{len(tickers)}")


if __name__ == "__main__":
    main()
