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

# 시장 레짐용으로 최소 SPY는 강추. VIX는 옵션이지만 있으면 좋음.
DEFAULT_EXTRA_TICKERS = ["SPY", "^VIX"]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_universe(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path} (run scripts/universe.py first)")

    uni = pd.read_csv(path)
    if "Ticker" not in uni.columns:
        raise ValueError("universe.csv must contain 'Ticker' column")
    if "Enabled" in uni.columns:
        uni = uni[uni["Enabled"] == True]  # noqa: E712

    tickers = (
        uni["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )
    return tickers


def safe_download_one(
    ticker: str,
    start: str | None,
    end: str | None,
    retries: int = 3,
    sleep_base: float = 1.2,
) -> pd.DataFrame:
    """
    Download OHLCV for a single ticker using yfinance.
    Returns a DataFrame indexed by Date with columns: Open, High, Low, Close, Adj Close, Volume
    """
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

            # yfinance index can be tz-aware DatetimeIndex; normalize to date
            df = df.copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"

            # Standardize columns
            # yfinance uses "Adj Close"
            expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            missing = [c for c in expected if c not in df.columns]
            if missing:
                raise ValueError(f"{ticker} missing columns: {missing}")

            df = df[expected].rename(columns={"Adj Close": "AdjClose"}).reset_index()
            df.insert(1, "Ticker", ticker)
            return df

        except Exception as e:
            last_err = e
            # backoff
            time.sleep(sleep_base * attempt)

    raise RuntimeError(f"Failed to download {ticker} after {retries} retries: {last_err}")


def load_existing_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"])

    try:
        df = pd.read_parquet(path)
    except Exception:
        # fallback: try CSV if parquet read fails
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["Date"])
        else:
            raise

    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df


def save_prices(df: pd.DataFrame, parquet_path: Path, csv_fallback_path: Path) -> str:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Try parquet first
    try:
        df.to_parquet(parquet_path, index=False)
        return f"parquet:{parquet_path}"
    except Exception as e:
        print(f"[WARN] Parquet save failed ({e}). Saving CSV fallback: {csv_fallback_path}")
        df.to_csv(csv_fallback_path, index=False)
        return f"csv:{csv_fallback_path}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OHLCV prices from Yahoo Finance (incremental).")
    parser.add_argument("--force-full", action="store_true", help="Ignore cache and download full history.")
    parser.add_argument("--start", type=str, default=None, help="Override start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="Override end date (YYYY-MM-DD).")
    parser.add_argument("--lookback-days", type=int, default=7, help="Overlap days for incremental refresh.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per ticker.")
    parser.add_argument("--sleep-base", type=float, default=1.2, help="Backoff base seconds.")
    parser.add_argument("--include-extra", action="store_true", help="Include default extra tickers (SPY, ^VIX).")
    args = parser.parse_args()

    tickers = read_universe(UNIVERSE_CSV)
    if args.include_extra:
        for t in DEFAULT_EXTRA_TICKERS:
            if t not in tickers:
                tickers.append(t)

    tickers = sorted(set(tickers))
    print(f"[INFO] tickers={len(tickers)} -> {tickers}")

    existing = load_existing_prices(OUT_PARQUET)
    has_existing = not existing.empty

    # Determine per-ticker start dates
    last_dates = {}
    if has_existing:
        grp = existing.groupby("Ticker")["Date"].max()
        last_dates = grp.to_dict()

    downloads = []
    failed = []

    for t in tickers:
        try:
            if args.force_full or not has_existing or t not in last_dates:
                start = args.start  # could be None -> yfinance will fetch max
            else:
                # incremental: overlap lookback-days to refresh adjusted data edges
                ld = pd.to_datetime(last_dates[t])
                start_dt = (ld - pd.Timedelta(days=args.lookback_days)).date()
                start = args.start or str(start_dt)

            end = args.end
            df_new = safe_download_one(
                ticker=t,
                start=start,
                end=end,
                retries=args.retries,
                sleep_base=args.sleep_base,
            )
            downloads.append(df_new)
            print(f"[OK] {t}: {df_new['Date'].min().date()} -> {df_new['Date'].max().date()} rows={len(df_new)}")

        except Exception as e:
            failed.append((t, str(e)))
            print(f"[FAIL] {t}: {e}")

    if not downloads:
        raise RuntimeError("No data downloaded. Check network/yfinance availability.")

    new_all = pd.concat(downloads, ignore_index=True)
    new_all["Date"] = pd.to_datetime(new_all["Date"])
    new_all["Ticker"] = new_all["Ticker"].astype(str).str.upper().str.strip()

    # Merge with existing and de-duplicate
    if has_existing:
        combined = pd.concat([existing, new_all], ignore_index=True)
    else:
        combined = new_all

    combined = combined.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last").reset_index(drop=True)

    # Save
    saved_to = save_prices(combined, OUT_PARQUET, OUT_CSV_FALLBACK)

    # Meta
    meta = {
        "updated_at_utc": now_utc_iso(),
        "saved_to": saved_to,
        "tickers_requested": tickers,
        "tickers_downloaded": sorted(set(new_all["Ticker"].unique().tolist())),
        "min_date": str(combined["Date"].min().date()) if not combined.empty else None,
        "max_date": str(combined["Date"].max().date()) if not combined.empty else None,
        "rows": int(len(combined)),
        "failed": [{"ticker": t, "error": err} for t, err in failed],
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] saved={saved_to} rows={len(combined)} range={meta['min_date']}..{meta['max_date']}")
    if failed:
        print("[WARN] Some tickers failed:")
        for t, err in failed:
            print(f"  - {t}: {err}")
        # 실패가 너무 많으면 파이프라인이 조용히 망가지는 걸 막기 위해, 절반 이상 실패 시 에러 처리
        if len(failed) >= max(3, len(tickers) // 2):
            raise RuntimeError(f"Too many ticker download failures: {len(failed)}/{len(tickers)}")


if __name__ == "__main__":
    main()
