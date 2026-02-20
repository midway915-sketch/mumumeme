#!/usr/bin/env python3
from __future__ import annotations

# ------------------------------------------------------------
# sys.path guard (avoid "No module named 'scripts'")
# ------------------------------------------------------------
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scripts.feature_spec import (
    get_feature_cols,
    read_feature_cols_meta,
    write_feature_cols_meta,
)

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "features"
META_DIR = DATA_DIR / "meta"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

OUT_PARQ = OUT_DIR / "features_model.parquet"
OUT_CSV = OUT_DIR / "features_model.csv"

MARKET_TICKER = "SPY"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def save_table(df: pd.DataFrame, parq: Path, csv: Path) -> str:
    parq.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parq, index=False)
        return f"parquet:{parq}"
    except Exception:
        df.to_csv(csv, index=False)
        return f"csv:{csv}"


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def compute_atr_ratio(g: pd.DataFrame, n: int = 14) -> pd.Series:
    hi = g["High"].astype(float)
    lo = g["Low"].astype(float)
    cl = g["Close"].astype(float)

    prev = cl.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=2 / (n + 1), adjust=False).mean()
    return (atr / cl.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_series_len(name: str, v, n: int) -> pd.Series:
    """
    DataFrame 생성 시 'All arrays must be of the same length' 방지용.
    - scalar -> n개로 broadcast
    - Series -> reset_index(drop=True)
    - array/list -> 길이 다르면 경고 후 pad/truncate
    """
    if isinstance(v, pd.Series):
        s = v.reset_index(drop=True)
        if len(s) != n:
            print(f"[WARN] {name}: series len mismatch {len(s)} != {n} -> pad/truncate")
            arr = s.to_numpy()
            if len(arr) > n:
                arr = arr[:n]
            else:
                pad = np.full(n - len(arr), np.nan)
                arr = np.concatenate([arr, pad])
            return pd.Series(arr)
        return s

    # scalar
    if np.isscalar(v):
        return pd.Series([v] * n)

    # array-like
    arr = np.asarray(v)
    if arr.ndim == 0:
        return pd.Series([float(arr)] * n)
    if len(arr) != n:
        print(f"[WARN] {name}: array len mismatch {len(arr)} != {n} -> pad/truncate")
        if len(arr) > n:
            arr = arr[:n]
        else:
            pad = np.full(n - len(arr), np.nan)
            arr = np.concatenate([arr, pad])
    return pd.Series(arr)


def compute_ticker_features(g: pd.DataFrame, market_map: dict) -> pd.DataFrame:
    g = g.sort_values("Date").reset_index(drop=True)

    dt = g["Date"]
    c = g["Close"].astype(float)

    roll_max_252 = c.rolling(252, min_periods=1).max()
    drawdown_252 = (c / roll_max_252 - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    roll_max_60 = c.rolling(60, min_periods=1).max()
    drawdown_60 = (c / roll_max_60 - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    atr_ratio = compute_atr_ratio(g, n=14)

    ma20 = c.rolling(20, min_periods=1).mean()
    ma20_slope = ((ma20 - ma20.shift(5)) / 5.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = (macd - signal).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    z = ((c - c.rolling(20, min_periods=1).mean()) / c.rolling(20, min_periods=1).std(ddof=0)).replace(
        [np.inf, -np.inf], np.nan
    )
    z_score = z.fillna(0.0)

    # market mapped (same length by construction)
    market_drawdown = g["Date"].map(market_map.get("Market_Drawdown", {})).astype(float)
    market_atr_ratio = g["Date"].map(market_map.get("Market_ATR_ratio", {})).astype(float)

    # new features
    ret_5 = c.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_10 = c.pct_change(10).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_20 = c.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    breakout_20 = (c >= c.rolling(20, min_periods=1).max().shift(1)).astype(int)
    vol = g["Volume"].astype(float) if "Volume" in g.columns else pd.Series([0.0] * len(g))
    vol_surge = (vol >= vol.rolling(20, min_periods=1).mean() * 1.5).astype(int)

    ema20 = c.ewm(span=20, adjust=False).mean()
    ema60 = c.ewm(span=60, adjust=False).mean()
    trend_align = (ema20 > ema60).astype(int)

    # beta_60 vs market_ret_1d
    ret_1d = c.pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mret_1d = g["Date"].map(market_map.get("Market_ret_1d", {})).astype(float).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    cov = ret_1d.rolling(60, min_periods=10).cov(mret_1d)
    var = mret_1d.rolling(60, min_periods=10).var(ddof=0)
    beta_60 = (cov / var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ret_score (simple)
    ret_score = ret_20.copy()

    # ---- SAFE DF build (prevents length mismatch crash)
    n = len(g)
    data = {
        "Date": dt,
        "Ticker": g["Ticker"],
        "Drawdown_252": drawdown_252,
        "Drawdown_60": drawdown_60,
        "ATR_ratio": atr_ratio,
        "Z_score": z_score,
        "MACD_hist": macd_hist,
        "MA20_slope": ma20_slope,
        "Market_Drawdown": market_drawdown,
        "Market_ATR_ratio": market_atr_ratio,
        "ret_score": ret_score,
        "ret_5": ret_5,
        "ret_10": ret_10,
        "ret_20": ret_20,
        "breakout_20": breakout_20,
        "vol_surge": vol_surge,
        "trend_align": trend_align,
        "beta_60": beta_60,
    }

    out = pd.DataFrame({k: _safe_series_len(k, v, n) for k, v in data.items()})
    return out


def compute_market_features(prices: pd.DataFrame, market_ticker: str = "SPY") -> dict:
    m = prices.loc[prices["Ticker"] == market_ticker].copy()
    m = m.sort_values("Date").reset_index(drop=True)
    if m.empty:
        return {"Market_Drawdown": {}, "Market_ATR_ratio": {}, "Market_ret_1d": {}}

    c = m["Close"].astype(float)
    roll_max = c.rolling(252, min_periods=1).max()
    dd = (c / roll_max - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    g = m.rename(columns={"High": "High", "Low": "Low", "Close": "Close"})
    atr_ratio = compute_atr_ratio(g, n=14)

    ret_1d = c.pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    d = m["Date"].tolist()
    return {
        "Market_Drawdown": dict(zip(d, dd.tolist())),
        "Market_ATR_ratio": dict(zip(d, atr_ratio.tolist())),
        "Market_ret_1d": dict(zip(d, ret_1d.tolist())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enable-sector-strength", action="store_true")
    args = ap.parse_args()

    prices = read_table(PRICES_PARQ, PRICES_CSV).copy()
    need = {"Date", "Ticker", "Open", "High", "Low", "Close"}
    miss = [c for c in need if c not in prices.columns]
    if miss:
        raise ValueError(f"prices missing columns: {miss}")

    prices["Date"] = norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices = prices.dropna(subset=["Date", "Ticker", "High", "Low", "Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    market_map = compute_market_features(prices, market_ticker=MARKET_TICKER)

    # feature cols SSOT
    feature_cols = get_feature_cols(sector_enabled=bool(args.enable_sector_strength))
    write_feature_cols_meta(feature_cols, sector_enabled=bool(args.enable_sector_strength))

    out = []
    for tkr, g in prices.groupby("Ticker", sort=False):
        out.append(compute_ticker_features(g, market_map))

    feats = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if feats.empty:
        raise RuntimeError("No features produced.")

    saved = save_table(feats, OUT_PARQ, OUT_CSV)

    meta = {
        "updated_at_utc": now_utc_iso(),
        "rows": int(len(feats)),
        "min_date": str(feats["Date"].min().date()),
        "max_date": str(feats["Date"].max().date()),
        "saved_to": saved,
        "market_ticker": MARKET_TICKER,
        "sector_enabled": bool(args.enable_sector_strength),
        "feature_cols": feature_cols,
    }
    META_DIR.mkdir(parents=True, exist_ok=True)
    (META_DIR / "features_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] saved={saved} rows={meta['rows']} range={meta['min_date']}..{meta['max_date']}")


if __name__ == "__main__":
    main()