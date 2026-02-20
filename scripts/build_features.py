#!/usr/bin/env python3
# scripts/build_features.py
from __future__ import annotations

# ✅ FIX: "python scripts/xxx.py"로 실행될 때도 scripts.* import가 되도록 repo root를 sys.path에 추가
import sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from scripts.feature_spec import (
    get_feature_cols,
    write_feature_cols_meta,
)

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
UNIVERSE_CSV = DATA_DIR / "universe.csv"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

OUT_PARQ = FEAT_DIR / "features_model.parquet"
OUT_CSV = FEAT_DIR / "features_model.csv"

MARKET_TICKER = "SPY"   # market proxy (must exist in raw prices)


# -----------------------------
# IO helpers
# -----------------------------
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

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"prices missing required column: {c}")

    df = df.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def read_universe_groups_strict() -> dict[str, str]:
    """
    ✅ sector features를 '무조건' 쓰려면 Group 매핑이 필수라서,
    universe.csv 없거나 Group 컬럼 없으면 여기서 바로 에러.
    """
    if not UNIVERSE_CSV.exists():
        raise FileNotFoundError(f"Missing {UNIVERSE_CSV}. Sector features require universe.csv with Group column.")

    uni = pd.read_csv(UNIVERSE_CSV)
    if "Ticker" not in uni.columns or "Group" not in uni.columns:
        raise ValueError("universe.csv must include columns: Ticker, Group (required for sector features).")

    uni = uni.copy()
    uni["Ticker"] = uni["Ticker"].astype(str).str.upper().str.strip()
    uni["Group"] = uni["Group"].astype(str).str.strip()
    uni = uni.dropna(subset=["Ticker", "Group"])

    m = dict(zip(uni["Ticker"].tolist(), uni["Group"].tolist()))
    if not m:
        raise ValueError("universe.csv Group mapping is empty. Sector features require non-empty Group mapping.")
    return m


# -----------------------------
# feature building blocks
# -----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_atr_ratio(g: pd.DataFrame, n: int = 14) -> pd.Series:
    high = pd.to_numeric(g["High"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(g["Low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(g["Close"], errors="coerce").to_numpy(dtype=float)
    prev_close = np.r_[np.nan, close[:-1]]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.nanmax(np.vstack([tr1, tr2, tr3]), axis=0)

    tr_s = pd.Series(tr, index=g.index)
    atr = tr_s.rolling(n, min_periods=n).mean()
    atr_ratio = atr / pd.Series(close, index=g.index)
    return atr_ratio


def compute_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    m = prices.loc[prices["Ticker"] == MARKET_TICKER].sort_values("Date").copy()
    if m.empty:
        raise ValueError(f"Market ticker {MARKET_TICKER} not found. Use fetch_prices.py --include-extra")

    c = pd.to_numeric(m["Close"], errors="coerce")

    roll_max_252 = c.rolling(252, min_periods=252).max()
    mdd = (c / roll_max_252) - 1.0

    atr_ratio = compute_atr_ratio(m, n=14)

    # daily returns for beta
    mret = c.pct_change()

    out = pd.DataFrame({
        "Date": pd.to_datetime(m["Date"], errors="coerce").dt.tz_localize(None).values,
        "Market_Drawdown": mdd.values,
        "Market_ATR_ratio": atr_ratio.values,
        "Market_ret_1d": mret.values,
    }).sort_values("Date").reset_index(drop=True)

    return out


def compute_ticker_features(g: pd.DataFrame, market_ret_by_date: pd.Series) -> pd.DataFrame:
    g = g.sort_values("Date").copy()
    dt = pd.to_datetime(g["Date"], errors="coerce").dt.tz_localize(None)

    c = pd.to_numeric(g["Close"], errors="coerce")
    v = pd.to_numeric(g["Volume"], errors="coerce")

    roll_max_252 = c.rolling(252, min_periods=252).max()
    dd_252 = (c / roll_max_252) - 1.0

    roll_max_60 = c.rolling(60, min_periods=60).max()
    dd_60 = (c / roll_max_60) - 1.0

    atr_ratio = compute_atr_ratio(g, n=14)

    ma20 = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std(ddof=0)
    z = (c - ma20) / std20

    ema12 = ema(c, 12)
    ema26 = ema(c, 26)
    macd = ema12 - ema26
    signal = ema(macd, 9)
    macd_hist = macd - signal

    ma20_slope = (ma20 / ma20.shift(5)) - 1.0

    ret_20 = (c / c.shift(20)) - 1.0
    ret_score = ret_20.copy()

    ret_5 = (c / c.shift(5)) - 1.0
    ret_10 = (c / c.shift(10)) - 1.0

    roll_max_20 = c.rolling(20, min_periods=20).max()
    breakout_20 = (c / roll_max_20) - 1.0

    vol_ma20 = v.rolling(20, min_periods=20).mean()
    vol_surge = v / vol_ma20

    ema50 = ema(c, 50)
    trend_align = (c / ema50) - 1.0

    r = c.pct_change()
    mret = market_ret_by_date.reindex(dt).astype(float)
    cov = r.rolling(60, min_periods=60).cov(mret)
    var = mret.rolling(60, min_periods=60).var()
    beta_60 = cov / var

    out = pd.DataFrame({
        "Date": dt.values,
        "Ticker": g["Ticker"].values,

        "Drawdown_252": dd_252.values,
        "Drawdown_60": dd_60.values,
        "ATR_ratio": atr_ratio.values,
        "Z_score": z.values,
        "MACD_hist": macd_hist.values,
        "MA20_slope": ma20_slope.values,
        "ret_score": ret_score.values,

        "ret_5": ret_5.values,
        "ret_10": ret_10.values,
        "ret_20": ret_20.values,
        "breakout_20": breakout_20.values,
        "vol_surge": vol_surge.values,
        "trend_align": trend_align.values,
        "beta_60": beta_60.values,

        "Volume": v.values,
        "Close": c.values,
    })

    return out


def add_sector_strength_strict(feats: pd.DataFrame, ticker_to_group: dict[str, str]) -> pd.DataFrame:
    """
    ✅ sector features 필수:
      - Group 매핑이 없으면 이미 read_universe_groups_strict()에서 실패
      - 계산 후 Sector_Ret_20 / RelStrength가 안 생기면 여기서도 실패
    """
    if "ret_20" not in feats.columns:
        raise RuntimeError("feats missing ret_20 -> cannot build Sector_Ret_20 / RelStrength")

    x = feats.copy()
    x["Group"] = x["Ticker"].map(ticker_to_group).fillna("UNKNOWN")

    sector_ret = (
        x.groupby(["Date", "Group"], as_index=False)["ret_20"]
        .mean()
        .rename(columns={"ret_20": "Sector_Ret_20"})
    )
    x = x.merge(sector_ret, on=["Date", "Group"], how="left")
    x["RelStrength"] = pd.to_numeric(x["ret_20"], errors="coerce") - pd.to_numeric(x["Sector_Ret_20"], errors="coerce")

    if "Sector_Ret_20" not in x.columns or "RelStrength" not in x.columns:
        raise RuntimeError("Sector features not created as expected (Sector_Ret_20/RelStrength missing).")

    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", type=str, default=None, help="only output rows with Date >= start-date (YYYY-MM-DD)")
    ap.add_argument("--max-window", type=int, default=260, help="max rolling window used (controls lookback)")
    ap.add_argument("--buffer-days", type=int, default=40, help="extra days added to lookback for safety")
    ap.add_argument("--min-volume", type=float, default=0.0, help="optional: drop rows with Volume < min-volume")

    # ✅ 섹터 피처는 '무조건' 사용 (옵션 제거/무시)
    args = ap.parse_args()

    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    prices = read_prices()

    start_date = None
    if args.start_date:
        start_date = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(start_date):
            raise ValueError(f"Invalid --start-date: {args.start_date}")

        lookback_days = int(args.max_window + args.buffer_days)
        compute_start = start_date - pd.Timedelta(days=lookback_days)
        prices = prices.loc[prices["Date"] >= compute_start].copy()

    market = compute_market_features(prices)
    market = market.sort_values("Date").reset_index(drop=True)
    market_ret_by_date = pd.Series(
        market["Market_ret_1d"].values,
        index=pd.to_datetime(market["Date"]).dt.tz_localize(None),
    )

    feats_list = []
    for _t, g in prices.groupby("Ticker", sort=False):
        feats_list.append(compute_ticker_features(g, market_ret_by_date=market_ret_by_date))
    feats = pd.concat(feats_list, ignore_index=True)

    feats = feats.copy()
    feats["Date"] = pd.to_datetime(feats["Date"], errors="coerce").dt.tz_localize(None)
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()

    market_merge_cols = ["Date", "Market_Drawdown", "Market_ATR_ratio"]
    feats = feats.merge(market[market_merge_cols], on="Date", how="left")

    if args.min_volume and args.min_volume > 0:
        feats = feats.loc[pd.to_numeric(feats["Volume"], errors="coerce") >= float(args.min_volume)].copy()

    # ✅ sector features are REQUIRED
    ticker_to_group = read_universe_groups_strict()
    feats = add_sector_strength_strict(feats, ticker_to_group=ticker_to_group)

    # ✅ 무조건 True
    sector_enabled = True

    # ✅ SSOT 피처셋(섹터 포함)
    FEATURE_COLS = get_feature_cols(sector_enabled=sector_enabled)

    # dropna (strict)
    feats = feats.dropna(subset=FEATURE_COLS + ["Date", "Ticker"]).copy()

    if start_date is not None:
        feats = feats.loc[pd.to_datetime(feats["Date"], errors="coerce") >= start_date].copy()

    feats = feats.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last").reset_index(drop=True)

    # save
    try:
        feats.to_parquet(OUT_PARQ, index=False)
    except Exception as e:
        print(f"[WARN] parquet save failed ({e}) -> writing csv only")
    feats.to_csv(OUT_CSV, index=False)

    # ✅ FIX: feature_spec.py 시그니처에 맞게 cols= 로 호출
    meta_path = write_feature_cols_meta(cols=FEATURE_COLS, sector_enabled=sector_enabled)

    print(f"[DONE] wrote: {OUT_PARQ} rows={len(feats)}")
    if len(feats):
        dmin = pd.to_datetime(feats["Date"]).min().date()
        dmax = pd.to_datetime(feats["Date"]).max().date()
        print(f"[INFO] range: {dmin}..{dmax}")
        print(f"[INFO] sector strength: ENABLED (required)")
        print(f"[INFO] feature_cols_meta: {meta_path}")
        print(f"[INFO] feature_cols({len(FEATURE_COLS)}): {FEATURE_COLS}")


if __name__ == "__main__":
    main()