# scripts/build_features.py
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

UNIVERSE_CSV = DATA_DIR / "universe.csv"
PRICES_PARQUET = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

OUT_PARQUET = FEATURE_DIR / "features.parquet"
OUT_CSV_FALLBACK = FEATURE_DIR / "features.csv"
META_JSON = FEATURE_DIR / "features_meta.json"

MARKET_TICKER = "SPY"
VIX_TICKER = "^VIX"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_prices() -> pd.DataFrame:
    if PRICES_PARQUET.exists():
        df = pd.read_parquet(PRICES_PARQUET)
    elif PRICES_CSV.exists():
        df = pd.read_csv(PRICES_CSV)
    else:
        raise FileNotFoundError("No prices found. Run scripts/fetch_prices.py first.")

    # normalize
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    required = {"Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prices missing columns: {missing}. cols={list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    # raw prices는 채우지 않음: 그냥 정렬만
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def load_universe() -> pd.DataFrame:
    if not UNIVERSE_CSV.exists():
        raise FileNotFoundError("universe.csv not found. Run scripts/universe.py first.")

    uni = pd.read_csv(UNIVERSE_CSV)
    if "Ticker" not in uni.columns:
        raise ValueError("universe.csv must contain Ticker column")

    uni["Ticker"] = uni["Ticker"].astype(str).str.upper().str.strip()

    if "Enabled" in uni.columns:
        uni = uni[uni["Enabled"] == True]  # noqa: E712

    # defaults if missing
    if "MinHistoryDays" not in uni.columns:
        uni["MinHistoryDays"] = 756
    if "MinAvgDollarVol20" not in uni.columns:
        uni["MinAvgDollarVol20"] = 2_000_000

    return uni.reset_index(drop=True)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder smoothing (EMA with alpha=1/window)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    sig = macd_line.ewm(span=signal, adjust=False, min_periods=slow + signal).mean()
    return macd_line - sig


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(window=window, min_periods=window).mean()
    sd = close.rolling(window=window, min_periods=window).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    return ma, upper, lower


def rolling_beta(x: pd.Series, m: pd.Series, window: int = 60) -> pd.Series:
    # beta = cov(x, m) / var(m)
    mean_x = x.rolling(window, min_periods=window).mean()
    mean_m = m.rolling(window, min_periods=window).mean()
    cov = (x * m).rolling(window, min_periods=window).mean() - mean_x * mean_m
    var_m = (m * m).rolling(window, min_periods=window).mean() - mean_m * mean_m
    return cov / var_m.replace(0, np.nan)


def compute_market_frames(prices: pd.DataFrame, max_window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - market_feat_by_date: Date-indexed market features from SPY
      - vix_by_date: Date-indexed VIX Close + changes
    """
    # Market (SPY)
    mkt = prices[prices["Ticker"] == MARKET_TICKER].copy()
    if mkt.empty:
        raise ValueError(f"Market ticker {MARKET_TICKER} not found. Use fetch_prices.py --include-extra")

    mkt = mkt.sort_values("Date").reset_index(drop=True)
    m_close = mkt["AdjClose"].astype(float)

    m_ret1 = m_close.pct_change(1)
    m_ret5 = m_close.pct_change(5)
    m_ret20 = m_close.pct_change(20)
    m_ret60 = m_close.pct_change(60)

    m_atr14 = atr(mkt["High"].astype(float), mkt["Low"].astype(float), mkt["Close"].astype(float), 14)
    m_atr_ratio = m_atr14 / mkt["Close"].astype(float)

    m_roll_max_252 = m_close.rolling(252, min_periods=252).max()
    m_dd252 = m_close / m_roll_max_252 - 1.0
    m_roll_max_60 = m_close.rolling(60, min_periods=60).max()
    m_dd60 = m_close / m_roll_max_60 - 1.0

    m_rvol20 = m_ret1.rolling(20, min_periods=20).std() * np.sqrt(252)

    market_feat = pd.DataFrame(
        {
            "Date": mkt["Date"],
            "Market_ret_1d": m_ret1,
            "Market_ret_5d": m_ret5,
            "Market_ret_20d": m_ret20,
            "Market_ret_60d": m_ret60,
            "Market_ATR_ratio": m_atr_ratio,
            "Market_Drawdown_252": m_dd252,
            "Market_Drawdown_60": m_dd60,
            "Market_realized_vol_20": m_rvol20,
        }
    )

    # VIX
    vix = prices[prices["Ticker"] == VIX_TICKER].copy()
    if vix.empty:
        # VIX 없이도 돌아가게는 하되, 컬럼은 NaN으로
        vix_feat = pd.DataFrame({"Date": market_feat["Date"], "VIX": np.nan, "VIX_chg_5d": np.nan})
    else:
        vix = vix.sort_values("Date").reset_index(drop=True)
        v = vix["Close"].astype(float)
        vix_feat = pd.DataFrame(
            {
                "Date": vix["Date"],
                "VIX": v,
                "VIX_chg_5d": v.pct_change(5),
            }
        )

    # Date 정렬
    market_feat = market_feat.sort_values("Date").reset_index(drop=True)
    vix_feat = vix_feat.sort_values("Date").reset_index(drop=True)

    return market_feat, vix_feat


def eligibility_filter(prices: pd.DataFrame, universe: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """
    MinHistoryDays / MinAvgDollarVol20 기반으로 제외할 티커를 걸러냄.
    raw 가격은 채우지 않음. 결측이 많은 종목은 여기서 자연스럽게 탈락.
    """
    reasons = []
    eligible = []

    overall_max_date = prices["Date"].max()

    for _, row in universe.iterrows():
        t = row["Ticker"]
        min_hist = int(row.get("MinHistoryDays", 756))
        min_dv = float(row.get("MinAvgDollarVol20", 2_000_000))

        d = prices[prices["Ticker"] == t].copy()
        d = d.sort_values("Date")

        # raw 결측은 채우지 않음: Close/Volume 없으면 그 행은 “거래 불가”로 보고 제외
        d_valid = d.dropna(subset=["Close", "Volume"])

        if len(d_valid) < min_hist:
            reasons.append({"Ticker": t, "Reason": "min_history", "Have": int(len(d_valid)), "Need": min_hist})
            continue

        # 최근 20일 평균 거래대금
        dv20 = (d_valid["Close"].astype(float) * d_valid["Volume"].astype(float)).tail(20).mean()
        if not np.isfinite(dv20) or dv20 < min_dv:
            reasons.append({"Ticker": t, "Reason": "min_dollar_vol20", "Have": float(dv20), "Need": min_dv})
            continue

        # 데이터가 너무 오래 끊긴 티커는 제외(옵션인데 실전성 때문에 넣음)
        last_date = d_valid["Date"].max()
        if (overall_max_date - last_date).days > 14:
            reasons.append({"Ticker": t, "Reason": "stale_data", "LastDate": str(last_date.date())})
            continue

        eligible.append(t)

    return eligible, reasons


def build_features_for_ticker(df: pd.DataFrame, market_feat: pd.DataFrame, vix_feat: pd.DataFrame) -> pd.DataFrame:
    """
    단일 티커 피처 생성. raw 가격은 채우지 않고, 계산 결과 NaN은 나중에 drop.
    """
    d = df.sort_values("Date").reset_index(drop=True).copy()

    close = d["AdjClose"].astype(float)  # 수익률은 adj가 더 안정적
    px = d["Close"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    open_ = d["Open"].astype(float)
    vol = d["Volume"].astype(float)

    ret1 = close.pct_change(1)
    ret5 = close.pct_change(5)
    ret10 = close.pct_change(10)
    ret20 = close.pct_change(20)
    ret60 = close.pct_change(60)

    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    ma200 = close.rolling(200, min_periods=200).mean()

    ma20_slope = ma20.pct_change(5)
    ma60_slope = ma60.pct_change(10)

    std20 = close.rolling(20, min_periods=20).std()
    z20 = (close - ma20) / std20

    atr14 = atr(high, low, px, 14)
    atr_ratio = atr14 / px

    rvol20 = ret1.rolling(20, min_periods=20).std() * np.sqrt(252)
    down_vol20 = ret1.where(ret1 < 0).rolling(20, min_periods=20).std() * np.sqrt(252)

    # Drawdown (현재 drawdown)
    roll_max_252 = close.rolling(252, min_periods=252).max()
    dd252 = close / roll_max_252 - 1.0
    roll_max_60 = close.rolling(60, min_periods=60).max()
    dd60 = close / roll_max_60 - 1.0

    # Rolling max drawdown
    # (drawdown series 대비 rolling min)
    dd_series = close / close.cummax() - 1.0
    maxdd_60 = dd_series.rolling(60, min_periods=60).min()
    maxdd_252 = dd_series.rolling(252, min_periods=252).min()

    # Bollinger
    bb_ma, bb_up, bb_low = bollinger(close, 20, 2.0)
    bb_width = (bb_up - bb_low) / bb_ma
    bb_pos = (close - bb_low) / (bb_up - bb_low).replace(0, np.nan)

    # MACD hist
    macd_h = macd_hist(close)

    # RSI
    rsi14 = rsi(close, 14)

    # Candles / gap
    prev_close = px.shift(1)
    gap = open_ / prev_close - 1.0
    true_range = (high - low).abs()
    rng = true_range / px
    body = (px - open_).abs() / px
    upper_wick = (high - np.maximum(px, open_)) / px
    lower_wick = (np.minimum(px, open_) - low) / px
    close_pos = (px - low) / (high - low).replace(0, np.nan)

    # Volume / liquidity proxies
    dollar_vol = px * vol
    dollar_vol_ma20 = dollar_vol.rolling(20, min_periods=20).mean()
    dollar_vol_ma252 = dollar_vol.rolling(252, min_periods=252).mean()
    dollar_vol_ratio = dollar_vol_ma20 / dollar_vol_ma252

    vol_ma20 = vol.rolling(20, min_periods=20).mean()
    vol_sd20 = vol.rolling(20, min_periods=20).std()
    vol_z20 = (vol - vol_ma20) / vol_sd20

    # OBV & slope
    obv = (np.sign(ret1.fillna(0)) * vol.fillna(0)).cumsum()
    obv_slope20 = obv.pct_change(20)

    # Tail risk
    worst_ret_20 = ret1.rolling(20, min_periods=20).min()
    max_gap_down_60 = gap.rolling(60, min_periods=60).min()

    out = pd.DataFrame(
        {
            "Date": d["Date"],
            "Ticker": d["Ticker"],
            # --- 기존 호환 이름 (너가 쓰던 것) ---
            "Drawdown_252": dd252,
            "Drawdown_60": dd60,
            "ATR_ratio": atr_ratio,
            "Z_score": z20,
            "MACD_hist": macd_h,
            "MA20_slope": ma20_slope,
            # --- 확장 피처 ---
            "ret_1d": ret1,
            "ret_5d": ret5,
            "ret_10d": ret10,
            "ret_20d": ret20,
            "ret_60d": ret60,
            "price_vs_ma20": close / ma20 - 1.0,
            "price_vs_ma60": close / ma60 - 1.0,
            "price_vs_ma200": close / ma200 - 1.0,
            "MA60_slope": ma60_slope,
            "realized_vol_20": rvol20,
            "downside_vol_20": down_vol20,
            "max_drawdown_60": maxdd_60,
            "max_drawdown_252": maxdd_252,
            "bb_width": bb_width,
            "bb_pos": bb_pos,
            "rsi_14": rsi14,
            "gap": gap,
            "range": rng,
            "body": body,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "close_position": close_pos,
            "dollar_vol_ma20": dollar_vol_ma20,
            "dollar_vol_ratio": dollar_vol_ratio,
            "vol_zscore_20": vol_z20,
            "obv_slope_20": obv_slope20,
            "worst_ret_20": worst_ret_20,
            "max_gap_down_60": max_gap_down_60,
        }
    )

    # merge market + vix by Date
    out = out.merge(market_feat, on="Date", how="left")
    out = out.merge(vix_feat, on="Date", how="left")

    # corr/beta with market (using daily returns)
    # (merge 후 계산해도 되지만, 여기서 간단히)
    m = market_feat.set_index("Date")["Market_ret_1d"]
    out = out.set_index("Date")
    out["corr_mkt_60"] = out["ret_1d"].rolling(60, min_periods=60).corr(m.reindex(out.index))
    out["beta_mkt_60"] = rolling_beta(out["ret_1d"], m.reindex(out.index), 60)
    out = out.reset_index()

    return out


def save_features(df: pd.DataFrame) -> str:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(OUT_PARQUET, index=False)
        return f"parquet:{OUT_PARQUET}"
    except Exception as e:
        print(f"[WARN] parquet save failed ({e}). saving csv fallback: {OUT_CSV_FALLBACK}")
        df.to_csv(OUT_CSV_FALLBACK, index=False)
        return f"csv:{OUT_CSV_FALLBACK}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-window", type=int, default=252, help="Max rolling window; implicit warm-up.")
    parser.add_argument("--strict-dropna", action="store_true", help="Drop any row with NaN in feature columns.")
    args = parser.parse_args()

    prices = load_prices()
    universe = load_universe()

    # eligibility filter
    eligible, excluded = eligibility_filter(prices, universe)

    # market frames
    market_feat, vix_feat = compute_market_frames(prices, args.max_window)

    # build per ticker
    feats_all = []
    for t in eligible:
        d = prices[prices["Ticker"] == t].copy()

        # raw 결측은 채우지 않음: 피처 계산 전에 최소한의 유효값만 남김
        d = d.dropna(subset=["Open", "High", "Low", "Close", "AdjClose", "Volume"])
        if d.empty:
            excluded.append({"Ticker": t, "Reason": "all_rows_missing_after_dropna"})
            continue

        f = build_features_for_ticker(d, market_feat, vix_feat)
        feats_all.append(f)

    if not feats_all:
        raise RuntimeError("No eligible tickers produced features. Check eligibility thresholds and raw data.")

    features_df = pd.concat(feats_all, ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # strict NaN drop (features columns only)
    feature_cols = [c for c in features_df.columns if c not in ("Date", "Ticker")]
    if args.strict_dropna or True:
        before = len(features_df)
        features_df = features_df.dropna(subset=feature_cols).reset_index(drop=True)
        after = len(features_df)
        print(f"[INFO] strict dropna: {before} -> {after} rows")

    # Start only after max window is effectively satisfied:
    # (dropna already handles warm-up, but this makes intent explicit)
    # We'll still keep it simple and trust dropna.

    saved_to = save_features(features_df)

    meta = {
        "updated_at_utc": now_utc_iso(),
        "saved_to": saved_to,
        "rows": int(len(features_df)),
        "min_date": str(features_df["Date"].min().date()),
        "max_date": str(features_df["Date"].max().date()),
        "eligible_tickers": eligible,
        "excluded": excluded,
        "feature_columns": feature_cols,
        "market_ticker": MARKET_TICKER,
        "vix_ticker": VIX_TICKER,
    }
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] features saved={saved_to} rows={len(features_df)} tickers={len(eligible)}")
    print(features_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
