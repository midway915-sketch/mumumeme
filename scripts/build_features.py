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

PRICES_PARQUET = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

OUT_PARQUET = FEATURE_DIR / "features.parquet"
OUT_CSV = FEATURE_DIR / "features.csv"
META_JSON = FEATURE_DIR / "features_meta.json"

MARKET_TICKER = "SPY"
VIX_TICKER = "^VIX"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    # 기본 컬럼 보장
    for c in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    df = df.sort_values(["Ticker", "Date"]).drop_duplicates(["Ticker", "Date"], keep="last").reset_index(drop=True)
    return df


def load_prices() -> pd.DataFrame:
    if PRICES_PARQUET.exists():
        df = pd.read_parquet(PRICES_PARQUET)
    elif PRICES_CSV.exists():
        df = pd.read_csv(PRICES_CSV)
    else:
        raise FileNotFoundError("No prices found. Run scripts/fetch_prices.py first.")
    return normalize_prices(df)


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def compute_atr_ratio(df: pd.DataFrame, atr_window: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = true_range(df["High"], df["Low"], prev_close)
    atr = tr.rolling(atr_window, min_periods=atr_window).mean()
    return atr / df["Close"]


def compute_drawdown(close: pd.Series, window: int) -> pd.Series:
    roll_max = close.rolling(window, min_periods=window).max()
    return close / roll_max - 1.0


def compute_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    return (close - ma) / sd


def compute_macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd = ema(close, fast) - ema(close, slow)
    sig = ema(macd, signal)
    return macd - sig


def compute_ma20_slope(close: pd.Series, ma_window: int = 20, slope_lookback: int = 5) -> pd.Series:
    ma = close.rolling(ma_window, min_periods=ma_window).mean()
    # 단순 기울기(퍼센트)
    return (ma - ma.shift(slope_lookback)) / ma.shift(slope_lookback)


def compute_market_frames(prices: pd.DataFrame, max_window: int) -> pd.DataFrame:
    m = prices[prices["Ticker"] == MARKET_TICKER].sort_values("Date").copy()
    if m.empty:
        raise ValueError(f"Market ticker {MARKET_TICKER} not found. Run fetch_prices.py --include-extra")

    m["Market_ret_1d"] = m["Close"].pct_change()
    m["Market_realized_vol_20"] = m["Market_ret_1d"].rolling(20, min_periods=20).std() * np.sqrt(252)

    # drawdown windows(60/252는 킬스위치와 모델에서 유용)
    m["Market_Drawdown_60"] = compute_drawdown(m["Close"], 60)
    m["Market_Drawdown_252"] = compute_drawdown(m["Close"], 252)

    m["Market_ATR_ratio"] = compute_atr_ratio(m, atr_window=14)

    out = m[["Date", "Market_ret_1d", "Market_realized_vol_20", "Market_Drawdown_60", "Market_Drawdown_252", "Market_ATR_ratio"]]
    return out


def compute_vix_frame(prices: pd.DataFrame) -> pd.DataFrame:
    v = prices[prices["Ticker"] == VIX_TICKER].sort_values("Date").copy()
    if v.empty:
        # VIX 없으면 그냥 NaN으로 붙이기 (죽지 않게)
        return pd.DataFrame(columns=["Date", "VIX"])
    v = v[["Date", "Close"]].rename(columns={"Close": "VIX"})
    return v


def is_eligible(
    df_t: pd.DataFrame,
    min_history_days: int,
    min_dollar_vol: float,
) -> tuple[bool, str]:
    df_t = df_t.dropna(subset=["Close", "Volume"]).copy()
    if len(df_t) < min_history_days:
        return False, f"short_history<{min_history_days}"

    dvol = (df_t["Close"] * df_t["Volume"]).rolling(20, min_periods=20).median()
    last = float(dvol.iloc[-1]) if len(dvol) else np.nan
    if not np.isfinite(last):
        return False, "no_dollarvol"
    if last < min_dollar_vol:
        return False, f"illiquid<{min_dollar_vol:g}"
    return True, "ok"


def compute_features_for_ticker(df_t: pd.DataFrame) -> pd.DataFrame:
    df_t = df_t.sort_values("Date").copy()
    close = df_t["Close"]

    df_t["Drawdown_252"] = compute_drawdown(close, 252)
    df_t["Drawdown_60"] = compute_drawdown(close, 60)
    df_t["ATR_ratio"] = compute_atr_ratio(df_t, atr_window=14)
    df_t["Z_score"] = compute_zscore(close, window=20)
    df_t["MACD_hist"] = compute_macd_hist(close, 12, 26, 9)
    df_t["MA20_slope"] = compute_ma20_slope(close, 20, 5)

    return df_t


def save_features(df: pd.DataFrame) -> str:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(OUT_PARQUET, index=False)
        return f"parquet:{OUT_PARQUET}"
    except Exception as e:
        print(f"[WARN] parquet save failed ({e}), saving csv: {OUT_CSV}")
        df.to_csv(OUT_CSV, index=False)
        return f"csv:{OUT_CSV}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-window", type=int, default=252)
    ap.add_argument("--min-history-days", type=int, default=260)  # 레버ETF 고려(대충 1년+)
    ap.add_argument("--min-dollar-vol", type=float, default=0.0)  # 처음엔 0으로 두고, 필요하면 올려
    ap.add_argument("--min-feature-rows", type=int, default=120)  # dropna 이후 최소 남아야 할 행
    ap.add_argument("--strict-dropna", action="store_true", help="drop rows with any NaN in feature columns")
    args = ap.parse_args()

    prices = load_prices()

    market_feat = compute_market_frames(prices, args.max_window)
    vix_feat = compute_vix_frame(prices)

    tickers = sorted(set(prices["Ticker"].unique().tolist()))
    # 시장 티커는 피처 생성 대상에서 제외(시장/보조 용도)
    tickers = [t for t in tickers if t not in (MARKET_TICKER, VIX_TICKER)]

    reasons = {}
    feats = []

    def run_with_thresholds(min_hist: int, min_dvol: float) -> int:
        nonlocal reasons, feats
        reasons = {}
        feats = []

        for t in tickers:
            df_t = prices[prices["Ticker"] == t].copy()
            ok, why = is_eligible(df_t, min_hist, min_dvol)
            if not ok:
                reasons[why] = reasons.get(why, 0) + 1
                continue

            df_f = compute_features_for_ticker(df_t)
            # market/vix merge
            df_f = df_f.merge(market_feat, on="Date", how="left").merge(vix_feat, on="Date", how="left")

            df_f["Ticker"] = t

            # strict mode: feature NaN 제거
            feature_cols = [c for c in df_f.columns if c not in ("Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume")]
            if args.strict_dropna:
                df_f = df_f.dropna(subset=feature_cols).reset_index(drop=True)

            if len(df_f) < args.min_feature_rows:
                reasons["too_few_feature_rows"] = reasons.get("too_few_feature_rows", 0) + 1
                continue

            keep_cols = ["Date", "Ticker"] + feature_cols
            feats.append(df_f[keep_cols])

        return len(feats)

    # 1차 시도
    n_ok = run_with_thresholds(args.min_history_days, args.min_dollar_vol)

    # 0개면 자동 완화(너처럼 액션에서 디버깅 힘들 때 이게 진짜 도움됨)
    if n_ok == 0:
        print("[WARN] 0 eligible tickers. Relaxing thresholds and retrying...")
        relaxed_min_hist = max(90, min(args.min_history_days, args.max_window + 30))
        n_ok = run_with_thresholds(relaxed_min_hist, 0.0)

    if n_ok == 0:
        print("[DEBUG] eligibility reject reasons:", reasons)
        # 가격 데이터 자체를 점검할 수 있게 표본 로그
        counts = prices.groupby("Ticker")["Date"].count().sort_values(ascending=False).head(20)
        print("[DEBUG] top ticker row counts:\n", counts.to_string())
        raise RuntimeError("No eligible tickers produced features. Check eligibility thresholds and raw data.")

    out = pd.concat(feats, ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    saved_to = save_features(out)

    meta = {
        "updated_at_utc": now_utc_iso(),
        "saved_to": saved_to,
        "rows": int(len(out)),
        "tickers_used": sorted(out["Ticker"].unique().tolist()),
        "tickers_used_count": int(out["Ticker"].nunique()),
        "min_date": str(out["Date"].min().date()),
        "max_date": str(out["Date"].max().date()),
        "params": {
            "max_window": args.max_window,
            "min_history_days": args.min_history_days,
            "min_dollar_vol": args.min_dollar_vol,
            "min_feature_rows": args.min_feature_rows,
            "strict_dropna": bool(args.strict_dropna),
        },
    }
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] features saved={saved_to} rows={meta['rows']} tickers={meta['tickers_used_count']}")
    print(f"[RANGE] {meta['min_date']}..{meta['max_date']}")
    print(f"[TICKERS] sample={meta['tickers_used'][:20]}")


if __name__ == "__main__":
    main()
