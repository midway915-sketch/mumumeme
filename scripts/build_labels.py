# scripts/build_labels.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
LABELS_DIR = DATA_DIR / "labels"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

FEATURES_MODEL_PARQ = FEATURES_DIR / "features_model.parquet"
FEATURES_MODEL_CSV = FEATURES_DIR / "features_model.csv"

OUT_RAW_DATA = DATA_DIR / "raw_data.csv"
OUT_LABELS_PARQ = LABELS_DIR / "labels_model.parquet"
OUT_LABELS_CSV = LABELS_DIR / "labels_model.csv"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def read_prices() -> pd.DataFrame:
    df = read_table(PRICES_PARQ, PRICES_CSV).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    need = {"Date", "Ticker", "Open", "High", "Low", "Close"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"prices missing columns: {miss}")
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def ensure_market_features(feats: pd.DataFrame, prices: pd.DataFrame, market_ticker: str = "SPY") -> pd.DataFrame:
    """
    Ensure Market_Drawdown and Market_ATR_ratio exist.
    - No lookahead: rolling max and ATR use past window only.
    - Join by Date onto all tickers.
    """
    out = feats.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()

    m = prices[prices["Ticker"] == market_ticker].sort_values("Date").copy()
    if m.empty:
        raise ValueError(f"Market ticker {market_ticker} not found in prices. Run fetch_prices.py --include-extra")

    close = pd.to_numeric(m["Close"], errors="coerce")
    high = pd.to_numeric(m["High"], errors="coerce")
    low = pd.to_numeric(m["Low"], errors="coerce")

    # Drawdown_252 on market: close / rolling_max(close,252) - 1
    roll_max_252 = close.rolling(252, min_periods=20).max()
    m["Market_Drawdown"] = (close / roll_max_252) - 1.0

    # ATR_ratio (14): ATR14 / Close
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=10).mean()
    m["Market_ATR_ratio"] = atr14 / close

    m_feat = m[["Date", "Market_Drawdown", "Market_ATR_ratio"]].dropna(subset=["Date"]).copy()

    # merge; if feats already has them, combine_first
    out = out.merge(m_feat, on="Date", how="left", suffixes=("", "_mkt"))

    if "Market_Drawdown" in out.columns and "Market_Drawdown_mkt" in out.columns:
        out["Market_Drawdown"] = pd.to_numeric(out["Market_Drawdown"], errors="coerce").combine_first(
            pd.to_numeric(out["Market_Drawdown_mkt"], errors="coerce")
        )
        out.drop(columns=["Market_Drawdown_mkt"], inplace=True, errors="ignore")
    elif "Market_Drawdown" not in out.columns:
        out["Market_Drawdown"] = pd.to_numeric(out["Market_Drawdown_mkt"], errors="coerce")
        out.drop(columns=["Market_Drawdown_mkt"], inplace=True, errors="ignore")

    if "Market_ATR_ratio" in out.columns and "Market_ATR_ratio_mkt" in out.columns:
        out["Market_ATR_ratio"] = pd.to_numeric(out["Market_ATR_ratio"], errors="coerce").combine_first(
            pd.to_numeric(out["Market_ATR_ratio_mkt"], errors="coerce")
        )
        out.drop(columns=["Market_ATR_ratio_mkt"], inplace=True, errors="ignore")
    elif "Market_ATR_ratio" not in out.columns:
        out["Market_ATR_ratio"] = pd.to_numeric(out["Market_ATR_ratio_mkt"], errors="coerce")
        out.drop(columns=["Market_ATR_ratio_mkt"], inplace=True, errors="ignore")

    return out


def compute_path_dependent_labels(
    prices: pd.DataFrame,
    horizon_days: int,
    profit_target: float,
    stop_level: float,
) -> pd.DataFrame:
    """
    For each (Ticker, Date):
      Success = target hit before stop within horizon_days
      Tau = days to success (1..horizon), NaN if not success
      Tail = stop touched at any point within horizon (risk label)
      MinFutureRet = min(low/entry - 1) within horizon
    Uses daily bars; conservative assumption: if both hit on same day -> stop wins.
    """
    prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True).copy()
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
    prices["High"] = pd.to_numeric(prices["High"], errors="coerce")
    prices["Low"] = pd.to_numeric(prices["Low"], errors="coerce")

    out_rows = []

    for t, g in prices.groupby("Ticker", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        n = len(g)
        close = g["Close"].to_numpy(dtype=float)
        high = g["High"].to_numpy(dtype=float)
        low = g["Low"].to_numpy(dtype=float)
        dates = g["Date"].to_numpy()

        success = np.zeros(n, dtype=np.int8)
        tau = np.full(n, np.nan, dtype=float)
        tail = np.zeros(n, dtype=np.int8)
        min_future_ret = np.full(n, np.nan, dtype=float)

        for i in range(n):
            entry = close[i]
            if not np.isfinite(entry) or entry <= 0:
                continue

            target_px = entry * (1.0 + profit_target)
            stop_px = entry * (1.0 + stop_level)

            j_end = min(n, i + 1 + horizon_days)  # lookahead excludes today (i+1..)
            if i + 1 >= j_end:
                continue

            # future window
            f_high = high[i + 1 : j_end]
            f_low = low[i + 1 : j_end]

            # risk stats
            mf = np.nanmin(f_low / entry - 1.0) if len(f_low) else np.nan
            min_future_ret[i] = mf

            # tail: stop touched anywhere in horizon
            stop_hits = np.where(f_low <= stop_px)[0]
            if len(stop_hits) > 0:
                tail[i] = 1
                first_stop = int(stop_hits[0]) + 1  # day count (1..)
            else:
                first_stop = None

            # success: target hit before stop; conservative: same-day tie -> stop wins
            tgt_hits = np.where(f_high >= target_px)[0]
            if len(tgt_hits) > 0:
                first_tgt = int(tgt_hits[0]) + 1
            else:
                first_tgt = None

            if first_tgt is not None:
                if first_stop is None or first_tgt < first_stop:
                    success[i] = 1
                    tau[i] = float(first_tgt)

        out = pd.DataFrame(
            {
                "Date": pd.to_datetime(dates),
                "Ticker": t,
                "EntryClose": close,
                "Success": success,
                "Tau_Success": tau,
                "Tail": tail,
                "MinFutureRet": min_future_ret,
            }
        )
        out_rows.append(out)

    return pd.concat(out_rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)  # tag만 맞추고, 여기서는 미사용
    ap.add_argument("--market-ticker", type=str, default="SPY")
    args = ap.parse_args()

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    prices = read_prices()

    feats = read_table(FEATURES_MODEL_PARQ, FEATURES_MODEL_CSV).copy()
    feats["Date"] = pd.to_datetime(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()

    # Market_* 강제 보정
    feats = ensure_market_features(feats, prices, market_ticker=args.market_ticker)

    # 라벨 생성(경로 의존)
    lbl = compute_path_dependent_labels(
        prices=prices,
        horizon_days=int(args.max_days),
        profit_target=float(args.profit_target),
        stop_level=float(args.stop_level),
    )

    # features + labels merge
    merged = feats.merge(lbl, on=["Date", "Ticker"], how="inner")

    # raw_data.csv: train_model.py가 읽는 형태로 저장
    # (features + Target/Success 포함)
    merged_out = merged.copy()
    merged_out["Target"] = merged_out["Success"].astype(int)

    # 저장
    try:
        merged_out.to_parquet(OUT_LABELS_PARQ, index=False)
        labels_saved = str(OUT_LABELS_PARQ)
    except Exception as e:
        print(f"[WARN] parquet save failed: {e} -> saving CSV {OUT_LABELS_CSV}")
        merged_out.to_csv(OUT_LABELS_CSV, index=False)
        labels_saved = str(OUT_LABELS_CSV)

    merged_out.to_csv(OUT_RAW_DATA, index=False)

    # 검증 로그
    need_cols = ["Market_Drawdown", "Market_ATR_ratio"]
    missing = [c for c in need_cols if c not in merged_out.columns]
    if missing:
        raise ValueError(f"[ERROR] still missing market columns after ensure_market_features: {missing}")

    print("=" * 60)
    print("[DONE] build_labels.py")
    print("saved labels:", labels_saved)
    print("saved raw_data:", OUT_RAW_DATA)
    print("rows:", len(merged_out))
    print("success rate:", round(float(merged_out["Success"].mean()), 4))
    print("tail rate:", round(float(merged_out["Tail"].mean()), 4))
    print("=" * 60)


if __name__ == "__main__":
    main()