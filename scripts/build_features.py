# scripts/build_features.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# sys.path guard (avoid "No module named 'scripts'")
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

try:
    from scripts.feature_spec import get_feature_cols, write_feature_cols_meta  # type: ignore
except Exception:
    # fallback when scripts isn't a package
    from feature_spec import get_feature_cols, write_feature_cols_meta  # type: ignore


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"
OUT_PARQ = FEAT_DIR / "features_model.parquet"
OUT_CSV = FEAT_DIR / "features_model.csv"

MARKET_TICKER = "SPY"


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

    mret = c.pct_change().fillna(0.0)

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(m["Date"], errors="coerce").dt.tz_localize(None),
            "Market_Drawdown": mdd.to_numpy(dtype=float),
            "Market_ATR_ratio": atr_ratio.to_numpy(dtype=float),
            "Market_ret_1d": mret.to_numpy(dtype=float),
        }
    )
    out = out.dropna(subset=["Date"]).drop_duplicates(["Date"], keep="last").sort_values("Date").reset_index(drop=True)
    return out


def compute_ticker_features(g: pd.DataFrame, market_ret_by_date: pd.Series) -> pd.DataFrame:
    g = g.sort_values("Date").copy()
    dt = pd.to_datetime(g["Date"], errors="coerce").dt.tz_localize(None)

    c = pd.to_numeric(g["Close"], errors="coerce")
    h = pd.to_numeric(g["High"], errors="coerce")
    l = pd.to_numeric(g["Low"], errors="coerce")

    roll_max_252 = c.rolling(252, min_periods=252).max()
    dd_252 = (c / roll_max_252) - 1.0

    roll_max_60 = c.rolling(60, min_periods=60).max()
    dd_60 = (c / roll_max_60) - 1.0

    atr_ratio = compute_atr_ratio(g, n=14)

    ema12 = ema(c, 12)
    ema26 = ema(c, 26)
    macd = ema12 - ema26
    signal = ema(macd, 9)
    macd_hist = (macd - signal)

    ma20 = c.rolling(20, min_periods=20).mean()
    ma20_slope = ma20.diff()

    z_score = (c - c.rolling(60, min_periods=60).mean()) / c.rolling(60, min_periods=60).std()

    ret_5 = c.pct_change(5)
    ret_10 = c.pct_change(10)
    ret_20 = c.pct_change(20)

    breakout_20 = (c / c.rolling(20, min_periods=20).max()) - 1.0
    vol = g["Volume"] if "Volume" in g.columns else pd.Series(np.nan, index=g.index)
    vol_surge = (pd.to_numeric(vol, errors="coerce") / pd.to_numeric(vol, errors="coerce").rolling(20, min_periods=20).mean()) - 1.0

    trend_align = ((c > ma20).astype(float)).rolling(20, min_periods=20).mean()

    # beta_60 (✅ 인덱스 정렬이 핵심)
    r = c.pct_change()
    mret_vals = market_ret_by_date.reindex(dt.to_numpy()).to_numpy(dtype=float)
    mret_s = pd.Series(mret_vals, index=g.index).fillna(0.0)

    cov = r.rolling(60, min_periods=60).cov(mret_s)
    var = mret_s.rolling(60, min_periods=60).var().replace(0.0, np.nan)
    beta_60 = cov / var

    out = pd.DataFrame(
        {
            "Date": dt.to_numpy(),
            "Ticker": g["Ticker"].astype(str).to_numpy(),
            "Drawdown_252": dd_252.to_numpy(dtype=float),
            "Drawdown_60": dd_60.to_numpy(dtype=float),
            "ATR_ratio": atr_ratio.to_numpy(dtype=float),
            "Z_score": z_score.to_numpy(dtype=float),
            "MACD_hist": macd_hist.to_numpy(dtype=float),
            "MA20_slope": ma20_slope.to_numpy(dtype=float),
            "ret_5": ret_5.to_numpy(dtype=float),
            "ret_10": ret_10.to_numpy(dtype=float),
            "ret_20": ret_20.to_numpy(dtype=float),
            "breakout_20": breakout_20.to_numpy(dtype=float),
            "vol_surge": vol_surge.to_numpy(dtype=float),
            "trend_align": trend_align.to_numpy(dtype=float),
            "beta_60": beta_60.to_numpy(dtype=float),
        }
    )
    out = out.dropna(subset=["Date"]).reset_index(drop=True)
    return out


def load_universe_group_map(universe_csv: Path) -> dict[str, str]:
    if not universe_csv.exists():
        return {}
    u = pd.read_csv(universe_csv)
    if "Ticker" not in u.columns or "Group" not in u.columns:
        return {}
    u["Ticker"] = u["Ticker"].astype(str).str.upper().str.strip()
    u["Group"] = u["Group"].astype(str).str.strip()
    u = u.dropna(subset=["Ticker", "Group"])
    return dict(zip(u["Ticker"].tolist(), u["Group"].tolist()))


def add_sector_strength(feats: pd.DataFrame, ticker_to_group: dict[str, str]) -> pd.DataFrame:
    if not ticker_to_group:
        raise ValueError(
            "Sector features requested but universe.csv has no usable Group mapping. "
            "Fix data/universe.csv to include a 'Group' column."
        )

    df = feats.copy()
    df["Group"] = df["Ticker"].map(ticker_to_group).fillna("")

    grp = df[df["Group"] != ""].copy()
    if grp.empty:
        raise ValueError("Sector features requested but no rows have Group assigned. Check universe.csv Group values.")

    grp = grp.sort_values(["Group", "Date"]).reset_index(drop=True)

    grp["Close"] = pd.to_numeric(grp.get("Close", np.nan), errors="coerce")
    if grp["Close"].isna().all():
        # Close가 없으면 최소한 ret_20 기반으로 계산
        grp["Close"] = np.nan

    # group 평균 ret_20
    grp_ret20 = grp.groupby(["Group", "Date"], sort=False)["ret_20"].mean().rename("Sector_Ret_20").reset_index()

    # RelStrength = ret_20 - sector_ret_20
    df = df.merge(grp_ret20, on=["Group", "Date"], how="left")
    df["RelStrength"] = pd.to_numeric(df["ret_20"], errors="coerce") - pd.to_numeric(df["Sector_Ret_20"], errors="coerce")

    df = df.drop(columns=["Group"], errors="ignore")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enable-sector-strength", action="store_true")
    args = ap.parse_args()

    prices = read_table(PRICES_PARQ, PRICES_CSV).copy()
    for c in ["Date", "Ticker", "High", "Low", "Close"]:
        if c not in prices.columns:
            raise ValueError(f"prices missing required col: {c}")

    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.tz_localize(None)
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices = prices.dropna(subset=["Date", "Ticker", "High", "Low", "Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    mkt = compute_market_features(prices)
    market_ret_by_date = mkt.set_index("Date")["Market_ret_1d"].astype(float)

    feats = []
    for tkr, g in prices.groupby("Ticker", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        f = compute_ticker_features(g, market_ret_by_date)
        feats.append(f)

    out = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
    if out.empty:
        raise RuntimeError("No features produced.")

    # attach market columns
    out = out.merge(mkt[["Date", "Market_Drawdown", "Market_ATR_ratio"]], on="Date", how="left")

    # ret_score (simple composite)
    out["ret_score"] = (
        pd.to_numeric(out["ret_20"], errors="coerce").fillna(0.0)
        - 0.5 * pd.to_numeric(out["Drawdown_60"], errors="coerce").fillna(0.0)
    )

    sector_enabled = bool(args.enable_sector_strength)
    if sector_enabled:
        ticker_to_group = load_universe_group_map(DATA_DIR / "universe.csv")
        out = add_sector_strength(out, ticker_to_group)

    # SSOT feature cols write
    feature_cols = get_feature_cols(sector_enabled=sector_enabled)
    write_feature_cols_meta(feature_cols=feature_cols, sector_enabled=sector_enabled)

    saved = save_table(out, OUT_PARQ, OUT_CSV)
    print(f"[DONE] saved={saved} rows={len(out)} range={out['Date'].min().date()}..{out['Date'].max().date()}")


if __name__ == "__main__":
    main()