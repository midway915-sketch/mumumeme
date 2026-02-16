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
FEAT_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"

PRICES_PARQUET = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

FEATURES_PARQUET = FEAT_DIR / "features_model.parquet"
FEATURES_CSV = FEAT_DIR / "features_model.csv"

OUT_RAW_CSV = DATA_DIR / "raw_data.csv"  # train_model.py 기본 입력

DEFAULT_FEATURE_COLS = [
    "Drawdown_252",
    "Drawdown_60",
    "ATR_ratio",
    "Z_score",
    "MACD_hist",
    "MA20_slope",
    "Market_Drawdown",
    "Market_ATR_ratio",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_tag(pt: float, max_days: int, sl: float, ex: int) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{max_days}_sl{sl_tag}_ex{ex}"


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
        return str(parq)
    except Exception:
        df.to_csv(csv, index=False)
        return str(csv)


def auto_load_features(features_path: str | None) -> tuple[pd.DataFrame, str]:
    """
    1) --features-path가 있으면 그걸 사용
    2) 기본 features_model.* 있으면 사용
    3) 없으면 data/features 내에서 Date/Ticker 컬럼 있는 파일 자동탐색
    """
    if features_path:
        fp = Path(features_path)
        if not fp.exists():
            raise FileNotFoundError(f"features-path not found: {fp}")
        df = pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
        return df, str(fp)

    if FEATURES_PARQUET.exists():
        return pd.read_parquet(FEATURES_PARQUET), str(FEATURES_PARQUET)
    if FEATURES_CSV.exists():
        return pd.read_csv(FEATURES_CSV), str(FEATURES_CSV)

    if not FEAT_DIR.exists():
        raise FileNotFoundError(f"features dir not found: {FEAT_DIR}")

    # 후보: parquet 우선, 없으면 csv. 최신 수정 파일부터
    candidates = []
    for ext in ("*.parquet", "*.csv"):
        candidates.extend(FEAT_DIR.glob(ext))
    candidates = [p for p in candidates if p.is_file()]

    if not candidates:
        raise FileNotFoundError(
            f"Missing file: {FEATURES_PARQUET} (or {FEATURES_CSV}) and no candidates in {FEAT_DIR}"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    last_err = None
    for p in candidates:
        try:
            df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
            cols = set(df.columns)
            if {"Date", "Ticker"}.issubset(cols):
                return df, str(p)
        except Exception as e:
            last_err = e

    raise FileNotFoundError(
        f"Could not find a usable features file in {FEAT_DIR}. "
        f"Need columns Date/Ticker. Last error: {last_err}"
    )


def forward_roll_max_excl_today(s: pd.Series, window: int) -> pd.Series:
    # next window days (t+1 .. t+window) max
    r = s[::-1].rolling(window, min_periods=window).max()[::-1].shift(-1)
    return r


def forward_roll_min_excl_today(s: pd.Series, window: int) -> pd.Series:
    r = s[::-1].rolling(window, min_periods=window).min()[::-1].shift(-1)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description="Build success labels (hit profit within horizon).")
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, default=30, help="tag용(파일명). 라벨 계산엔 직접 사용 안 함.")
    ap.add_argument("--features-path", type=str, default=None)
    ap.add_argument("--feature-cols", type=str, default="",
                    help="콤마로 feature 컬럼 지정(비우면 기본 8개)")
    args = ap.parse_args()

    pt = float(args.profit_target)
    max_days = int(args.max_days)
    sl = float(args.stop_level)
    ex = int(args.max_extend_days)
    tag = fmt_tag(pt, max_days, sl, ex)

    # load prices
    prices = read_table(PRICES_PARQUET, PRICES_CSV).copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices = (
        prices.sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    need_px = {"Date", "Ticker", "High", "Low", "Close"}
    miss_px = [c for c in need_px if c not in prices.columns]
    if miss_px:
        raise ValueError(f"prices missing columns: {miss_px}")

    # load features (auto)
    feats, feats_src = auto_load_features(args.features_path)
    feats = feats.copy()
    feats["Date"] = pd.to_datetime(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = (
        feats.sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    feature_cols = DEFAULT_FEATURE_COLS
    if str(args.feature_cols).strip():
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]

    missing_feat = [c for c in feature_cols if c not in feats.columns]
    if missing_feat:
        raise ValueError(f"features missing feature columns: {missing_feat} (src={feats_src})")

    # merge prices into features to label
    base = feats[["Date", "Ticker"] + feature_cols].merge(
        prices[["Date", "Ticker", "High", "Low", "Close"]],
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
    )

    base = base.dropna(subset=["Close"] + feature_cols).reset_index(drop=True)
    base = base.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # compute labels per ticker
    out = []
    for tkr, g in base.groupby("Ticker", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        close = g["Close"].astype(float)
        high = g["High"].astype(float)
        low = g["Low"].astype(float)

        fut_max_high = forward_roll_max_excl_today(high, max_days)
        fut_min_low = forward_roll_min_excl_today(low, max_days)

        success = (fut_max_high >= close * (1.0 + pt)).astype("float")
        fut_dd = (fut_min_low / close) - 1.0

        g["Success"] = success
        g["FutureMinDD"] = fut_dd

        g = g.dropna(subset=["Success", "FutureMinDD"]).reset_index(drop=True)
        out.append(g)

    labeled = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if labeled.empty:
        raise RuntimeError("No labeled rows produced. Check raw prices/features and horizons.")

    labeled["Success"] = labeled["Success"].astype(int)

    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    out_parq = LABEL_DIR / f"raw_data_{tag}.parquet"
    out_csv = LABEL_DIR / f"raw_data_{tag}.csv"
    saved_to = save_table(labeled, out_parq, out_csv)

    labeled.to_csv(OUT_RAW_CSV, index=False)

    meta = {
        "updated_at_utc": now_utc_iso(),
        "tag": tag,
        "profit_target": pt,
        "max_days": max_days,
        "stop_level": sl,
        "rows": int(len(labeled)),
        "base_success_rate": float(labeled["Success"].mean()),
        "min_date": str(labeled["Date"].min().date()),
        "max_date": str(labeled["Date"].max().date()),
        "saved_to": saved_to,
        "also_written": str(OUT_RAW_CSV),
        "feature_cols": feature_cols,
        "features_source": feats_src,
    }
    (LABEL_DIR / f"raw_data_{tag}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] labels saved -> {saved_to}")
    print(f"[DONE] also wrote -> {OUT_RAW_CSV}")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
