# scripts/build_tail_labels.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"
META_DIR = DATA_DIR / "meta"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def _to_dt(x):
    return pd.to_datetime(x, errors="coerce")


def forward_min(s: pd.Series, window: int) -> pd.Series:
    return s.iloc[::-1].rolling(window, min_periods=window).min().iloc[::-1]


def forward_max(s: pd.Series, window: int) -> pd.Series:
    return s.iloc[::-1].rolling(window, min_periods=window).max().iloc[::-1]


def make_tag(profit_target: float, max_days: int, stop_level: float, max_extend_days: int) -> str:
    pt_tag = int(round(profit_target * 100))
    sl_tag = int(round(abs(stop_level) * 100))
    return f"pt{pt_tag}_h{max_days}_sl{sl_tag}_ex{max_extend_days}"


def normalize_prices_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure prices has: Date, Ticker, High, Low, Close
    Accepts variants: date/Datetime, ticker, AdjClose/Adj Close, close, high, low
    """
    df = df.copy()

    # flatten multiindex cols
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.columns = [str(c).strip() for c in df.columns]

    # Date
    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "Date"})
        elif isinstance(df.index, pd.DatetimeIndex) or df.index.name in ("Date", "Datetime"):
            idx_name = df.index.name or "Date"
            df = df.reset_index().rename(columns={idx_name: "Date"})

    # Ticker
    if "Ticker" not in df.columns:
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker": "Ticker"})

    # Common OHLC renames
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "high":
            rename_map[c] = "High"
        elif cl == "low":
            rename_map[c] = "Low"
        elif cl == "close":
            rename_map[c] = "Close"
        elif cl in ("adj close", "adjclose", "adj_close"):
            rename_map[c] = "AdjClose"
        elif cl == "open":
            rename_map[c] = "Open"
        elif cl == "volume":
            rename_map[c] = "Volume"
    if rename_map:
        df = df.rename(columns=rename_map)

    # If Close missing but AdjClose exists, use AdjClose as Close (strategy uses Close)
    if "Close" not in df.columns and "AdjClose" in df.columns:
        df["Close"] = df["AdjClose"]

    # If still missing, hard fail with diagnostic
    needed = {"Date", "Ticker", "High", "Low", "Close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"prices missing columns: {sorted(missing)}; cols={list(df.columns)[:40]}")

    df["Date"] = _to_dt(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    # numeric coercion
    for c in ["High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def normalize_features_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("features_model must contain Date and Ticker")
    df["Date"] = _to_dt(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build tail labels aligned with the infinite DCA strategy.")
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--tail-horizon-days", default=None, type=int,
                    help="Tail check horizon in days. Default: max-days (strategy-aligned).")
    ap.add_argument("--start-date", default=None, type=str)
    ap.add_argument("--buffer-days", default=0, type=int)

    ap.add_argument("--prices-parq", default=str(RAW_DIR / "prices.parquet"), type=str)
    ap.add_argument("--prices-csv", default=str(RAW_DIR / "prices.csv"), type=str)

    ap.add_argument("--features-parq", default=str(FEAT_DIR / "features_model.parquet"), type=str)
    ap.add_argument("--features-csv", default=str(FEAT_DIR / "features_model.csv"), type=str)

    ap.add_argument("--out-parq", default=None, type=str)
    ap.add_argument("--out-csv", default=None, type=str)

    args = ap.parse_args()

    max_days = int(args.max_days)
    tail_h = int(args.tail_horizon_days) if args.tail_horizon_days is not None else max_days
    stop_level = float(args.stop_level)
    profit_target = float(args.profit_target)

    tag = make_tag(profit_target, max_days, stop_level, int(args.max_extend_days))

    prices = normalize_prices_schema(read_table(Path(args.prices_parq), Path(args.prices_csv)))
    feats = normalize_features_schema(read_table(Path(args.features_parq), Path(args.features_csv)))

    prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    feats = feats.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # incremental window (optional)
    if args.start_date:
        start = pd.to_datetime(args.start_date)
        if int(args.buffer_days) > 0:
            start = start - pd.Timedelta(days=int(args.buffer_days))
        prices = prices.loc[prices["Date"] >= start].copy()
        feats = feats.loc[feats["Date"] >= start].copy()

    # forward stats computed on full price (per ticker)
    forward_stats = []
    for t, df_t in prices.groupby("Ticker", sort=False):
        df_t = df_t.sort_values("Date").reset_index(drop=True)

        close = df_t["Close"]
        high = df_t["High"]
        low = df_t["Low"]

        max_high_h = forward_max(high, max_days)
        min_low_tail = forward_min(low, tail_h)

        out = pd.DataFrame({
            "Date": df_t["Date"],
            "Ticker": t,
            "FWD_MAX_HIGH_H": max_high_h,
            "FWD_MIN_LOW_TAIL": min_low_tail,
            "ENTRY_CLOSE": close,
        })
        forward_stats.append(out)

    fwd = pd.concat(forward_stats, ignore_index=True)
    fwd = fwd.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    base = feats.merge(fwd, on=["Date", "Ticker"], how="left", validate="1:1")

    entry = pd.to_numeric(base["ENTRY_CLOSE"], errors="coerce")

    succ_thr = entry * (1.0 + profit_target)
    tail_thr = entry * (1.0 + stop_level)

    fwd_max_high = pd.to_numeric(base["FWD_MAX_HIGH_H"], errors="coerce")
    fwd_min_low = pd.to_numeric(base["FWD_MIN_LOW_TAIL"], errors="coerce")

    base["Success"] = ((fwd_max_high >= succ_thr) & np.isfinite(fwd_max_high) & np.isfinite(succ_thr)).astype(int)
    base["TailHit"] = ((fwd_min_low <= tail_thr) & np.isfinite(fwd_min_low) & np.isfinite(tail_thr)).astype(int)
    base["TailTarget"] = ((base["TailHit"] == 1) & (base["Success"] == 0)).astype(int)

    # drop rows without full forward windows
    base = base.loc[np.isfinite(fwd_max_high) & np.isfinite(fwd_min_low) & np.isfinite(entry)].copy()

    keep_cols = [c for c in feats.columns] + ["Success", "TailHit", "TailTarget"]
    out_df = base[keep_cols].copy()

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    out_parq = Path(args.out_parq) if args.out_parq else (LABEL_DIR / f"labels_tail_{tag}.parquet")
    out_csv = Path(args.out_csv) if args.out_csv else (LABEL_DIR / f"labels_tail_{tag}.csv")

    saved_parq = None
    try:
        out_df.to_parquet(out_parq, index=False)
        saved_parq = str(out_parq)
    except Exception as e:
        print(f"[WARN] parquet write failed: {e}")

    out_df.to_csv(out_csv, index=False)

    meta = {
        "tag": tag,
        "profit_target": profit_target,
        "max_days": max_days,
        "stop_level": stop_level,
        "tail_horizon_days": tail_h,
        "rows": int(len(out_df)),
        "saved_parquet": saved_parq,
        "saved_csv": str(out_csv),
        "label_def": {
            "Success": "FWD(max_days) max High >= entry Close*(1+profit_target)",
            "TailHit": "FWD(tail_horizon_days) min Low <= entry Close*(1+stop_level)",
            "TailTarget": "TailHit==1 AND Success==0 (strategy-aligned bad tail path)",
        },
    }
    (META_DIR / f"build_tail_labels_{tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] build_tail_labels.py tag={tag} rows={len(out_df)}")
    if saved_parq:
        print(f"[OK] {saved_parq}")
    print(f"[OK] {out_csv}")


if __name__ == "__main__":
    main()