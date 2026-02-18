# scripts/build_tau_labels.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# IO helpers
# -----------------------------
def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def read_universe(universe_csv: str) -> list[str]:
    path = Path(universe_csv)
    if not path.exists():
        raise FileNotFoundError(f"Missing universe file: {universe_csv}")
    uni = pd.read_csv(path)
    if "Ticker" not in uni.columns:
        raise ValueError("universe.csv must contain 'Ticker' column")
    if "Enabled" in uni.columns:
        uni = uni[uni["Enabled"] == True]  # noqa: E712
    return (
        uni["Ticker"].astype(str).str.upper().str.strip().dropna().unique().tolist()
    )


# -----------------------------
# τ label computation
# -----------------------------
def compute_tau_for_ticker(
    px: pd.DataFrame,
    profit_target: float,
    horizon_days: int,
) -> pd.DataFrame:
    """
    For a single ticker price frame sorted by Date:
      entry at Close[t]
      success if any future High within horizon satisfies: High >= Close[t]*(1+profit_target)
      tau_days = first hit day distance in trading days (inclusive: entry day counts as 1 if hit on same day)
      If never hits within horizon => tau_days = NaN
    Returns: DataFrame with Date, Ticker, tau_days
    """
    if px.empty:
        return px

    px = px.sort_values("Date").reset_index(drop=True)

    close = pd.to_numeric(px["Close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(px["High"], errors="coerce").to_numpy(dtype=float)

    n = len(px)
    tau = np.full(n, np.nan, dtype=float)

    # if Close is missing, tau remains NaN
    valid_close = np.isfinite(close) & (close > 0)

    # horizon in trading steps (index distance). if horizon_days=0 => no future considered
    H = int(max(horizon_days, 0))

    for i in range(n):
        if not valid_close[i]:
            continue

        thr = close[i] * (1.0 + float(profit_target))

        # search in [i, i+H] inclusive, clipped
        j_end = min(n - 1, i + H)
        if j_end < i:
            continue

        seg = high[i : j_end + 1]
        # find first index where seg >= thr
        hit = np.where(np.isfinite(seg) & (seg >= thr))[0]
        if hit.size:
            # inclusive trading-day count
            tau[i] = float(hit[0] + 1)

    out = px[["Date", "Ticker"]].copy()
    out["tau_days"] = tau
    return out


def add_bucket_cols(df: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    df = df.copy()
    tau = pd.to_numeric(df["tau_days"], errors="coerce")
    for k in ks:
        k = int(k)
        if k <= 0:
            continue
        df[f"tau_le_{k}"] = ((tau <= k) & np.isfinite(tau)).astype(int)
    return df


# -----------------------------
# main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build τ labels (time-to-profit-target) from prices.")

    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)

    # required in your pipeline for spec consistency (not used in τ calc directly)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    # bucket cutoffs (your yml uses these)
    ap.add_argument("--k1", type=int, default=10)
    ap.add_argument("--k2", type=int, default=20)

    # IO
    ap.add_argument("--features-parq", default="data/features/features_model.parquet", type=str)
    ap.add_argument("--features-csv", default="data/features/features_model.csv", type=str)
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)
    ap.add_argument("--universe-csv", default="data/universe.csv", type=str)
    ap.add_argument("--exclude-tickers", default="SPY,^VIX", type=str)

    ap.add_argument("--out-parq", default="data/labels/labels_tau.parquet", type=str)
    ap.add_argument("--out-csv", default="data/labels/labels_tau.csv", type=str)

    # optional incremental slice
    ap.add_argument("--start-date", default="", type=str)
    ap.add_argument("--buffer-days", default=120, type=int)

    args = ap.parse_args()

    # -------------------------
    # load minimal features index (Date/Ticker universe)
    # (we trust features_model as the master calendar for gating days)
    # -------------------------
    feats = read_table(args.features_parq, args.features_csv).copy()
    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_model must contain Date and Ticker columns")

    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # optional slice for speed (keep buffer for forward window stability)
    if args.start_date:
        start = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(start):
            raise ValueError(f"Invalid --start-date: {args.start_date}")
        cut = start - pd.Timedelta(days=int(args.buffer_days))
        feats = feats[feats["Date"] >= cut].reset_index(drop=True)

    # -------------------------
    # load prices (need Close/High)
    # -------------------------
    prices = read_table(args.prices_parq, args.prices_csv).copy()
    if "Date" not in prices.columns or "Ticker" not in prices.columns:
        raise ValueError("prices must contain Date and Ticker columns")
    for c in ["Close", "High"]:
        if c not in prices.columns:
            raise ValueError(f"prices missing required column: {c}")

    prices["Date"] = norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices = prices.dropna(subset=["Date", "Ticker"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # universe-only + exclude extra tickers
    universe = set(read_universe(args.universe_csv))
    excludes = set([t.strip().upper() for t in str(args.exclude_tickers).split(",") if t.strip()])

    prices = prices[prices["Ticker"].isin(universe)].copy()
    prices = prices[~prices["Ticker"].isin(excludes)].copy()

    feats = feats[feats["Ticker"].isin(universe)].copy()
    feats = feats[~feats["Ticker"].isin(excludes)].copy()

    if feats.empty:
        raise RuntimeError("No feature rows after universe/exclude filter. Check universe.csv and exclude tickers.")
    if prices.empty:
        raise RuntimeError("No price rows after universe/exclude filter. Check prices fetch and universe.")

    # -------------------------
    # compute tau on price calendar
    # horizon = max_days + max_extend_days (tag semantics)
    # -------------------------
    horizon = int(args.max_days) + int(args.max_extend_days)
    if horizon <= 0:
        raise ValueError("horizon (max_days + max_extend_days) must be > 0")

    out_list = []
    for t, px_t in prices.groupby("Ticker", sort=False):
        px_t = px_t[["Date", "Ticker", "Close", "High"]].copy()
        tau_t = compute_tau_for_ticker(px_t, profit_target=float(args.profit_target), horizon_days=horizon)
        out_list.append(tau_t)

    tau_all = pd.concat(out_list, ignore_index=True)
    tau_all = tau_all.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # -------------------------
    # align τ back to features dates only (master calendar)
    # -------------------------
    base = feats[["Date", "Ticker"]].drop_duplicates(["Date", "Ticker"]).copy()
    base = base.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    merged = base.merge(tau_all, on=["Date", "Ticker"], how="left")

    # buckets: k1,k2 plus max_days and horizon
    ks = sorted(set([int(args.k1), int(args.k2), int(args.max_days), int(horizon)]))
    merged = add_bucket_cols(merged, ks)

    # traceability columns
    merged["profit_target"] = float(args.profit_target)
    merged["max_days"] = int(args.max_days)
    merged["stop_level"] = float(args.stop_level)
    merged["max_extend_days"] = int(args.max_extend_days)
    merged["tau_horizon_days"] = int(horizon)

    # Save
    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_csv)
    out_parq.parent.mkdir(parents=True, exist_ok=True)

    # parquet preferred
    try:
        merged.to_parquet(out_parq, index=False)
        print(f"[DONE] wrote {out_parq} rows={len(merged)}")
    except Exception as e:
        print(f"[WARN] parquet save failed: {e}")

    merged.to_csv(out_csv, index=False)
    print(f"[DONE] wrote {out_csv} rows={len(merged)}")

    # quick sanity
    non_na = int(pd.to_numeric(merged["tau_days"], errors="coerce").notna().sum())
    print(f"[INFO] tau non-NaN rows: {non_na}/{len(merged)} horizon={horizon} k1={args.k1} k2={args.k2}")


if __name__ == "__main__":
    main()