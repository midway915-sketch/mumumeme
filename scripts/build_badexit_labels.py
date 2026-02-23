#!/usr/bin/env python3
# scripts/build_badexit_labels.py
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _read_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing trades file: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported trades format: {path.suffix} (use .parquet or .csv)")

    if df.empty:
        return df

    need = {"EntryDate", "ExitDate", "Tickers", "Reason"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"trades missing columns {missing}. cols={list(df.columns)[:50]}")

    df = df.copy()
    df["EntryDate"] = _norm_date(df["EntryDate"])
    df["ExitDate"] = _norm_date(df["ExitDate"])
    df["Reason"] = df["Reason"].astype(str)
    df["Tickers"] = df["Tickers"].astype(str)

    df = df.dropna(subset=["EntryDate", "ExitDate"]).reset_index(drop=True)
    return df


def _is_bad_reason(reason: str) -> bool:
    """
    BadExit = 1 for:
      - REVAL_FAIL(...)
      - GRACE_END_EXIT(...)
    """
    r = (reason or "").strip().upper()
    return r.startswith("REVAL_FAIL") or r.startswith("GRACE_END_EXIT")


def _explode_tickers(tickers: str) -> list[str]:
    # "TQQQ" or "TQQQ,UPRO" 형태
    parts = [p.strip().upper() for p in (tickers or "").split(",") if p.strip()]
    return parts


def build_badexit_labels(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Output schema:
      Date, Ticker, BadExit
    Label date = EntryDate (cycle 시작일 기준)
    """
    if trades.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "BadExit"])

    rows = []
    for _, r in trades.iterrows():
        entry = r["EntryDate"]
        tickers = _explode_tickers(r.get("Tickers", ""))
        if not tickers:
            continue

        bad = 1 if _is_bad_reason(str(r.get("Reason", ""))) else 0

        for t in tickers:
            rows.append({"Date": entry, "Ticker": t, "BadExit": bad})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "BadExit"])

    out["Date"] = _norm_date(out["Date"])
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out["BadExit"] = pd.to_numeric(out["BadExit"], errors="coerce").fillna(0).astype(int)

    # 한 Date,Ticker가 여러 번 나오면(TopK=2 등) "한 번이라도 bad면 bad"로 합침
    out = (
        out.dropna(subset=["Date", "Ticker"])
        .groupby(["Date", "Ticker"], as_index=False)["BadExit"]
        .max()
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build BadExit labels from sim_engine_trades_* files.")
    ap.add_argument("--trades-path", required=True, type=str, help="sim_engine_trades_*.parquet or .csv")
    ap.add_argument("--out-parquet", default="", type=str)
    ap.add_argument("--out-csv", default="", type=str)
    ap.add_argument("--tag", default="", type=str, help="optional tag for output filename")
    ap.add_argument("--suffix", default="", type=str, help="optional suffix for output filename")
    ap.add_argument("--out-dir", default="data/labels", type=str)
    args = ap.parse_args()

    trades_path = Path(args.trades_path)
    trades = _read_trades(trades_path)

    labels = build_badexit_labels(trades)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out_parquet.strip() or args.out_csv.strip():
        out_p = Path(args.out_parquet) if args.out_parquet.strip() else None
        out_c = Path(args.out_c) if args.out_csv.strip() else None  # type: ignore
    else:
        tag = args.tag.strip() or "run"
        suffix = args.suffix.strip() or "unknown"
        base = out_dir / f"labels_badexit_{tag}_gate_{suffix}"
        out_p = base.with_suffix(".parquet")
        out_c = base.with_suffix(".csv")

    # write
    if out_p is not None:
        labels.to_parquet(out_p, index=False)
        print(f"[DONE] wrote: {out_p} rows={len(labels)}")
    if out_c is not None:
        labels.to_csv(out_c, index=False)
        print(f"[DONE] wrote: {out_c} rows={len(labels)}")

    if len(labels):
        print(f"[INFO] range: {labels['Date'].min().date()}..{labels['Date'].max().date()}")
        print(f"[INFO] BadExit rate: {labels['BadExit'].mean():.4f}")


if __name__ == "__main__":
    main()