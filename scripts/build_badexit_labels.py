#!/usr/bin/env python3
# scripts/build_badexit_labels.py
from __future__ import annotations

# ✅ FIX: "python scripts/xxx.py"로 실행될 때도 scripts.* import가 되도록 repo root를 sys.path에 추가
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feature_spec import get_feature_cols


DATA_DIR = Path("data")
SIGNALS_DIR = DATA_DIR / "signals"
FEAT_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"

FEAT_PARQ = FEAT_DIR / "features_model.parquet"
FEAT_CSV = FEAT_DIR / "features_model.csv"

OUT_PARQ = LABEL_DIR / "labels_badexit.parquet"
OUT_CSV = LABEL_DIR / "labels_badexit.csv"


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def _is_badexit_reason(reason: str) -> int:
    """
    ✅ 결정된 규칙:
      BadExit = 1 if Reason starts with REVAL_FAIL OR GRACE_END_EXIT
      BadExit = 0 otherwise
    """
    r = str(reason or "").strip().upper()
    if r.startswith("REVAL_FAIL"):
        return 1
    if r.startswith("GRACE_END_EXIT"):
        return 1
    return 0


def _parse_trades_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if df.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "BadExit"])

    need = {"EntryDate", "Tickers", "Reason"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"trades file missing cols={miss}: {path}")

    df = df.copy()
    df["EntryDate"] = _norm_date(df["EntryDate"])
    df["Reason"] = df["Reason"].astype(str)
    df["BadExit"] = df["Reason"].apply(_is_badexit_reason).astype(int)

    # Tickes: "TQQQ" or "TQQQ,UPRO"
    rows = []
    for _, r in df.iterrows():
        d = r["EntryDate"]
        if pd.isna(d):
            continue
        tickers = [t.strip().upper() for t in str(r["Tickers"]).split(",") if t.strip()]
        if not tickers:
            continue
        y = int(r["BadExit"])
        for t in tickers:
            rows.append({"Date": d, "Ticker": t, "BadExit": y})

    if not rows:
        return pd.DataFrame(columns=["Date", "Ticker", "BadExit"])

    out = pd.DataFrame(rows)
    out["Date"] = _norm_date(out["Date"])
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out["BadExit"] = pd.to_numeric(out["BadExit"], errors="coerce").fillna(0).astype(int)

    # 혹시 같은 (Date,Ticker)가 여러 trades 파일/중복으로 생기면:
    # - BadExit는 "한 번이라도 bad면 1" 로 OR 처리
    out = (
        out.dropna(subset=["Date", "Ticker"])
        .groupby(["Date", "Ticker"], as_index=False)["BadExit"]
        .max()
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build BadExit labels from sim_engine_trades files.")
    ap.add_argument("--signals-dir", type=str, default=str(SIGNALS_DIR))
    ap.add_argument("--pattern", type=str, default="sim_engine_trades_*.parquet")
    ap.add_argument("--also-read-csv", action="store_true", help="Also ingest sim_engine_trades_*.csv")

    ap.add_argument("--out-parq", type=str, default=str(OUT_PARQ))
    ap.add_argument("--out-csv", type=str, default=str(OUT_CSV))

    ap.add_argument("--start-date", type=str, default=None, help="keep rows with Date >= start-date (YYYY-MM-DD)")
    ap.add_argument("--buffer-days", type=int, default=120, help="extra past days for stable joins")

    args = ap.parse_args()

    signals_dir = Path(args.signals_dir)
    if not signals_dir.exists():
        raise FileNotFoundError(f"signals dir not found: {signals_dir}")

    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    # ✅ 18개 SSOT 강제(섹터 포함)
    feature_cols = get_feature_cols(sector_enabled=True)
    if len(feature_cols) != 18:
        raise RuntimeError(f"SSOT feature cols must be 18, got {len(feature_cols)}: {feature_cols}")

    feats = read_table(FEAT_PARQ, FEAT_CSV).copy()
    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_model must include Date and Ticker")

    feats["Date"] = _norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = (
        feats.dropna(subset=["Date", "Ticker"])
        .sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    missing = [c for c in feature_cols if c not in feats.columns]
    if missing:
        raise ValueError(
            f"features_model missing SSOT feature cols (must have all 18): {missing}\n"
            f"-> Fix build_features.py / feature_spec.py consistency."
        )

    # start-date handling (buffer 포함)
    start_date = None
    compute_start = None
    if args.start_date:
        start_date = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(start_date):
            raise ValueError(f"Invalid --start-date: {args.start_date}")
        compute_start = start_date - pd.Timedelta(days=int(args.buffer_days))
        feats = feats.loc[feats["Date"] >= compute_start].copy()

    # ---- ingest trades files
    paths = sorted(signals_dir.glob(args.pattern))
    if args.also_read_csv:
        paths += sorted(signals_dir.glob(re.sub(r"\.parquet$", ".csv", args.pattern)))

    if not paths:
        raise FileNotFoundError(f"No trades files found: {signals_dir}/{args.pattern}")

    label_parts = []
    for p in paths:
        try:
            part = _parse_trades_file(p)
            if not part.empty:
                label_parts.append(part)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")

    if not label_parts:
        raise RuntimeError("No BadExit labels produced. Check trades inputs / pattern.")

    labels = pd.concat(label_parts, ignore_index=True)
    labels["Date"] = _norm_date(labels["Date"])
    labels["Ticker"] = labels["Ticker"].astype(str).str.upper().str.strip()
    labels["BadExit"] = pd.to_numeric(labels["BadExit"], errors="coerce").fillna(0).astype(int)

    # compute_start 적용 (buffer 고려)
    if compute_start is not None:
        labels = labels.loc[labels["Date"] >= compute_start].copy()

    labels = (
        labels.dropna(subset=["Date", "Ticker"])
        .groupby(["Date", "Ticker"], as_index=False)["BadExit"]
        .max()
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    # ---- merge to features dates (one-to-one expected)
    merged = feats[["Date", "Ticker"] + feature_cols].merge(
        labels[["Date", "Ticker", "BadExit"]],
        on=["Date", "Ticker"],
        how="inner",
        validate="one_to_one",
    )

    # output cut
    if start_date is not None:
        merged = merged.loc[merged["Date"] >= start_date].copy()

    merged = (
        merged.sort_values(["Date", "Ticker"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_csv)
    out_parq.parent.mkdir(parents=True, exist_ok=True)

    merged.to_parquet(out_parq, index=False)
    merged.to_csv(out_csv, index=False)

    print(f"[DONE] wrote: {out_parq} rows={len(merged)}")
    if len(merged):
        print(f"[INFO] range: {merged['Date'].min().date()}..{merged['Date'].max().date()}")
        print(f"[INFO] BadExit positive rate={merged['BadExit'].mean():.4f}")
        print(f"[INFO] feature_cols(18, forced): {feature_cols}")


if __name__ == "__main__":
    main()