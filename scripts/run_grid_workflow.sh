#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# IO helpers
# ----------------------------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_table(parq_path: Path, csv_path: Path) -> pd.DataFrame:
    if parq_path.exists():
        return pd.read_parquet(parq_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing file: {parq_path} (or {csv_path})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def coerce_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return x.astype(float)


def parse_require_files(s: str) -> List[Path]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    return [Path(p) for p in parts]


def ensure_required_files_exist(require_files: List[Path]) -> None:
    missing = [str(p) for p in require_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


# ----------------------------
# Column normalization
# ----------------------------
def ensure_date_ticker_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'Date' and 'Ticker' exist as columns.
    Robust to:
      - Date as index (DatetimeIndex / convertible index)
      - date/datetime/timestamp column names
      - ticker/symbol column names
    """
    out = df.copy()

    if "Date" not in out.columns:
        if not isinstance(out.index, pd.RangeIndex):
            idx = out.index
            if isinstance(idx, pd.DatetimeIndex):
                out = out.reset_index()
                if "index" in out.columns and "Date" not in out.columns:
                    out = out.rename(columns={"index": "Date"})
            else:
                try:
                    tmp = pd.to_datetime(idx, errors="coerce")
                    if tmp.notna().any():
                        out = out.reset_index()
                        if "index" in out.columns and "Date" not in out.columns:
                            out = out.rename(columns={"index": "Date"})
                except Exception:
                    pass

    colmap = {c.lower(): c for c in out.columns}

    if "date" in colmap and "Date" not in out.columns:
        out = out.rename(columns={colmap["date"]: "Date"})
    if "datetime" in colmap and "Date" not in out.columns:
        out = out.rename(columns={colmap["datetime"]: "Date"})
    if "timestamp" in colmap and "Date" not in out.columns:
        out = out.rename(columns={colmap["timestamp"]: "Date"})

    if "ticker" in colmap and "Ticker" not in out.columns:
        out = out.rename(columns={colmap["ticker"]: "Ticker"})
    if "symbol" in colmap and "Ticker" not in out.columns:
        out = out.rename(columns={colmap["symbol"]: "Ticker"})

    if "Date" not in out.columns:
        raise KeyError(
            f"Date column missing. cols(head)={list(out.columns)[:30]} "
            f"index.name={out.index.name} index.type={type(out.index).__name__}"
        )
    if "Ticker" not in out.columns:
        raise KeyError(
            f"Ticker column missing. cols(head)={list(out.columns)[:30]} "
            f"index.name={out.index.name} index.type={type(out.index).__name__}"
        )
    return out


def apply_exclude_tickers(df: pd.DataFrame, exclude_csv: str) -> pd.DataFrame:
    ex = [t.strip().upper() for t in (exclude_csv or "").split(",") if t.strip()]
    if not ex:
        return df
    return df[~df["Ticker"].isin(ex)].copy()


# ----------------------------
# Features source selection (require-files 기반)
# ----------------------------
def pick_features_source(require_files: List[Path]) -> Tuple[Path, Path, str]:
    """
    1) require_files 안에 features_scored(.parquet/.csv)가 있으면 그걸 우선
    2) 없으면 data/features/features_scored.* 를 시도
    3) 그것도 없으면 data/features/features_model.* 로 fallback
    """
    for p in require_files:
        name = p.name
        if "features_scored" in name and (name.endswith(".parquet") or name.endswith(".csv")):
            if name.endswith(".parquet"):
                return p, p.with_suffix(".csv"), f"require_files:{p.name}"
            return p.with_suffix(".parquet"), p, f"require_files:{p.name}"

    scored_p = Path("data/features/features_scored.parquet")
    scored_c = Path("data/features/features_scored.csv")
    if scored_p.exists() or scored_c.exists():
        return scored_p, scored_c, "default:features_scored"

    return Path("data/features/features_model.parquet"), Path("data/features/features_model.csv"), "fallback:features_model"


# ----------------------------
# Utility / gating / ranking
# ----------------------------
def build_utility(df: pd.DataFrame, lambda_tail: float) -> pd.Series:
    # utility = ret_score - lambda * p_tail
    ret = coerce_num(df, "ret_score", 0.0)
    p_tail = coerce_num(df, "p_tail", 0.0)
    return ret - float(lambda_tail) * p_tail


def gate_filter(df: pd.DataFrame, mode: str, tail_threshold: float, utility_quantile: float) -> pd.DataFrame:
    """
    IMPORTANT: No groupby().apply().
    Use transform so 'Date' never disappears.
    """
    d = ensure_date_ticker_columns(df)
    mode = (mode or "none").strip().lower()

    if d.empty:
        return d

    if mode == "none":
        return d.copy()

    if mode == "tail":
        return d[d["p_tail"] <= float(tail_threshold)].copy()

    if mode == "utility":
        q = float(utility_quantile)
        thr = d.groupby("Date")["utility"].transform(lambda s: s.quantile(q))
        return d[d["utility"] >= thr].copy()

    if mode == "tail_utility":
        q = float(utility_quantile)
        d2 = d[d["p_tail"] <= float(tail_threshold)].copy()
        if d2.empty:
            return d2
        thr = d2.groupby("Date")["utility"].transform(lambda s: s.quantile(q))
        return d2[d2["utility"] >= thr].copy()

    raise ValueError(f"Unknown mode: {mode} (expected none|tail|utility|tail_utility)")


def parse_rank_by_list(rank_by: str) -> List[str]:
    # accept "utility" or "utility,ret_score" etc.
    items = [x.strip().lower() for x in (rank_by or "").split(",") if x.strip()]
    return items or ["utility"]


def rank_topk_per_day(df: pd.DataFrame, rank_by: str, topk: int) -> pd.DataFrame:
    d = ensure_date_ticker_columns(df)
    if d.empty:
        return d

    metrics = parse_rank_by_list(rank_by)
    allowed = {"utility", "ret_score", "p_success"}
    bad = [m for m in metrics if m not in allowed]
    if bad:
        raise ValueError(f"Unknown rank_by metrics: {bad} (allowed: utility|ret_score|p_success)")

    # stable multi-key sort (desc) + tie-break by Ticker asc
    sort_cols = ["Date"] + metrics + ["Ticker"]
    ascending = [True] + [False] * len(metrics) + [True]

    d2 = d.copy()
    # make sure columns exist (utility exists already; others are coerced earlier)
    d2 = d2.sort_values(sort_cols, ascending=ascending)

    picks = (
        d2.groupby("Date", as_index=False, group_keys=False)
        .head(int(topk))
        .reset_index(drop=True)
    )
    return picks


def fmt_tag(pt: float, H: int, sl: float, ex: int) -> str:
    pt_i = int(round(float(pt) * 100))
    sl_i = int(round(abs(float(sl)) * 100))
    return f"pt{pt_i}_h{int(H)}_sl{sl_i}_ex{int(ex)}"


# ----------------------------
# main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Gate picks generator (grid-friendly).")

    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--mode", type=str, required=True, help="none|tail|utility|tail_utility")
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--suffix", type=str, required=True)

    ap.add_argument("--tail-threshold", type=float, required=True)
    ap.add_argument("--utility-quantile", type=float, required=True)
    ap.add_argument("--rank-by", type=str, required=True, help="utility|ret_score|p_success or comma-list")
    ap.add_argument("--lambda-tail", type=float, required=True)

    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--ps-min", type=float, required=True)

    ap.add_argument("--exclude-tickers", type=str, default="")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--require-files", type=str, required=True)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    require_files = parse_require_files(args.require_files)
    ensure_required_files_exist(require_files)

    # choose features file (prefer scored)
    f_parq, f_csv, source_hint = pick_features_source(require_files)

    # read features
    feats = read_table(f_parq, f_csv).copy()
    features_src = str(f_parq) if f_parq.exists() else str(f_csv)

    feats = ensure_date_ticker_columns(feats)
    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = (
        feats.dropna(subset=["Date", "Ticker"])
        .sort_values(["Date", "Ticker"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    # required columns (없으면 0으로 채움)
    feats["p_success"] = coerce_num(feats, "p_success", 0.0)
    feats["p_tail"] = coerce_num(feats, "p_tail", 0.0)
    feats["ret_score"] = coerce_num(feats, "ret_score", 0.0)

    feats = apply_exclude_tickers(feats, args.exclude_tickers)

    # base gate: p_success min
    feats = feats[feats["p_success"] >= float(args.ps_min)].copy()

    feats["utility"] = build_utility(feats, float(args.lambda_tail))

    gated = gate_filter(
        feats,
        mode=args.mode,
        tail_threshold=float(args.tail_threshold),
        utility_quantile=float(args.utility_quantile),
    )

    picks = rank_topk_per_day(gated, rank_by=args.rank_by, topk=int(args.topk))

    picks_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
    meta_path = out_dir / f"picks_meta_{args.tag}_gate_{args.suffix}.json"

    keep_cols = [c for c in ["Date", "Ticker", "p_success", "p_tail", "ret_score", "utility"] if c in picks.columns]
    picks_out = picks[keep_cols].copy()
    picks_out.to_csv(picks_path, index=False)

    meta = {
        "updated_at_utc": now_utc_iso(),
        "tag": args.tag,
        "suffix": args.suffix,
        "mode": args.mode,
        "rank_by": args.rank_by,
        "topk": int(args.topk),
        "ps_min": float(args.ps_min),
        "tail_threshold": float(args.tail_threshold),
        "utility_quantile": float(args.utility_quantile),
        "lambda_tail": float(args.lambda_tail),
        "exclude_tickers": args.exclude_tickers,
        "features_src": features_src,
        "features_src_hint": source_hint,
        "rows_in_features": int(len(feats)),
        "rows_after_gate": int(len(gated)),
        "rows_picks": int(len(picks_out)),
        "profit_target": float(args.profit_target),
        "max_days": int(args.max_days),
        "stop_level": float(args.stop_level),
        "max_extend_days": int(args.max_extend_days),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] wrote: {picks_path} rows={len(picks_out)}")
    print(f"[DONE] wrote: {meta_path}")


if __name__ == "__main__":
    main()