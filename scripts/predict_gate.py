#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

def ensure_date_ticker_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Date가 인덱스에 들어간 경우 -> 컬럼으로 복구
    if "Date" not in out.columns:
        idx_name = (out.index.name or "").lower()
        if idx_name in ("date", "datetime", "timestamp"):
            out = out.reset_index()

    # 대소문자/대체이름을 Date/Ticker로 통일
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
        raise KeyError(f"Date column missing. columns(head)={list(out.columns)[:30]} index.name={out.index.name}")
    if "Ticker" not in out.columns:
        raise KeyError(f"Ticker column missing. columns(head)={list(out.columns)[:30]} index.name={out.index.name}")

    return out

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


def pick_features_source(require_files: List[Path]) -> tuple[Path, Path]:
    """
    Prefer features_scored if present in require_files, else fallback to features_model.
    Return (parq_path, csv_path) candidates (one of them must exist or later fallback will try).
    """
    # 1) if require_files includes a features_scored parquet/csv, honor it
    for p in require_files:
        name = p.name
        if "features_scored" in name and (name.endswith(".parquet") or name.endswith(".csv")):
            if name.endswith(".parquet"):
                return p, p.with_suffix(".csv")
            else:
                return p.with_suffix(".parquet"), p

    # 2) default paths
    return Path("data/features/features_scored.parquet"), Path("data/features/features_scored.csv")


def apply_exclude_tickers(df: pd.DataFrame, exclude_csv: str) -> pd.DataFrame:
    ex = [t.strip().upper() for t in (exclude_csv or "").split(",") if t.strip()]
    if not ex:
        return df
    return df[~df["Ticker"].isin(ex)].copy()


def build_utility(df: pd.DataFrame, lambda_tail: float) -> pd.Series:
    # utility = ret_score - lambda * p_tail
    ret = coerce_num(df, "ret_score", 0.0)
    p_tail = coerce_num(df, "p_tail", 0.0)
    return ret - float(lambda_tail) * p_tail


def gate_filter(
    df: pd.DataFrame,
    mode: str,
    tail_threshold: float,
    utility_quantile: float,
) -> pd.DataFrame:
    mode = (mode or "none").strip().lower()
    d = df.copy()

    if mode == "none":
        return d

    if mode == "tail":
        return d[d["p_tail"] <= float(tail_threshold)].copy()

    if mode == "utility":
        q = float(utility_quantile)

        def _f(g: pd.DataFrame) -> pd.DataFrame:
            if len(g) == 0:
                return g
            thr = g["utility"].quantile(q)
            return g[g["utility"] >= thr]

        return d.groupby("Date", group_keys=False).apply(_f).reset_index(drop=True)

    if mode == "tail_utility":
        d = d[d["p_tail"] <= float(tail_threshold)].copy()
        q = float(utility_quantile)

        def _f(g: pd.DataFrame) -> pd.DataFrame:
            if len(g) == 0:
                return g
            thr = g["utility"].quantile(q)
            return g[g["utility"] >= thr]

        return d.groupby("Date", group_keys=False).apply(_f).reset_index(drop=True)

    raise ValueError(f"Unknown mode: {mode} (expected none|tail|utility|tail_utility)")


def rank_and_topk(df: pd.DataFrame, rank_by: str, topk: int) -> pd.DataFrame:
    rb = (rank_by or "utility").strip().lower()
    d = df.copy()

    if rb == "utility":
        metric = d["utility"]
    elif rb == "ret_score":
        metric = d["ret_score"]
    elif rb == "p_success":
        metric = d["p_success"]
    else:
        raise ValueError(f"Unknown rank_by: {rank_by} (expected utility|ret_score|p_success)")

    d["_metric"] = metric
    d = d.sort_values(["Date", "_metric"], ascending=[True, False])

    picks = (
        d.groupby("Date", as_index=False, group_keys=False)
        .head(int(topk))
        .drop(columns=["_metric"])
        .reset_index(drop=True)
    )
    return picks


def ensure_required_files_exist(require_files: List[Path]) -> None:
    missing = [str(p) for p in require_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate picks generator (reads features_scored if available).")

    # args used only for metadata compatibility with your pipeline
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--mode", type=str, required=True, help="none|tail|utility|tail_utility")
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--suffix", type=str, required=True)

    ap.add_argument("--tail-threshold", type=float, required=True)
    ap.add_argument("--utility-quantile", type=float, required=True)
    ap.add_argument("--rank-by", type=str, required=True, help="utility|ret_score|p_success")
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

    # decide which features file to read
    f_parq, f_csv = pick_features_source(require_files)

    # fallback: if scored not found, try features_model
    if (not f_parq.exists()) and (not f_csv.exists()):
        f_parq = Path("data/features/features_model.parquet")
        f_csv = Path("data/features/features_model.csv")

    feats = read_table(f_parq, f_csv).copy()
    features_src = str(f_parq) if f_parq.exists() else str(f_csv)

    # --- normalize Date/Ticker even if names differ or Date is index
    feats = ensure_date_ticker_columns(feats)

    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    # ensure key columns exist
    feats["p_success"] = coerce_num(feats, "p_success", 0.0)
    feats["p_tail"] = coerce_num(feats, "p_tail", 0.0)
    feats["ret_score"] = coerce_num(feats, "ret_score", 0.0)

    feats = apply_exclude_tickers(feats, args.exclude_tickers)

    # base filter: ps_min
    feats = feats[feats["p_success"] >= float(args.ps_min)].copy()

    # utility column
    feats["utility"] = build_utility(feats, float(args.lambda_tail))

    # gate
    gated = gate_filter(
        feats,
        mode=args.mode,
        tail_threshold=float(args.tail_threshold),
        utility_quantile=float(args.utility_quantile),
    )

    # rank + topk per day
    picks = rank_and_topk(gated, rank_by=args.rank_by, topk=int(args.topk))

    # output path MUST match run_grid_workflow.sh
    picks_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
    meta_path = out_dir / f"picks_meta_{args.tag}_gate_{args.suffix}.json"

    # Keep columns simple & stable for simulator
    keep_cols = []
    for c in ["Date", "Ticker", "p_success", "p_tail", "ret_score", "utility"]:
        if c in picks.columns:
            keep_cols.append(c)
    picks_out = picks[keep_cols].copy()

    # write CSV
    picks_out.to_csv(picks_path, index=False)

    meta = {
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