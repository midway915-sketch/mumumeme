#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
FEAT_DIR = DATA_DIR / "features"
SIG_DIR = DATA_DIR / "signals"

DEFAULT_FEATS_SCORED_PARQ = FEAT_DIR / "features_scored.parquet"
DEFAULT_FEATS_SCORED_CSV = FEAT_DIR / "features_scored.csv"
DEFAULT_FEATS_MODEL_PARQ = FEAT_DIR / "features_model.parquet"
DEFAULT_FEATS_MODEL_CSV = FEAT_DIR / "features_model.csv"


# ----------------------------
# misc
# ----------------------------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def read_table(parq_path: Path, csv_path: Path) -> pd.DataFrame:
    if parq_path.exists():
        return pd.read_parquet(parq_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing file: {parq_path} (or {csv_path})")


def coerce_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return x.astype(float)


def ensure_date_ticker_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'Date' and 'Ticker' exist as columns.
    Robust to Date as index / date-like alternative names.
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
        raise KeyError(f"Date column missing. cols(head)={list(out.columns)[:30]} index.type={type(out.index).__name__}")
    if "Ticker" not in out.columns:
        raise KeyError(f"Ticker column missing. cols(head)={list(out.columns)[:30]} index.type={type(out.index).__name__}")

    return out


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


def gate_filter(df: pd.DataFrame, mode: str, tail_threshold: float, utility_quantile: float) -> pd.DataFrame:
    """
    No groupby().apply() to avoid Date column loss.
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
    """
    Accept:
      - "utility"
      - "utility,ret_score"
      - "ret_score,p_success"
    """
    items = [x.strip().lower() for x in (rank_by or "").split(",") if x.strip()]
    return items or ["utility"]


def ensure_rank_columns(d: pd.DataFrame, metrics: List[str]) -> None:
    ok = {"utility", "ret_score", "p_success"}
    bad = [m for m in metrics if m not in ok]
    if bad:
        raise ValueError(f"Unknown rank_by metrics: {bad} (allowed: utility|ret_score|p_success)")
    # columns presence is handled by coerce_num + utility build


def rank_topk_per_day_multi(df: pd.DataFrame, rank_by: str, topk: int) -> pd.DataFrame:
    d = ensure_date_ticker_columns(df)
    if d.empty:
        return d

    metrics = parse_rank_by_list(rank_by)
    ensure_rank_columns(d, metrics)

    # build sort keys (descending)
    sort_cols = []
    for m in metrics:
        if m == "utility":
            sort_cols.append("utility")
        elif m == "ret_score":
            sort_cols.append("ret_score")
        elif m == "p_success":
            sort_cols.append("p_success")

    # 안정 tie-break
    sort_cols_full = ["Date"] + sort_cols + ["Ticker"]
    ascending = [True] + [False] * len(sort_cols) + [True]

    d2 = d.copy()
    d2 = d2.sort_values(sort_cols_full, ascending=ascending)

    picks = (
        d2.groupby("Date", as_index=False, group_keys=False)
        .head(int(topk))
        .reset_index(drop=True)
    )
    return picks


def resolve_features_source(features_path: str | None) -> tuple[Path, Path, str]:
    """
    Priority:
      1) --features-path (parquet/csv)
      2) data/features/features_scored.(parquet|csv)
      3) data/features/features_model.(parquet|csv)
    """
    if features_path:
        fp = Path(features_path)
        if not fp.exists():
            raise FileNotFoundError(f"--features-path not found: {fp}")
        if fp.suffix.lower() == ".parquet":
            return fp, fp.with_suffix(".csv"), str(fp)
        return fp.with_suffix(".parquet"), fp, str(fp)

    if DEFAULT_FEATS_SCORED_PARQ.exists() or DEFAULT_FEATS_SCORED_CSV.exists():
        return DEFAULT_FEATS_SCORED_PARQ, DEFAULT_FEATS_SCORED_CSV, "features_scored"

    return DEFAULT_FEATS_MODEL_PARQ, DEFAULT_FEATS_MODEL_CSV, "features_model"


def fmt_tag(pt: float, H: int, sl: float, ex: int) -> str:
    pt_i = int(round(pt * 100))
    sl_i = int(round(abs(sl) * 100))
    return f"pt{pt_i}_h{int(H)}_sl{sl_i}_ex{int(ex)}"


# ----------------------------
# main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Gate picks generator (features_scored preferred).")

    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--mode", type=str, required=True, help="none|tail|utility|tail_utility")
    ap.add_argument("--tail-threshold", type=float, required=True)
    ap.add_argument("--utility-quantile", type=float, required=True)

    # ✅ allow "utility,ret_score"
    ap.add_argument("--rank-by", type=str, required=True, help="utility|ret_score|p_success or comma-list")
    ap.add_argument("--lambda-tail", type=float, required=True)

    # grid-lib passes these; keep for compatibility even if unused here
    ap.add_argument("--tau-gamma", type=float, default=float(os.getenv("TAU_GAMMA", "0.0")))
    ap.add_argument("--suffix", type=str, required=True)

    # outputs / misc
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument("--meta-json", type=str, default="")
    ap.add_argument("--features-path", type=str, default=None)
    ap.add_argument("--exclude-tickers", type=str, default="")

    # ✅ grid-lib에서 안 넘겨도 돌아가게 기본값 제공
    ap.add_argument("--topk", type=int, default=int(os.getenv("TOPK", "1")))
    ap.add_argument("--ps-min", type=float, default=float(os.getenv("PS_MIN", "0.0")))

    args = ap.parse_args()

    pt = float(args.profit_target)
    H = int(args.max_days)
    sl = float(args.stop_level)
    ex = int(args.max_extend_days)

    tag = fmt_tag(pt, H, sl, ex)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    meta_json = Path(args.meta_json) if args.meta_json else out_csv.with_suffix(".json")

    f_parq, f_csv, src_name = resolve_features_source(args.features_path)
    feats = read_table(f_parq, f_csv).copy()
    feats_src = str(f_parq if f_parq.exists() else f_csv)

    feats = ensure_date_ticker_columns(feats)
    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"]).reset_index(drop=True)

    # required model scores (없으면 0으로 채우고, 이후 gate에서 자연스럽게 걸러짐)
    feats["p_success"] = coerce_num(feats, "p_success", 0.0)
    feats["p_tail"] = coerce_num(feats, "p_tail", 0.0)
    feats["ret_score"] = coerce_num(feats, "ret_score", 0.0)

    feats = apply_exclude_tickers(feats, args.exclude_tickers)

    # base gate: p_success min
    ps_min = float(args.ps_min)
    if ps_min > 0:
        feats = feats[feats["p_success"] >= ps_min].copy()

    # utility build
    feats["utility"] = build_utility(feats, float(args.lambda_tail))

    # gate
    gated = gate_filter(
        feats,
        mode=args.mode,
        tail_threshold=float(args.tail_threshold),
        utility_quantile=float(args.utility_quantile),
    )

    # rank + topk
    picks = rank_topk_per_day_multi(gated, rank_by=args.rank_by, topk=int(args.topk))

    keep_cols = [c for c in ["Date", "Ticker", "p_success", "p_tail", "ret_score", "utility"] if c in picks.columns]
    picks_out = picks[keep_cols].copy()

    picks_out.to_csv(out_csv, index=False)

    meta = {
        "updated_at_utc": now_utc_iso(),
        "tag": tag,
        "suffix": args.suffix,
        "mode": args.mode,
        "rank_by": args.rank_by,
        "topk": int(args.topk),
        "ps_min": float(ps_min),
        "tail_threshold": float(args.tail_threshold),
        "utility_quantile": float(args.utility_quantile),
        "lambda_tail": float(args.lambda_tail),
        "tau_gamma": float(args.tau_gamma),
        "exclude_tickers": args.exclude_tickers,
        "features_src": feats_src,
        "features_kind": src_name,
        "rows_in_features": int(len(feats)),
        "rows_after_gate": int(len(gated)),
        "rows_picks": int(len(picks_out)),
        "profit_target": float(pt),
        "max_days": int(H),
        "stop_level": float(sl),
        "max_extend_days": int(ex),
        "out_csv": str(out_csv),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] wrote: {out_csv} rows={len(picks_out)}")
    print(f"[DONE] wrote: {meta_json}")


if __name__ == "__main__":
    main()