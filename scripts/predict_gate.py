#!/usr/bin/env python3
from __future__ import annotations

"""
scripts/predict_gate.py

Gate picks generator.

- If tau_H exists and tail primitives (labels_tail_base_*.parquet/csv) are available,
  derive p_tail_effective using H=tau_H and EX=max_extend_days.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# IO helpers
# ----------------------------
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
    out = df.copy()

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
        raise KeyError("Date column missing.")
    if "Ticker" not in out.columns:
        raise KeyError("Ticker column missing.")

    return out


def apply_exclude_tickers(df: pd.DataFrame, exclude_csv: str) -> pd.DataFrame:
    ex = [t.strip().upper() for t in (exclude_csv or "").split(",") if t.strip()]
    if not ex:
        return df
    return df[~df["Ticker"].isin(ex)].copy()


# ----------------------------
# Tail primitives discovery
# ----------------------------
def _try_find_tail_base_file(pt: float, sl: float, exmax: int):
    pt100 = int(round(pt * 100))
    sl100 = int(round(abs(sl) * 100))

    base = Path("data/labels")
    if not base.exists():
        return None

    candidates = list(base.glob(
        f"labels_tail_base_pt{pt100}_sl{sl100}_hmax*_exmax{int(exmax)}.*"
    ))

    if not candidates:
        return None

    def _hmax_of(p: Path):
        try:
            part = p.stem.split("_hmax", 1)[1]
            return int(part.split("_exmax", 1)[0])
        except Exception:
            return -1

    candidates = sorted(candidates, key=_hmax_of, reverse=True)
    return candidates[0]


# ----------------------------
# Tail derivation (tau_H aware)
# ----------------------------
def _derive_p_tail_from_primitives(df: pd.DataFrame, ex: int, default_h: int):
    required = ["TauSLDays", "TauTPDays", "TauRecoverAfterSLDays"]
    for c in required:
        if c not in df.columns:
            return None, {"ok": False, "reason": f"missing {c}"}

    tau_sl = pd.to_numeric(df["TauSLDays"], errors="coerce")
    tau_tp = pd.to_numeric(df["TauTPDays"], errors="coerce")
    tau_rec = pd.to_numeric(df["TauRecoverAfterSLDays"], errors="coerce")

    if "tau_H" in df.columns:
        H_eff = pd.to_numeric(df["tau_H"], errors="coerce").fillna(default_h).astype(int)
    else:
        H_eff = pd.Series(default_h, index=df.index)

    sl_ok = tau_sl.notna() & (tau_sl <= H_eff)
    tp_before_sl = tau_tp.notna() & tau_sl.notna() & (tau_tp <= H_eff) & (tau_tp <= tau_sl)
    recovered = tau_rec.notna() & (tau_rec <= ex)

    p_tail_eff = (sl_ok & (~tp_before_sl) & recovered).astype(int)

    return p_tail_eff, {
        "ok": True,
        "tail_eff_rate": float(p_tail_eff.mean()) if len(p_tail_eff) else 0.0,
    }


# ----------------------------
# Utility / gating / ranking
# ----------------------------
def build_utility(df: pd.DataFrame, lambda_tail: float) -> pd.Series:
    ret = coerce_num(df, "ret_score", 0.0)
    p_tail = coerce_num(df, "p_tail", 0.0)
    return ret - lambda_tail * p_tail


def gate_filter(df: pd.DataFrame, mode: str, tail_threshold: float, utility_quantile: float):
    if df.empty:
        return df

    if mode == "none":
        return df

    if mode == "tail":
        return df[df["p_tail"] <= tail_threshold]

    if mode == "utility":
        thr = df.groupby("Date")["utility"].transform(lambda s: s.quantile(utility_quantile))
        return df[df["utility"] >= thr]

    if mode == "tail_utility":
        d = df[df["p_tail"] <= tail_threshold]
        if d.empty:
            return d
        thr = d.groupby("Date")["utility"].transform(lambda s: s.quantile(utility_quantile))
        return d[d["utility"] >= thr]

    raise ValueError("Invalid gate mode")


def rank_topk_per_day(df: pd.DataFrame, rank_by: str, topk: int):
    if df.empty:
        return df

    metric = df[rank_by]
    d = df.copy()
    d["_metric"] = metric
    d = d.sort_values(["Date", "_metric"], ascending=[True, False])

    return (
        d.groupby("Date", group_keys=False)
        .head(topk)
        .drop(columns="_metric")
    )


# ----------------------------
# MAIN
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--mode", required=True)
    ap.add_argument("--tail-threshold", type=float, required=True)
    ap.add_argument("--utility-quantile", type=float, required=True)
    ap.add_argument("--rank-by", required=True)
    ap.add_argument("--lambda-tail", type=float, required=True)

    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--ps-min", type=float, default=0.0)
    ap.add_argument("--badexit-max", type=float, default=1.0)

    ap.add_argument("--tag", required=True)
    ap.add_argument("--suffix", required=True)

    ap.add_argument("--exclude-tickers", default="")
    ap.add_argument("--features-path", default="")

    args = ap.parse_args()

    out_dir = Path("data/signals")
    out_dir.mkdir(parents=True, exist_ok=True)

    picks_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
    meta_path = out_dir / f"picks_meta_{args.tag}_gate_{args.suffix}.json"

    # load features
    if args.features_path:
        feats = pd.read_parquet(args.features_path)
    else:
        feats = read_table(
            Path("data/features/features_scored.parquet"),
            Path("data/features/features_scored.csv"),
        )

    feats = ensure_date_ticker_columns(feats)
    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()

    feats["p_success"] = coerce_num(feats, "p_success")
    feats["p_tail"] = coerce_num(feats, "p_tail")
    feats["ret_score"] = coerce_num(feats, "ret_score")
    feats["p_badexit"] = coerce_num(feats, "p_badexit")

    feats = feats[feats["p_success"] >= args.ps_min]
    feats = feats[feats["p_badexit"] <= args.badexit_max]

    # --- derive p_tail if primitives exist
    tail_base = _try_find_tail_base_file(
        args.profit_target,
        args.stop_level,
        args.max_extend_days,
    )

    tail_info = {"used": False}

    if tail_base:
        base = read_table(tail_base, tail_base)
        base = ensure_date_ticker_columns(base)
        base["Date"] = norm_date(base["Date"])
        base["Ticker"] = base["Ticker"].astype(str).str.upper().str.strip()

        merged = feats.merge(
            base,
            on=["Date", "Ticker"],
            how="left",
        )

        p_tail_eff, audit = _derive_p_tail_from_primitives(
            merged,
            args.max_extend_days,
            args.max_days,
        )

        if p_tail_eff is not None:
            merged["p_tail"] = p_tail_eff
            feats = merged
            tail_info = audit

    feats["utility"] = build_utility(feats, args.lambda_tail)

    gated = gate_filter(
        feats,
        args.mode,
        args.tail_threshold,
        args.utility_quantile,
    )

    picks = rank_topk_per_day(gated, args.rank_by, args.topk)

    picks.to_csv(picks_path, index=False)

    meta = {
        "rows_final": int(len(picks)),
        "tail_info": tail_info,
        "mode": args.mode,
        "rank_by": args.rank_by,
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] {picks_path} rows={len(picks)}")


if __name__ == "__main__":
    main()