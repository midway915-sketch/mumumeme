#!/usr/bin/env python3
from __future__ import annotations

"""
scripts/predict_gate.py

Gate picks generator.

✅ NEW (B):
- If tau_H exists and tail primitives (labels_tail_base_*.parquet/csv) are available,
  derive p_tail_effective using H=tau_H and EX=max_extend_days, and use it for gating/utility.
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
        raise KeyError(f"Date column missing. cols(head)={list(out.columns)[:30]} index.name={out.index.name}")
    if "Ticker" not in out.columns:
        raise KeyError(f"Ticker column missing. cols(head)={list(out.columns)[:30]} index.name={out.index.name}")

    return out


def apply_exclude_tickers(df: pd.DataFrame, exclude_csv: str) -> pd.DataFrame:
    ex = [t.strip().upper() for t in (exclude_csv or "").split(",") if t.strip()]
    if not ex:
        return df
    return df[~df["Ticker"].isin(ex)].copy()


# ----------------------------
# Features source selection
# ----------------------------
def pick_features_source(require_files: List[Path]) -> Tuple[Path, Path]:
    for p in require_files:
        name = p.name
        if "features_scored" in name and (name.endswith(".parquet") or name.endswith(".csv")):
            if name.endswith(".parquet"):
                return p, p.with_suffix(".csv")
            return p.with_suffix(".parquet"), p

    return Path("data/features/features_scored.parquet"), Path("data/features/features_scored.csv")


# ----------------------------
# Tail primitives discovery + derivation
# ----------------------------
def _try_find_tail_base_file(pt: float, sl: float, exmax: int) -> tuple[Path | None, str]:
    """
    Find newest/most-covering labels_tail_base_ file for given pt/sl/exmax.
    Pattern: data/labels/labels_tail_base_pt{pt100}_sl{sl100}_hmax*_exmax{exmax}.parquet|csv
    Chooses the largest hmax available.
    """
    pt100 = int(round(float(pt) * 100))
    sl100 = int(round(abs(float(sl)) * 100))
    base = Path("data/labels")
    if not base.exists():
        return None, "data/labels missing"

    # glob parquet first
    pats = [
        base / f"labels_tail_base_pt{pt100}_sl{sl100}_hmax*_exmax{int(exmax)}.parquet",
        base / f"labels_tail_base_pt{pt100}_sl{sl100}_hmax*_exmax{int(exmax)}.csv",
    ]

    candidates: list[Path] = []
    for pat in pats:
        candidates.extend(sorted(base.glob(pat.name)))

    if not candidates:
        return None, "no tail_base candidates"

    # pick with largest hmax in filename
    def _hmax_of(p: Path) -> int:
        s = p.stem  # no suffix
        # ...hmax{H}_exmax{EX}
        try:
            part = s.split("_hmax", 1)[1]
            h_str = part.split("_exmax", 1)[0]
            return int(float(h_str))
        except Exception:
            return -1

    candidates = sorted(candidates, key=_hmax_of, reverse=True)
    return candidates[0], f"auto:{candidates[0].name}"


def _derive_p_tail_from_primitives(df: pd.DataFrame, ex: int, default_h: int) -> tuple[pd.Series, dict]:
    """
    Derive p_tail_effective (0/1) using:
      H_eff = tau_H if exists else default_h
      sl_ok = TauSLDays <= H_eff
      tp_before_sl = (TauTPDays <= H_eff) and (TauTPDays <= TauSLDays)
      recovered = TauRecoverAfterSLDays <= EX
      p_tail = 1 if sl_ok and (not tp_before_sl) and recovered else 0
    """
    d = df.copy()

    # required columns check
    need = ["TauSLDays", "TauTPDays", "TauRecoverAfterSLDays"]
    have = [c for c in need if c in d.columns]
    if len(have) != len(need):
        return pd.Series([np.nan] * len(d), index=d.index), {"ok": False, "reason": f"missing primitives: {set(need)-set(have)}"}

    H_eff = None
    if "tau_H" in d.columns:
        H_eff = pd.to_numeric(d["tau_H"], errors="coerce").fillna(default_h).astype(int)
    else:
        H_eff = pd.Series([int(default_h)] * len(d), index=d.index, dtype=int)

    tau_sl = pd.to_numeric(d["TauSLDays"], errors="coerce")
    tau_tp = pd.to_numeric(d["TauTPDays"], errors="coerce")
    tau_rec = pd.to_numeric(d["TauRecoverAfterSLDays"], errors="coerce")

    sl_ok = tau_sl.notna() & (tau_sl <= H_eff)
    tp_before_sl = tau_tp.notna() & (tau_tp <= H_eff) & tau_sl.notna() & (tau_tp <= tau_sl)
    recovered = tau_rec.notna() & (tau_rec <= int(ex))

    p_tail_eff = (sl_ok & (~tp_before_sl) & recovered).astype(int)

    audit = {
        "ok": True,
        "default_h": int(default_h),
        "ex": int(ex),
        "rows_with_tauH": int(d["tau_H"].notna().sum()) if "tau_H" in d.columns else 0,
        "tail_eff_rate": float(p_tail_eff.mean()) if len(p_tail_eff) else 0.0,
    }
    return p_tail_eff, audit


# ----------------------------
# Utility / gating / ranking
# ----------------------------
def build_utility(df: pd.DataFrame, lambda_tail: float) -> pd.Series:
    ret = coerce_num(df, "ret_score", 0.0)
    p_tail = coerce_num(df, "p_tail", 0.0)
    return ret - float(lambda_tail) * p_tail


def gate_filter(df: pd.DataFrame, mode: str, tail_threshold: float, utility_quantile: float) -> pd.DataFrame:
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


def rank_topk_per_day(df: pd.DataFrame, rank_by: str, topk: int) -> pd.DataFrame:
    d = ensure_date_ticker_columns(df)
    if d.empty:
        return d

    rb = (rank_by or "utility").strip().lower()
    if rb == "utility":
        metric = d["utility"]
    elif rb == "ret_score":
        metric = d["ret_score"]
    elif rb == "p_success":
        metric = d["p_success"]
    else:
        raise ValueError(f"Unknown rank_by: {rank_by} (expected utility|ret_score|p_success)")

    d = d.copy()
    d["_metric"] = metric
    d = d.sort_values(["Date", "_metric"], ascending=[True, False])

    picks = (
        d.groupby("Date", as_index=False, group_keys=False)
        .head(int(topk))
        .drop(columns=["_metric"])
        .reset_index(drop=True)
    )
    return picks


def _default_out_paths(out_dir: Path, tag: str, suffix: str) -> tuple[Path, Path]:
    return (
        out_dir / f"picks_{tag}_gate_{suffix}.csv",
        out_dir / f"picks_meta_{tag}_gate_{suffix}.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate picks generator (reads features_scored if available).")

    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--mode", type=str, required=True, help="none|tail|utility|tail_utility")
    ap.add_argument("--tail-threshold", type=float, required=True)
    ap.add_argument("--utility-quantile", type=float, required=True)
    ap.add_argument("--rank-by", type=str, required=True, help="utility|ret_score|p_success")
    ap.add_argument("--lambda-tail", type=float, required=True)

    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--ps-min", type=float, default=0.0)

    ap.add_argument("--badexit-max", type=float, default=1.0, help="keep rows with p_badexit <= this (default 1.0)")

    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--suffix", type=str, required=True)

    ap.add_argument("--exclude-tickers", type=str, default="")

    # compat outputs
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--require-files", type=str, default="")
    ap.add_argument("--out-csv", type=str, default="")
    ap.add_argument("--meta-json", type=str, default="")

    ap.add_argument("--features-path", type=str, default="")

    args = ap.parse_args()

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else Path("data/signals")
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(args.out_csv).strip():
        picks_path = Path(args.out_csv)
        meta_path = Path(args.meta_json) if str(args.meta_json).strip() else picks_path.with_name(
            picks_path.name.replace("picks_", "picks_meta_").replace(".csv", ".json")
        )
    else:
        picks_path, meta_path = _default_out_paths(out_dir, args.tag, args.suffix)

    require_files = parse_require_files(args.require_files)
    if require_files:
        ensure_required_files_exist(require_files)

    # load features
    if str(args.features_path).strip():
        fp = Path(args.features_path)
        if not fp.exists():
            raise FileNotFoundError(f"features-path not found: {fp}")
        feats = pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
        features_src = str(fp)
    else:
        f_parq, f_csv = pick_features_source(require_files)
        if (not f_parq.exists()) and (not f_csv.exists()):
            f_parq = Path("data/features/features_model.parquet")
            f_csv = Path("data/features/features_model.csv")
        feats = read_table(f_parq, f_csv)
        features_src = str(f_parq) if f_parq.exists() else str(f_csv)

    feats = ensure_date_ticker_columns(feats)
    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # numeric columns (missing -> defaults)
    feats["p_success"] = coerce_num(feats, "p_success", 0.0)
    feats["p_tail"] = coerce_num(feats, "p_tail", 0.0)
    feats["ret_score"] = coerce_num(feats, "ret_score", 0.0)
    feats["p_badexit"] = coerce_num(feats, "p_badexit", 0.0)

    # tau_H (optional)
    if "tau_H" in feats.columns:
        feats["tau_H"] = pd.to_numeric(feats["tau_H"], errors="coerce").fillna(float(args.max_days)).astype(int)

    rows_loaded = int(len(feats))

    feats = apply_exclude_tickers(feats, args.exclude_tickers)
    rows_after_exclude = int(len(feats))

    # -------------------------
    # ✅ NEW: derive p_tail from primitives using tau_H
    # -------------------------
    tail_base_path, tail_base_src = _try_find_tail_base_file(
        pt=float(args.profit_target),
        sl=float(args.stop_level),
        exmax=int(args.max_extend_days),
    )
    tail_derivation = {"ok": False, "reason": "not attempted"}
    p_tail_source = "model_or_default"

    if tail_base_path is not None and tail_base_path.exists():
        # load base
        if tail_base_path.suffix.lower() == ".parquet":
            base_df = pd.read_parquet(tail_base_path)
        else:
            base_df = pd.read_csv(tail_base_path)

        base_df = ensure_date_ticker_columns(base_df)
        base_df["Date"] = norm_date(base_df["Date"])
        base_df["Ticker"] = base_df["Ticker"].astype(str).str.upper().str.strip()
        base_df = base_df.dropna(subset=["Date", "Ticker"]).reset_index(drop=True)

        # merge
        m = feats.merge(
            base_df[["Date", "Ticker", "TauSLDays", "TauTPDays", "TauRecoverAfterSLDays", "TailPmaxEXmax"]],
            on=["Date", "Ticker"],
            how="left",
            validate="many_to_one",
        )

        # derive
        p_tail_eff, audit = _derive_p_tail_from_primitives(
            m,
            ex=int(args.max_extend_days),
            default_h=int(args.max_days),
        )
        tail_derivation = {"base_src": tail_base_src, "base_path": str(tail_base_path), **audit}

        if audit.get("ok", False):
            m["p_tail"] = pd.to_numeric(p_tail_eff, errors="coerce").fillna(0).astype(int)
            feats = m
            p_tail_source = "derived(tau_H+tail_base)"
        else:
            feats = m
            p_tail_source = "model_or_default+base_merge_failed"
    else:
        tail_derivation = {"ok": False, "reason": "tail_base not found", "base_src": tail_base_src}
        p_tail_source = "model_or_default"

    # 1) p_success min
    ps_min = float(args.ps_min)
    feats = feats[feats["p_success"] >= ps_min].copy()
    rows_after_psmin = int(len(feats))

    # 2) p_badexit max
    badexit_max = float(args.badexit_max)
    feats = feats[feats["p_badexit"] <= badexit_max].copy()
    rows_after_badexit = int(len(feats))

    # 3) utility + gate
    feats["utility"] = build_utility(feats, float(args.lambda_tail))

    gated = gate_filter(
        feats,
        mode=args.mode,
        tail_threshold=float(args.tail_threshold),
        utility_quantile=float(args.utility_quantile),
    )
    rows_after_gate = int(len(gated))

    picks = rank_topk_per_day(gated, rank_by=args.rank_by, topk=int(args.topk))

    keep_cols = [
        c
        for c in [
            "Date",
            "Ticker",
            "p_success",
            "p_tail",
            "p_badexit",
            "ret_score",
            "utility",
            "tau_H",
            "TauSLDays",
            "TauTPDays",
            "TauRecoverAfterSLDays",
            "TailPmaxEXmax",
        ]
        if c in picks.columns
    ]
    picks_out = picks[keep_cols].copy()

    picks_path.parent.mkdir(parents=True, exist_ok=True)
    picks_out.to_csv(picks_path, index=False)

    meta = {
        "tag": args.tag,
        "suffix": args.suffix,
        "mode": args.mode,
        "rank_by": args.rank_by,
        "topk": int(args.topk),

        "ps_min": float(ps_min),
        "badexit_max": float(badexit_max),

        "tail_threshold": float(args.tail_threshold),
        "utility_quantile": float(args.utility_quantile),
        "lambda_tail": float(args.lambda_tail),

        "exclude_tickers": args.exclude_tickers,
        "features_src": features_src,
        "require_files": [str(p) for p in require_files],

        # ✅ tail source
        "p_tail_source": p_tail_source,
        "tail_derivation": tail_derivation,

        # ✅ row audit trail
        "rows_loaded": rows_loaded,
        "rows_after_exclude": rows_after_exclude,
        "rows_after_psmin": rows_after_psmin,
        "rows_after_badexit": rows_after_badexit,
        "rows_after_gate": rows_after_gate,
        "rows_picks": int(len(picks_out)),

        # params for traceability
        "profit_target": float(args.profit_target),
        "max_days": int(args.max_days),
        "stop_level": float(args.stop_level),
        "max_extend_days": int(args.max_extend_days),
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] wrote: {picks_path} rows={len(picks_out)}")
    print(f"[DONE] wrote: {meta_path}")


if __name__ == "__main__":
    main()