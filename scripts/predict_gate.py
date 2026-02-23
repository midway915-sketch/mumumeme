#!/usr/bin/env python3
"""
Generate daily picks (TopK) for gate.
Writes:
- picks_{tag}_gate_{suffix}.csv
- picks_{tag}_gate_{suffix}.debug.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path("app")
DATA_DIR = Path("data")
FEATURES_DIR = DATA_DIR / "features"


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def _load_features(features_path: str) -> pd.DataFrame:
    # if explicit features_path given, use it; else fallback to features_scored
    if features_path:
        p = Path(features_path)
        if p.exists():
            if p.suffix.lower() == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
        else:
            raise FileNotFoundError(f"--features-path not found: {p}")
    else:
        p_parq = FEATURES_DIR / "features_scored.parquet"
        p_csv = FEATURES_DIR / "features_scored.csv"
        df = read_table(p_parq, p_csv)

    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("features must include Date,Ticker")

    df = df.copy()
    df["Date"] = _norm_date(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["Date", "Ticker"]).reset_index(drop=True)
    return df


def _load_models():
    model_p = APP_DIR / "model.pkl"
    scaler_p = APP_DIR / "scaler.pkl"
    if not model_p.exists() or not scaler_p.exists():
        raise FileNotFoundError(f"Missing model/scaler: {model_p} {scaler_p}")
    import joblib

    model = joblib.load(model_p)
    scaler = joblib.load(scaler_p)
    return model, scaler


def _load_tail_models():
    model_p = APP_DIR / "tail_model.pkl"
    scaler_p = APP_DIR / "tail_scaler.pkl"
    if not model_p.exists() or not scaler_p.exists():
        raise FileNotFoundError(f"Missing tail model/scaler: {model_p} {scaler_p}")
    import joblib

    model = joblib.load(model_p)
    scaler = joblib.load(scaler_p)
    return model, scaler


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    # try to use meta feature_cols if present
    meta = DATA_DIR / "meta" / "feature_cols.json"
    if meta.exists():
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
            cols = payload.get("feature_cols") or payload.get("cols") or payload.get("features")
            if isinstance(cols, list) and cols:
                cols = [str(c).strip() for c in cols if str(c).strip()]
                missing = [c for c in cols if c not in df.columns]
                if missing:
                    raise ValueError(f"feature_cols.json contains missing cols in features: {missing[:20]}")
                return cols
        except Exception:
            pass

    # fallback: numeric columns except known keys/probs
    drop = {"Date", "Ticker", "p_success", "p_tail", "tau_H", "tau_class", "ret_score", "utility"}
    cols = [c for c in df.columns if c not in drop]
    # only keep columns that look numeric-ish
    out = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    if not out:
        raise RuntimeError("No numeric feature cols found.")
    return out


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--mode", required=True, choices=["none", "tail", "utility", "tail_utility"])
    ap.add_argument("--tail-threshold", required=True, type=float)
    ap.add_argument("--utility-quantile", required=True, type=float)
    ap.add_argument("--rank-by", required=True, choices=["utility", "ret_score", "p_success"])
    ap.add_argument("--lambda-tail", required=True, type=float)

    ap.add_argument("--topk", default=1, type=int)
    ap.add_argument("--ps-min", default=0.0, type=float)
    ap.add_argument("--badexit-max", default=None, type=float)

    ap.add_argument("--tag", required=True)
    ap.add_argument("--suffix", required=True)

    ap.add_argument("--exclude-tickers", default="")
    ap.add_argument("--features-path", default="")

    # ✅ workflow compat (run_grid_workflow.sh)
    ap.add_argument("--out-dir", default="data/signals", help="output directory for picks/meta")
    ap.add_argument(
        "--require-files",
        default="",
        help="comma-separated file paths that must exist; missing -> exit code 2",
    )

    args = ap.parse_args()

    # ---- fail-fast required files (CI safety)
    req = [x.strip() for x in str(args.require_files or "").split(",") if x.strip()]
    if req:
        missing = [p for p in req if not Path(p).exists()]
        if missing:
            print(f"[ERROR] Missing required files: {missing}")
            sys.exit(2)

    if int(args.topk) < 1:
        raise ValueError("--topk must be >= 1")

    feats = _load_features(args.features_path)

    # model probs (p_success)
    model, scaler = _load_models()
    feat_cols = _get_feature_cols(feats)
    feats2 = _coerce_numeric(feats, feat_cols)

    X = feats2[feat_cols].to_numpy(dtype=float)
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)
    # assume class 1 is success
    if proba.shape[1] >= 2:
        p_success = proba[:, 1]
    else:
        p_success = proba[:, 0]

    feats2["p_success"] = p_success

    # tail probs (p_tail)
    tail_model, tail_scaler = _load_tail_models()
    Xt = tail_scaler.transform(X)
    tproba = tail_model.predict_proba(Xt)
    # assume class 1 is tail (bad tail)
    if tproba.shape[1] >= 2:
        p_tail = tproba[:, 1]
    else:
        p_tail = tproba[:, 0]
    feats2["p_tail"] = p_tail

    # basic filters
    if args.exclude_tickers:
        ex = [x.strip().upper() for x in args.exclude_tickers.split(",") if x.strip()]
        if ex:
            feats2 = feats2[~feats2["Ticker"].isin(ex)].copy()

    feats2 = feats2[feats2["p_success"] >= float(args.ps_min)].copy()
    if feats2.empty:
        # still write empty picks for reproducibility
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        picks_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
        debug_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.debug.json"
        pd.DataFrame(columns=["Date", "Ticker"]).to_csv(picks_path, index=False)
        debug_path.write_text(json.dumps({"empty": True, "reason": "ps_min_filter"}, indent=2), encoding="utf-8")
        print(f"[WARN] empty picks (ps_min={args.ps_min}) -> wrote {picks_path}")
        return

    # compute score by mode
    mode = args.mode
    tmax = float(args.tail_threshold)
    uq = float(args.utility_quantile)
    lam = float(args.lambda_tail)

    # utility: penalize tail if enabled
    feats2["ret_score"] = feats2.get("ret_score", np.nan)
    feats2["utility_raw"] = feats2.get("utility", np.nan)

    # fallback utility if not present: use p_success
    if "utility" not in feats2.columns or feats2["utility_raw"].isna().all():
        feats2["utility_raw"] = feats2["p_success"].astype(float)

    # tail penalty
    feats2["tail_pen"] = np.maximum(0.0, feats2["p_tail"] - tmax)

    # quantile filter (utility)
    qv = feats2["utility_raw"].quantile(uq) if np.isfinite(uq) else feats2["utility_raw"].min()
    feats2["utility_qpass"] = feats2["utility_raw"] >= qv

    # final score by mode
    if mode == "none":
        feats2["score"] = feats2["utility_raw"]
    elif mode == "tail":
        feats2["score"] = feats2["utility_raw"] - lam * feats2["tail_pen"]
    elif mode == "utility":
        feats2 = feats2[feats2["utility_qpass"]].copy()
        feats2["score"] = feats2["utility_raw"]
    elif mode == "tail_utility":
        feats2 = feats2[feats2["utility_qpass"]].copy()
        feats2["score"] = feats2["utility_raw"] - lam * feats2["tail_pen"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # rank_by override
    if args.rank_by == "p_success":
        feats2["score"] = feats2["p_success"]
    elif args.rank_by == "ret_score":
        if "ret_score" not in feats2.columns or feats2["ret_score"].isna().all():
            feats2["ret_score"] = feats2["utility_raw"]
        feats2["score"] = feats2["ret_score"].fillna(-np.inf)

    # optional badexit filter (if column exists in features)
    if args.badexit_max is not None and "p_badexit" in feats2.columns:
        feats2 = feats2[feats2["p_badexit"] <= float(args.badexit_max)].copy()

    # pick topk per date
    feats2 = feats2.sort_values(["Date", "score"], ascending=[True, False]).reset_index(drop=True)

    picks = (
        feats2.groupby("Date", group_keys=False)
        .head(int(args.topk))[["Date", "Ticker", "score", "p_success", "p_tail"]]
        .copy()
    )

    # write outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    picks_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
    debug_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.debug.json"

    picks[["Date", "Ticker"]].to_csv(picks_path, index=False)

    dbg = {
        "tag": args.tag,
        "suffix": args.suffix,
        "mode": mode,
        "tail_threshold": tmax,
        "utility_quantile": uq,
        "rank_by": args.rank_by,
        "lambda_tail": lam,
        "topk": int(args.topk),
        "ps_min": float(args.ps_min),
        "badexit_max": float(args.badexit_max) if args.badexit_max is not None else None,
        "exclude_tickers": args.exclude_tickers,
        "features_path": args.features_path or "data/features/features_scored.(parquet/csv)",
        "out_dir": str(out_dir),
        "n_rows_features": int(len(feats2)),
        "n_rows_picks": int(len(picks)),
    }
    debug_path.write_text(json.dumps(dbg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] wrote picks: {picks_path} rows={len(picks)}")
    print(f"[DONE] wrote debug: {debug_path}")


if __name__ == "__main__":
    main()