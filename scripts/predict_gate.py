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
    drop = {"Date", "Ticker", "p_success", "p_tail", "p_badexit", "tau_H", "tau_class", "ret_score", "utility"}
    cols = [c for c in df.columns if c not in drop]
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


def _apply_regime_filter(
    df: pd.DataFrame,
    mode: str,
    dd_max: float,
    ret20_min: float,
    atr_max: float,
    leverage_mult: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Regime filter is applied per-row using Market_* columns (already in your feature set).
    - mode=off    : no filter
    - mode=basic  : requires Market_Drawdown >= -dd_max_adj AND Market_ATR_ratio <= atr_max_adj
    - mode=trend  : requires Market_ret_20 >= ret20_min
    - mode=dd     : requires Market_Drawdown >= -dd_max_adj
    - mode=combo  : basic + trend (all)
    Leverage handling:
      For leveraged universe (e.g. 3x), we tighten dd/atr thresholds:
        dd_max_adj  = dd_max / leverage_mult
        atr_max_adj = 1 + (atr_max - 1)/leverage_mult
      (keeps thresholds sane; not overly harsh like multiplying ret requirements)
    """
    mode = (mode or "off").strip().lower()
    audit = {
        "enabled": mode != "off",
        "mode": mode,
        "inputs": {
            "dd_max": float(dd_max),
            "ret20_min": float(ret20_min),
            "atr_max": float(atr_max),
            "leverage_mult": float(leverage_mult),
        },
        "used_cols": [],
        "thresholds_used": {},
        "n_before": int(len(df)),
        "n_after": int(len(df)),
        "n_dropped": 0,
        "reason": "",
    }
    if mode == "off":
        return df, audit

    # check required cols exist
    need_cols = set()
    if mode in ("basic", "dd", "combo"):
        need_cols.add("Market_Drawdown")
    if mode in ("basic", "combo"):
        need_cols.add("Market_ATR_ratio")
    if mode in ("trend", "combo"):
        need_cols.add("Market_ret_20")

    missing = [c for c in sorted(need_cols) if c not in df.columns]
    if missing:
        audit["reason"] = f"missing_cols:{missing}"
        # fail open (do not drop anything) to avoid accidental empty picks
        return df, audit

    lev = float(leverage_mult) if np.isfinite(leverage_mult) and leverage_mult > 0 else 1.0
    dd_max = float(dd_max) if np.isfinite(dd_max) and dd_max > 0 else 0.20
    atr_max = float(atr_max) if np.isfinite(atr_max) and atr_max > 0 else 1.30
    ret20_min = float(ret20_min) if np.isfinite(ret20_min) else 0.0

    dd_max_adj = dd_max / lev
    atr_max_adj = 1.0 + (atr_max - 1.0) / lev

    mask = pd.Series(True, index=df.index)

    if mode in ("basic", "dd", "combo"):
        md = pd.to_numeric(df["Market_Drawdown"], errors="coerce").fillna(-999.0)
        mask &= (md >= -abs(dd_max_adj))
        audit["used_cols"].append("Market_Drawdown")
        audit["thresholds_used"]["Market_Drawdown_min"] = -abs(dd_max_adj)

    if mode in ("basic", "combo"):
        ma = pd.to_numeric(df["Market_ATR_ratio"], errors="coerce").fillna(999.0)
        mask &= (ma <= atr_max_adj)
        audit["used_cols"].append("Market_ATR_ratio")
        audit["thresholds_used"]["Market_ATR_ratio_max"] = atr_max_adj

    if mode in ("trend", "combo"):
        mr = pd.to_numeric(df["Market_ret_20"], errors="coerce").fillna(-999.0)
        mask &= (mr >= ret20_min)
        audit["used_cols"].append("Market_ret_20")
        audit["thresholds_used"]["Market_ret_20_min"] = ret20_min

    out = df.loc[mask].copy()
    audit["n_after"] = int(len(out))
    audit["n_dropped"] = int(audit["n_before"] - audit["n_after"])
    return out, audit


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

    # ✅ Regime filter (gate-level)
    ap.add_argument(
        "--regime-mode",
        default="off",
        choices=["off", "basic", "trend", "dd", "combo"],
        help="regime filter mode applied before ranking",
    )
    ap.add_argument("--regime-dd-max", default=0.20, type=float, help="max allowed market drawdown (e.g. 0.20 => -20%)")
    ap.add_argument("--regime-ret20-min", default=0.00, type=float, help="min market 20d return (e.g. 0.00)")
    ap.add_argument("--regime-atr-max", default=1.30, type=float, help="max market ATR ratio (e.g. 1.30)")
    ap.add_argument(
        "--regime-leverage-mult",
        default=3.0,
        type=float,
        help="universe leverage multiplier (e.g. 3.0 for 3x ETFs) to tighten dd/atr thresholds",
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
    if proba.shape[1] >= 2:
        p_success = proba[:, 1]
    else:
        p_success = proba[:, 0]
    feats2["p_success"] = p_success

    # tail probs (p_tail)
    tail_model, tail_scaler = _load_tail_models()
    Xt = tail_scaler.transform(X)
    tproba = tail_model.predict_proba(Xt)
    if tproba.shape[1] >= 2:
        p_tail = tproba[:, 1]
    else:
        p_tail = tproba[:, 0]
    feats2["p_tail"] = p_tail

    n0 = int(len(feats2))

    # basic filters
    if args.exclude_tickers:
        ex = [x.strip().upper() for x in args.exclude_tickers.split(",") if x.strip()]
        if ex:
            feats2 = feats2[~feats2["Ticker"].isin(ex)].copy()

    feats2 = feats2[feats2["p_success"] >= float(args.ps_min)].copy()
    n_after_ps = int(len(feats2))

    # ✅ regime filter (gate-level)
    feats2, regime_audit = _apply_regime_filter(
        feats2,
        mode=str(args.regime_mode),
        dd_max=float(args.regime_dd_max),
        ret20_min=float(args.regime_ret20_min),
        atr_max=float(args.regime_atr_max),
        leverage_mult=float(args.regime_leverage_mult),
    )
    n_after_regime = int(len(feats2))

    if feats2.empty:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        picks_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
        debug_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.debug.json"
        pd.DataFrame(columns=["Date", "Ticker"]).to_csv(picks_path, index=False)
        debug_path.write_text(
            json.dumps(
                {
                    "empty": True,
                    "reason": "filters",
                    "n0": n0,
                    "n_after_ps": n_after_ps,
                    "regime_audit": regime_audit,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[WARN] empty picks after filters -> wrote {picks_path}")
        return

    # compute score by mode
    mode = args.mode
    tmax = float(args.tail_threshold)
    uq = float(args.utility_quantile)
    lam = float(args.lambda_tail)

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
        "n0": n0,
        "n_after_ps": n_after_ps,
        "regime_audit": regime_audit,
        "n_after_regime": n_after_regime,
        "n_rows_final_pool": int(len(feats2)),
        "n_rows_picks": int(len(picks)),
    }
    debug_path.write_text(json.dumps(dbg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] wrote picks: {picks_path} rows={len(picks)}")
    print(f"[DONE] wrote debug: {debug_path}")


if __name__ == "__main__":
    main()