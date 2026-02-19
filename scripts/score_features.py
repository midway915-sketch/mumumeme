#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "Drawdown_252",
    "Drawdown_60",
    "ATR_ratio",
    "Z_score",
    "MACD_hist",
    "MA20_slope",
    "ret_score",
    "Volume",
    "Close",
    "Market_Drawdown",
    "Market_ATR_ratio",
]

# Tau (holding horizon class) mapping
# 0: FAST  -> 20 days
# 1: MID   -> 40 days
# 2: SLOW  -> 60 days
TAU_CLASS_TO_H = {0: 20, 1: 40, 2: 60}


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def ensure_features_exist(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
    return out


def load_report_features(report_path: Path, fallback: list[str]) -> list[str]:
    if not report_path.exists():
        return fallback
    try:
        j = json.loads(report_path.read_text(encoding="utf-8"))
        cols = j.get("feature_cols", None)
        if isinstance(cols, list) and cols:
            return [str(x) for x in cols]
    except Exception:
        pass
    return fallback


def main() -> None:
    ap = argparse.ArgumentParser(description="Score features_model -> features_scored with p_success/p_tail (+ optional tau_H).")

    ap.add_argument("--tag", default="", type=str, help="label key tag (for train report lookups)")
    ap.add_argument("--features-parq", default="data/features/features_model.parquet", type=str)
    ap.add_argument("--features-csv", default="data/features/features_model.csv", type=str)
    ap.add_argument("--out-parq", default="data/features/features_scored.parquet", type=str)
    ap.add_argument("--out-csv", default="data/features/features_scored.csv", type=str)

    # ---- p_success
    ap.add_argument("--model", default="app/model.pkl", type=str)
    ap.add_argument("--scaler", default="app/scaler.pkl", type=str)
    ap.add_argument("--ps-features", default=",".join(DEFAULT_FEATURES), type=str)

    # ---- p_tail (optional)
    ap.add_argument("--tail-model", default="app/tail_model.pkl", type=str)
    ap.add_argument("--tail-scaler", default="app/tail_scaler.pkl", type=str)
    ap.add_argument(
        "--tail-features",
        default="",
        type=str,
        help="comma-separated override. default=read from train_tail_report_{tag}.json or fallback to ps-features",
    )

    # ---- tau (optional)
    ap.add_argument("--tau-model", default="app/tau_model.pkl", type=str)
    ap.add_argument("--tau-scaler", default="app/tau_scaler.pkl", type=str)
    ap.add_argument("--tau-features", default=",".join(DEFAULT_FEATURES), type=str)

    args = ap.parse_args()

    f_parq = Path(args.features_parq)
    f_csv = Path(args.features_csv)
    feats = read_table(f_parq, f_csv).copy()

    # normalize required keys
    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise KeyError(f"features_model must include Date,Ticker. cols={list(feats.columns)[:40]}")
    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # ---- p_success
    model_path = Path(args.model)
    scaler_path = Path(args.scaler)
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Missing p_success model/scaler: {model_path}, {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    ps_cols = [c.strip() for c in str(args.ps_features).split(",") if c.strip()]
    feats_ps = ensure_features_exist(feats, ps_cols)
    X = feats_ps[ps_cols].to_numpy(dtype=float)
    Xs = scaler.transform(X)
    p = model.predict_proba(Xs)
    # binary classifier assumed: class1 probability
    if p.shape[1] >= 2:
        feats["p_success"] = p[:, 1].astype(float)
    else:
        feats["p_success"] = p[:, 0].astype(float)

    # ---- p_tail (optional)
    tail_model_path = Path(args.tail_model)
    tail_scaler_path = Path(args.tail_scaler)

    if tail_model_path.exists() and tail_scaler_path.exists():
        tail_model = joblib.load(tail_model_path)
        tail_scaler = joblib.load(tail_scaler_path)

        # tail feature cols: try report first
        tag = args.tag.strip()
        report_path = Path("data/meta") / f"train_tail_report_{tag}.json" if tag else Path("data/meta/train_tail_report.json")

        if str(args.tail_features).strip():
            tail_cols = [c.strip() for c in str(args.tail_features).split(",") if c.strip()]
        else:
            tail_cols = load_report_features(report_path, fallback=ps_cols)

        feats_tail = ensure_features_exist(feats, tail_cols)
        Xt = feats_tail[tail_cols].to_numpy(dtype=float)
        Xts = tail_scaler.transform(Xt)
        pt = tail_model.predict_proba(Xts)
        if pt.shape[1] >= 2:
            feats["p_tail"] = pt[:, 1].astype(float)
        else:
            feats["p_tail"] = pt[:, 0].astype(float)
    else:
        feats["p_tail"] = 0.0

    # ---- tau (optional)
    tau_model_path = Path(args.tau_model)
    tau_scaler_path = Path(args.tau_scaler)
    if tau_model_path.exists() and tau_scaler_path.exists():
        tau_model = joblib.load(tau_model_path)
        tau_scaler = joblib.load(tau_scaler_path)

        tau_cols = [c.strip() for c in str(args.tau_features).split(",") if c.strip()]
        feats_tau = ensure_features_exist(feats, tau_cols)
        Xu = feats_tau[tau_cols].to_numpy(dtype=float)
        Xus = tau_scaler.transform(Xu)
        pu = tau_model.predict_proba(Xus)

        feats["tau_p_fast"] = pu[:, 0].astype(float)
        feats["tau_p_mid"] = pu[:, 1].astype(float)
        feats["tau_p_slow"] = pu[:, 2].astype(float)
        feats["tau_class"] = np.argmax(pu, axis=1).astype(int)
        feats["tau_H"] = feats["tau_class"].map(TAU_CLASS_TO_H).fillna(0).astype(int)
    else:
        feats["tau_p_fast"] = 0.0
        feats["tau_p_mid"] = 0.0
        feats["tau_p_slow"] = 0.0
        feats["tau_class"] = 1
        feats["tau_H"] = 0

    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_csv)
    out_parq.parent.mkdir(parents=True, exist_ok=True)

    feats.to_parquet(out_parq, index=False)
    feats.to_csv(out_csv, index=False)

    print(f"[DONE] wrote: {out_parq} rows={len(feats)}")
    print(f"[DONE] wrote: {out_csv}")
    print("[INFO] cols(head) =", list(feats.columns)[:30])


if __name__ == "__main__":
    main()