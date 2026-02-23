#!/usr/bin/env python3
"""
WF lite badexit training (half-year walk-forward).

Writes:
- app/badexit_model.pkl
- app/badexit_scaler.pkl
- data/meta/train_badexit_report.json

Assumes input dataset has:
- Date, Ticker
- features columns (per SSOT feature_cols.json or DEFAULT_FEATURES fallback)
- label column: y_badexit (0/1)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


DATA_DIR = Path("data")
META_DIR = DATA_DIR / "meta"
APP_DIR = Path("app")


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _load_ssot_cols() -> list[str] | None:
    p = META_DIR / "feature_cols.json"
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
        cols = j.get("feature_cols")
        if isinstance(cols, list) and cols:
            return [str(c).strip() for c in cols if str(c).strip()]
    except Exception:
        return None
    return None


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


def _halfyear_key(d: pd.Timestamp) -> str:
    y = int(d.year)
    h = 1 if int(d.month) <= 6 else 2
    return f"{y}H{h}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True, type=str, help="parquet/csv containing features+label y_badexit")
    ap.add_argument("--label-col", default="y_badexit", type=str)
    ap.add_argument("--min-train-halfyears", default=4, type=int, help="minimum half-years to start training")
    ap.add_argument("--out-model", default=str(APP_DIR / "badexit_model.pkl"))
    ap.add_argument("--out-scaler", default=str(APP_DIR / "badexit_scaler.pkl"))
    ap.add_argument("--out-report", default=str(META_DIR / "train_badexit_report.json"))
    args = ap.parse_args()

    p = Path(args.data_path)
    if not p.exists():
        raise FileNotFoundError(f"missing --data-path: {p}")

    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
    if "Date" not in df.columns:
        raise ValueError("data must include Date")
    if args.label_col not in df.columns:
        raise ValueError(f"data must include label col: {args.label_col}")

    df = df.copy()
    df["Date"] = _norm_date(df["Date"])
    df = df.dropna(subset=["Date"]).reset_index(drop=True)

    # features columns: SSOT 우선
    feat_cols = _load_ssot_cols()
    if not feat_cols:
        # fallback: numeric cols excluding keys/labels
        drop = {"Date", "Ticker", args.label_col, "p_success", "p_tail", "p_badexit", "tau_H", "tau_class"}
        feat_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
        if not feat_cols:
            raise RuntimeError("no feature cols found (SSOT missing and no numeric cols detected)")

    # ensure present
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"SSOT feature_cols missing in data: {missing[:30]}")

    df = _coerce_numeric(df, feat_cols)
    y = pd.to_numeric(df[args.label_col], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()
    X = df[feat_cols].to_numpy(dtype=float)

    # half-year walk-forward splits
    df["_hy"] = df["Date"].apply(_halfyear_key)
    hy_list = sorted(df["_hy"].unique())

    fold_logs = []
    oof_pred = np.full(len(df), np.nan, dtype=float)

    # WF: train on past half-years, validate on next half-year
    for i in range(len(hy_list)):
        val_hy = hy_list[i]
        train_hys = hy_list[:i]
        if len(train_hys) < int(args.min_train_halfyears):
            continue

        tr_idx = df.index[df["_hy"].isin(train_hys)].to_numpy()
        va_idx = df.index[df["_hy"] == val_hy].to_numpy()
        if tr_idx.size == 0 or va_idx.size == 0:
            continue

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_idx])
        Xva = scaler.transform(X[va_idx])

        # n_jobs는 제거(스킵해도 영향 없음)
        model = LogisticRegression(max_iter=200, solver="lbfgs")
        model.fit(Xtr, y[tr_idx])

        pva = model.predict_proba(Xva)[:, 1]
        oof_pred[va_idx] = pva

        ll = log_loss(y[va_idx], np.clip(pva, 1e-6, 1 - 1e-6))
        fold_logs.append({"val_halfyear": val_hy, "logloss": float(ll), "n_tr": int(tr_idx.size), "n_va": int(va_idx.size)})
        print(f"[INFO] val={val_hy} logloss={ll:.6f} n_tr={tr_idx.size} n_va={va_idx.size}")

    # final train on all available data (for production scoring)
    final_scaler = StandardScaler()
    Xs = final_scaler.fit_transform(X)
    final_model = LogisticRegression(max_iter=200, solver="lbfgs")
    final_model.fit(Xs, y)

    APP_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, Path(args.out_model))
    joblib.dump(final_scaler, Path(args.out_scaler))

    # report
    report = {
        "label_col": args.label_col,
        "feature_cols": feat_cols,
        "splits": "half-year walk-forward",
        "folds": fold_logs,
        "oof_coverage": float(np.isfinite(oof_pred).mean()),
        "n_rows": int(len(df)),
        "n_pos": int(y.sum()),
    }
    Path(args.out_report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] saved model : {args.out_model}")
    print(f"[DONE] saved scaler: {args.out_scaler}")
    print(f"[DONE] wrote report: {args.out_report}")


if __name__ == "__main__":
    main()