#!/usr/bin/env python3
"""
WF lite badexit training (half-year walk-forward).

Writes:
- app/badexit_model.pkl
- app/badexit_scaler.pkl
- data/meta/train_badexit_report.json

Assumes input dataset has:
- Date, Ticker
- features columns (per SSOT feature_cols.json or numeric fallback)
- label column: y_badexit (0/1) OR user-provided --label-col
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
            out = [str(c).strip() for c in cols if str(c).strip()]
            return out if out else None
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


def _pick_feature_cols(df: pd.DataFrame, label_col: str) -> tuple[list[str], dict]:
    """
    Prefer SSOT feature cols, but make it robust:
      - use intersection (SSOT ∩ df.columns)
      - if intersection empty -> fallback to numeric columns excluding known keys/labels
    """
    debug = {}

    ssot = _load_ssot_cols()
    if ssot:
        ssot_in_df = [c for c in ssot if c in df.columns]
        debug["ssot_cols_n"] = len(ssot)
        debug["ssot_in_df_n"] = len(ssot_in_df)
        debug["ssot_missing_sample"] = [c for c in ssot if c not in df.columns][:20]
        if ssot_in_df:
            return ssot_in_df, debug

    # numeric fallback
    drop = {
        "Date",
        "Ticker",
        label_col,
        # scored cols / other possible non-features
        "p_success",
        "p_tail",
        "p_badexit",
        "tau_H",
        "tau_class",
        "_hy",
    }
    numeric_cols = []
    for c in df.columns:
        if c in drop:
            continue
        # allow numeric or numeric-looking
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    debug["fallback_numeric_n"] = len(numeric_cols)
    return numeric_cols, debug


def _fit_lr_safe(X: np.ndarray, y: np.ndarray, *, max_iter: int = 200) -> LogisticRegression:
    """
    Fit LogisticRegression even if y has only 1 class:
      - append 1 synthetic opposite-class sample (zeros) with tiny weight
    """
    y = np.asarray(y).astype(int)
    uniq = np.unique(y)

    model = LogisticRegression(max_iter=max_iter, solver="lbfgs")

    if uniq.size >= 2:
        model.fit(X, y)
        return model

    # single-class: add synthetic opposite
    nfeat = X.shape[1]
    x_syn = np.zeros((1, nfeat), dtype=float)
    y_syn = np.array([1 - int(uniq[0])], dtype=int)

    X_aug = np.vstack([X, x_syn])
    y_aug = np.concatenate([y, y_syn])

    # tiny weight for synthetic sample so it doesn't distort much
    w = np.ones(len(y_aug), dtype=float)
    w[-1] = 1e-6

    model.fit(X_aug, y_aug, sample_weight=w)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True, type=str, help="parquet/csv containing features+label")
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

    feat_cols, feat_debug = _pick_feature_cols(df, args.label_col)
    if not feat_cols:
        raise RuntimeError(
            f"no feature cols found. columns={list(df.columns)[:50]} (total={len(df.columns)})"
        )

    # ensure numeric coercion
    df = _coerce_numeric(df, feat_cols)

    y = (
        pd.to_numeric(df[args.label_col], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(0, 1)
        .to_numpy()
    )
    X = df[feat_cols].to_numpy(dtype=float)

    # half-year walk-forward splits
    df["_hy"] = df["Date"].apply(_halfyear_key)
    hy_list = sorted(df["_hy"].unique())

    fold_logs = []
    oof_pred = np.full(len(df), np.nan, dtype=float)

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

        ytr = y[tr_idx]
        yva = y[va_idx]
        uniq_tr = np.unique(ytr)

        # if train is single-class, still fit safely
        model = _fit_lr_safe(Xtr, ytr, max_iter=200)
        pva = model.predict_proba(Xva)[:, 1]
        oof_pred[va_idx] = pva

        # logloss can fail if yva is single-class and pva is extreme; keep it robust
        try:
            ll = log_loss(yva, np.clip(pva, 1e-6, 1 - 1e-6))
            ll_f = float(ll)
        except Exception:
            ll_f = float("nan")

        fold_logs.append(
            {
                "val_halfyear": val_hy,
                "logloss": ll_f,
                "n_tr": int(tr_idx.size),
                "n_va": int(va_idx.size),
                "train_classes": [int(x) for x in uniq_tr.tolist()],
            }
        )
        print(f"[INFO] val={val_hy} logloss={ll_f} n_tr={tr_idx.size} n_va={va_idx.size} train_classes={uniq_tr.tolist()}")

    # final train on all available data
    final_scaler = StandardScaler()
    Xs = final_scaler.fit_transform(X)

    uniq_all = np.unique(y)
    print(f"[INFO] final train classes={uniq_all.tolist()} n_rows={len(df)} n_pos={int(y.sum())}")

    final_model = _fit_lr_safe(Xs, y, max_iter=200)

    APP_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, Path(args.out_model))
    joblib.dump(final_scaler, Path(args.out_scaler))

    report = {
        "label_col": args.label_col,
        "feature_cols": feat_cols,
        "feature_cols_source_debug": feat_debug,
        "splits": "half-year walk-forward",
        "folds": fold_logs,
        "oof_coverage": float(np.isfinite(oof_pred).mean()),
        "n_rows": int(len(df)),
        "n_pos": int(y.sum()),
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    }
    Path(args.out_report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] saved model : {args.out_model}")
    print(f"[DONE] saved scaler: {args.out_scaler}")
    print(f"[DONE] wrote report: {args.out_report}")


if __name__ == "__main__":
    main()