#!/usr/bin/env python3
"""
WF lite badexit training (half-year walk-forward).

Writes:
- app/badexit_model.pkl
- app/badexit_scaler.pkl
- data/meta/train_badexit_report.json

Assumes input dataset has:
- Date, Ticker (Ticker optional)
- features columns (SSOT feature_cols.json preferred; fallback numeric autodetect)
- label column (default: y_badexit)
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
    """
    Supports these SSOT formats:
      - ["c1","c2",...]
      - {"feature_cols":[...]}
      - {"cols":[...]}
      - {"features":[...]}
      - {"p_success_cols":[...]}  (fallback key)
    """
    p = META_DIR / "feature_cols.json"
    if not p.exists():
        return None

    try:
        j = json.loads(p.read_text(encoding="utf-8"))

        cols = None
        if isinstance(j, list):
            cols = j
        elif isinstance(j, dict):
            for k in ("feature_cols", "cols", "features", "p_success_cols"):
                v = j.get(k)
                if isinstance(v, list) and v:
                    cols = v
                    break

        if isinstance(cols, list) and cols:
            out = [str(c).strip() for c in cols if str(c).strip()]
            return out if out else None
    except Exception:
        return None

    return None


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
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


def _detect_numeric_feature_cols(df: pd.DataFrame, label_col: str) -> list[str]:
    drop = {
        "Date",
        "Ticker",
        label_col,
        # common scored/meta cols
        "p_success",
        "p_tail",
        "p_badexit",
        "tau_H",
        "tau_class",
    }

    cand = [c for c in df.columns if c not in drop]
    feat_cols: list[str] = []

    # robust: "numeric dtype" OR "can be coerced to numeric with not-all-NaN"
    for c in cand:
        if pd.api.types.is_numeric_dtype(df[c]):
            feat_cols.append(c)
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            feat_cols.append(c)

    # keep stable ordering
    return feat_cols


def _fit_lr_safe(X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression, dict]:
    """
    Fit StandardScaler + LogisticRegression safely even if y has 1 class.
    If one-class: add 1 synthetic opposite sample with tiny weight.
    """
    info = {"n_classes": int(np.unique(y).size)}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=200, solver="lbfgs")

    uniq = np.unique(y)
    if uniq.size >= 2:
        model.fit(Xs, y)
        return scaler, model, info

    only = int(uniq[0])
    opp = 1 - only

    X_syn = X[:1].copy()
    X_syn = X_syn + 1e-9  # tiny jitter to avoid exact duplicate
    y_syn = np.array([opp], dtype=int)

    X_aug = np.vstack([X, X_syn])
    y_aug = np.concatenate([y, y_syn])

    scaler2 = StandardScaler()
    Xs_aug = scaler2.fit_transform(X_aug)

    sw = np.ones(len(y_aug), dtype=float)
    sw[-1] = 1e-6  # tiny weight on synthetic sample

    model2 = LogisticRegression(max_iter=200, solver="lbfgs")
    model2.fit(Xs_aug, y_aug, sample_weight=sw)

    info.update({"one_class": True, "only": only, "synthetic_added": True, "synthetic_weight": float(sw[-1])})
    return scaler2, model2, info


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

    # 1) feature cols
    ssot_cols = _load_ssot_cols()
    if ssot_cols:
        # keep only those present; if none present -> fallback
        present = [c for c in ssot_cols if c in df.columns]
        if present:
            feat_cols = present
        else:
            feat_cols = _detect_numeric_feature_cols(df, args.label_col)
    else:
        feat_cols = _detect_numeric_feature_cols(df, args.label_col)

    if not feat_cols:
        raise RuntimeError(f"no feature cols found. columns={list(df.columns)}")

    df = _coerce_numeric(df, feat_cols)

    y = pd.to_numeric(df[args.label_col], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()
    X = df[feat_cols].to_numpy(dtype=float)

    # 2) half-year walk-forward
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

        scaler, model, info = _fit_lr_safe(X[tr_idx], y[tr_idx])
        Xva = scaler.transform(X[va_idx])
        pva = model.predict_proba(Xva)[:, 1]
        oof_pred[va_idx] = pva

        ll = log_loss(y[va_idx], np.clip(pva, 1e-6, 1 - 1e-6), labels=[0, 1])
        fold_logs.append(
            {
                "val_halfyear": val_hy,
                "logloss": float(ll),
                "n_tr": int(tr_idx.size),
                "n_va": int(va_idx.size),
                "train_fit_info": info,
            }
        )
        print(f"[INFO] val={val_hy} logloss={ll:.6f} n_tr={tr_idx.size} n_va={va_idx.size} info={info}")

    # 3) final fit (all data)
    final_scaler, final_model, final_info = _fit_lr_safe(X, y)

    APP_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, Path(args.out_model))
    joblib.dump(final_scaler, Path(args.out_scaler))

    report = {
        "label_col": args.label_col,
        "feature_cols": feat_cols,
        "splits": "half-year walk-forward",
        "min_train_halfyears": int(args.min_train_halfyears),
        "folds": fold_logs,
        "oof_coverage": float(np.isfinite(oof_pred).mean()),
        "n_rows": int(len(df)),
        "n_pos": int(y.sum()),
        "final_fit_info": final_info,
        "ssot_loaded": bool(ssot_cols),
    }
    Path(args.out_report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] saved model : {args.out_model}")
    print(f"[DONE] saved scaler: {args.out_scaler}")
    print(f"[DONE] wrote report: {args.out_report}")


if __name__ == "__main__":
    main()