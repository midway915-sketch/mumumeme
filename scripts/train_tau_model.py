#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss


DATA_DIR = Path("data")
META_DIR = DATA_DIR / "meta"
APP_DIR = Path("app")


def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def parse_csv_list(s: str) -> list[str]:
    items = [x.strip() for x in str(s or "").split(",")]
    return [x for x in items if x]


def ensure_features_exist(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feat_cols:
        if c not in out.columns:
            out[c] = 0.0
    for c in feat_cols:
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    return out


# ✅ build_features.py 기준(16) + (옵션) 섹터(2)
DEFAULT_FEATURES = [
    # base (9)
    "Drawdown_252",
    "Drawdown_60",
    "ATR_ratio",
    "Z_score",
    "MACD_hist",
    "MA20_slope",
    "Market_Drawdown",
    "Market_ATR_ratio",
    "ret_score",
    # new (7)
    "ret_5",
    "ret_10",
    "ret_20",
    "breakout_20",
    "vol_surge",
    "trend_align",
    "beta_60",
    # optional sector (2) - 있으면 쓰고, 없으면 0으로 생성됨
    "Sector_Ret_20",
    "RelStrength",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train tau (FAST/MID/SLOW) multi-class model.")
    ap.add_argument("--tag", default="", type=str, help="e.g. pt10_h40_sl10_ex30 (used for report/filenames)")

    ap.add_argument("--features-parq", default="data/features/features_model.parquet", type=str)
    ap.add_argument("--features-csv", default="data/features/features_model.csv", type=str)
    ap.add_argument("--labels-parq", default="data/labels/labels_tau.parquet", type=str)
    ap.add_argument("--labels-csv", default="data/labels/labels_tau.csv", type=str)

    ap.add_argument("--date-col", default="Date", type=str)
    ap.add_argument("--ticker-col", default="Ticker", type=str)
    ap.add_argument("--target-col", default="TauClass", type=str)

    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES), type=str)

    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=800, type=int)

    ap.add_argument("--out-model", default="", type=str)
    ap.add_argument("--out-scaler", default="", type=str)

    args = ap.parse_args()

    feats = read_table(args.features_parq, args.features_csv).copy()
    labs = read_table(args.labels_parq, args.labels_csv).copy()

    for df, name in [(feats, "features_model"), (labs, "labels_tau")]:
        if args.date_col not in df.columns or args.ticker_col not in df.columns:
            raise KeyError(f"{name} must have {args.date_col},{args.ticker_col}")

    feats[args.date_col] = norm_date(feats[args.date_col])
    labs[args.date_col] = norm_date(labs[args.date_col])
    feats[args.ticker_col] = feats[args.ticker_col].astype(str).str.upper().str.strip()
    labs[args.ticker_col] = labs[args.ticker_col].astype(str).str.upper().str.strip()

    feat_cols = parse_csv_list(args.features) or DEFAULT_FEATURES
    feats = ensure_features_exist(feats, feat_cols)

    if args.target_col not in labs.columns:
        raise KeyError(f"labels_tau missing target column: {args.target_col}")

    # merge
    df = pd.merge(
        feats[[args.date_col, args.ticker_col] + feat_cols].copy(),
        labs[[args.date_col, args.ticker_col, args.target_col]].copy(),
        on=[args.date_col, args.ticker_col],
        how="inner",
    )

    df = df.dropna(subset=[args.date_col, args.ticker_col, args.target_col]).copy()
    if df.empty:
        raise RuntimeError("No training rows after merge. Check labels_tau and features_model overlap.")

    # sort time
    df = df.sort_values(args.date_col).reset_index(drop=True)

    X = df[feat_cols].to_numpy(dtype=float)
    y = pd.to_numeric(df[args.target_col], errors="coerce").fillna(2).astype(int).to_numpy()

    n = len(df)
    if n < 200:
        raise RuntimeError(f"Not enough training rows: {n}")

    split = int(n * float(args.train_ratio))
    split = max(50, min(split, n - 50))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # multinomial logistic
    model = LogisticRegression(
        max_iter=int(args.max_iter),
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
    )

    # time-series CV sanity (train only)
    tscv = TimeSeriesSplit(n_splits=int(args.n_splits))
    cv_losses = []
    for tr, va in tscv.split(X_train_s):
        m = LogisticRegression(
            max_iter=int(args.max_iter),
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
        )
        m.fit(X_train_s[tr], y_train[tr])
        p_va = m.predict_proba(X_train_s[va])
        cv_losses.append(log_loss(y_train[va], p_va, labels=[0, 1, 2]))

    model.fit(X_train_s, y_train)

    # evaluate
    ll_test = float("nan")
    if len(X_test_s) > 0:
        p_test = model.predict_proba(X_test_s)
        ll_test = float(log_loss(y_test, p_test, labels=[0, 1, 2]))

    print("=" * 60)
    print(f"[INFO] rows={n} train={len(X_train)} test={len(X_test)}")
    print(f"[INFO] CV logloss (train folds): {np.round(cv_losses, 4).tolist()} mean={float(np.mean(cv_losses)):.4f}")
    print(f"[INFO] Test logloss: {ll_test:.4f}")
    print(f"[INFO] Class distribution (all): {pd.Series(y).value_counts().to_dict()}")
    print("=" * 60)

    # outputs
    APP_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    tag = (args.tag or "").strip()
    out_model = Path(args.out_model) if args.out_model else (APP_DIR / (f"tau_model_{tag}.pkl" if tag else "tau_model.pkl"))
    out_scaler = Path(args.out_scaler) if args.out_scaler else (APP_DIR / (f"tau_scaler_{tag}.pkl" if tag else "tau_scaler.pkl"))

    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)
    print(f"[DONE] saved model: {out_model}")
    print(f"[DONE] saved scaler: {out_scaler}")

    # ✅ report for score_features.py (to avoid feature mismatch)
    report = {
        "tag": tag,
        "feature_cols": feat_cols,
        "rows": int(n),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cv_logloss": [float(x) for x in cv_losses],
        "cv_logloss_mean": float(np.mean(cv_losses)) if cv_losses else float("nan"),
        "test_logloss": ll_test,
        "class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
        "out_model": str(out_model),
        "out_scaler": str(out_scaler),
    }
    report_path = META_DIR / f"train_tau_report_{tag}.json" if tag else (META_DIR / "train_tau_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote report: {report_path}")


if __name__ == "__main__":
    main()