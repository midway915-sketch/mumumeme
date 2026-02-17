# scripts/train_tau_model.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


LABELS_DIR = Path("data/labels")
APP_DIR = Path("app")
META_DIR = Path("data/meta")

DEFAULT_FEATURES = [
    "Drawdown_252",
    "Drawdown_60",
    "ATR_ratio",
    "Z_score",
    "MACD_hist",
    "MA20_slope",
    "Market_Drawdown",
    "Market_ATR_ratio",
]


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def choose_features(df: pd.DataFrame, requested: list[str]) -> list[str]:
    cols = set(df.columns)
    feats = [c for c in requested if c in cols]
    if len(feats) < 4:
        banned = {"TauDays","TauLE1","TauLE2","TauLE3","TauCut1","TauCut2","TauCut3","TauHorizon","Date","Ticker"}
        numeric = [c for c in df.columns if c not in banned and pd.api.types.is_numeric_dtype(df[c])]
        feats = feats + [c for c in numeric if c not in feats][:12]
    out, seen = [], set()
    for c in feats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _fit_one(X_train, y_train, n_splits: int, max_iter: int):
    base = LogisticRegression(max_iter=max_iter)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    try:
        clf = CalibratedClassifierCV(estimator=base, method="isotonic", cv=tscv)
    except TypeError:
        clf = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=tscv)
    clf.fit(X_train, y_train)
    return clf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES), type=str)
    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=500, type=int)
    args = ap.parse_args()

    pt_tag = int(round(args.profit_target * 100))
    sl_tag = int(round(abs(args.stop_level) * 100))
    tag = f"pt{pt_tag}_h{args.max_days}_sl{sl_tag}_ex{args.max_extend_days}"

    parq = LABELS_DIR / f"labels_tau_{tag}.parquet"
    csv = LABELS_DIR / f"labels_tau_{tag}.csv"
    df = read_table(parq, csv).copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date")

    # dynamic labels should exist
    required = ["TauLE1", "TauLE2", "TauLE3"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"labels_tau missing {col}")

    # read cutoffs (just to store)
    cut1 = int(pd.to_numeric(df.get("TauCut1", 10), errors="coerce").dropna().iloc[0]) if "TauCut1" in df.columns else 10
    cut2 = int(pd.to_numeric(df.get("TauCut2", 20), errors="coerce").dropna().iloc[0]) if "TauCut2" in df.columns else 20
    cut3 = int(pd.to_numeric(df.get("TauCut3", args.max_days), errors="coerce").dropna().iloc[0]) if "TauCut3" in df.columns else int(args.max_days)

    requested = [x.strip() for x in args.features.split(",") if x.strip()]
    feature_cols = choose_features(df, requested)

    use = df.dropna(subset=feature_cols + required).copy()
    if len(use) < 1000:
        # inference-safe fallback
        use = df.copy()
        for c in feature_cols:
            use[c] = pd.to_numeric(use[c], errors="coerce").fillna(0.0)
        for col in required:
            use[col] = pd.to_numeric(use[col], errors="coerce").fillna(0).astype(int)

    X = use[feature_cols]
    split_idx = int(len(use) * float(args.train_ratio))
    split_idx = max(1, min(split_idx, len(use) - 1))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {}
    aucs = {}

    for label in required:
        y = use[label].astype(int)
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        clf = _fit_one(X_train_s, y_train, n_splits=int(args.n_splits), max_iter=int(args.max_iter))
        models[label] = clf

        probs = clf.predict_proba(X_test_s)[:, 1]
        auc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else float("nan")
        aucs[label] = auc

    APP_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(models, APP_DIR / "tau_cdf_models.pkl")
    joblib.dump(scaler, APP_DIR / "tau_scaler.pkl")

    report = {
        "tag": tag,
        "cuts": {"cut1": cut1, "cut2": cut2, "cut3": cut3},
        "rows_total": int(len(df)),
        "rows_used": int(len(use)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_cols": feature_cols,
        "aucs": {k: (round(v, 6) if np.isfinite(v) else None) for k, v in aucs.items()},
        "paths": {
            "labels": str(parq if parq.exists() else csv),
            "models": str(APP_DIR / "tau_cdf_models.pkl"),
            "scaler": str(APP_DIR / "tau_scaler.pkl"),
        },
    }
    (META_DIR / f"train_tau_report_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print("[DONE] train_tau_model.py")
    print("tag:", tag)
    print("cuts:", report["cuts"])
    print("AUCs:", report["aucs"])
    print("features:", feature_cols)
    print("=" * 60)


if __name__ == "__main__":
    main()