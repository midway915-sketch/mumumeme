# scripts/train_model.py
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


LABELS_PARQUET = Path("data/labels/labels_model.parquet")
LABELS_CSV = Path("data/labels/labels_model.csv")

MODEL_PATH = Path("app/model.pkl")
SCALER_PATH = Path("app/scaler.pkl")
REPORT_PATH = Path("data/meta/train_model_report.json")


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
    raise FileNotFoundError(f"Training data not found: {parq} (or {csv})")


def choose_features(df: pd.DataFrame, requested: list[str]) -> list[str]:
    cols = set(df.columns)
    feats = [c for c in requested if c in cols]

    # 그래도 너무 적으면: 숫자형 후보에서 Target/Date/Ticker 제외하고 상위 몇 개
    if len(feats) < 4:
        banned = {"Target", "target", "Date", "Ticker"}
        numeric = []
        for c in df.columns:
            if c in banned:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric.append(c)
        # 안정적으로 일부만
        extras = [c for c in numeric if c not in feats][:12]
        feats = feats + extras

    # 최종 중복 제거
    seen = set()
    out = []
    for c in feats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-col", default="Target", type=str)
    ap.add_argument("--date-col", default="Date", type=str)
    ap.add_argument("--ticker-col", default="Ticker", type=str)
    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES), type=str)
    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=500, type=int)
    args = ap.parse_args()

    df = read_table(LABELS_PARQUET, LABELS_CSV).copy()

    # date 정렬 (time split)
    if args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
        df = df.sort_values(args.date_col)
    else:
        # date 없으면 그냥 index 기준
        df = df.reset_index(drop=True)

    # target 준비
    if args.target_col not in df.columns:
        # 혹시 Success 컬럼만 있을 때
        if "Success" in df.columns:
            df[args.target_col] = pd.to_numeric(df["Success"], errors="coerce")
        elif "success" in df.columns:
            df[args.target_col] = pd.to_numeric(df["success"], errors="coerce")
        else:
            raise ValueError(f"Missing target column: {args.target_col} (and no Success/success)")

    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce")

    requested = [x.strip() for x in args.features.split(",") if x.strip()]
    feature_cols = choose_features(df, requested)

    # 결측 제거
    use = df.dropna(subset=feature_cols + [args.target_col]).copy()
    if len(use) < 500:
        # 너무 적으면: feature 결측을 0으로 대체하는 fallback
        use = df.copy()
        for c in feature_cols:
            if c in use.columns:
                use[c] = pd.to_numeric(use[c], errors="coerce").fillna(0.0)
        use[args.target_col] = pd.to_numeric(use[args.target_col], errors="coerce").fillna(0.0)

    X = use[feature_cols]
    y = use[args.target_col].astype(int)

    # train/test split (time order)
    split_idx = int(len(use) * float(args.train_ratio))
    split_idx = max(1, min(split_idx, len(use) - 1))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base = LogisticRegression(max_iter=int(args.max_iter))

    tscv = TimeSeriesSplit(n_splits=int(args.n_splits))

    # sklearn 버전에 따라 파라미터 이름이 estimator 로 바뀜
    try:
        model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=tscv)
    except TypeError:
        model = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=tscv)

    model.fit(X_train_scaled, y_train)

    probs = model.predict_proba(X_test_scaled)[:, 1]
    auc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else float("nan")

    report = {
        "rows_total": int(len(df)),
        "rows_used": int(len(use)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_mean_test": float(np.mean(y_test)) if len(y_test) else float("nan"),
        "pred_mean_test": float(np.mean(probs)) if len(probs) else float("nan"),
        "roc_auc_test": auc,
        "feature_cols": feature_cols,
        "labels_source": str(LABELS_PARQUET if LABELS_PARQUET.exists() else LABELS_CSV),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print("[DONE] train_model.py")
    print("saved:", str(MODEL_PATH), str(SCALER_PATH))
    print("roc_auc_test:", round(report["roc_auc_test"], 4) if np.isfinite(report["roc_auc_test"]) else "nan")
    print("target_mean_test:", round(report["target_mean_test"], 4) if np.isfinite(report["target_mean_test"]) else "nan")
    print("pred_mean_test:", round(report["pred_mean_test"], 4) if np.isfinite(report["pred_mean_test"]) else "nan")
    print("features:", feature_cols)
    print("=" * 60)


if __name__ == "__main__":
    main()