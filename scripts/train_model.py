# scripts/train_model.py
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


DATA_DIR = Path("data")
FEATURE_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"
MODEL_DIR = Path("app") / "model"

FEATURES_PARQUET = FEATURE_DIR / "features.parquet"
FEATURES_CSV = FEATURE_DIR / "features.csv"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_tag(pt: float, h: int, sl: float) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{h}_sl{sl_tag}"


def load_features() -> pd.DataFrame:
    if FEATURES_PARQUET.exists():
        f = pd.read_parquet(FEATURES_PARQUET)
    elif FEATURES_CSV.exists():
        f = pd.read_csv(FEATURES_CSV)
    else:
        raise FileNotFoundError("features not found. Run scripts/build_features.py first.")

    f = f.copy()
    f["Date"] = pd.to_datetime(f["Date"])
    f["Ticker"] = f["Ticker"].astype(str).str.upper().str.strip()
    return f


def load_labels(tag: str) -> pd.DataFrame:
    p = LABEL_DIR / f"labels_{tag}.parquet"
    c = LABEL_DIR / f"labels_{tag}.csv"
    if p.exists():
        df = pd.read_parquet(p)
    elif c.exists():
        df = pd.read_csv(c)
    else:
        raise FileNotFoundError(f"labels not found for tag={tag}. Run scripts/build_labels.py first.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profit-target", type=float, required=True)
    parser.add_argument("--holding-days", type=int, required=True)
    parser.add_argument("--stop-level", type=float, required=True)
    parser.add_argument("--features", type=str, default="", help="comma-separated feature columns (optional)")
    args = parser.parse_args()

    tag = fmt_tag(args.profit_target, args.holding_days, args.stop_level)

    feat = load_features()
    lab = load_labels(tag)

    df = feat.merge(lab[["Date", "Ticker", "Success"]], on=["Date", "Ticker"], how="inner")

    # 라벨 NaN(미래 부족 구간)은 제거
    df = df.dropna(subset=["Success"]).reset_index(drop=True)
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # feature columns
    if args.features.strip():
        feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    else:
        feature_cols = [c for c in feat.columns if c not in ("Date", "Ticker")]

    # 혹시라도 숫자 아닌 컬럼 섞이면 제거
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = num_cols

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["Success"].to_numpy(dtype=int)

    # 날짜 기준 80/20 split (cross-sectional leakage 방지)
    unique_dates = np.array(sorted(df["Date"].unique()))
    split_i = int(len(unique_dates) * 0.8)
    split_i = max(1, min(split_i, len(unique_dates) - 1))
    train_dates = set(unique_dates[:split_i])
    test_dates = set(unique_dates[split_i:])

    is_train = df["Date"].isin(train_dates).to_numpy()
    is_test = df["Date"].isin(test_dates).to_numpy()

    X_train, y_train = X[is_train], y[is_train]
    X_test, y_test = X[is_test], y[is_test]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base_model = LogisticRegression(max_iter=800)
    tscv = TimeSeriesSplit(n_splits=3)

    model = CalibratedClassifierCV(
        base_model,
        method="isotonic",
        cv=tscv,
    )
    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:, 1]

    print("=" * 70)
    if len(np.unique(y_test)) >= 2:
        auc = roc_auc_score(y_test, probs)
        print("Test ROC-AUC:", round(float(auc), 4))
    else:
        print("Test ROC-AUC: (skipped - single class in test)")
    print("Base Success Rate:", round(float(y_test.mean()), 4))
    print("Predicted Mean Probability:", round(float(probs.mean()), 4))
    print("Rows train/test:", len(X_train), "/", len(X_test))
    print("Features:", len(feature_cols))
    print("=" * 70)

    # save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"model_{tag}.pkl"
    scaler_path = MODEL_DIR / f"scaler_{tag}.pkl"
    meta_path = MODEL_DIR / f"model_{tag}_meta.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "trained_at_utc": now_utc_iso(),
        "tag": tag,
        "profit_target": args.profit_target,
        "holding_days": args.holding_days,
        "stop_level": args.stop_level,
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "base_success_rate_test": float(y_test.mean()),
        "pred_mean_prob_test": float(probs.mean()),
    }
    meta_path.write_text(pd.Series(meta).to_json(), encoding="utf-8")

    print(f"✅ saved: {model_path}")
    print(f"✅ saved: {scaler_path}")
    print(f"✅ saved: {meta_path}")


if __name__ == "__main__":
    main()
