# scripts/train_strategy_models.py
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor


DATA_DIR = Path("data")
FEATURE_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"
MODEL_DIR = Path("app") / "model"

FEATURES_PARQUET = FEATURE_DIR / "features.parquet"
FEATURES_CSV = FEATURE_DIR / "features.csv"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_tag(pt: float, max_days: int, sl: float) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{max_days}_sl{sl_tag}"


def load_features() -> pd.DataFrame:
    p = Path("data/features/features_model.parquet")
    c = Path("data/features/features_model.csv")
    if p.exists():
        f = pd.read_parquet(p)
    elif c.exists():
        f = pd.read_csv(c)
    else:
        # fallback
        p0 = Path("data/features/features.parquet")
        c0 = Path("data/features/features.csv")
        f = pd.read_parquet(p0) if p0.exists() else pd.read_csv(c0)

    f = f.copy()
    f["Date"] = pd.to_datetime(f["Date"])
    f["Ticker"] = f["Ticker"].astype(str).str.upper().str.strip()
    return f


def load_strategy_labels(tag: str) -> pd.DataFrame:
    p = LABEL_DIR / f"strategy_labels_{tag}.parquet"
    c = LABEL_DIR / f"strategy_labels_{tag}.csv"
    if p.exists():
        df = pd.read_parquet(p)
    elif c.exists():
        df = pd.read_csv(c)
    else:
        raise FileNotFoundError(f"strategy labels not found for tag={tag}. Run scripts/build_strategy_labels.py first.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profit-target", type=float, required=True)
    p.add_argument("--max-days", type=int, required=True)
    p.add_argument("--stop-level", type=float, required=True)
    args = p.parse_args()

    tag = fmt_tag(args.profit_target, args.max_days, args.stop_level)

    feat = load_features()
    lab = load_strategy_labels(tag)

    df = feat.merge(lab[["Date", "Ticker", "CycleReturn", "ExtendDays", "MinCycleRet"]], on=["Date", "Ticker"], how="inner")
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # 엄격모드: features에서 이미 dropna 했지만, 혹시 섞인 NaN은 제거
    feature_cols = [c for c in feat.columns if c not in ("Date", "Ticker")]
    feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    df = df.dropna(subset=feature_cols + ["CycleReturn"]).reset_index(drop=True)

    # Targets
    y_pos = (df["CycleReturn"] > 0).astype(int).to_numpy()
    y_ret = df["CycleReturn"].astype(float).to_numpy()

    X = df[feature_cols].to_numpy(dtype=float)

    # 날짜 기준 80/20 split
    unique_dates = np.array(sorted(df["Date"].unique()))
    split_i = int(len(unique_dates) * 0.8)
    split_i = max(1, min(split_i, len(unique_dates) - 1))
    train_dates = set(unique_dates[:split_i])
    test_dates = set(unique_dates[split_i:])

    is_train = df["Date"].isin(train_dates).to_numpy()
    is_test = df["Date"].isin(test_dates).to_numpy()

    X_train, X_test = X[is_train], X[is_test]
    y_pos_train, y_pos_test = y_pos[is_train], y_pos[is_test]
    y_ret_train, y_ret_test = y_ret[is_train], y_ret[is_test]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 1) Classifier (p_pos)
    base = LogisticRegression(max_iter=800)
    tscv = TimeSeriesSplit(n_splits=3)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=tscv)
    clf.fit(X_train_s, y_pos_train)

    p_test = clf.predict_proba(X_test_s)[:, 1]
    auc = None
    if len(np.unique(y_pos_test)) >= 2:
        auc = float(roc_auc_score(y_pos_test, p_test))

    # 2) Regressor (E[CycleReturn])
    reg = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.08,
        max_iter=300,
        random_state=42,
    )
    reg.fit(X_train_s, y_ret_train)
    rhat_test = reg.predict(X_test_s)

    print("=" * 80)
    print("TAG:", tag)
    print("Rows train/test:", len(X_train), "/", len(X_test))
    print("Features:", len(feature_cols))
    if auc is not None:
        print("p_pos ROC-AUC:", round(auc, 4))
    else:
        print("p_pos ROC-AUC: (skipped - single class in test)")
    print("Test base pos rate:", round(float(y_pos_test.mean()), 4))
    print("Test mean p_pos:", round(float(p_test.mean()), 4))
    print("Test mean realized return:", round(float(np.mean(y_ret_test)), 6))
    print("Test mean predicted return:", round(float(np.mean(rhat_test)), 6))
    print("=" * 80)

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    clf_path = MODEL_DIR / f"clf_pos_{tag}.pkl"
    reg_path = MODEL_DIR / f"reg_ret_{tag}.pkl"
    scaler_path = MODEL_DIR / f"scaler_{tag}.pkl"
    meta_path = MODEL_DIR / f"models_{tag}_meta.json"

    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "trained_at_utc": now_utc_iso(),
        "tag": tag,
        "profit_target": args.profit_target,
        "max_days": args.max_days,
        "stop_level": args.stop_level,
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "auc_pos": auc,
        "test_base_pos_rate": float(y_pos_test.mean()),
        "test_mean_p_pos": float(p_test.mean()),
        "test_mean_realized_ret": float(np.mean(y_ret_test)),
        "test_mean_pred_ret": float(np.mean(rhat_test)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"✅ saved: {clf_path}")
    print(f"✅ saved: {reg_path}")
    print(f"✅ saved: {scaler_path}")
    print(f"✅ saved: {meta_path}")


if __name__ == "__main__":
    import json
    main()
