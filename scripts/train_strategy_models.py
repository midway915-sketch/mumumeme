# scripts/train_strategy_models.py
from __future__ import annotations

import argparse
import json
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
from sklearn.ensemble import HistGradientBoostingRegressor


DATA_DIR = Path("data")
FEATURE_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"
MODEL_DIR = Path("app") / "model"

FEATURES_MODEL_PARQUET = FEATURE_DIR / "features_model.parquet"
FEATURES_MODEL_CSV = FEATURE_DIR / "features_model.csv"
FEATURES_PARQUET = FEATURE_DIR / "features.parquet"
FEATURES_CSV = FEATURE_DIR / "features.csv"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_tag(pt: float, max_days: int, sl: float, max_ext: int) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{max_days}_sl{sl_tag}_ex{max_ext}"


def load_features() -> pd.DataFrame:
    # EV feature 포함된 features_model 우선
    if FEATURES_MODEL_PARQUET.exists():
        f = pd.read_parquet(FEATURES_MODEL_PARQUET)
    elif FEATURES_MODEL_CSV.exists():
        f = pd.read_csv(FEATURES_MODEL_CSV)
    else:
        # fallback
        if FEATURES_PARQUET.exists():
            f = pd.read_parquet(FEATURES_PARQUET)
        else:
            f = pd.read_csv(FEATURES_CSV)

    f = f.copy()
    f["Date"] = pd.to_datetime(f["Date"])
    f["Ticker"] = f["Ticker"].astype(str).str.upper().str.strip()
    return f.sort_values(["Date", "Ticker"]).reset_index(drop=True)


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
    df["ExitDate"] = pd.to_datetime(df["ExitDate"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def make_date_cv_splits(train_df: pd.DataFrame, n_splits: int = 3):
    """
    CalibratedClassifierCV에 넣을 수 있는 (train_idx, test_idx) 리스트 생성.
    날짜 단위로 fold를 나눠서 같은 날짜가 train/test에 섞이는 걸 줄임.
    """
    unique_dates = np.array(sorted(train_df["Date"].unique()))
    if len(unique_dates) < (n_splits + 1):
        # 데이터 너무 적으면 그냥 3-fold로 단순화
        return 3

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for tr_d, te_d in tscv.split(unique_dates):
        tr_dates = set(unique_dates[tr_d])
        te_dates = set(unique_dates[te_d])
        tr_idx = np.where(train_df["Date"].isin(tr_dates).to_numpy())[0]
        te_idx = np.where(train_df["Date"].isin(te_dates).to_numpy())[0]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        splits.append((tr_idx, te_idx))

    return splits if len(splits) >= 2 else 3


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)
    ap.add_argument("--tail-threshold", type=float, default=-0.30, help="MinCycleRet가 이 값 이하이면 tail=1")
    args = ap.parse_args()

    tag = fmt_tag(args.profit_target, args.max_days, args.stop_level, args.max_extend_days)

    feat = load_features()
    lab = load_strategy_labels(tag)

    # 학습에 필요한 컬럼만 merge
    df = feat.merge(
        lab[["Date", "Ticker", "CycleReturn", "MinCycleRet", "ExtendDays", "ExtendedFlag", "ForcedExitFlag"]],
        on=["Date", "Ticker"],
        how="inner",
    ).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # feature columns: 숫자만
    feature_cols = [c for c in feat.columns if c not in ("Date", "Ticker")]
    feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # NaN 제거(엄격)
    df = df.dropna(subset=feature_cols + ["CycleReturn", "MinCycleRet"]).reset_index(drop=True)

    # Targets
    y_pos = (df["CycleReturn"].to_numpy(dtype=float) > 0).astype(int)
    y_tail = (df["MinCycleRet"].to_numpy(dtype=float) <= float(args.tail_threshold)).astype(int)
    y_ret = df["CycleReturn"].to_numpy(dtype=float)
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
    y_tail_train, y_tail_test = y_tail[is_train], y_tail[is_test]
    y_ret_train, y_ret_test = y_ret[is_train], y_ret[is_test]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 날짜 기반 CV splits(캘리브레이션)
    train_df = df.loc[is_train, ["Date"]].reset_index(drop=True)
    cv_splits = make_date_cv_splits(train_df, n_splits=3)

    # 1) p_pos 모델
    base_pos = LogisticRegression(max_iter=800)
    clf_pos = CalibratedClassifierCV(base_pos, method="isotonic", cv=cv_splits)
    clf_pos.fit(X_train_s, y_pos_train)
    p_pos_test = clf_pos.predict_proba(X_test_s)[:, 1]
    auc_pos = safe_auc(y_pos_test, p_pos_test)

    # 2) tail(큰 손실) 확률 모델
    base_tail = LogisticRegression(max_iter=800)
    clf_tail = CalibratedClassifierCV(base_tail, method="isotonic", cv=cv_splits)
    clf_tail.fit(X_train_s, y_tail_train)
    p_tail_test = clf_tail.predict_proba(X_test_s)[:, 1]
    auc_tail = safe_auc(y_tail_test, p_tail_test)

    # 3) 기대수익 회귀
    reg = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.08,
        max_iter=350,
        random_state=42,
    )
    reg.fit(X_train_s, y_ret_train)
    rhat_test = reg.predict(X_test_s)

    # 출력
    print("=" * 90)
    print("TAG:", tag)
    print("Rows train/test:", len(X_train), "/", len(X_test))
    print("Features:", len(feature_cols))
    print("Test pos rate:", round(float(y_pos_test.mean()), 4), "  mean p_pos:", round(float(p_pos_test.mean()), 4))
    if auc_pos is not None:
        print("p_pos ROC-AUC:", round(auc_pos, 4))
    else:
        print("p_pos ROC-AUC: (skipped - single class in test)")
    print("Test tail rate:", round(float(y_tail_test.mean()), 4), " mean p_tail:", round(float(p_tail_test.mean()), 4))
    if auc_tail is not None:
        print("p_tail ROC-AUC:", round(auc_tail, 4))
    else:
        print("p_tail ROC-AUC: (skipped - single class in test)")
    print("Test mean realized return:", round(float(np.mean(y_ret_test)), 6))
    print("Test mean predicted return:", round(float(np.mean(rhat_test)), 6))
    print("=" * 90)

    # 저장
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    thr_tag = int(round(abs(args.tail_threshold) * 100))

    clf_pos_path = MODEL_DIR / f"clf_pos_{tag}.pkl"
    clf_tail_path = MODEL_DIR / f"clf_tail_{tag}_thr{thr_tag}.pkl"
    reg_path = MODEL_DIR / f"reg_ret_{tag}.pkl"
    scaler_path = MODEL_DIR / f"scaler_{tag}.pkl"
    meta_path = MODEL_DIR / f"models_{tag}_meta.json"

    joblib.dump(clf_pos, clf_pos_path)
    joblib.dump(clf_tail, clf_tail_path)
    joblib.dump(reg, reg_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "trained_at_utc": now_utc_iso(),
        "tag": tag,
        "profit_target": args.profit_target,
        "max_days": args.max_days,
        "stop_level": args.stop_level,
        "max_extend_days": args.max_extend_days,
        "tail_threshold": args.tail_threshold,
        "tail_threshold_tag": thr_tag,
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "auc_pos": auc_pos,
        "auc_tail": auc_tail,
        "test_pos_rate": float(y_pos_test.mean()),
        "test_tail_rate": float(y_tail_test.mean()),
        "test_mean_p_pos": float(p_pos_test.mean()),
        "test_mean_p_tail": float(p_tail_test.mean()),
        "test_mean_realized_ret": float(np.mean(y_ret_test)),
        "test_mean_pred_ret": float(np.mean(rhat_test)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("✅ saved:", clf_pos_path)
    print("✅ saved:", clf_tail_path)
    print("✅ saved:", reg_path)
    print("✅ saved:", scaler_path)
    print("✅ saved:", meta_path)


if __name__ == "__main__":
    main()
