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
    train_dates = set(un_
