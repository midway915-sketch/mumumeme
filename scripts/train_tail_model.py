#!/usr/bin/env python3
from __future__ import annotations

# ✅ FIX(A): "python scripts/xxx.py" 실행에서도 scripts.* import 되도록 repo root를 sys.path에 추가
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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

from scripts.feature_spec import read_feature_cols_meta, get_feature_cols


APP_DIR = Path("app")
META_DIR = Path("data/meta")
LABELS_DIR = Path("data/labels")


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def parse_csv_list(s: str) -> list[str]:
    if s is None:
        return []
    items = [x.strip() for x in str(s).split(",")]
    return [x for x in items if x]


def resolve_feature_cols(args_features: str) -> tuple[list[str], str]:
    """
    Priority:
      1) --features (explicit override)
      2) data/meta/feature_cols.json (SSOT written by build_features.py)
      3) fallback SSOT default (sector disabled)
    Returns: (feature_cols, source_string)
    """
    override = parse_csv_list(args_features)
    if override:
        return [str(c).strip() for c in override if str(c).strip()], "--features"

    cols_meta, _sector_enabled = read_feature_cols_meta()
    if cols_meta:
        return cols_meta, "data/meta/feature_cols.json"

    return get_feature_cols(sector_enabled=False), "feature_spec.py (fallback)"


def ensure_feature_columns_strict(df: pd.DataFrame, feat_cols: list[str], source_hint: str = "") -> None:
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        hint = f" (src={source_hint})" if source_hint else ""
        raise ValueError(
            f"Missing feature columns{hint}: {missing}\n"
            f"-> build_tail_labels.py가 SSOT feature_cols를 포함하도록 먼저 맞춰져 있어야 함."
        )


def coerce_features_numeric(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in feat_cols:
        df[c] = (
            pd.to_numeric(df[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    return df


def _date_based_train_test_split(df: pd.DataFrame, date_col: str, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    ✅ row split 금지: Date 단위로 split
    Returns: (train_df, test_df, cut_date_str)
    """
    d = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    uniq = pd.Series(d.dropna().unique()).sort_values().to_list()
    if len(uniq) < 10:
        raise RuntimeError(f"Not enough unique dates for split: {len(uniq)}")

    cut_i = int(len(uniq) * float(train_ratio)) - 1
    cut_i = max(0, min(cut_i, len(uniq) - 2))  # test 최소 1일 확보
    cut_date = pd.Timestamp(uniq[cut_i])

    train_df = df.loc[pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None) <= cut_date].copy()
    test_df = df.loc[pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None) > cut_date].copy()

    if len(train_df) < 50 or len(test_df) < 50:
        raise RuntimeError(f"Split too small. train={len(train_df)} test={len(test_df)} cut_date={cut_date.date()}")

    return train_df, test_df, str(cut_date.date())


def _make_date_cv_splits(df_train: pd.DataFrame, date_col: str, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    ✅ CV도 Date 단위로만 진행 (같은 날짜 혼재 방지)
    CalibratedClassifierCV(cv=...)에 넣을 수 있는 (train_idx, val_idx) list 반환
    """
    d = pd.to_datetime(df_train[date_col], errors="coerce").dt.tz_localize(None)
    uniq_dates = pd.Series(d.dropna().unique()).sort_values().to_list()

    if len(uniq_dates) < (n_splits + 2):
        # 데이터가 적으면 splits를 줄임
        n_splits = max(2, min(int(n_splits), max(2, len(uniq_dates) - 2)))

    if len(uniq_dates) < 4:
        raise RuntimeError(f"Not enough unique dates for CV: {len(uniq_dates)}")

    date_to_id = {pd.Timestamp(x): i for i, x in enumerate(uniq_dates)}
    date_ids = d.map(lambda x: date_to_id.get(pd.Timestamp(x), -1)).to_numpy(dtype=int)

    tscv = TimeSeriesSplit(n_splits=int(n_splits))
    U = len(uniq_dates)
    dummy = np.arange(U)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for tr_u, te_u in tscv.split(dummy):
        tr_mask = np.isin(date_ids, tr_u)
        te_mask = np.isin(date_ids, te_u)
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        splits.append((tr_idx, te_idx))

    if not splits:
        raise RuntimeError("Failed to build date-based CV splits.")
    return splits


def _fit_model(X_train_s, y_train, cv_splits, max_iter: int):
    base = LogisticRegression(max_iter=int(max_iter))
    try:
        clf = CalibratedClassifierCV(estimator=base, method="isotonic", cv=cv_splits)
    except TypeError:
        clf = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=cv_splits)
    clf.fit(X_train_s, y_train)
    return clf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    # default: SSOT/meta. override only if you really want.
    ap.add_argument("--features", default="", type=str, help="comma-separated feature cols (override SSOT/meta)")

    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=500, type=int)
    args = ap.parse_args()

    pt_tag = int(round(args.profit_target * 100))
    sl_tag = int(round(abs(args.stop_level) * 100))
    tag = f"pt{pt_tag}_h{args.max_days}_sl{sl_tag}_ex{args.max_extend_days}"

    # ✅ A안 SSOT: labels_tail_{tag}만 허용 (fallback 제거)
    parq = LABELS_DIR / f"labels_tail_{tag}.parquet"
    csv = LABELS_DIR / f"labels_tail_{tag}.csv"
    if not parq.exists() and not csv.exists():
        raise FileNotFoundError(
            f"Missing tail labels for tag={tag}.\n"
            f"-> expected: {parq} (or {csv})\n"
            f"-> run: python scripts/build_tail_labels.py --profit-target ... --max-days ... --stop-level ... --max-extend-days ..."
        )

    df = read_table(parq, csv).copy()
    src = str(parq if parq.exists() else csv)

    if "Date" not in df.columns:
        raise ValueError(f"Date column missing (labels_src={src})")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["Date"]).sort_values(["Date"]).reset_index(drop=True)

    if "TailTarget" not in df.columns:
        raise ValueError(f"TailTarget column missing (labels_src={src})")

    feat_cols, feat_src = resolve_feature_cols(args.features)
    feat_cols = [str(c).strip() for c in feat_cols if str(c).strip()]

    # STRICT: missing features -> fail fast
    ensure_feature_columns_strict(df, feat_cols, source_hint=f"{feat_src}, labels_src={src}")

    df = coerce_features_numeric(df, feat_cols)

    use = df.dropna(subset=feat_cols + ["TailTarget"]).copy()
    y_all = pd.to_numeric(use["TailTarget"], errors="coerce").fillna(0).astype(int).to_numpy()

    if len(use) < 200:
        raise RuntimeError(f"Not enough training rows: {len(use)} (labels_src={src})")

    if len(np.unique(y_all)) < 2:
        raise RuntimeError(f"TailTarget has only one class in training data (labels_src={src}).")

    # ✅ Date-based train/test split
    train_df, test_df, cut_date = _date_based_train_test_split(use, date_col="Date", train_ratio=float(args.train_ratio))

    y_train = pd.to_numeric(train_df["TailTarget"], errors="coerce").fillna(0).astype(int).to_numpy()
    X_train = train_df[feat_cols].to_numpy(dtype=float)

    y_test = pd.to_numeric(test_df["TailTarget"], errors="coerce").fillna(0).astype(int).to_numpy()
    X_test = test_df[feat_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ✅ Date-based CV splits
    cv_splits = _make_date_cv_splits(train_df, date_col="Date", n_splits=int(args.n_splits))

    model = _fit_model(X_train_s, y_train, cv_splits=cv_splits, max_iter=int(args.max_iter))

    probs = model.predict_proba(X_test_s)[:, 1]
    auc = float("nan")
    if len(np.unique(y_test)) > 1:
        auc = float(roc_auc_score(y_test, probs))

    APP_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    out_model = APP_DIR / "tail_model.pkl"
    out_scaler = APP_DIR / "tail_scaler.pkl"
    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)

    report = {
        "tag": tag,
        "labels_src": src,

        "rows_used": int(len(use)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "cut_date": cut_date,

        "feature_cols_source": feat_src,
        "feature_cols": feat_cols,

        "auc": (round(auc, 6) if np.isfinite(auc) else None),
        "base_rate_test": (round(float(np.mean(y_test)), 6) if len(y_test) else None),
        "pred_mean_test": (round(float(np.mean(probs)), 6) if len(probs) else None),

        "n_splits": int(len(cv_splits)),
        "paths": {"model": str(out_model), "scaler": str(out_scaler)},
    }
    (META_DIR / f"train_tail_report_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print("[DONE] train_tail_model.py (A + date-split/CV)")
    print("tag:", tag)
    print("labels_src:", src)
    print("cut_date(train<=):", cut_date)
    print("AUC:", report["auc"])
    print("base_rate_test:", report["base_rate_test"])
    print("pred_mean_test:", report["pred_mean_test"])
    print("feature_cols_source:", feat_src)
    print("features:", feat_cols)
    print("n_splits(date-cv):", report["n_splits"])
    print("=" * 60)


if __name__ == "__main__":
    main()