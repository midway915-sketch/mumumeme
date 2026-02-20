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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

from scripts.feature_spec import read_feature_cols_meta, get_feature_cols


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
            f"-> features_model/build_features와 labels_tau/build_tau_labels가 같은 SSOT feature_cols를 쓰도록 맞춰야 함."
        )


def coerce_features_numeric(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feat_cols:
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train tau (FAST/MID/SLOW) multi-class model.")

    # tag는 리포트/파일명용 (워크플로에서 LABEL_KEY+ex 붙여서 넘겨도 됨)
    ap.add_argument("--tag", default="", type=str, help="e.g. pt10_h40_sl10_ex20")

    ap.add_argument("--features-parq", default="data/features/features_model.parquet", type=str)
    ap.add_argument("--features-csv", default="data/features/features_model.csv", type=str)
    ap.add_argument("--labels-parq", default="data/labels/labels_tau.parquet", type=str)
    ap.add_argument("--labels-csv", default="data/labels/labels_tau.csv", type=str)

    ap.add_argument("--date-col", default="Date", type=str)
    ap.add_argument("--ticker-col", default="Ticker", type=str)
    ap.add_argument("--target-col", default="TauClass", type=str)

    # ✅ 기본은 SSOT(meta) 사용. 필요할 때만 override.
    ap.add_argument("--features", default="", type=str, help="comma-separated feature cols (override SSOT/meta)")

    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=800, type=int)

    ap.add_argument("--out-model", default="", type=str)
    ap.add_argument("--out-scaler", default="", type=str)

    args = ap.parse_args()

    feats = read_table(args.features_parq, args.features_csv).copy()
    labs = read_table(args.labels_parq, args.labels_csv).copy()

    # basic schema checks
    for df, name in [(feats, "features_model"), (labs, "labels_tau")]:
        if args.date_col not in df.columns or args.ticker_col not in df.columns:
            raise KeyError(f"{name} must have {args.date_col},{args.ticker_col}")

    if args.target_col not in labs.columns:
        raise KeyError(f"labels_tau missing target column: {args.target_col}")

    # normalize keys
    feats[args.date_col] = norm_date(feats[args.date_col])
    labs[args.date_col] = norm_date(labs[args.date_col])
    feats[args.ticker_col] = feats[args.ticker_col].astype(str).str.upper().str.strip()
    labs[args.ticker_col] = labs[args.ticker_col].astype(str).str.upper().str.strip()

    feats = feats.dropna(subset=[args.date_col, args.ticker_col]).copy()
    labs = labs.dropna(subset=[args.date_col, args.ticker_col, args.target_col]).copy()

    # ✅ SSOT feature cols
    feat_cols, feat_src = resolve_feature_cols(args.features)
    feat_cols = [c.strip() for c in feat_cols if c.strip()]

    # ✅ STRICT: features_model에 피처가 없으면 바로 에러
    ensure_feature_columns_strict(feats, feat_cols, source_hint=f"{feat_src}, feats_src={args.features_parq}")

    # numeric coercion
    feats = coerce_features_numeric(feats, feat_cols)

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

    ll_test = float("nan")
    if len(X_test_s) > 0:
        p_test = model.predict_proba(X_test_s)
        ll_test = float(log_loss(y_test, p_test, labels=[0, 1, 2]))

    print("=" * 60)
    print(f"[INFO] rows={n} train={len(X_train)} test={len(X_test)}")
    print(f"[INFO] feature_cols_source: {feat_src}")
    print(f"[INFO] CV logloss (train folds): {np.round(cv_losses, 4).tolist()} mean={float(np.mean(cv_losses)):.4f}")
    print(f"[INFO] Test logloss: {ll_test:.4f}")
    print(f"[INFO] Class distribution (all): {pd.Series(y).value_counts().to_dict()}")
    print("=" * 60)

    # outputs
    APP_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    tag = (args.tag or "").strip()

    # ✅ 파이프라인 호환: 기본은 고정 파일명
    out_model = Path(args.out_model) if args.out_model else (APP_DIR / "tau_model.pkl")
    out_scaler = Path(args.out_scaler) if args.out_scaler else (APP_DIR / "tau_scaler.pkl")

    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)
    print(f"[DONE] saved model: {out_model}")
    print(f"[DONE] saved scaler: {out_scaler}")

    report = {
        "tag": tag,
        "feature_cols_source": feat_src,
        "feature_cols": feat_cols,
        "rows": int(n),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cv_logloss": [float(x) for x in cv_losses],
        "cv_logloss_mean": float(np.mean(cv_losses)) if cv_losses else None,
        "test_logloss": ll_test,
        "class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
        "out_model": str(out_model),
        "out_scaler": str(out_scaler),
        "labels_src": str(Path(args.labels_parq) if Path(args.labels_parq).exists() else Path(args.labels_csv)),
        "features_src": str(Path(args.features_parq) if Path(args.features_parq).exists() else Path(args.features_csv)),
    }
    report_path = META_DIR / (f"train_tau_report_{tag}.json" if tag else "train_tau_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote report: {report_path}")


if __name__ == "__main__":
    main()