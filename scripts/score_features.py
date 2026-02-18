# scripts/score_features.py
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


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


def ensure_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def coerce_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _model_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-z))
    raise TypeError("Model does not support probability prediction")


def load_joblib(path: str):
    if joblib is None:
        raise RuntimeError("joblib is not available. pip install joblib")
    return joblib.load(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build features_scored by attaching p_success/p_tail predictions to features_model.")
    ap.add_argument("--features-parq", default="data/features/features_model.parquet", type=str)
    ap.add_argument("--features-csv", default="data/features/features_model.csv", type=str)

    ap.add_argument("--out-parq", default="data/features/features_scored.parquet", type=str)
    ap.add_argument("--out-csv", default="data/features/features_scored.csv", type=str)

    ap.add_argument("--success-model", default="app/model.pkl", type=str)
    ap.add_argument("--success-scaler", default="app/scaler.pkl", type=str)

    ap.add_argument("--tail-model", default="app/tail_model.pkl", type=str)

    ap.add_argument("--require-success", action="store_true", help="Fail if success model/scaler missing.")
    ap.add_argument("--require-tail", action="store_true", help="Fail if tail model missing.")

    ap.add_argument("--lambda-tail", default=0.05, type=float, help="utility = ret_score - lambda_tail*p_tail")
    ap.add_argument("--meta-out", default="data/features/features_scored_meta.json", type=str)
    args = ap.parse_args()

    feats = read_table(args.features_parq, args.features_csv).copy()
    ensure_cols(feats, ["Date", "Ticker"], "features_model")

    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # --- minimal numeric set: we will score using "all numeric columns except Date/Ticker"
    # but you can later hardcode FEATURE_COLS if you want.
    num_cols = []
    for c in feats.columns:
        if c in ("Date", "Ticker"):
            continue
        if pd.api.types.is_numeric_dtype(feats[c]):
            num_cols.append(c)
        else:
            # try numeric coercion if looks like numbers
            try:
                _ = pd.to_numeric(feats[c].head(50), errors="coerce")
                # if at least some numeric exists, include coerced
                if _.notna().any():
                    feats[c] = pd.to_numeric(feats[c], errors="coerce")
                    num_cols.append(c)
            except Exception:
                pass

    X = feats[num_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- p_success
    succ_model_path = Path(args.success_model)
    succ_scaler_path = Path(args.success_scaler)
    has_success = succ_model_path.exists() and succ_scaler_path.exists()

    if args.require_success and not has_success:
        raise FileNotFoundError(f"Missing success model/scaler: {succ_model_path} / {succ_scaler_path}")

    if has_success:
        model = load_joblib(str(succ_model_path))
        scaler = load_joblib(str(succ_scaler_path))
        Xs = scaler.transform(X.to_numpy(dtype=float))
        # scaler returns ndarray -> keep order
        p_success = _model_predict_proba(model, pd.DataFrame(Xs))
        feats["p_success"] = np.clip(p_success.astype(float), 0.0, 1.0)
    else:
        feats["p_success"] = 0.0

    # --- p_tail
    tail_model_path = Path(args.tail_model)
    has_tail = tail_model_path.exists()

    if args.require_tail and not has_tail:
        raise FileNotFoundError(f"Missing tail model: {tail_model_path}")

    if has_tail:
        tail_model = load_joblib(str(tail_model_path))
        p_tail = _model_predict_proba(tail_model, X)
        feats["p_tail"] = np.clip(p_tail.astype(float), 0.0, 1.0)
    else:
        feats["p_tail"] = 0.0

    # --- ret_score fallback (if missing)
    if "ret_score" not in feats.columns:
        feats["ret_score"] = 0.0
    feats["ret_score"] = coerce_num(feats, "ret_score", 0.0)

    # --- utility
    lam = float(args.lambda_tail)
    feats["utility"] = feats["ret_score"] - lam * coerce_num(feats, "p_tail", 0.0)

    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_csv)
    out_parq.parent.mkdir(parents=True, exist_ok=True)

    feats.to_parquet(out_parq, index=False)
    feats.to_csv(out_csv, index=False)

    meta = {
        "rows": int(len(feats)),
        "date_min": str(pd.to_datetime(feats["Date"]).min().date()) if len(feats) else None,
        "date_max": str(pd.to_datetime(feats["Date"]).max().date()) if len(feats) else None,
        "has_success_model": bool(has_success),
        "has_tail_model": bool(has_tail),
        "success_model": str(succ_model_path),
        "success_scaler": str(succ_scaler_path),
        "tail_model": str(tail_model_path),
        "lambda_tail": lam,
        "numeric_feature_cols_used": num_cols,
    }
    Path(args.meta_out).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] wrote: {out_parq} / {out_csv}")
    print(f"[DONE] meta : {args.meta_out}")
    print(f"[INFO] has_success={has_success} has_tail={has_tail} lambda_tail={lam}")


if __name__ == "__main__":
    main()