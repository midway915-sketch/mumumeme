# scripts/predict_gate.py
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import joblib


DATA_DIR = Path("data")
FEAT_DIR = DATA_DIR / "features"
SIG_DIR = DATA_DIR / "signals"
APP_DIR = Path("app")

FEATURE_COLS = [
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


def _to_dt(x):
    return pd.to_datetime(x, errors="coerce")


def parse_csv_list(s: str) -> list[str]:
    if s is None:
        return []
    items = [x.strip() for x in str(s).split(",")]
    return [x for x in items if x != ""]


def parse_float_list(s: str) -> list[float]:
    out = []
    for x in parse_csv_list(s):
        out.append(float(x))
    return out


def sanitize_suffix(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def ensure_feature_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    # ✅ 누락 컬럼은 0으로 생성해서 pipeline 안 죽게
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def load_tagged_model(kind: str, tag: str):
    if kind == "success":
        m1 = APP_DIR / f"model_{tag}.pkl"
        s1 = APP_DIR / f"scaler_{tag}.pkl"
        m0 = APP_DIR / "model.pkl"
        s0 = APP_DIR / "scaler.pkl"
    elif kind == "tail":
        m1 = APP_DIR / f"tail_model_{tag}.pkl"
        s1 = APP_DIR / f"tail_scaler_{tag}.pkl"
        m0 = APP_DIR / "tail_model.pkl"
        s0 = APP_DIR / "tail_scaler.pkl"
    else:
        raise ValueError("kind must be success|tail")

    if m1.exists() and s1.exists():
        return joblib.load(m1), joblib.load(s1), str(m1), str(s1)
    if m0.exists() and s0.exists():
        return joblib.load(m0), joblib.load(s0), str(m0), str(s0)
    raise FileNotFoundError(f"Missing {kind} model files for tag={tag}")


def try_load_tail(tag: str):
    try:
        return load_tagged_model("tail", tag)
    except FileNotFoundError:
        return None


def predict_prob(df: pd.DataFrame, model, scaler, feature_cols: list[str]) -> np.ndarray:
    X = df[feature_cols].to_numpy(dtype=float)
    Xs = scaler.transform(X)
    p = model.predict_proba(Xs)[:, 1]
    return np.asarray(p, dtype=float)


def compute_ret_score(feats: pd.DataFrame) -> np.ndarray:
    z = pd.to_numeric(feats.get("Z_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dd60 = pd.to_numeric(feats.get("Drawdown_60", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dd252 = pd.to_numeric(feats.get("Drawdown_252", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    atr = pd.to_numeric(feats.get("ATR_ratio", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    macd = pd.to_numeric(feats.get("MACD_hist", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    slope = pd.to_numeric(feats.get("MA20_slope", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    score = (
        0.35 * (-dd60) +
        0.15 * (-dd252) +
        0.20 * (macd) +
        0.20 * (slope) +
        0.10 * (atr) +
        0.05 * (-np.abs(z))
    )
    return score.astype(float)


def make_utility(p_success: np.ndarray, p_tail: np.ndarray, lambda_tail: float) -> np.ndarray:
    lam = float(lambda_tail)
    return (p_success - lam * p_tail).astype(float)


def gate_and_pick_one_day(day_df: pd.DataFrame, mode: str, tail_max: float,
                         utility_quantiles: list[float], rank_by: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = day_df.copy()
    for c in ["p_success", "p_tail", "utility", "ret_score"]:
        if c not in df.columns:
            df[c] = np.nan

    qs = utility_quantiles[:] if utility_quantiles else [0.75, 0.60, 0.50]
    qs = sorted([float(q) for q in qs], reverse=True)  # strict -> loose

    if rank_by not in ("utility", "ret_score", "p_success"):
        raise ValueError("rank_by must be utility|ret_score|p_success")
    rank_col = rank_by

    def pick_top(cands: pd.DataFrame) -> pd.DataFrame:
        if cands is None or len(cands) == 0:
            return df.iloc[:0].copy()
        cc = cands.copy()
        cc[rank_col] = pd.to_numeric(cc[rank_col], errors="coerce").fillna(-1e18)
        cc = cc.sort_values(rank_col, ascending=False)
        return cc.head(1).copy()

    def tail_filter(x: pd.DataFrame) -> pd.DataFrame:
        t = x.copy()
        t["p_tail"] = pd.to_numeric(t["p_tail"], errors="coerce")
        return t.loc[t["p_tail"].notna() & (t["p_tail"] <= float(tail_max))].copy()

    def utility_cut(q: float) -> float:
        u = pd.to_numeric(df["utility"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(u) == 0:
            return float("inf")
        return float(u.quantile(q))

    if mode == "none":
        return pick_top(df), df

    # tail-only
    if mode == "tail":
        c = tail_filter(df)
        return pick_top(c), c

    # utility or tail_utility: stepwise relax utility by DAY quantile
    base = df
    if mode == "tail_utility":
        base = tail_filter(df)

    last_cands = base.iloc[:0].copy()

    for q in qs:
        thr = utility_cut(q)  # based on day df (not base), but threshold applies on base rows
        c = base.copy()
        c["utility"] = pd.to_numeric(c["utility"], errors="coerce")
        c = c.loc[c["utility"].notna() & (c["utility"] >= thr)].copy()
        last_cands = c
        if len(c) > 0:
            return pick_top(c), c

    # fallback:
    if mode == "tail_utility":
        # utility 조건 완전 제거, tail 통과자 중 top-1
        if len(base) > 0:
            return pick_top(base), base
        return df.iloc[:0].copy(), last_cands

    # mode == "utility": 끝까지 없으면 스킵
    return df.iloc[:0].copy(), last_cands


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--mode", required=True, choices=["none", "tail", "utility", "tail_utility"])
    ap.add_argument("--tag", required=True, type=str)
    ap.add_argument("--suffix", required=True, type=str)
    ap.add_argument("--out-dir", default=str(SIG_DIR), type=str)

    ap.add_argument("--features-parq", default=str(FEAT_DIR / "features_model.parquet"), type=str)
    ap.add_argument("--features-csv", default=str(FEAT_DIR / "features_model.csv"), type=str)

    ap.add_argument("--tail-threshold", required=True, type=float)
    ap.add_argument("--utility-quantile", required=True, type=str)  # "0.75,0.60,0.50"
    ap.add_argument("--rank-by", required=True, choices=["utility", "ret_score", "p_success"])
    ap.add_argument("--lambda-tail", required=True, type=float)

    ap.add_argument("--require-files", default="", type=str)
    args = ap.parse_args()

    # optional required files check
    for p in parse_csv_list(args.require_files):
        if p and (not Path(p).exists()):
            raise FileNotFoundError(f"--require-files missing: {p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag.strip()
    suffix = sanitize_suffix(args.suffix)

    feats = read_table(Path(args.features_parq), Path(args.features_csv)).copy()
    feats.columns = [str(c).strip() for c in feats.columns]
    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_model must contain Date and Ticker")

    feats["Date"] = _to_dt(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    feats = ensure_feature_cols(feats, FEATURE_COLS)

    # success model must exist (this is core)
    success_model, success_scaler, ms, ss = load_tagged_model("success", tag)
    p_success = predict_prob(feats, success_model, success_scaler, FEATURE_COLS)
    feats["p_success"] = p_success

    # tail model is optional (we keep pipeline alive if missing)
    tail_pack = try_load_tail(tag)
    if tail_pack is None:
        feats["p_tail"] = 0.0
        print(f"[WARN] tail model not found for tag={tag}. Using p_tail=0.0 (tail gates become no-op).")
        mt = st = "(missing)"
    else:
        tail_model, tail_scaler, mt, st = tail_pack
        feats["p_tail"] = predict_prob(feats, tail_model, tail_scaler, FEATURE_COLS)

    feats["ret_score"] = compute_ret_score(feats)
    feats["utility"] = make_utility(
        feats["p_success"].to_numpy(dtype=float),
        pd.to_numeric(feats["p_tail"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        float(args.lambda_tail),
    )

    utility_qs = parse_float_list(args.utility_quantile) or [0.75, 0.60, 0.50]

    picks = []
    picks_full = []

    for d, day_df in feats.groupby("Date", sort=False):
        picked, cands = gate_and_pick_one_day(
            day_df=day_df,
            mode=args.mode,
            tail_max=float(args.tail_threshold),
            utility_quantiles=utility_qs,
            rank_by=args.rank_by,
        )

        cands = (cands if cands is not None else day_df.iloc[:0].copy()).copy()
        cands["GateMode"] = args.mode
        cands["GateSuffix"] = suffix
        picks_full.append(cands)

        if picked is not None and len(picked) > 0:
            r = picked.iloc[0].to_dict()
            r["GateMode"] = args.mode
            r["GateSuffix"] = suffix
            picks.append(r)

    picks_df = pd.DataFrame(picks)
    if len(picks_df) == 0:
        picks_df = pd.DataFrame(columns=["Date", "Ticker", "p_success", "p_tail", "utility", "ret_score", "GateMode", "GateSuffix"])

    picks_df = picks_df.sort_values(["Date"]).reset_index(drop=True)
    out_picks = out_dir / f"picks_{tag}_gate_{suffix}.csv"
    picks_df.to_csv(out_picks, index=False)

    full_df = pd.concat(picks_full, ignore_index=True) if picks_full else pd.DataFrame()
    out_full = out_dir / f"picks_full_{tag}_gate_{suffix}.csv"
    full_df.to_csv(out_full, index=False)

    print("=" * 60)
    print("[DONE] predict_gate.py")
    print("mode:", args.mode)
    print("tag:", tag)
    print("suffix:", suffix)
    print("rank_by:", args.rank_by)
    print("utility_quantiles(day-wise):", utility_qs)
    print("tail_max:", float(args.tail_threshold))
    print("models:", ms, mt)
    print("picks rows:", len(picks_df))
    print("wrote:", out_picks)
    print("wrote:", out_full)
    print("=" * 60)


if __name__ == "__main__":
    main()