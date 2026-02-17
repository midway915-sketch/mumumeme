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


def make_tag(profit_target: float, max_days: int, stop_level: float, max_extend_days: int) -> str:
    pt_tag = int(round(float(profit_target) * 100))
    sl_tag = int(round(abs(float(stop_level)) * 100))
    return f"pt{pt_tag}_h{int(max_days)}_sl{sl_tag}_ex{int(max_extend_days)}"


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
    # safe filename segment
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def ensure_feature_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"features_model missing required FEATURE_COLS: {missing}")
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def load_tagged_model(kind: str, tag: str):
    """
    kind: 'success' or 'tail'
    expects:
      app/model_{tag}.pkl + app/scaler_{tag}.pkl
      app/tail_model_{tag}.pkl + app/tail_scaler_{tag}.pkl
    fallback to app/model.pkl if tagged not found (but we try tagged first).
    """
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
    raise FileNotFoundError(f"Missing {kind} model files for tag={tag} (checked {m1},{s1} and {m0},{s0})")


def predict_prob(df: pd.DataFrame, model, scaler, feature_cols: list[str]) -> np.ndarray:
    X = df[feature_cols].to_numpy(dtype=float)
    Xs = scaler.transform(X)
    p = model.predict_proba(Xs)[:, 1]
    return np.asarray(p, dtype=float)


def compute_ret_score(feats: pd.DataFrame) -> np.ndarray:
    """
    Simple, stable heuristic score (no ML).
    Higher is better.
    """
    z = pd.to_numeric(feats.get("Z_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dd60 = pd.to_numeric(feats.get("Drawdown_60", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dd252 = pd.to_numeric(feats.get("Drawdown_252", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    atr = pd.to_numeric(feats.get("ATR_ratio", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    macd = pd.to_numeric(feats.get("MACD_hist", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    slope = pd.to_numeric(feats.get("MA20_slope", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # contrarian-ish + momentum blend (works decently for lever ETFs)
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
    # utility = success minus tail penalty (tail is "bad path" probability)
    lam = float(lambda_tail)
    return (p_success - lam * p_tail).astype(float)


def gate_and_pick_one_day(
    day_df: pd.DataFrame,
    mode: str,
    tail_max: float,
    utility_quantiles: list[float],
    rank_by: str,
    lambda_tail: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (picked_row_df, candidates_df_after_filters)
    picked_row_df is empty if no pick.
    Strategy:
      - compute u_cuts by day quantiles (descending strictness if needed)
      - primary filters depend on mode
      - if 0 passes:
          - relax ONLY utility cut stepwise (tail kept) for modes involving utility
          - final fallback: if still none and tail filter exists, pick top-1 from tail survivors
    """
    df = day_df.copy()

    # Always ensure required columns exist
    for c in ["p_success", "p_tail", "utility", "ret_score"]:
        if c not in df.columns:
            df[c] = np.nan

    # sort utility_quantiles from strict -> loose (e.g. 0.90,0.75,0.60)
    qs = [float(q) for q in utility_quantiles] if utility_quantiles else [0.75, 0.60, 0.50]
    qs = sorted(qs, reverse=True)

    # day-wise utility cut function
    def u_cut(q: float) -> float:
        u = pd.to_numeric(df["utility"], errors="coerce")
        u = u.replace([np.inf, -np.inf], np.nan).dropna()
        if len(u) == 0:
            return float("inf")  # makes everyone fail utility gate
        return float(u.quantile(q))

    # rank score
    if rank_by == "utility":
        rank_col = "utility"
    elif rank_by == "ret_score":
        rank_col = "ret_score"
    elif rank_by == "p_success":
        rank_col = "p_success"
    else:
        raise ValueError("rank_by must be utility|ret_score|p_success")

    def pick_top(cands: pd.DataFrame) -> pd.DataFrame:
        if cands is None or len(cands) == 0:
            return cands.iloc[:0].copy()
        cc = cands.copy()
        cc[rank_col] = pd.to_numeric(cc[rank_col], errors="coerce").fillna(-1e18)
        cc = cc.sort_values(rank_col, ascending=False)
        return cc.head(1).copy()

    # build base filters
    def apply_filters(u_threshold: float | None) -> pd.DataFrame:
        cands = df.copy()

        if mode in ("tail", "tail_utility"):
            cands["p_tail"] = pd.to_numeric(cands["p_tail"], errors="coerce")
            cands = cands.loc[cands["p_tail"].notna() & (cands["p_tail"] <= float(tail_max))].copy()

        if mode in ("utility", "tail_utility"):
            if u_threshold is None or not np.isfinite(u_threshold):
                # no valid threshold -> empty
                return cands.iloc[:0].copy()
            cands["utility"] = pd.to_numeric(cands["utility"], errors="coerce")
            cands = cands.loc[cands["utility"].notna() & (cands["utility"] >= float(u_threshold))].copy()

        # mode=none passes all
        return cands

    # mode=none: no gate, just pick top by rank
    if mode == "none":
        picked = pick_top(df)
        return picked, df

    # primary attempt(s)
    tried = []
    last_cands = None

    if mode in ("utility", "tail_utility"):
        # strict -> loose utility cuts
        for q in qs:
            thr = u_cut(q)
            cands = apply_filters(thr)
            tried.append((q, thr, len(cands)))
            last_cands = cands
            if len(cands) > 0:
                return pick_top(cands), cands

        # still none: final fallback for utility modes (keep tail if applicable)
        if mode == "utility":
            # no tail constraint to save us -> skip day
            return df.iloc[:0].copy(), (last_cands if last_cands is not None else df.iloc[:0].copy())

        # tail_utility: keep tail, drop utility gate entirely, pick top-1 from tail survivors
        tail_only = apply_filters(u_threshold=None)  # will keep tail, utility gate disabled -> returns empty with current impl
        # so do tail-only explicitly
        tail_surv = df.copy()
        tail_surv["p_tail"] = pd.to_numeric(tail_surv["p_tail"], errors="coerce")
        tail_surv = tail_surv.loc[tail_surv["p_tail"].notna() & (tail_surv["p_tail"] <= float(tail_max))].copy()
        if len(tail_surv) > 0:
            return pick_top(tail_surv), tail_surv
        return df.iloc[:0].copy(), (last_cands if last_cands is not None else df.iloc[:0].copy())

    # mode == "tail": tail gate only
    tail_cands = apply_filters(u_threshold=None)
    # apply_filters(None) for tail mode should keep tail and not apply utility
    if len(tail_cands) > 0:
        return pick_top(tail_cands), tail_cands
    return df.iloc[:0].copy(), tail_cands


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

    ap.add_argument("--tail-threshold", required=True, type=float, help="tail_max (p_tail <= tail_max)")
    ap.add_argument("--utility-quantile", required=True, type=str,
                    help="comma-separated quantiles for stepwise relax (e.g. 0.75,0.60,0.50)")
    ap.add_argument("--rank-by", required=True, choices=["utility", "ret_score", "p_success"])
    ap.add_argument("--lambda-tail", required=True, type=float)

    ap.add_argument("--require-files", default="", type=str,
                    help="comma-separated required paths; if any missing -> error. (optional)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # optional required files check (avoid empty flag error)
    req = parse_csv_list(args.require_files) if hasattr(args, "require_files") else []
    for p in req:
        if p and (not Path(p).exists()):
            raise FileNotFoundError(f"--require-files missing: {p}")

    tag = args.tag
    suffix = sanitize_suffix(args.suffix)

    feats = read_table(Path(args.features_parq), Path(args.features_csv)).copy()
    feats.columns = [str(c).strip() for c in feats.columns]
    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_model must contain Date and Ticker")
    feats["Date"] = _to_dt(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    feats = ensure_feature_cols(feats, FEATURE_COLS)

    # models
    success_model, success_scaler, ms, ss = load_tagged_model("success", tag)
    tail_model, tail_scaler, mt, st = load_tagged_model("tail", tag)

    p_success = predict_prob(feats, success_model, success_scaler, FEATURE_COLS)
    p_tail = predict_prob(feats, tail_model, tail_scaler, FEATURE_COLS)

    feats["p_success"] = p_success
    feats["p_tail"] = p_tail
    feats["ret_score"] = compute_ret_score(feats)
    feats["utility"] = make_utility(feats["p_success"].to_numpy(dtype=float), feats["p_tail"].to_numpy(dtype=float), args.lambda_tail)

    # gate + pick day by day (Top-1)
    utility_qs = parse_float_list(args.utility_quantile)
    if not utility_qs:
        utility_qs = [0.75, 0.60, 0.50]

    picks = []
    picks_full = []

    for d, day_df in feats.groupby("Date", sort=False):
        picked, cands = gate_and_pick_one_day(
            day_df=day_df,
            mode=args.mode,
            tail_max=float(args.tail_threshold),
            utility_quantiles=utility_qs,
            rank_by=args.rank_by,
            lambda_tail=float(args.lambda_tail),
        )

        # full candidates after final filtering attempt (for debugging)
        if cands is None:
            cands = day_df.iloc[:0].copy()
        cands = cands.copy()
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
        # still write empty file to keep pipeline consistent
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