# scripts/predict_gate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def _to_dt(x):
    return pd.to_datetime(x, errors="coerce")


def _make_tau_cutoffs(max_days: int) -> tuple[int, int, int]:
    c3 = int(max_days)
    c1 = min(10, c3)
    c2 = min(20, c3)
    if c1 >= c3:
        c1 = max(1, c3 - 2)
    if c2 <= c1:
        mid = (c1 + c3) // 2
        c2 = mid if mid > c1 else min(c3 - 1, c1 + 1)
    if c2 >= c3:
        c2 = max(c1 + 1, c3 - 1)
    if not (c1 < c2 < c3):
        c1 = max(1, int(round(c3 * 0.33)))
        c2 = max(c1 + 1, int(round(c3 * 0.66)))
        c2 = min(c2, c3 - 1)
    return int(c1), int(c2), int(c3)


def _pick_feature_cols(model, fallback: list[str], df_cols: list[str]) -> list[str]:
    cols = set(df_cols)
    if hasattr(model, "feature_names_in_"):
        feats = [c for c in list(getattr(model, "feature_names_in_")) if c in cols]
        if feats:
            return feats
    feats = [c for c in fallback if c in cols]
    if feats:
        return feats
    banned = {"Date", "Ticker"}
    return [c for c in df_cols if c not in banned][:8]


def _load_tau_cuts(tag: str, max_days: int) -> tuple[int, int, int]:
    report = Path(f"data/meta/train_tau_report_{tag}.json")
    if report.exists():
        try:
            j = json.loads(report.read_text(encoding="utf-8"))
            cuts = j.get("cuts", {})
            c1 = int(cuts.get("cut1", 10))
            c2 = int(cuts.get("cut2", 20))
            c3 = int(cuts.get("cut3", max_days))
            if c1 < c2 < c3:
                return c1, c2, c3
        except Exception:
            pass
    return _make_tau_cutoffs(int(max_days))


def _require_files(spec: str | None) -> None:
    if not spec:
        return
    paths = [p.strip() for p in str(spec).split(",") if p.strip()]
    if not paths:
        return
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"[require-files] Missing: {missing}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--mode", required=True, type=str, choices=["none", "tail", "utility", "tail_utility"])
    ap.add_argument("--tag", required=True, type=str)
    ap.add_argument("--suffix", required=True, type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    ap.add_argument("--features-parq", default="data/features/features_model.parquet", type=str)
    ap.add_argument("--features-csv", default="data/features/features_model.csv", type=str)

    ap.add_argument("--tail-threshold", required=True, type=float)
    ap.add_argument("--utility-quantile", required=True, type=float)
    ap.add_argument("--rank-by", required=True, type=str, choices=["utility", "ret_score", "p_success"])
    ap.add_argument("--lambda-tail", required=True, type=float)

    # ✅ 핵심: 값 없이 들어와도 OK
    ap.add_argument(
        "--require-files",
        nargs="?",
        const="",
        default="",
        type=str,
        help="comma-separated paths that must exist before running (optional)",
    )

    args = ap.parse_args()

    _require_files(args.require_files)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = read_table(args.features_parq, args.features_csv).copy()
    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_model must contain Date and Ticker")

    feats["Date"] = _to_dt(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    success_model = joblib.load("app/model.pkl")
    success_scaler = joblib.load("app/scaler.pkl")

    tail_model = joblib.load("app/tail_model.pkl")
    tail_scaler = joblib.load("app/tail_scaler.pkl")

    tau_models = joblib.load("app/tau_cdf_models.pkl")
    tau_scaler = joblib.load("app/tau_scaler.pkl")

    fallback_feats = [
        "Drawdown_252","Drawdown_60","ATR_ratio","Z_score","MACD_hist","MA20_slope","Market_Drawdown","Market_ATR_ratio"
    ]
    cols = list(feats.columns)

    succ_feats = _pick_feature_cols(success_model, fallback_feats, cols)
    tail_feats = _pick_feature_cols(tail_model, fallback_feats, cols)

    ref_tau_model = tau_models.get("TauLE3", next(iter(tau_models.values())))
    tau_feats = _pick_feature_cols(ref_tau_model, fallback_feats, cols)

    Xs = feats[succ_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    Xt = feats[tail_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    Xtau = feats[tau_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    Xs_s = success_scaler.transform(Xs)
    Xt_s = tail_scaler.transform(Xt)
    Xtau_s = tau_scaler.transform(Xtau)

    p_success = success_model.predict_proba(Xs_s)[:, 1]
    p_tail = tail_model.predict_proba(Xt_s)[:, 1]

    cut1, cut2, cut3 = _load_tau_cuts(args.tag, int(args.max_days))

    p_le1 = tau_models["TauLE1"].predict_proba(Xtau_s)[:, 1]
    p_le2 = tau_models["TauLE2"].predict_proba(Xtau_s)[:, 1]
    p_le3 = tau_models["TauLE3"].predict_proba(Xtau_s)[:, 1]

    p_le1 = np.clip(p_le1, 0.0, 1.0)
    p_le2 = np.clip(np.maximum(p_le2, p_le1), 0.0, 1.0)
    p_le3 = np.clip(np.maximum(p_le3, p_le2), 0.0, 1.0)

    p1 = p_le1
    p2 = np.clip(p_le2 - p_le1, 0.0, 1.0)
    p3 = np.clip(p_le3 - p_le2, 0.0, 1.0)
    p4 = np.clip(1.0 - p_le3, 0.0, 1.0)

    mid1 = max(1.0, cut1 * 0.7)
    mid2 = (cut1 + cut2) / 2.0
    mid3 = (cut2 + cut3) / 2.0
    mid4 = cut3 + max(5.0, (cut3 - cut2) / 2.0)
    tau_exp = (mid1 * p1) + (mid2 * p2) + (mid3 * p3) + (mid4 * p4)

    out = feats[["Date", "Ticker"]].copy()
    out["p_success"] = pd.to_numeric(p_success, errors="coerce")
    out["p_tail"] = pd.to_numeric(p_tail, errors="coerce")

    out["tau_cut1"] = cut1
    out["tau_cut2"] = cut2
    out["tau_cut3"] = cut3

    out["p_tau_le1"] = pd.to_numeric(p_le1, errors="coerce")
    out["p_tau_le2"] = pd.to_numeric(p_le2, errors="coerce")
    out["p_tau_le3"] = pd.to_numeric(p_le3, errors="coerce")

    out["p_tau1"] = pd.to_numeric(p1, errors="coerce")
    out["p_tau2"] = pd.to_numeric(p2, errors="coerce")
    out["p_tau3"] = pd.to_numeric(p3, errors="coerce")
    out["p_tau4"] = pd.to_numeric(p4, errors="coerce")
    out["tau_exp"] = pd.to_numeric(tau_exp, errors="coerce")

    if "ret_score" in feats.columns:
        out["ret_score"] = pd.to_numeric(feats["ret_score"], errors="coerce").fillna(0.0)
    else:
        out["ret_score"] = out["p_success"].fillna(0.0) - out["p_tail"].fillna(0.0)

    lam = float(args.lambda_tail)
    out["utility"] = out["p_success"].fillna(0.0) - lam * out["p_tail"].fillna(0.0)

    use = out.copy()
    use["Skipped"] = 0

    if args.mode in ("tail", "tail_utility"):
        ok_tail = (use["p_tail"].fillna(1.0) <= float(args.tail_threshold))
    else:
        ok_tail = pd.Series([True] * len(use))

    if args.mode in ("utility", "tail_utility"):
        q = float(args.utility_quantile)
        cut = use.groupby("Date")["utility"].transform(lambda s: s.quantile(q))
        ok_util = (use["utility"] >= cut)
    else:
        ok_util = pd.Series([True] * len(use))

    ok = ok_tail & ok_util
    use.loc[~ok, "Skipped"] = 1

    rank_col = args.rank_by
    if rank_col not in use.columns:
        raise ValueError(f"rank_by column missing: {rank_col}")

    def pick_one(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        cand = g[g["Skipped"] == 0]
        g["Pick"] = 0
        if len(cand) == 0:
            return g
        idx = cand[rank_col].astype(float).idxmax()
        g.loc[idx, "Pick"] = 1
        return g

    use = use.groupby("Date", group_keys=False).apply(pick_one)

    picks = use[use["Pick"] == 1].copy()
    picks = picks.sort_values(["Date", rank_col], ascending=[True, False]).drop_duplicates(["Date"], keep="first")

    out_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
    full_path = out_dir / f"picks_full_{args.tag}_gate_{args.suffix}.csv"

    use = use.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    use.to_csv(full_path, index=False)

    picks_out = picks[
        ["Date","Ticker","p_success","p_tail","utility","ret_score","tau_exp",
         "tau_cut1","tau_cut2","tau_cut3","p_tau_le1","p_tau_le2","p_tau_le3","p_tau1","p_tau2","p_tau3","p_tau4"]
    ].copy()
    picks_out.to_csv(out_path, index=False)

    print(f"[DONE] wrote picks: {out_path} (rows={len(picks_out)})")
    print(f"[INFO] full daily table: {full_path} (rows={len(use)})")


if __name__ == "__main__":
    main()