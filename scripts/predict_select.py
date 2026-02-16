# scripts/predict_select.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

DATA_DIR = Path("data")
FEATURE_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"
SIGNAL_DIR = DATA_DIR / "signals"
MODEL_DIR = Path("app") / "model"

FEATURES_PARQUET = FEATURE_DIR / "features.parquet"
FEATURES_CSV = FEATURE_DIR / "features.csv"


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


def load_labels(tag: str) -> pd.DataFrame:
    p = LABEL_DIR / f"strategy_labels_{tag}.parquet"
    c = LABEL_DIR / f"strategy_labels_{tag}.csv"
    if p.exists():
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(c)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    tag = fmt_tag(args.profit_target, args.max_days, args.stop_level)

    clf = joblib.load(MODEL_DIR / f"clf_pos_{tag}.pkl")
    reg = joblib.load(MODEL_DIR / f"reg_ret_{tag}.pkl")
    scaler = joblib.load(MODEL_DIR / f"scaler_{tag}.pkl")

    feat = load_features()
    lab = load_labels(tag)

    feature_cols = [c for c in feat.columns if c not in ("Date", "Ticker")]
    feature_cols = feat[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = feat[feature_cols].to_numpy(dtype=float)
    Xs = scaler.transform(X)

    feat = feat.copy()
    feat["p_pos"] = clf.predict_proba(Xs)[:, 1]
    feat["pred_ret"] = reg.predict(Xs)

    # 날짜별 랭킹
    def pick_for_day(g: pd.DataFrame) -> pd.Series:
        # ret-only
        ret_pick = g.loc[g["pred_ret"].idxmax(), "Ticker"]

        # p-only
        p_pick = g.loc[g["p_pos"].idxmax(), "Ticker"]

        # gate(p Top-K) + rank(ret)
        gg = g.sort_values("p_pos", ascending=False).head(max(1, min(args.topk, len(g))))
        gate_pick = gg.loc[gg["pred_ret"].idxmax(), "Ticker"]

        # score 결합(옵션): 0.5 rank(p) + 0.5 rank(ret)
        # rank는 1이 최고로 맞추기 위해 ascending=False
        g2 = g.copy()
        g2["rank_p"] = g2["p_pos"].rank(ascending=False, method="average")
        g2["rank_r"] = g2["pred_ret"].rank(ascending=False, method="average")
        g2["blend_score"] = -0.5 * g2["rank_p"] - 0.5 * g2["rank_r"]  # 클수록 좋게
        blend_pick = g2.loc[g2["blend_score"].idxmax(), "Ticker"]

        return pd.Series(
            {
                "pick_ret_only": ret_pick,
                "pick_p_only": p_pick,
                "pick_gate": gate_pick,
                "pick_blend": blend_pick,
            }
        )

    picks = feat.groupby("Date", sort=False).apply(pick_for_day).reset_index()

    # 평가(라벨과 조인)
    lab_eval = lab[["Date", "Ticker", "CycleReturn", "MinCycleRet", "ExtendDays"]].copy()

    def attach(method_col: str, name: str) -> pd.DataFrame:
        tmp = picks[["Date", method_col]].rename(columns={method_col: "Ticker"})
        out = tmp.merge(lab_eval, on=["Date", "Ticker"], how="left")
        out["method"] = name
        return out

    eval_df = pd.concat(
        [
            attach("pick_ret_only", "ret_only"),
            attach("pick_p_only", "p_only"),
            attach("pick_gate", "gate_topk"),
            attach("pick_blend", "blend_rank"),
        ],
        ignore_index=True,
    )

    # 요약
    summary = (
        eval_df.dropna(subset=["CycleReturn"])
        .groupby("method")
        .agg(
            n=("CycleReturn", "count"),
            win_rate=("CycleReturn", lambda x: float((x > 0).mean())),
            avg_ret=("CycleReturn", "mean"),
            worst_min_ret=("MinCycleRet", "min"),
            avg_extend_days=("ExtendDays", "mean"),
        )
        .reset_index()
        .sort_values("avg_ret", ascending=False)
    )

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    picks_path = SIGNAL_DIR / f"picks_{tag}.csv"
    eval_path = SIGNAL_DIR / f"eval_{tag}.csv"
    summary_path = SIGNAL_DIR / f"summary_{tag}.csv"

    picks.to_csv(picks_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Saved:", picks_path)
    print("Saved:", eval_path)
    print("Saved:", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
