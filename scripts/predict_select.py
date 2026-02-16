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

FEATURES_MODEL_PARQUET = FEATURE_DIR / "features_model.parquet"
FEATURES_MODEL_CSV = FEATURE_DIR / "features_model.csv"
FEATURES_PARQUET = FEATURE_DIR / "features.parquet"
FEATURES_CSV = FEATURE_DIR / "features.csv"


def fmt_tag(pt: float, max_days: int, sl: float, max_ext: int) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{max_days}_sl{sl_tag}_ex{max_ext}"


def load_features() -> pd.DataFrame:
    if FEATURES_MODEL_PARQUET.exists():
        f = pd.read_parquet(FEATURES_MODEL_PARQUET)
    elif FEATURES_MODEL_CSV.exists():
        f = pd.read_csv(FEATURES_MODEL_CSV)
    else:
        if FEATURES_PARQUET.exists():
            f = pd.read_parquet(FEATURES_PARQUET)
        else:
            f = pd.read_csv(FEATURES_CSV)

    f["Date"] = pd.to_datetime(f["Date"])
    f["Ticker"] = f["Ticker"].astype(str).str.upper().str.strip()
    return f.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def load_labels(tag: str) -> pd.DataFrame:
    p = LABEL_DIR / f"strategy_labels_{tag}.parquet"
    c = LABEL_DIR / f"strategy_labels_{tag}.csv"
    if p.exists():
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(c)
    df["Date"] = pd.to_datetime(df["Date"])
    df["ExitDate"] = pd.to_datetime(df["ExitDate"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def get_market_row(day_df: pd.DataFrame) -> dict:
    """
    같은 Date의 모든 티커가 동일한 market 피처를 갖는 구조라면
    아무 행 1개에서 market 정보를 뽑아도 됨.
    """
    cols = [
        "VIX",
        "Market_Drawdown_60",
        "Market_Drawdown_252",
        "Market_realized_vol_20",
        "Market_ret_1d",
    ]
    row = {}
    first = day_df.iloc[0]
    for c in cols:
        row[c] = float(first[c]) if c in day_df.columns and pd.notna(first[c]) else np.nan
    return row


def apply_killswitch(market: dict, args) -> tuple[bool, int, float, str]:
    """
    returns: (skip, topk_used, lambda_used, reason)
    """
    vix = market.get("VIX", np.nan)
    dd60 = market.get("Market_Drawdown_60", np.nan)

    topk_used = args.topk
    lam_used = args.lambda_risk
    reason = ""

    # 하드 스킵 조건
    if np.isfinite(vix) and vix >= args.hard_vix:
        return True, topk_used, lam_used, f"hard_skip_vix>={args.hard_vix} (vix={vix:.2f})"
    if np.isfinite(dd60) and dd60 <= -abs(args.hard_dd60):
        return True, topk_used, lam_used, f"hard_skip_dd60<={-abs(args.hard_dd60):.2f} (dd60={dd60:.3f})"

    # 소프트 리스크 모드
    if np.isfinite(vix) and vix >= args.soft_vix:
        topk_used = max(1, min(topk_used, args.soft_topk))
        lam_used = max(lam_used, args.soft_lambda)
        reason = f"soft_risk_vix>={args.soft_vix} (vix={vix:.2f})"
    elif np.isfinite(dd60) and dd60 <= -abs(args.soft_dd60):
        topk_used = max(1, min(topk_used, args.soft_topk))
        lam_used = max(lam_used, args.soft_lambda)
        reason = f"soft_risk_dd60<={-abs(args.soft_dd60):.2f} (dd60={dd60:.3f})"

    return False, topk_used, lam_used, reason


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--tail-threshold", type=float, default=-0.30)
    ap.add_argument("--lambda-risk", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=5)

    # killswitch params
    ap.add_argument("--hard-vix", type=float, default=40.0)
    ap.add_argument("--hard-dd60", type=float, default=0.18)  # abs value, dd60 <= -0.18이면 스킵
    ap.add_argument("--soft-vix", type=float, default=30.0)
    ap.add_argument("--soft-dd60", type=float, default=0.12)
    ap.add_argument("--soft-topk", type=int, default=2)
    ap.add_argument("--soft-lambda", type=float, default=0.10)

    args = ap.parse_args()

    tag = fmt_tag(args.profit_target, args.max_days, args.stop_level, args.max_extend_days)
    thr_tag = int(round(abs(args.tail_threshold) * 100))

    clf_pos = joblib.load(MODEL_DIR / f"clf_pos_{tag}.pkl")
    clf_tail = joblib.load(MODEL_DIR / f"clf_tail_{tag}_thr{thr_tag}.pkl")
    reg = joblib.load(MODEL_DIR / f"reg_ret_{tag}.pkl")
    scaler = joblib.load(MODEL_DIR / f"scaler_{tag}.pkl")

    feat = load_features()
    lab = load_labels(tag)

    feature_cols = [c for c in feat.columns if c not in ("Date", "Ticker")]
    feature_cols = feat[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = feat[feature_cols].to_numpy(dtype=float)
    Xs = scaler.transform(X)

    feat = feat.copy()
    feat["p_pos"] = clf_pos.predict_proba(Xs)[:, 1]
    feat["p_tail"] = clf_tail.predict_proba(Xs)[:, 1]
    feat["pred_ret"] = reg.predict(Xs)

    # 날짜별 picks
    pick_rows = []
    for date, g in feat.groupby("Date", sort=False):
        g = g.reset_index(drop=True)

        market = get_market_row(g)
        skip, topk_used, lam_used, reason = apply_killswitch(market, args)

        if skip:
            pick_rows.append(
                {
                    "Date": date,
                    "Skipped": 1,
                    "SkipReason": reason,
                    "topk_used": topk_used,
                    "lambda_used": lam_used,
                    "pick_ret_only": None,
                    "pick_p_only": None,
                    "pick_gate_ret": None,
                    "pick_utility": None,
                    "pick_gate_utility": None,
                    "pick_blend": None,
                    "VIX": market.get("VIX", np.nan),
                    "Market_Drawdown_60": market.get("Market_Drawdown_60", np.nan),
                }
            )
            continue

        # utility 계산(리스크 조절 반영)
        g = g.copy()
        g["utility"] = g["pred_ret"] - float(lam_used) * g["p_tail"]

        # ret-only / p-only
        pick_ret = g.loc[g["pred_ret"].idxmax(), "Ticker"]
        pick_p = g.loc[g["p_pos"].idxmax(), "Ticker"]

        # gate topk by p_pos
        gg = g.sort_values("p_pos", ascending=False).head(max(1, min(topk_used, len(g))))
        pick_gate_ret = gg.loc[gg["pred_ret"].idxmax(), "Ticker"]
        pick_gate_utility = gg.loc[gg["utility"].idxmax(), "Ticker"]

        # utility only
        pick_utility = g.loc[g["utility"].idxmax(), "Ticker"]

        # blend rank(p, pred_ret)
        g2 = g.copy()
        g2["rank_p"] = g2["p_pos"].rank(ascending=False, method="average")
        g2["rank_r"] = g2["pred_ret"].rank(ascending=False, method="average")
        g2["blend_score"] = -(0.5 * g2["rank_p"] + 0.5 * g2["rank_r"])
        pick_blend = g2.loc[g2["blend_score"].idxmax(), "Ticker"]

        pick_rows.append(
            {
                "Date": date,
                "Skipped": 0,
                "SkipReason": reason,
                "topk_used": topk_used,
                "lambda_used": lam_used,
                "pick_ret_only": pick_ret,
                "pick_p_only": pick_p,
                "pick_gate_ret": pick_gate_ret,
                "pick_utility": pick_utility,
                "pick_gate_utility": pick_gate_utility,
                "pick_blend": pick_blend,
                "VIX": market.get("VIX", np.nan),
                "Market_Drawdown_60": market.get("Market_Drawdown_60", np.nan),
            }
        )

    picks = pd.DataFrame(pick_rows).sort_values("Date").reset_index(drop=True)

    # 평가
    lab_eval = lab[["Date", "Ticker", "CycleReturn", "MinCycleRet", "ExtendDays", "ForcedExitFlag"]].copy()
    lab_eval["TailFlag"] = (lab_eval["MinCycleRet"].astype(float) <= float(args.tail_threshold)).astype(int)

    def attach(method_col: str, name: str) -> pd.DataFrame:
        tmp = picks[picks["Skipped"] == 0][["Date", method_col]].rename(columns={method_col: "Ticker"})
        out = tmp.merge(lab_eval, on=["Date", "Ticker"], how="left")
        out["method"] = name
        return out

    eval_df = pd.concat(
        [
            attach("pick_ret_only", "ret_only"),
            attach("pick_p_only", "p_only"),
            attach("pick_gate_ret", "gate_topk_then_ret"),
            attach("pick_utility", "utility"),
            attach("pick_gate_utility", "gate_topk_then_utility"),
            attach("pick_blend", "blend_rank"),
        ],
        ignore_index=True,
    )

    summary = (
        eval_df.dropna(subset=["CycleReturn"])
        .groupby("method")
        .agg(
            n=("CycleReturn", "count"),
            win_rate=("CycleReturn", lambda x: float((x > 0).mean())),
            avg_ret=("CycleReturn", "mean"),
            tail_rate=("TailFlag", "mean"),
            worst_min_ret=("MinCycleRet", "min"),
            avg_extend_days=("ExtendDays", "mean"),
            forced_exit_rate=("ForcedExitFlag", "mean"),
        )
        .reset_index()
        .sort_values("avg_ret", ascending=False)
    )

    # 스킵 통계
    skip_stats = picks.agg(
        total_days=("Date", "count"),
        skipped_days=("Skipped", "sum"),
    )
    skip_stats = {
        "total_days": int(skip_stats["total_days"]),
        "skipped_days": int(skip_stats["skipped_days"]),
        "skip_rate": float(skip_stats["skipped_days"] / max(1, skip_stats["total_days"])),
    }

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    picks_path = SIGNAL_DIR / f"picks_{tag}_ks.csv"
    eval_path = SIGNAL_DIR / f"eval_{tag}_ks.csv"
    summary_path = SIGNAL_DIR / f"summary_{tag}_ks.csv"
    skip_path = SIGNAL_DIR / f"skip_{tag}_ks.json"

    picks.to_csv(picks_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    summary.to_csv(summary_path, index=False)
    skip_path.write_text(pd.Series(skip_stats).to_json(), encoding="utf-8")

    print("Saved:", picks_path)
    print("Saved:", eval_path)
    print("Saved:", summary_path)
    print("Saved:", skip_path)
    print("\nSkip stats:", skip_stats)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
