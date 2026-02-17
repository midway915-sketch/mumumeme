# scripts/predict_gate.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


DATA_DIR = Path("data")
FEATURE_DIR = DATA_DIR / "features"
SIGNALS_DIR = DATA_DIR / "signals"
APP_DIR = Path("app")

FEATS_PARQ = FEATURE_DIR / "features_model.parquet"
FEATS_CSV = FEATURE_DIR / "features_model.csv"

SUCCESS_MODEL = APP_DIR / "model.pkl"
SUCCESS_SCALER = APP_DIR / "scaler.pkl"

TAIL_MODEL = APP_DIR / "tail_model.pkl"
TAIL_SCALER = APP_DIR / "tail_scaler.pkl"

TAU_PRED_PARQ = SIGNALS_DIR / "tau_pred.parquet"
TAU_PRED_CSV = SIGNALS_DIR / "tau_pred.csv"
TAU_MODEL = APP_DIR / "tau_model.pkl"
TAU_SCALER = APP_DIR / "tau_scaler.pkl"

# 기본 feature 셋(없으면 더미로라도 맞추게)
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

DEFAULT_TAU_GAMMA = 0.05  # utility_time 페널티 강도


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def make_tag(profit_target: float, max_days: int, stop_level: float, max_extend_days: int) -> str:
    pt = int(round(profit_target * 100))
    sl = int(round(abs(stop_level) * 100))
    return f"pt{pt}_h{int(max_days)}_sl{sl}_ex{int(max_extend_days)}"


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"features_model missing required FEATURE_COLS: {missing}")


def predict_proba_if_possible(df: pd.DataFrame, model_path: Path, scaler_path: Path, out_col: str) -> pd.Series:
    if not model_path.exists() or not scaler_path.exists():
        # 모델이 없으면 NaN으로 반환
        return pd.Series([np.nan] * len(df), index=df.index)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X = df[FEATURE_COLS].astype(float)
    Xs = scaler.transform(X)

    # sklearn CalibratedClassifierCV / LogisticRegression 등 predict_proba 지원 가정
    p = model.predict_proba(Xs)[:, 1]
    return pd.Series(p, index=df.index, name=out_col)


def attach_tau_hat(df: pd.DataFrame) -> pd.DataFrame:
    """
    우선순위:
      1) data/signals/tau_pred.(parquet/csv) 가 있으면 merge
      2) 없으면 app/tau_model.pkl + scaler로 여기서 예측
      3) 둘 다 없으면 tau_hat NaN
    """
    out = df.copy()

    if TAU_PRED_PARQ.exists() or TAU_PRED_CSV.exists():
        tau = read_table(TAU_PRED_PARQ, TAU_PRED_CSV)
        tau["Date"] = pd.to_datetime(tau["Date"])
        tau["Ticker"] = tau["Ticker"].astype(str).str.upper().str.strip()
        tau = tau[["Date", "Ticker", "tau_hat"]].drop_duplicates(["Date", "Ticker"], keep="last")
        out = out.merge(tau, on=["Date", "Ticker"], how="left")
        return out

    if TAU_MODEL.exists() and TAU_SCALER.exists():
        model = joblib.load(TAU_MODEL)
        scaler = joblib.load(TAU_SCALER)
        X = out[FEATURE_COLS].astype(float)
        Xs = scaler.transform(X)
        tau_hat = model.predict(Xs)
        out["tau_hat"] = pd.Series(tau_hat, index=out.index).astype(float)
        return out

    out["tau_hat"] = np.nan
    return out


def daily_pick_top1(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    df: candidate rows (already filtered by gate), columns Date, Ticker, score_col
    returns: Date -> top1 ticker (score desc), ties by score then ticker
    """
    df2 = df.copy()
    df2["_score"] = pd.to_numeric(df2[score_col], errors="coerce")
    df2 = df2.dropna(subset=["_score"])
    if df2.empty:
        return pd.DataFrame(columns=["Date", "pick_custom"])

    df2 = df2.sort_values(["Date", "_score", "Ticker"], ascending=[True, False, True])
    top = df2.groupby("Date", sort=False).head(1)
    out = top[["Date", "Ticker"]].rename(columns={"Ticker": "pick_custom"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--gate-mode", type=str, required=True,
                    choices=["none", "tail", "utility", "tail_utility"])
    ap.add_argument("--tail-max", type=float, default=0.20)
    ap.add_argument("--u-quantile", type=float, default=0.75)

    ap.add_argument("--rank-by", type=str, required=True,
                    choices=["utility", "ret_score", "p_success", "utility_time"])
    ap.add_argument("--lambda-tail", type=float, default=0.05)

    ap.add_argument("--out-suffix", type=str, required=True)
    ap.add_argument("--tag", type=str, default=None)

    # utility_time 페널티 강도(워크플로 input 유지해야 하니 기본값만 사용해도 됨)
    ap.add_argument("--tau-gamma", type=float, default=DEFAULT_TAU_GAMMA)

    args = ap.parse_args()

    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    tag = args.tag or make_tag(args.profit_target, args.max_days, args.stop_level, args.max_extend_days)

    feats = read_table(FEATS_PARQ, FEATS_CSV)
    feats["Date"] = pd.to_datetime(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()

    # 필수 feature 확보 (없으면 즉시 에러 -> 너가 원한 것처럼 조용히 틀어지지 않게)
    ensure_cols(feats, FEATURE_COLS)

    use = feats.copy()

    # === p_success / p_tail 예측 ===
    use["p_success"] = predict_proba_if_possible(use, SUCCESS_MODEL, SUCCESS_SCALER, "p_success")

    # tail gate를 쓰는데 tail model이 없으면 바로 에러 (silent fail 방지)
    if args.gate_mode in ("tail", "tail_utility"):
        if not (TAIL_MODEL.exists() and TAIL_SCALER.exists()):
            raise FileNotFoundError(
                "gate-mode requires tail model: missing app/tail_model.pkl or app/tail_scaler.pkl"
            )

    use["p_tail"] = predict_proba_if_possible(use, TAIL_MODEL, TAIL_SCALER, "p_tail")

    # === tau_hat attach (utility_time용) ===
    use = attach_tau_hat(use)

    # === 기본 점수 구성 ===
    # ret_score 컬럼이 있으면 사용, 없으면 p_success로 fallback
    if "ret_score" not in use.columns:
        use["ret_score"] = use["p_success"]

    # utility 컬럼이 있으면 사용, 없으면 (p_success - lambda * p_tail)로 구성
    if "utility" not in use.columns:
        ptail = pd.to_numeric(use["p_tail"], errors="coerce").fillna(0.0)
        psucc = pd.to_numeric(use["p_success"], errors="coerce").fillna(0.0)
        use["utility"] = psucc - float(args.lambda_tail) * ptail

    # utility_time: utility - gamma*log1p(tau_hat)
    tau = pd.to_numeric(use["tau_hat"], errors="coerce")
    tau_fill = tau.fillna(tau.median() if np.isfinite(tau.median()) else float(args.max_days))
    use["utility_time"] = pd.to_numeric(use["utility"], errors="coerce").fillna(0.0) - float(args.tau_gamma) * np.log1p(
        tau_fill.clip(lower=1.0)
    )

    # === Gate 필터 ===
    candidates = use.copy()

    # tail gate
    if args.gate_mode in ("tail", "tail_utility"):
        candidates = candidates[pd.to_numeric(candidates["p_tail"], errors="coerce") <= float(args.tail_max)]

    # utility gate (daily quantile)
    if args.gate_mode in ("utility", "tail_utility"):
        base = pd.to_numeric(candidates["utility"], errors="coerce")
        candidates = candidates.assign(_u=base)
        # 날짜별 분위수 컷
        q = float(args.u_quantile)
        u_cut = candidates.groupby("Date")["_u"].transform(lambda s: s.quantile(q) if len(s) else np.nan)
        candidates = candidates[candidates["_u"] >= u_cut].drop(columns=["_u"], errors="ignore")

    # rank score 선택
    score_col = args.rank_by

    # === 날짜별 top1 pick ===
    picks = daily_pick_top1(candidates[["Date", "Ticker", score_col]].copy(), score_col=score_col)

    # 모든 날짜에 대해 “스킵 여부”를 포함한 pick 테이블 만들기
    all_dates = pd.DataFrame({"Date": sorted(use["Date"].dropna().unique())})
    out = all_dates.merge(picks, on="Date", how="left")
    out["Skipped"] = out["pick_custom"].isna().astype(int)

    # 메타 컬럼
    out["tag"] = tag
    out["gate_mode"] = args.gate_mode
    out["rank_by"] = args.rank_by
    out["tail_max"] = float(args.tail_max)
    out["u_quantile"] = float(args.u_quantile)
    out["lambda_tail"] = float(args.lambda_tail)
    out["profit_target"] = float(args.profit_target)
    out["max_days"] = int(args.max_days)
    out["stop_level"] = float(args.stop_level)
    out["max_extend_days"] = int(args.max_extend_days)
    out["out_suffix"] = args.out_suffix

    # 저장
    out_path = SIGNALS_DIR / f"picks_{tag}_gate_{args.out_suffix}.csv"
    out.to_csv(out_path, index=False)

    # 디버그(짧게)
    skip_rate = float(out["Skipped"].mean()) if len(out) else 1.0
    print("=" * 60)
    print(f"[DONE] wrote {out_path}")
    print("rows:", len(out), "skip_rate:", round(skip_rate, 4))
    print("example picks:\n", out.dropna(subset=["pick_custom"]).head(5).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()