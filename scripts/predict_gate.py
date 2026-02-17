# scripts/predict_gate.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
SIGNALS_DIR = DATA_DIR / "signals"
APP_DIR = Path("app")

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

FEATURES_MODEL_PARQ = FEATURES_DIR / "features_model.parquet"
FEATURES_MODEL_CSV = FEATURES_DIR / "features_model.csv"

MODEL_PKL = APP_DIR / "model.pkl"
SCALER_PKL = APP_DIR / "scaler.pkl"

TAIL_MODEL_PKL = APP_DIR / "tail_model.pkl"
TAIL_SCALER_PKL = APP_DIR / "tail_scaler.pkl"

TAU_MODEL_PKL = APP_DIR / "tau_model.pkl"
TAU_SCALER_PKL = APP_DIR / "tau_scaler.pkl"


# 기본 feature 컬럼(없으면 자동 0 채움)
BASE_FEATURE_COLS = [
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


def read_prices() -> pd.DataFrame:
    df = read_table(PRICES_PARQ, PRICES_CSV).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    need = {"Date", "Ticker", "High", "Low", "Close"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"prices missing columns: {miss}")
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def ensure_market_features(feats: pd.DataFrame, prices: pd.DataFrame, market_ticker: str = "SPY") -> pd.DataFrame:
    """
    Ensure Market_Drawdown and Market_ATR_ratio exist by computing from SPY daily bars,
    then merging by Date.
    """
    out = feats.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()

    # 이미 둘 다 있으면 그대로
    if "Market_Drawdown" in out.columns and "Market_ATR_ratio" in out.columns:
        return out

    m = prices[prices["Ticker"] == market_ticker].sort_values("Date").copy()
    if m.empty:
        raise ValueError(f"Market ticker {market_ticker} not found. Run fetch_prices.py --include-extra")

    close = pd.to_numeric(m["Close"], errors="coerce")
    high = pd.to_numeric(m["High"], errors="coerce")
    low = pd.to_numeric(m["Low"], errors="coerce")

    # Market_Drawdown: close / rolling_max_252 - 1
    roll_max_252 = close.rolling(252, min_periods=20).max()
    m["Market_Drawdown"] = (close / roll_max_252) - 1.0

    # Market_ATR_ratio: ATR14 / close
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=10).mean()
    m["Market_ATR_ratio"] = atr14 / close

    m_feat = m[["Date", "Market_Drawdown", "Market_ATR_ratio"]].copy()
    out = out.merge(m_feat, on="Date", how="left", suffixes=("", "_mkt"))

    # combine_first로 채우기
    if "Market_Drawdown" in out.columns and "Market_Drawdown_mkt" in out.columns:
        out["Market_Drawdown"] = pd.to_numeric(out["Market_Drawdown"], errors="coerce").combine_first(
            pd.to_numeric(out["Market_Drawdown_mkt"], errors="coerce")
        )
        out.drop(columns=["Market_Drawdown_mkt"], inplace=True, errors="ignore")
    elif "Market_Drawdown" not in out.columns:
        out["Market_Drawdown"] = pd.to_numeric(out["Market_Drawdown_mkt"], errors="coerce")
        out.drop(columns=["Market_Drawdown_mkt"], inplace=True, errors="ignore")

    if "Market_ATR_ratio" in out.columns and "Market_ATR_ratio_mkt" in out.columns:
        out["Market_ATR_ratio"] = pd.to_numeric(out["Market_ATR_ratio"], errors="coerce").combine_first(
            pd.to_numeric(out["Market_ATR_ratio_mkt"], errors="coerce")
        )
        out.drop(columns=["Market_ATR_ratio_mkt"], inplace=True, errors="ignore")
    elif "Market_ATR_ratio" not in out.columns:
        out["Market_ATR_ratio"] = pd.to_numeric(out["Market_ATR_ratio_mkt"], errors="coerce")
        out.drop(columns=["Market_ATR_ratio_mkt"], inplace=True, errors="ignore")

    return out


def safe_load_model(model_path: Path, scaler_path: Path):
    if model_path.exists() and scaler_path.exists():
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None


def build_tag(profit_target: float, max_days: int, stop_level: float, max_extend_days: int) -> str:
    pt = int(round(profit_target * 100))
    sl = int(round(abs(stop_level) * 100))
    return f"pt{pt}_h{max_days}_sl{sl}_ex{max_extend_days}"


def prepare_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def per_date_quantile_threshold(x: pd.Series, q: float) -> float:
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return np.nan
    return float(x.quantile(q))


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate picks by p_tail/utility then rank; outputs daily picks CSV.")
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--mode", type=str, default="none", choices=["none", "tail", "utility", "tail_utility"])
    ap.add_argument("--tail-threshold", dest="tail_max", type=float, default=0.20)
    ap.add_argument("--utility-quantile", dest="u_quantile", type=float, default=0.75)
    ap.add_argument("--rank-by", type=str, default="utility", choices=["utility", "ret_score", "p_success", "utility_time"])
    ap.add_argument("--lambda-tail", type=float, default=0.05)
    ap.add_argument("--tau-gamma", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--market-ticker", type=str, default="SPY")
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--out-csv", type=str, default="")

    # run_grid_workflow.sh가 추가 인자를 줘도 안 깨지게
    args, _unknown = ap.parse_known_args()

    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    tag = build_tag(args.profit_target, args.max_days, args.stop_level, args.max_extend_days)
    suffix = args.suffix.strip()
    if not suffix:
        # 기본 suffix 생성(로그/파일명 일관)
        # 예: tail_utility_t0p30_q0p75_rutility_time_g0p05
        def tok(x: float) -> str:
            s = f"{x:.4f}".rstrip("0").rstrip(".")
            return s.replace(".", "p").replace("-", "m")
        suffix = f"{args.mode}_t{tok(args.tail_max)}_q{tok(args.u_quantile)}_r{args.rank_by}_g{tok(args.tau_gamma)}"

    print("=" * 30)
    print(f"[RUN] mode={args.mode} tail_max={args.tail_max} u_q={args.u_quantile} rank_by={args.rank_by} gamma={args.tau_gamma} suffix={suffix}")
    print("=" * 30)

    feats = read_table(FEATURES_MODEL_PARQ, FEATURES_MODEL_CSV).copy()
    feats["Date"] = pd.to_datetime(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()

    prices = read_prices()
    feats = ensure_market_features(feats, prices, market_ticker=args.market_ticker)

    # 모델 로드(없으면 NaN으로 진행)
    success_model, success_scaler = safe_load_model(MODEL_PKL, SCALER_PKL)
    tail_model, tail_scaler = safe_load_model(TAIL_MODEL_PKL, TAIL_SCALER_PKL)
    tau_model, tau_scaler = safe_load_model(TAU_MODEL_PKL, TAU_SCALER_PKL)

    # 모델 입력 feature 준비
    feats = prepare_features(feats, BASE_FEATURE_COLS)
    X = feats[BASE_FEATURE_COLS].to_numpy(dtype=float)

    # p_success
    if success_model is not None and success_scaler is not None:
        Xs = success_scaler.transform(X)
        p_success = success_model.predict_proba(Xs)[:, 1]
    else:
        p_success = np.full(len(feats), np.nan)

    # p_tail
    if tail_model is not None and tail_scaler is not None:
        Xt = tail_scaler.transform(X)
        p_tail = tail_model.predict_proba(Xt)[:, 1]
    else:
        # tail 모델이 없으면 보수적으로 NaN (tail gate 쓰면 자동 skip 많이 날 수 있음)
        p_tail = np.full(len(feats), np.nan)

    # tau_pred(성공까지 걸릴 기간 예측) — 없으면 NaN
    if tau_model is not None and tau_scaler is not None:
        Xtau = tau_scaler.transform(X)
        # 회귀모델/분류모델 모두 대응: predict 있으면 그걸로
        try:
            tau_pred = tau_model.predict(Xtau).astype(float)
        except Exception:
            tau_pred = np.full(len(feats), np.nan)
    else:
        tau_pred = np.full(len(feats), np.nan)

    # ret_score: 없으면 간단히 만들기(변동성/추세/낙폭 조합)
    if "ret_score" in feats.columns:
        ret_score = pd.to_numeric(feats["ret_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        # Drawdown_60 낮을수록(+), MA20_slope 높을수록(+), Z_score 너무 과열이면(-)
        ret_score = (
            (-feats["Drawdown_60"].to_numpy(dtype=float))
            + (feats["MA20_slope"].to_numpy(dtype=float) * 5.0)
            - (np.abs(feats["Z_score"].to_numpy(dtype=float)) * 0.2)
        )

    # utility(기본): p_success - lambda*p_tail
    # utility_time: utility - gamma*(tau_pred/max_days)
    util = p_success.copy()
    if np.isfinite(args.lambda_tail) and args.lambda_tail != 0:
        util = util - args.lambda_tail * p_tail

    util_time = util.copy()
    if np.isfinite(args.tau_gamma) and args.tau_gamma != 0:
        # tau_pred를 horizon 스케일로 정규화(0~1 근사)
        tau_norm = np.where(np.isfinite(tau_pred), np.clip(tau_pred / max(1, args.max_days), 0, 10), np.nan)
        util_time = util_time - args.tau_gamma * tau_norm

    use = feats[["Date", "Ticker"]].copy()
    use["p_success"] = p_success
    use["p_tail"] = p_tail
    use["ret_score"] = ret_score
    use["utility"] = util
    use["utility_time"] = util_time
    use["Skipped"] = 0

    # gate 통과 여부
    # tail gate: p_tail <= tail_max (NaN이면 실패 처리)
    use["pass_tail"] = (pd.to_numeric(use["p_tail"], errors="coerce") <= float(args.tail_max)).astype(int)

    # utility gate: per-date quantile 이상
    # 기준 컬럼은 rank_by가 utility_time이면 utility_time, 아니면 utility
    gate_util_col = "utility_time" if args.rank_by == "utility_time" else "utility"
    use["_gate_util_thr"] = use.groupby("Date")[gate_util_col].transform(lambda s: per_date_quantile_threshold(s, float(args.u_quantile)))
    use["pass_utility"] = (pd.to_numeric(use[gate_util_col], errors="coerce") >= pd.to_numeric(use["_gate_util_thr"], errors="coerce")).astype(int)

    if args.mode == "none":
        use["pass_gate"] = 1
    elif args.mode == "tail":
        use["pass_gate"] = use["pass_tail"]
    elif args.mode == "utility":
        use["pass_gate"] = use["pass_utility"]
    else:  # tail_utility
        use["pass_gate"] = (use["pass_tail"] & use["pass_utility"]).astype(int)

    # rank score
    rank_col = args.rank_by
    use["_rank"] = pd.to_numeric(use[rank_col], errors="coerce")

    # 날짜별 TopK 선정
    out_rows = []
    for d, g in use.groupby("Date", sort=True):
        gg = g[g["pass_gate"] == 1].copy()
        if gg.empty:
            out_rows.append(
                {
                    "Date": d,
                    "Ticker": "",
                    "Skipped": 1,
                    "rank_by": rank_col,
                    "score": np.nan,
                    "mode": args.mode,
                    "tail_max": args.tail_max,
                    "u_quantile": args.u_quantile,
                    "tau_gamma": args.tau_gamma,
                    "lambda_tail": args.lambda_tail,
                    "label": suffix,
                    "TAG": tag,
                }
            )
            continue

        gg = gg.sort_values("_rank", ascending=False)
        pick = gg.iloc[0]
        out_rows.append(
            {
                "Date": d,
                "Ticker": pick["Ticker"],
                "Skipped": 0,
                "rank_by": rank_col,
                "score": float(pick["_rank"]) if np.isfinite(pick["_rank"]) else np.nan,
                "mode": args.mode,
                "tail_max": args.tail_max,
                "u_quantile": args.u_quantile,
                "tau_gamma": args.tau_gamma,
                "lambda_tail": args.lambda_tail,
                "p_success": float(pick["p_success"]) if np.isfinite(pick["p_success"]) else np.nan,
                "p_tail": float(pick["p_tail"]) if np.isfinite(pick["p_tail"]) else np.nan,
                "utility": float(pick["utility"]) if np.isfinite(pick["utility"]) else np.nan,
                "utility_time": float(pick["utility_time"]) if np.isfinite(pick["utility_time"]) else np.nan,
                "ret_score": float(pick["ret_score"]) if np.isfinite(pick["ret_score"]) else np.nan,
                "label": suffix,
                "TAG": tag,
            }
        )

    out = pd.DataFrame(out_rows).sort_values("Date").reset_index(drop=True)

    # 저장
    if args.out_csv.strip():
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = SIGNALS_DIR / f"picks_{tag}_{suffix}.csv"

    out.to_csv(out_path, index=False)
    print(f"[DONE] wrote picks: {out_path} rows={len(out)} skipped_days={int(out['Skipped'].sum())}")


if __name__ == "__main__":
    main()