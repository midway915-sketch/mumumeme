# scripts/predict_gate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib


# ===============================
# Paths
# ===============================
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
SIG_DIR = DATA_DIR / "signals"
APP_DIR = Path("app")

PRICES_PARQUET = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

DEFAULT_FEATURES_PARQUET = FEAT_DIR / "features_model.parquet"
DEFAULT_FEATURES_CSV = FEAT_DIR / "features_model.csv"

SUCCESS_MODEL = APP_DIR / "model.pkl"
SUCCESS_SCALER = APP_DIR / "scaler.pkl"

TAIL_MODEL = APP_DIR / "tail_model.pkl"
TAIL_SCALER = APP_DIR / "tail_scaler.pkl"


# ===============================
# Config
# ===============================
MARKET_TICKER = "SPY"
EXCLUDE_TICKERS = {MARKET_TICKER, "^VIX"}  # market/vix는 매수 후보에서 제외

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


# ===============================
# Utils
# ===============================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_tag(pt: float, max_days: int, sl: float, ex: int) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{max_days}_sl{sl_tag}_ex{ex}"


def read_prices() -> pd.DataFrame:
    if PRICES_PARQUET.exists():
        px = pd.read_parquet(PRICES_PARQUET)
    elif PRICES_CSV.exists():
        px = pd.read_csv(PRICES_CSV)
    else:
        raise FileNotFoundError(
            "Missing prices: data/raw/prices.parquet or data/raw/prices.csv (run fetch_prices.py first)"
        )

    px = px.copy()
    px["Date"] = pd.to_datetime(px["Date"])
    px["Ticker"] = px["Ticker"].astype(str).str.upper().str.strip()
    px = (
        px.sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )
    return px


def auto_load_features(features_path: str | None) -> tuple[pd.DataFrame, str]:
    """
    1) --features-path 지정 시 그 파일 사용
    2) data/features/features_model.* 있으면 사용
    3) 없으면 data/features 내 Date/Ticker 있는 파일 자동탐색
    """
    if features_path:
        fp = Path(features_path)
        if not fp.exists():
            raise FileNotFoundError(f"features-path not found: {fp}")
        df = pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
        return df, str(fp)

    if DEFAULT_FEATURES_PARQUET.exists():
        return pd.read_parquet(DEFAULT_FEATURES_PARQUET), str(DEFAULT_FEATURES_PARQUET)
    if DEFAULT_FEATURES_CSV.exists():
        return pd.read_csv(DEFAULT_FEATURES_CSV), str(DEFAULT_FEATURES_CSV)

    if not FEAT_DIR.exists():
        raise FileNotFoundError(f"features dir not found: {FEAT_DIR}")

    candidates = []
    candidates += list(FEAT_DIR.glob("*.parquet"))
    candidates += list(FEAT_DIR.glob("*.csv"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No feature files found in {FEAT_DIR}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    last_err = None
    for p in candidates:
        try:
            df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
            if {"Date", "Ticker"}.issubset(df.columns):
                return df, str(p)
        except Exception as e:
            last_err = e

    raise FileNotFoundError(
        f"Could not find a usable features file in {FEAT_DIR} (need Date/Ticker). Last error: {last_err}"
    )


def compute_market_frame_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Market_Drawdown: SPY 기준 rolling high 대비 drawdown
    Market_ATR_ratio: SPY ATR(14)/Close
    - 히스토리가 짧아도 전부 NaN이 되지 않게 min_periods 완화
    """
    m = prices[prices["Ticker"] == MARKET_TICKER].sort_values("Date").copy()
    if m.empty:
        raise RuntimeError(f"Market ticker {MARKET_TICKER} not found. Run fetch_prices.py --include-extra")

    for c in ["High", "Low", "Close"]:
        if c not in m.columns:
            raise ValueError(f"prices missing {c} for {MARKET_TICKER}")

    close = m["Close"].astype(float)
    high = m["High"].astype(float)
    low = m["Low"].astype(float)

    # drawdown
    # 252를 쓰되, 데이터가 짧으면 최소 20일 정도부터라도 값이 생기게
    dd_minp = min(len(close), 252)
    dd_minp = max(20, min(252, dd_minp))
    roll_max = close.rolling(252, min_periods=dd_minp).max()
    market_dd = (close / roll_max) - 1.0

    # atr ratio
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_minp = max(5, min(14, len(close)))
    atr14 = tr.rolling(14, min_periods=atr_minp).mean()
    market_atr_ratio = atr14 / close

    return pd.DataFrame(
        {
            "Date": m["Date"].values,
            "Market_Drawdown": market_dd.values,
            "Market_ATR_ratio": market_atr_ratio.values,
        }
    )


def ensure_market_features(feats: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    feats에 Market_* 없으면 SPY로 계산해서 Date 기준으로 붙임.
    merge 충돌 방지: suffixes=('', '_m') 후 coalesce 처리.
    """
    need = [c for c in ["Market_Drawdown", "Market_ATR_ratio"] if c not in feats.columns]
    if not need:
        return feats

    market = compute_market_frame_from_prices(prices)
    merged = feats.merge(market, on="Date", how="left", suffixes=("", "_m"))

    for col in ["Market_Drawdown", "Market_ATR_ratio"]:
        col_m = f"{col}_m"
        if col in merged.columns and col_m in merged.columns:
            merged[col] = merged[col].combine_first(merged[col_m])
            merged.drop(columns=[col_m], inplace=True)
        elif col not in merged.columns and col_m in merged.columns:
            merged.rename(columns={col_m: col}, inplace=True)

    return merged


def load_model_and_scaler(model_path: Path, scaler_path: Path):
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Missing model or scaler: {model_path} / {scaler_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_proba_df(df: pd.DataFrame, model, scaler, feature_cols: list[str]) -> np.ndarray:
    X = df[feature_cols].to_numpy(dtype=np.float64)
    Xs = scaler.transform(X)
    p = model.predict_proba(Xs)[:, 1]
    return p.astype(np.float64)


def safe_rank_column(df: pd.DataFrame, rank_by: str) -> str:
    # rank_by가 우리 계산 컬럼이 아닐 수도 있어서 방어
    if rank_by in df.columns:
        return rank_by
    # 흔한 대체
    for alt in ["utility", "p_success", "ret_score"]:
        if alt in df.columns:
            return alt
    return "p_success"


# ===============================
# Main
# ===============================
def main() -> None:
    ap = argparse.ArgumentParser(description="Gate + Rank picker (writes daily Top-1 picks).")
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--gate-mode", type=str, default="none",
                    choices=["none", "tail", "utility", "tail_utility"])
    ap.add_argument("--tail-max", type=float, default=0.30,
                    help="Pass if p_tail <= tail_max (lower tail risk)")
    ap.add_argument("--u-quantile", type=float, default=0.75,
                    help="Utility gate threshold by daily quantile (pass if utility >= q)")
    ap.add_argument("--rank-by", type=str, default="utility",
                    help="Ranking column: utility|p_success|ret_score|<existing_column>")
    ap.add_argument("--lambda-tail", type=float, default=0.05,
                    help="utility = p_success - lambda_tail * p_tail (if tail model exists)")

    ap.add_argument("--features-path", type=str, default=None)
    ap.add_argument("--out-suffix", type=str, default="run")

    ap.add_argument("--out-dir", type=str, default=str(SIG_DIR))
    ap.add_argument("--write-scores", action="store_true",
                    help="Also write per-date per-ticker score table for debugging.")
    args = ap.parse_args()

    pt = float(args.profit_target)
    h = int(args.max_days)
    sl = float(args.stop_level)
    ex = int(args.max_extend_days)
    tag = fmt_tag(pt, h, sl, ex)

    gate_mode = args.gate_mode
    tail_max = float(args.tail_max)
    u_q = float(args.u_quantile)
    lambda_tail = float(args.lambda_tail)
    out_suffix = str(args.out_suffix)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load features & prices
    prices = read_prices()
    feats, feats_src = auto_load_features(args.features_path)

    feats = feats.copy()
    feats["Date"] = pd.to_datetime(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = (
        feats.sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    # --- ensure Market_* columns exist (SPY derived)
    feats = ensure_market_features(feats, prices)

    # --- required columns check (but allow NaN rows to be dropped)
    missing_cols = [c for c in ["Date", "Ticker"] + FEATURE_COLS if c not in feats.columns]
    if missing_cols:
        raise ValueError(f"features_model missing required columns: {missing_cols} (src={feats_src})")

    # --- drop excluded tickers
    feats = feats[~feats["Ticker"].isin(EXCLUDE_TICKERS)].reset_index(drop=True)

    # --- drop rows with NaNs in feature cols
    feats = feats.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if feats.empty:
        raise RuntimeError("After dropping NaNs, no rows left in features. Check build_features output / prices history.")

    # --- load success model
    success_model, success_scaler = load_model_and_scaler(SUCCESS_MODEL, SUCCESS_SCALER)
    feats["p_success"] = predict_proba_df(feats, success_model, success_scaler, FEATURE_COLS)

    # --- load tail model if exists (optional for baseline/utility)
    tail_available = TAIL_MODEL.exists() and TAIL_SCALER.exists()
    if tail_available:
        tail_model, tail_scaler = load_model_and_scaler(TAIL_MODEL, TAIL_SCALER)
        feats["p_tail"] = predict_proba_df(feats, tail_model, tail_scaler, FEATURE_COLS)
    else:
        feats["p_tail"] = 0.0

    # mode tail/tail_utility는 tail model 필요
    if gate_mode in ("tail", "tail_utility") and not tail_available:
        raise RuntimeError(f"gate-mode={gate_mode} requires tail_model.pkl & tail_scaler.pkl, but they are missing.")

    # --- utility
    if tail_available:
        feats["utility"] = feats["p_success"] - (lambda_tail * feats["p_tail"])
    else:
        feats["utility"] = feats["p_success"]

    # optional ret_score fallback (없으면 p_success로)
    if "ret_score" not in feats.columns:
        feats["ret_score"] = feats["p_success"]

    # --- pick per date
    picks = []
    score_rows = []  # optional debugging table

    rank_col = safe_rank_column(feats, args.rank_by)

    for d, day in feats.groupby("Date", sort=True):
        day = day.copy()

        # daily utility threshold
        # (pass if utility >= quantile)
        if len(day) > 0:
            u_cut = float(day["utility"].quantile(u_q))
        else:
            u_cut = float("nan")

        # gate flags
        pass_tail = (day["p_tail"] <= tail_max)
        pass_utility = (day["utility"] >= u_cut)

        if gate_mode == "none":
            passed = np.ones(len(day), dtype=bool)
        elif gate_mode == "tail":
            passed = pass_tail.to_numpy()
        elif gate_mode == "utility":
            passed = pass_utility.to_numpy()
        else:  # tail_utility
            passed = (pass_tail & pass_utility).to_numpy()

        day["Passed"] = passed.astype(int)
        day["u_cut"] = u_cut
        day["tail_max"] = tail_max
        day["gate_mode"] = gate_mode
        day["rank_col"] = rank_col

        cand = day[day["Passed"] == 1].copy()
        skipped = 0

        if cand.empty:
            # gate 통과자가 없으면 skip
            pick_ticker = ""
            skipped = 1
            chosen_score = np.nan
        else:
            cand = cand.sort_values(rank_col, ascending=False)
            pick_ticker = str(cand.iloc[0]["Ticker"])
            chosen_score = float(cand.iloc[0][rank_col])

        picks.append(
            {
                "Date": d,
                "pick_custom": pick_ticker,
                "Skipped": skipped,
                "gate_mode": gate_mode,
                "tail_max": tail_max,
                "u_quantile": u_q,
                "u_cut": u_cut,
                "rank_by": args.rank_by,
                "rank_col_used": rank_col,
                "lambda_tail": lambda_tail,
                "tail_model_available": int(tail_available),
                "chosen_score": chosen_score,
            }
        )

        if args.write_scores:
            score_rows.append(
                day[
                    [
                        "Date", "Ticker", "Passed",
                        "p_success", "p_tail", "utility", "ret_score",
                        "u_cut", "tail_max"
                    ]
                ]
            )

    picks_df = pd.DataFrame(picks).sort_values("Date").reset_index(drop=True)

    out_path = out_dir / f"picks_{tag}_gate_{out_suffix}.csv"
    picks_df.to_csv(out_path, index=False)

    meta = {
        "updated_at_utc": now_utc_iso(),
        "tag": tag,
        "out_suffix": out_suffix,
        "gate_mode": gate_mode,
        "tail_max": tail_max,
        "u_quantile": u_q,
        "rank_by": args.rank_by,
        "rank_col_used": rank_col,
        "lambda_tail": lambda_tail,
        "features_source": feats_src,
        "rows_features": int(len(feats)),
        "rows_picks": int(len(picks_df)),
        "tail_model_available": bool(tail_available),
        "price_source": str(PRICES_PARQUET if PRICES_PARQUET.exists() else PRICES_CSV),
    }
    (out_dir / f"picks_{tag}_gate_{out_suffix}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"[DONE] wrote picks -> {out_path}")
    print(json.dumps(meta, ensure_ascii=False))

    if args.write_scores and score_rows:
        scores_df = pd.concat(score_rows, ignore_index=True)
        scores_path = out_dir / f"scores_{tag}_gate_{out_suffix}.csv"
        scores_df.to_csv(scores_path, index=False)
        print(f"[DONE] wrote scores -> {scores_path}")


if __name__ == "__main__":
    main()
