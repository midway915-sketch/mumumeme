# scripts/predict_gate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib


DATA_DIR = Path("data")
FEAT_DIR = DATA_DIR / "features"
SIGNAL_DIR = DATA_DIR / "signals"

DEFAULT_FEATURES_MODEL_PARQUET = FEAT_DIR / "features_model.parquet"
DEFAULT_FEATURES_MODEL_CSV = FEAT_DIR / "features_model.csv"

DEFAULT_SUCCESS_MODEL = Path("app/model.pkl")
DEFAULT_SUCCESS_SCALER = Path("app/scaler.pkl")

# tail 모델 파일명은 네 레포에 맞게 바꿔도 됨 (없으면 gate_mode에 따라 에러/대체)
DEFAULT_TAIL_MODEL = Path("app/tail_model.pkl")
DEFAULT_TAIL_SCALER = Path("app/tail_scaler.pkl")


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


def fmt_tag(pt: float, max_days: int, sl: float, max_ext: int) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{max_days}_sl{sl_tag}_ex{max_ext}"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing table: {parq} (or {csv})")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, default=30)  # tag용

    ap.add_argument("--gate-mode", type=str, default="none",
                    choices=["none", "tail", "utility", "tail_utility"])
    ap.add_argument("--tail-max", type=float, default=0.25)
    ap.add_argument("--u-quantile", type=float, default=0.75)

    ap.add_argument("--rank-by", type=str, default="utility",
                    choices=["utility", "ret_score", "p_success"])
    ap.add_argument("--lambda-tail", type=float, default=0.05)

    ap.add_argument("--features-path", type=str, default=None,
                    help="기본: data/features/features_model.parquet (없으면 csv fallback)")
    ap.add_argument("--success-model", type=str, default=str(DEFAULT_SUCCESS_MODEL))
    ap.add_argument("--success-scaler", type=str, default=str(DEFAULT_SUCCESS_SCALER))
    ap.add_argument("--tail-model", type=str, default=str(DEFAULT_TAIL_MODEL))
    ap.add_argument("--tail-scaler", type=str, default=str(DEFAULT_TAIL_SCALER))

    ap.add_argument("--out-suffix", type=str, default="",
                    help="파일명 구분자. 예: none_t025_q075")
    args = ap.parse_args()

    pt = float(args.profit_target)
    max_days = int(args.max_days)
    sl = float(args.stop_level)
    tag = fmt_tag(pt, max_days, sl, int(args.max_extend_days))

    # load features_model
    if args.features_path:
        fp = Path(args.features_path)
        if not fp.exists():
            raise FileNotFoundError(f"features-path not found: {fp}")
        if fp.suffix.lower() == ".parquet":
            df = pd.read_parquet(fp)
        else:
            df = pd.read_csv(fp)
    else:
        df = read_table(DEFAULT_FEATURES_MODEL_PARQUET, DEFAULT_FEATURES_MODEL_CSV)

    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("features_model must contain Date, Ticker columns")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # basic cleaning: drop rows with missing feature cols
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"features_model missing required FEATURE_COLS: {missing}")

    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # load models
    success_model_path = Path(args.success_model)
    success_scaler_path = Path(args.success_scaler)
    if not success_model_path.exists() or not success_scaler_path.exists():
        raise FileNotFoundError("success model/scaler not found. Run train_model.py first.")

    success_model = joblib.load(success_model_path)
    success_scaler = joblib.load(success_scaler_path)

    tail_model_path = Path(args.tail_model)
    tail_scaler_path = Path(args.tail_scaler)

    tail_available = tail_model_path.exists() and tail_scaler_path.exists()
    if (args.gate_mode in ["tail", "tail_utility"]) and not tail_available:
        raise FileNotFoundError(
            "tail gate 모드를 쓰려면 tail model/scaler가 필요해.\n"
            f"찾는 경로: {tail_model_path} / {tail_scaler_path}\n"
            "파일명이 다르면 --tail-model / --tail-scaler로 지정해줘."
        )

    tail_model = joblib.load(tail_model_path) if tail_available else None
    tail_scaler = joblib.load(tail_scaler_path) if tail_available else None

    # predict probabilities
    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)

    Xs = success_scaler.transform(X)
    p_success = success_model.predict_proba(Xs)[:, 1].astype(np.float64)

    if tail_available:
        Xt = tail_scaler.transform(X)
        p_tail = tail_model.predict_proba(Xt)[:, 1].astype(np.float64)
    else:
        # tail 모델 없을 때: utility gate만 돌리고 싶으면 일단 0으로 둠(보수적으로는 0.5로 둬도 됨)
        p_tail = np.zeros(len(df), dtype=np.float64)

    df["p_success"] = p_success
    df["p_tail"] = p_tail

    # expected-ish score (단순 근사)
    # 성공하면 +pt, 실패하면 sl(음수)로 본 기대수익
    df["ret_score"] = df["p_success"] * pt + (1.0 - df["p_success"]) * sl

    # utility: 기대수익 - 꼬리 페널티
    lam = float(args.lambda_tail)
    df["utility"] = df["ret_score"] - lam * df["p_tail"]

    # per-day gating & pick
    out_rows = []
    grouped = df.groupby("Date", sort=True)

    for date, day in grouped:
        day = day.copy()

        n_all = len(day)
        if n_all == 0:
            out_rows.append({
                "Date": date, "Skipped": 1, "SkipReason": "NO_CANDIDATE",
                "pick_custom": "", "n_all": 0, "n_after_gate": 0,
            })
            continue

        u_min = float(day["utility"].quantile(float(args.u_quantile)))

        gated = day
        if args.gate_mode in ["tail", "tail_utility"]:
            gated = gated[gated["p_tail"] <= float(args.tail_max)]
        if args.gate_mode in ["utility", "tail_utility"]:
            gated = gated[gated["utility"] >= u_min]

        n_after = len(gated)

        if n_after == 0:
            out_rows.append({
                "Date": date, "Skipped": 1, "SkipReason": "GATE_EMPTY",
                "pick_custom": "", "n_all": int(n_all), "n_after_gate": 0,
                "u_min_used": u_min,
                "gate_mode": args.gate_mode,
                "tail_max": float(args.tail_max),
                "u_quantile": float(args.u_quantile),
                "rank_by": args.rank_by,
                "lambda_tail": lam,
            })
            continue

        rank_col = args.rank_by
        if rank_col not in gated.columns:
            out_rows.append({
                "Date": date, "Skipped": 1, "SkipReason": f"MISSING_RANK_COL:{rank_col}",
                "pick_custom": "", "n_all": int(n_all), "n_after_gate": int(n_after),
            })
            continue

        # deterministic: rank_by desc, then p_success desc, then p_tail asc
        gated = gated.sort_values(
            [rank_col, "p_success", "p_tail", "Ticker"],
            ascending=[False, False, True, True],
        )

        top = gated.iloc[0]
        out_rows.append({
            "Date": date,
            "Skipped": 0,
            "SkipReason": "",
            "pick_custom": str(top["Ticker"]),
            "n_all": int(n_all),
            "n_after_gate": int(n_after),
            "u_min_used": u_min,
            "gate_mode": args.gate_mode,
            "tail_max": float(args.tail_max),
            "u_quantile": float(args.u_quantile),
            "rank_by": args.rank_by,
            "lambda_tail": lam,
            "top_p_success": float(top["p_success"]),
            "top_p_tail": float(top["p_tail"]),
            "top_ret_score": float(top["ret_score"]),
            "top_utility": float(top["utility"]),
        })

    out = pd.DataFrame(out_rows).sort_values("Date").reset_index(drop=True)

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.out_suffix}" if str(args.out_suffix).strip() else ""
    out_csv = SIGNAL_DIR / f"picks_{tag}_gate{suffix}.csv"
    out.to_csv(out_csv, index=False)

    meta = {
        "updated_at_utc": now_utc_iso(),
        "tag": tag,
        "gate_mode": args.gate_mode,
        "tail_max": float(args.tail_max),
        "u_quantile": float(args.u_quantile),
        "rank_by": args.rank_by,
        "lambda_tail": lam,
        "rows": int(len(out)),
        "skipped_days": int((out["Skipped"] == 1).sum()),
        "saved_to": str(out_csv),
    }
    (SIGNAL_DIR / f"picks_{tag}_gate{suffix}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"[DONE] gate picks saved: {out_csv}")
    print(f"[META] {json.dumps(meta, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
