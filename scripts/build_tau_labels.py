# scripts/build_tau_labels.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


FEATURES_PARQ = "data/features/features_model.parquet"
FEATURES_CSV = "data/features/features_model.csv"

OUT_PARQ = "data/labels/labels_tau.parquet"
OUT_CSV = "data/labels/labels_tau.csv"


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build tau labels (time-to-success).")

    # keep consistent schema with other steps
    ap.add_argument("--profit-target", type=float, default=0.10)
    ap.add_argument("--max-days", type=int, default=40)

    # ✅ these caused your error -> make them optional with defaults
    ap.add_argument("--stop-level", type=float, default=-0.10)
    ap.add_argument("--max-extend-days", type=int, default=30)

    # label options
    ap.add_argument("--horizons", type=str, default="10,20,40",
                    help="comma-separated horizons for bucket labels (e.g. 10,20,40)")
    ap.add_argument("--out-parq", type=str, default=OUT_PARQ)
    ap.add_argument("--out-csv", type=str, default=OUT_CSV)

    # optional incremental slice
    ap.add_argument("--start-date", type=str, default="")
    ap.add_argument("--buffer-days", type=int, default=120)

    args = ap.parse_args()

    feats = read_table(FEATURES_PARQ, FEATURES_CSV).copy()
    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_model must have Date and Ticker")

    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Optional slice (for speed)
    if args.start_date:
        start = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(start):
            raise ValueError(f"Invalid --start-date: {args.start_date}")
        cut = start - pd.Timedelta(days=int(args.buffer_days))
        feats = feats[feats["Date"] >= cut].reset_index(drop=True)

    # ------------------------------------------------------------
    # IMPORTANT:
    # 이 라벨은 "성공까지 걸린 기간(τ)"을 만들기 위한 자리야.
    # 너네 파이프라인에서는 labels_model(성공/실패) 만들 때처럼
    # 미래 경로(40일 내 +PT 달성)를 보고 τ를 계산해야 함.
    #
    # 여기서는 최소 동작 버전(placeholder)로,
    # features에 이미 아래 컬럼 중 하나가 있다고 가정할 때만 라벨을 만든다:
    # - TimeToSuccessDays / time_to_success_days / Tau / tau
    #
    # (없으면 에러로 알려주고, 실제 τ 계산은 build_labels.py 쪽에서
    #  가격기반으로 만드는 방식으로 붙이는 게 정석)
    # ------------------------------------------------------------
    tau_col = None
    for c in ["TimeToSuccessDays", "time_to_success_days", "Tau", "tau"]:
        if c in feats.columns:
            tau_col = c
            break

    if tau_col is None:
        raise RuntimeError(
            "No tau column found in features_model. "
            "Expected one of: TimeToSuccessDays, time_to_success_days, Tau, tau. "
            "=> τ를 가격 기반으로 계산하도록 라벨 빌더를 확장해야 함."
        )

    tau = pd.to_numeric(feats[tau_col], errors="coerce")

    # bucket labels for stability: e.g. success within 10/20/40 days
    horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
    horizons = sorted(set([h for h in horizons if h > 0]))
    if not horizons:
        horizons = [10, 20, 40]

    out = feats[["Date", "Ticker"]].copy()
    out["tau_days"] = tau

    for h in horizons:
        out[f"tau_le_{h}"] = ((tau <= h) & np.isfinite(tau)).astype(int)

    # add spec columns (for traceability)
    out["profit_target"] = float(args.profit_target)
    out["max_days"] = int(args.max_days)
    out["stop_level"] = float(args.stop_level)
    out["max_extend_days"] = int(args.max_extend_days)

    out_dir = Path(args.out_parq).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save parquet first, csv fallback
    try:
        out.to_parquet(args.out_parq, index=False)
        print(f"[DONE] wrote {args.out_parq} rows={len(out)}")
    except Exception as e:
        print(f"[WARN] parquet save failed: {e}")
    out.to_csv(args.out_csv, index=False)
    print(f"[DONE] wrote {args.out_csv} rows={len(out)}")


if __name__ == "__main__":
    main()