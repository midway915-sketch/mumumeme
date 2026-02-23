#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
FEATURES_DIR = DATA_DIR / "features"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def parse_csv_list(s: str) -> list[str]:
    items = [x.strip() for x in str(s or "").split(",")]
    return [x for x in items if x]


@dataclass
class GateConfig:
    mode: str
    tail_threshold: float
    utility_quantile: float
    rank_by: str
    lambda_tail: float
    topk: int
    ps_min: float
    badexit_max: float


def compute_utility(df: pd.DataFrame, lam: float) -> pd.Series:
    # utility = p_success - lam * p_tail
    ps = pd.to_numeric(df.get("p_success", 0.0), errors="coerce").fillna(0.0).astype(float)
    pt = pd.to_numeric(df.get("p_tail", 0.0), errors="coerce").fillna(0.0).astype(float)
    return ps - float(lam) * pt


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate picks builder")

    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--mode", required=True, type=str, choices=["none", "tail", "utility", "tail_utility"])
    ap.add_argument("--tail-threshold", required=True, type=float)
    ap.add_argument("--utility-quantile", required=True, type=float)
    ap.add_argument("--rank-by", required=True, type=str, choices=["utility", "ret_score", "p_success"])
    ap.add_argument("--lambda-tail", required=True, type=float)

    ap.add_argument("--topk", default=1, type=int)
    ap.add_argument("--ps-min", default=0.0, type=float)
    ap.add_argument("--badexit-max", default=1.0, type=float)

    ap.add_argument("--tag", required=True, type=str)
    ap.add_argument("--suffix", required=True, type=str)

    ap.add_argument("--exclude-tickers", default="")
    ap.add_argument("--features-path", default="")

    # ✅ workflow compat (run_grid_workflow.sh / gate-grid.yml)
    ap.add_argument("--out-dir", default="data/signals", type=str)
    ap.add_argument(
        "--require-files",
        default="",
        type=str,
        help="comma-separated list of files that must exist; fail-fast if missing",
    )

    args = ap.parse_args()

    # ---- require-files (fail fast)
    req = str(getattr(args, "require_files", "") or "").strip()
    if req:
        missing: list[str] = []
        for p in [x.strip() for x in req.split(",") if x.strip()]:
            if not Path(p).exists():
                missing.append(p)
        if missing:
            print(f"[ERROR] missing required files: {missing}")
            raise SystemExit(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # features_scored load
    if args.features_path:
        fp = Path(args.features_path)
        feats = read_table(fp, fp.with_suffix(".csv")).copy()
    else:
        feats = read_table(FEATURES_DIR / "features_scored.parquet", FEATURES_DIR / "features_scored.csv").copy()

    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_scored must contain Date,Ticker")

    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).copy()

    # exclude tickers
    excl = set([x.upper().strip() for x in parse_csv_list(args.exclude_tickers)])
    if excl:
        feats = feats.loc[~feats["Ticker"].isin(excl)].copy()

    cfg = GateConfig(
        mode=str(args.mode),
        tail_threshold=float(args.tail_threshold),
        utility_quantile=float(args.utility_quantile),
        rank_by=str(args.rank_by),
        lambda_tail=float(args.lambda_tail),
        topk=int(args.topk),
        ps_min=float(args.ps_min),
        badexit_max=float(args.badexit_max),
    )

    # 필수 컬럼 최소 체크 (없으면 0으로 처리되는 형태는 score_features에서 이미 보장하긴 함)
    for c in ["p_success", "p_tail", "p_badexit", "ret_score"]:
        if c not in feats.columns:
            feats[c] = 0.0

    feats["p_success"] = pd.to_numeric(feats["p_success"], errors="coerce").fillna(0.0).astype(float)
    feats["p_tail"] = pd.to_numeric(feats["p_tail"], errors="coerce").fillna(0.0).astype(float)
    feats["p_badexit"] = pd.to_numeric(feats["p_badexit"], errors="coerce").fillna(0.0).astype(float)
    feats["ret_score"] = pd.to_numeric(feats["ret_score"], errors="coerce").fillna(0.0).astype(float)

    # gating filters
    df = feats.copy()

    # p_success min
    df = df.loc[df["p_success"] >= float(cfg.ps_min)].copy()

    # p_badexit max
    df = df.loc[df["p_badexit"] <= float(cfg.badexit_max)].copy()

    # tail threshold condition depending on mode
    if cfg.mode in ("tail", "tail_utility"):
        df = df.loc[df["p_tail"] <= float(cfg.tail_threshold)].copy()

    # utility condition depending on mode
    if cfg.mode in ("utility", "tail_utility"):
        df["utility"] = compute_utility(df, lam=float(cfg.lambda_tail))
        # keep top quantile per day AFTER ranking metric is computed
    else:
        df["utility"] = compute_utility(df, lam=float(cfg.lambda_tail))

    if df.empty:
        # still write empty picks file for downstream consistency
        picks_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
        pd.DataFrame(columns=["Date", "Ticker"]).to_csv(picks_path, index=False)
        print(f"[DONE] wrote empty picks: {picks_path}")
        return

    # rank metric
    if cfg.rank_by == "utility":
        score_col = "utility"
    elif cfg.rank_by == "ret_score":
        score_col = "ret_score"
    else:
        score_col = "p_success"

    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).astype(float)

    # per-day quantile filter (utility mode only)
    if cfg.mode in ("utility", "tail_utility"):
        uq = float(cfg.utility_quantile)

        def _keep_quantile(g: pd.DataFrame) -> pd.DataFrame:
            if len(g) <= 1:
                return g
            thr = float(np.quantile(g[score_col].to_numpy(dtype=float), uq))
            return g.loc[g[score_col] >= thr]

        df = df.groupby("Date", group_keys=False).apply(_keep_quantile).reset_index(drop=True)

    # topk per day
    df = df.sort_values(["Date", score_col], ascending=[True, False])
    picks = df.groupby("Date", group_keys=False).head(int(cfg.topk))[["Date", "Ticker"]].copy()

    picks_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
    picks.to_csv(picks_path, index=False)
    print(f"[DONE] wrote picks: {picks_path} rows={len(picks)}")


if __name__ == "__main__":
    main()