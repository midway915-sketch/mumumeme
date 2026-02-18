# scripts/build_tau_labels.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


FEATURES_PARQ = "data/features/features_model.parquet"
FEATURES_CSV  = "data/features/features_model.csv"

PRICES_PARQ = "data/raw/prices.parquet"
PRICES_CSV  = "data/raw/prices.csv"

OUT_PARQ = "data/labels/labels_tau.parquet"
OUT_CSV  = "data/labels/labels_tau.csv"


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


def ensure_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def compute_tau_labels(
    prices: pd.DataFrame,
    profit_target: float,
    max_days: int,
    max_extend_days: int,
    k1: int,
    k2: int,
) -> pd.DataFrame:
    """
    TauDays: entry day(0) 포함하여 '목표가(PT) 도달까지 걸린 거래일 수' (1..H)
    TauClass:
      0 = within horizon(=max_days) 성공 못함
      1 = <= k1 일 이내 성공
      2 = k1 < tau <= k2
      3 = k2 < tau <= max_days
    """
    H = int(max_days)
    if H <= 1:
        raise ValueError("max_days must be >= 2")
    if not (1 <= k1 < k2 <= H):
        raise ValueError(f"Require 1 <= k1 < k2 <= max_days. got k1={k1}, k2={k2}, max_days={H}")

    # 정렬/정규화
    p = prices.copy()
    p["Date"] = norm_date(p["Date"])
    p["Ticker"] = p["Ticker"].astype(str).str.upper().str.strip()
    p = p.dropna(subset=["Date", "Ticker", "Close", "High"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    out_rows = []

    # ticker별로 forward high를 rolling 해서 "최초 도달일" 찾기
    for t, g in p.groupby("Ticker", sort=False):
        g = g.reset_index(drop=True)
        close = pd.to_numeric(g["Close"], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(g["High"], errors="coerce").to_numpy(dtype=float)

        n = len(g)
        if n < H + 5:
            continue

        # 각 i에서 목표가 = close[i]*(1+PT)
        # tau = min d in [1..H] s.t. high[i+d] >= target
        tau_days = np.full(n, np.nan, dtype=float)
        success = np.zeros(n, dtype=int)

        for i in range(0, n - 1):
            c0 = close[i]
            if not np.isfinite(c0) or c0 <= 0:
                continue
            target = c0 * (1.0 + float(profit_target))

            # lookahead 범위: i+1 .. i+H (거래일 기준)
            j_end = min(n - 1, i + H)
            # high가 target 넘는 첫 지점 찾기
            # (i+1부터)
            window = high[i + 1 : j_end + 1]
            if window.size == 0:
                continue

            hit = np.where(np.isfinite(window) & (window >= target))[0]
            if hit.size > 0:
                d = int(hit[0] + 1)  # +1 because window starts at i+1
                tau_days[i] = float(d)
                success[i] = 1

        # TauClass 만들기
        tau_class = np.zeros(n, dtype=int)
        # 성공한 애들만 분류
        mask = np.isfinite(tau_days)
        td = tau_days[mask].astype(int)

        # 1/2/3 bucket
        cls = np.zeros_like(td, dtype=int)
        cls[(td >= 1) & (td <= k1)] = 1
        cls[(td > k1) & (td <= k2)] = 2
        cls[(td > k2) & (td <= H)] = 3
        # (이론상 td>H는 없음)
        tau_class[mask] = cls

        tmp = pd.DataFrame({
            "Date": g["Date"],
            "Ticker": t,
            "TauDays": tau_days,      # float (NaN 가능)
            "TauClass": tau_class,    # int (0..3)
            "TauSuccess": success,    # 0/1
            "ProfitTarget": float(profit_target),
            "MaxDays": int(max_days),
            "StopLevel": np.nan,      # 참고용(계산에는 안 씀)
            "MaxExtendDays": int(max_extend_days),  # 참고용(계산에는 안 씀)
            "K1": int(k1),
            "K2": int(k2),
        })
        out_rows.append(tmp)

    if not out_rows:
        raise RuntimeError("No tau labels produced. Check price history length and tickers.")

    out = pd.concat(out_rows, ignore_index=True)
    out = out.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build tau classification labels (TauClass) from prices.")
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)         # ✅ CLI 호환 (tau 계산엔 미사용)
    ap.add_argument("--max-extend-days", required=True, type=int)      # ✅ CLI 호환 (tau 계산엔 미사용)

    ap.add_argument("--k1", default=10, type=int)
    ap.add_argument("--k2", default=20, type=int)

    ap.add_argument("--features-parq", default=FEATURES_PARQ, type=str)
    ap.add_argument("--features-csv", default=FEATURES_CSV, type=str)
    ap.add_argument("--prices-parq", default=PRICES_PARQ, type=str)
    ap.add_argument("--prices-csv", default=PRICES_CSV, type=str)

    ap.add_argument("--out-parq", default=OUT_PARQ, type=str)
    ap.add_argument("--out-csv", default=OUT_CSV, type=str)

    args = ap.parse_args()

    # prices로 tau 만들고, features 존재하면 Date/Ticker로 intersect만 남김(누수/정합성 방지)
    prices = read_table(args.prices_parq, args.prices_csv)
    ensure_cols(prices, ["Date", "Ticker", "Close", "High"], "prices")

    tau = compute_tau_labels(
        prices=prices,
        profit_target=args.profit_target,
        max_days=args.max_days,
        max_extend_days=args.max_extend_days,
        k1=args.k1,
        k2=args.k2,
    )

    # features_model이 있으면 그 (Date,Ticker)만 유지
    try:
        feats = read_table(args.features_parq, args.features_csv)
        ensure_cols(feats, ["Date", "Ticker"], "features_model")
        feats = feats.copy()
        feats["Date"] = norm_date(feats["Date"])
        feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
        feats = feats.dropna(subset=["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"])
        tau = tau.merge(feats[["Date", "Ticker"]], on=["Date", "Ticker"], how="inner")
    except FileNotFoundError:
        pass

    # ✅ 최종 필수 컬럼 보장
    ensure_cols(tau, ["Date", "Ticker", "TauClass"], "labels_tau")

    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_csv)
    out_parq.parent.mkdir(parents=True, exist_ok=True)

    # parquet 우선 저장
    try:
        tau.to_parquet(out_parq, index=False)
        print(f"[DONE] wrote {out_parq} rows={len(tau)}")
    except Exception as e:
        print(f"[WARN] parquet save failed: {e}")

    tau.to_csv(out_csv, index=False)
    print(f"[DONE] wrote {out_csv} rows={len(tau)}")

    # sanity
    vc = tau["TauClass"].value_counts(dropna=False).sort_index()
    print("[INFO] TauClass counts:")
    print(vc.to_string())


if __name__ == "__main__":
    main()