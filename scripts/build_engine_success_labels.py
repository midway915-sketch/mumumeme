#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
LBL_DIR = DATA_DIR / "labels"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

OUT_PARQ = LBL_DIR / "labels_success.parquet"
OUT_CSV = LBL_DIR / "labels_success.csv"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def engine_success_for_ticker(g: pd.DataFrame, H: int, pt: float) -> pd.DataFrame:
    """
    엔진형 성공(=TP1 발생) 라벨.
    - 진입일 i의 '첫 매수'는 i일 Close로 실행(엔진 동일)
    - 이후 매수는 매일 Close에서:
        close <= avg: full
        avg < close <= avg*1.05: half
        close > avg*1.05: 0
    - TP1 판정은 '그날 High'가 avg*(1+pt) 이상이면 성공(1) (엔진 동일)
      (TP1 체결가는 avg*(1+pt)라고 가정)
    - H일 동안(holding_days 1..H) TP1 한번이라도 뜨면 성공
    """
    g = g.sort_values("Date").reset_index(drop=True).copy()

    close = pd.to_numeric(g["Close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(g["High"], errors="coerce").to_numpy(dtype=float)

    n = len(g)
    success = np.zeros(n, dtype=np.int8)

    # scale invariance: entry_seed=1.0 (평단/조건은 스케일 무관)
    # unit = entry_seed / H
    entry_seed = 1.0
    unit = entry_seed / float(H)

    for i in range(n):
        if i + H >= n:
            # 미래 H일을 다 볼 수 없으면 라벨 불가
            success[i] = 0
            continue

        # state for this entry
        shares = 0.0
        invested = 0.0
        tp1_done = False

        # entry buy (day i close)
        px0 = close[i]
        if not np.isfinite(px0) or px0 <= 0:
            continue

        invest0 = unit  # topk=1 가정, weight=1
        shares += invest0 / px0
        invested += invest0

        # day i counts as holding_days=1 in engine
        # TP1 check uses day i High as well? 엔진은 cycle in progress에서 당일 체크.
        # 여기서는 i일 entry 직후 그날 High로 TP1 가능하다고 보려면 아래를 포함.
        avg = invested / shares
        if np.isfinite(high[i]) and high[i] >= avg * (1.0 + pt):
            tp1_done = True
            success[i] = 1
            continue

        # forward days: i+1 .. i+H-1 (총 H일 보유 구간)
        for d in range(i + 1, i + H):
            # TP1 check first (엔진은 intraday high로 TP1 발생 가능)
            avg = invested / shares if shares > 0 else np.nan
            if (not tp1_done) and np.isfinite(avg) and avg > 0:
                if np.isfinite(high[d]) and high[d] >= avg * (1.0 + pt):
                    tp1_done = True
                    success[i] = 1
                    break

            # DCA buy at close (엔진 정상모드의 가격 필터만 반영)
            px = close[d]
            if not np.isfinite(px) or px <= 0:
                continue

            avg = invested / shares if shares > 0 else np.nan
            desired = unit

            if np.isfinite(avg) and avg > 0:
                if px <= avg:
                    pass
                elif px <= avg * 1.05:
                    desired = desired / 2.0
                else:
                    desired = 0.0

            if desired > 0:
                shares += desired / px
                invested += desired

    out = pd.DataFrame({
        "Date": g["Date"].values,
        "Ticker": g["Ticker"].values,
        "Success": success,
        "profit_target": float(pt),
        "max_days": int(H),
        "label_type": "engine_tp1",
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ENGINE-consistent success labels (TP1 hit within H days, avg-price DCA).")
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--start-date", default=None, type=str)
    ap.add_argument("--prices-parq", default=str(PRICES_PARQ), type=str)
    ap.add_argument("--prices-csv", default=str(PRICES_CSV), type=str)
    ap.add_argument("--out-parq", default=str(OUT_PARQ), type=str)
    ap.add_argument("--out-csv", default=str(OUT_CSV), type=str)
    args = ap.parse_args()

    LBL_DIR.mkdir(parents=True, exist_ok=True)

    prices = read_table(Path(args.prices_parq), Path(args.prices_csv)).copy()
    need = ["Date", "Ticker", "High", "Close"]
    missing = [c for c in need if c not in prices.columns]
    if missing:
        raise ValueError(f"prices missing required columns: {missing}")

    prices["Date"] = norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices["High"] = pd.to_numeric(prices["High"], errors="coerce")
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")

    prices = (
        prices.dropna(subset=["Date", "Ticker", "High", "Close"])
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    H = int(args.max_days)
    pt = float(args.profit_target)

    out = prices.groupby("Ticker", group_keys=False).apply(lambda g: engine_success_for_ticker(g, H=H, pt=pt))

    # tail rows (미래 부족) 제거: i+H >= n 부분은 success=0으로 남아있을 수 있어서, 엄격히 자를지 선택
    # 여기서는 "미래 H일 없는 구간"은 빼는 게 정답 (라벨 정보가 없으니까)
    # engine_success_for_ticker는 마지막 H일은 의미가 없으니 drop.
    # (정확히는 각 티커별로 마지막 H일을 drop)
    out2 = []
    for t, g in out.groupby("Ticker", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        if len(g) > H:
            out2.append(g.iloc[:-H].copy())
    labels = pd.concat(out2, ignore_index=True) if out2 else pd.DataFrame()

    if args.start_date and not labels.empty:
        sd = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(sd):
            raise ValueError(f"Invalid --start-date: {args.start_date}")
        labels = labels.loc[labels["Date"] >= sd].copy()

    labels = labels.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_c
sv)
    out_parq.parent.mkdir(parents=True, exist_ok=True)

    labels.to_parquet(out_parq, index=False)
    labels.to_csv(out_csv, index=False)

    dmin = pd.to_datetime(labels["Date"]).min().date() if len(labels) else None
    dmax = pd.to_datetime(labels["Date"]).max().date() if len(labels) else None
    print(f"[DONE] wrote: {out_parq} rows={len(labels)} range={dmin}..{dmax} pt={pt} H={H} label=engine_tp1")


if __name__ == "__main__":
    main()