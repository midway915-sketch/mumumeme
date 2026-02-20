#!/usr/bin/env python3
from __future__ import annotations

# ✅ FIX(A): "python scripts/xxx.py" 실행에서도 scripts.* import 되도록 repo root를 sys.path에 추가
import sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scripts.feature_spec import read_feature_cols_meta, get_feature_cols


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"

PRICES_PARQUET = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

FEATURES_PARQUET = FEAT_DIR / "features_model.parquet"
FEATURES_CSV = FEAT_DIR / "features_model.csv"

# ✅ FIX: 루트 data/ 가 아니라 data/labels/ 로
OUT_STRATEGY_RAW = LABEL_DIR / "strategy_raw_data.csv"

MARKET_TICKER = "SPY"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_tag(pt: float, max_days: int, sl: float, ex: int) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{max_days}_sl{sl_tag}_ex{ex}"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def save_table(df: pd.DataFrame, parq: Path, csv: Path) -> str:
    parq.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parq, index=False)
        return str(parq)
    except Exception:
        df.to_csv(csv, index=False)
        return str(csv)


def auto_load_features(features_path: str | None) -> tuple[pd.DataFrame, str]:
    if features_path:
        fp = Path(features_path)
        if not fp.exists():
            raise FileNotFoundError(f"features-path not found: {fp}")
        df = pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
        return df, str(fp)

    if FEATURES_PARQUET.exists():
        return pd.read_parquet(FEATURES_PARQUET), str(FEATURES_PARQUET)
    if FEATURES_CSV.exists():
        return pd.read_csv(FEATURES_CSV), str(FEATURES_CSV)

    if not FEAT_DIR.exists():
        raise FileNotFoundError(f"features dir not found: {FEAT_DIR}")

    candidates = list(FEAT_DIR.glob("*.parquet")) + list(FEAT_DIR.glob("*.csv"))
    candidates = [p for p in candidates if p.is_file()]

    if not candidates:
        raise FileNotFoundError(
            f"Missing file: {FEATURES_PARQUET} (or {FEATURES_CSV}) and no candidates in {FEAT_DIR}"
        )

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
        f"Could not find a usable features file in {FEAT_DIR}. Need columns Date/Ticker. Last error: {last_err}"
    )


def forward_roll_max_excl_today(s: pd.Series, window: int) -> pd.Series:
    return s[::-1].rolling(window, min_periods=window).max()[::-1].shift(-1)


def forward_roll_min_excl_today(s: pd.Series, window: int) -> pd.Series:
    return s[::-1].rolling(window, min_periods=window).min()[::-1].shift(-1)


# ---------------------------
# A안 Tail(엔진 평단 기반) 계산
# ---------------------------
def clamp_invest_by_leverage(seed: float, entry_seed: float, desired: float, max_leverage_pct: float) -> float:
    if desired <= 0:
        return 0.0
    if not np.isfinite(entry_seed) or entry_seed <= 0:
        return float(min(desired, max(seed, 0.0)))

    borrow_limit = float(entry_seed) * float(max_leverage_pct)
    floor_seed = -borrow_limit
    room = float(seed) - float(floor_seed)  # seed + borrow_limit
    if room <= 0:
        return 0.0
    return float(min(desired, room))


def tail_A_series_for_ticker(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    H: int,
    ex: int,
    profit_target: float,
    tail_threshold: float,
    tp1_frac: float = 0.50,
    max_leverage_pct: float = 1.00,
    near_zone_mult: float = 1.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      tail_hit[i] (0/1): TP1 전까지 low <= avg*(1+tail_thr) 한번이라도 발생하면 1
      fut_dd_close[i]: 참고용 (기존과 호환) entry close 기준 (t+1..t+H+ex) 최저 Low DD
                       = min(low/close_i - 1)
    """
    n = len(close)
    tail_hit = np.zeros(n, dtype=np.int8)
    fut_dd_close = np.full(n, np.nan, dtype=float)

    pt_mult = 1.0 + float(profit_target)
    tail_mult = 1.0 + float(tail_threshold)

    H = int(max(1, H))
    ex = int(max(0, ex))
    tail_h = int(H + ex)

    tp1_frac = float(min(1.0, max(0.0, tp1_frac)))

    for i in range(n):
        entry_px = float(close[i])
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        # (참고용) entry close 기준 미래 최저 DD
        end0 = min(n - 1, i + tail_h)
        if end0 >= i + 1:
            lo_min = np.nanmin(low[i + 1 : end0 + 1])
            if np.isfinite(lo_min) and lo_min > 0:
                fut_dd_close[i] = (lo_min / entry_px) - 1.0

        # 엔진형 DCA(스케일 불변) - seed=1로 정규화
        entry_seed = 1.0
        seed = 1.0
        unit = entry_seed / float(H)

        shares = 0.0
        invested = 0.0
        tp1_done = False

        # entry buy at day i close
        invest0 = clamp_invest_by_leverage(seed, entry_seed, unit, max_leverage_pct)
        if invest0 > 0:
            seed -= invest0
            invested += invest0
            shares += invest0 / entry_px

        if shares <= 0 or invested <= 0:
            continue

        avg = invested / shares
        holding_days = 1

        end = min(n - 1, i + tail_h)
        hit = False

        for j in range(i + 1, end + 1):
            holding_days += 1
            lo = float(low[j])
            hi = float(high[j])
            cl = float(close[j])

            # (1) Tail check first (same-day TP1 vs Tail => Tail 우선 보수)
            if (not tp1_done) and np.isfinite(lo) and np.isfinite(avg) and avg > 0:
                if lo <= avg * tail_mult:
                    hit = True
                    break

            # (2) TP1 check => TP1 나오면 Tail 스캔 종료(=0)
            if (not tp1_done) and np.isfinite(hi) and np.isfinite(avg) and avg > 0:
                if hi >= avg * pt_mult:
                    # partial sell at tp_px
                    tp_px = avg * pt_mult
                    sell_shares = float(min(shares, max(0.0, shares * tp1_frac)))
                    if sell_shares > 0 and tp_px > 0:
                        proceeds = sell_shares * tp_px
                        shares_after = shares - sell_shares
                        if shares_after > 0:
                            invested *= (shares_after / shares)
                        else:
                            invested = 0.0
                        shares = shares_after
                        seed += proceeds
                    tp1_done = True
                    break

            # (3) DCA only until H (H 이후 DCA stop)
            if (not tp1_done) and (holding_days <= H):
                if not (np.isfinite(cl) and cl > 0):
                    continue

                desired = unit

                # 엔진 룰: <=avg full, <=avg*1.05 half, else 0
                if np.isfinite(avg) and avg > 0:
                    if cl <= avg:
                        pass
                    elif cl <= avg * float(near_zone_mult):
                        desired = desired / 2.0
                    else:
                        desired = 0.0

                invest = clamp_invest_by_leverage(seed, entry_seed, desired, max_leverage_pct)
                if invest > 0:
                    seed -= invest
                    invested += invest
                    shares += invest / cl
                    if shares > 0:
                        avg = invested / shares

        tail_hit[i] = 1 if hit else 0

    return tail_hit, fut_dd_close


def main() -> None:
    ap = argparse.ArgumentParser(description="Build strategy labels (Success + Tail(A) + ExtendNeeded).")
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument(
        "--tail-threshold",
        type=float,
        default=-0.30,
        help="A안 tail threshold vs avg (e.g. -0.30 means low <= avg*(1-0.30) before TP1).",
    )

    # 엔진 파라미터(기본값은 simulate 엔진과 맞춤)
    ap.add_argument("--tp1-frac", type=float, default=0.50)
    ap.add_argument("--max-leverage-pct", type=float, default=1.00)
    ap.add_argument("--near-zone-mult", type=float, default=1.05)

    ap.add_argument("--features-path", type=str, default=None)
    ap.add_argument("--feature-cols", type=str, default="", help="콤마로 feature 컬럼 지정(비우면 meta/SSOT 사용)")
    args = ap.parse_args()

    pt = float(args.profit_target)
    max_days = int(args.max_days)
    sl = float(args.stop_level)
    ex = int(args.max_extend_days)
    tail_thr = float(args.tail_threshold)

    tag = fmt_tag(pt, max_days, sl, ex)

    prices = read_table(PRICES_PARQUET, PRICES_CSV).copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices = (
        prices.sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    need_px = {"Date", "Ticker", "High", "Low", "Close"}
    miss_px = [c for c in need_px if c not in prices.columns]
    if miss_px:
        raise ValueError(f"prices missing columns: {miss_px}")

    feats, feats_src = auto_load_features(args.features_path)
    feats = feats.copy()
    feats["Date"] = pd.to_datetime(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = (
        feats.sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    # ✅ feature cols 결정 정책:
    # 1) args.feature_cols 명시 > 2) data/meta/feature_cols.json > 3) SSOT 기본 (sector off)
    if str(args.feature_cols).strip():
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    else:
        cols_meta, _sector_enabled = read_feature_cols_meta()
        if cols_meta:
            feature_cols = cols_meta
        else:
            feature_cols = get_feature_cols(sector_enabled=False)

    missing_feat = [c for c in feature_cols if c not in feats.columns]
    if missing_feat:
        raise ValueError(f"features missing feature columns: {missing_feat} (src={feats_src})")

    base = feats[["Date", "Ticker"] + feature_cols].merge(
        prices[["Date", "Ticker", "High", "Low", "Close"]],
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
    )
    base = base.dropna(subset=["Close"] + feature_cols).reset_index(drop=True)
    base = base.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    success_h = max_days
    tail_h = max_days + ex

    out = []
    for _tkr, g in base.groupby("Ticker", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)

        close = g["Close"].astype(float)
        high = g["High"].astype(float)
        low = g["Low"].astype(float)

        # --- Success (기존 그대로: entry Close 기준 미래 max High)
        fut_max_high_success = forward_roll_max_excl_today(high, success_h)
        success = (fut_max_high_success >= close * (1.0 + pt)).astype("float")

        # --- Tail (A안: 엔진 평단 기반)
        tail_A, fut_dd_close = tail_A_series_for_ticker(
            close=close.to_numpy(dtype=float),
            high=high.to_numpy(dtype=float),
            low=low.to_numpy(dtype=float),
            H=max_days,
            ex=ex,
            profit_target=pt,
            tail_threshold=tail_thr,
            tp1_frac=float(args.tp1_frac),
            max_leverage_pct=float(args.max_leverage_pct),
            near_zone_mult=float(args.near_zone_mult),
        )
        tail = tail_A.astype("float")

        # --- 참고용 FutureMinDD_TailH (호환 위해 entry Close 기준 DD 유지)
        fut_dd_tail = pd.Series(fut_dd_close, index=g.index).astype(float)

        # --- ExtendNeeded (기존 그대로: H일 시점 ret < SL and success==0)
        offset = max_days - 1
        close_at_max = close.shift(-offset)
        ret_at_max = (close_at_max / close) - 1.0
        extend_needed = ((success == 0) & (ret_at_max < sl)).astype("float")

        g["Success"] = success
        g["Tail"] = tail
        g["FutureMinDD_TailH"] = fut_dd_tail
        g["RetAtMaxDays"] = ret_at_max
        g["ExtendNeeded"] = extend_needed

        g = g.dropna(
            subset=["Success", "Tail", "FutureMinDD_TailH", "RetAtMaxDays", "ExtendNeeded"]
        ).reset_index(drop=True)
        out.append(g)

    labeled = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if labeled.empty:
        raise RuntimeError("No strategy labeled rows produced. Check horizons and raw data.")

    labeled["Success"] = labeled["Success"].astype(int)
    labeled["Tail"] = labeled["Tail"].astype(int)
    labeled["ExtendNeeded"] = labeled["ExtendNeeded"].astype(int)

    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    out_parq = LABEL_DIR / f"strategy_raw_data_{tag}.parquet"
    out_csv = LABEL_DIR / f"strategy_raw_data_{tag}.csv"
    saved_to = save_table(labeled, out_parq, out_csv)

    # ✅ FIX: 루트 data/가 아니라 data/labels/ 아래로
    labeled.to_csv(OUT_STRATEGY_RAW, index=False)

    meta = {
        "updated_at_utc": now_utc_iso(),
        "tag": tag,
        "profit_target": pt,
        "max_days": max_days,
        "stop_level": sl,
        "max_extend_days": ex,
        "tail_threshold": tail_thr,
        "rows": int(len(labeled)),
        "base_success_rate": float(labeled["Success"].mean()),
        "base_tail_rate": float(labeled["Tail"].mean()),
        "base_extend_needed_rate": float(labeled["ExtendNeeded"].mean()),
        "min_date": str(labeled["Date"].min().date()),
        "max_date": str(labeled["Date"].max().date()),
        "saved_to": saved_to,
        "also_written": str(OUT_STRATEGY_RAW),
        "feature_cols": feature_cols,
        "features_source": feats_src,
        "tail_horizon_days": int(tail_h),
        "market_ticker": MARKET_TICKER,
        "tail_label_mode": "A(avg_price_before_TP1)",
        "engine_params": {
            "tp1_frac": float(args.tp1_frac),
            "max_leverage_pct": float(args.max_leverage_pct),
            "near_zone_mult": float(args.near_zone_mult),
        },
    }
    (LABEL_DIR / f"strategy_raw_data_{tag}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"[DONE] strategy labels saved -> {saved_to}")
    print(f"[DONE] also wrote -> {OUT_STRATEGY_RAW}")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()