#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# sys.path guard (avoid "No module named 'scripts'")
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

try:
    # prefer SSOT feature spec if available
    from scripts.feature_spec import read_feature_cols_meta, get_feature_cols  # type: ignore
except Exception:
    try:
        # fallback when scripts isn't a package
        from feature_spec import read_feature_cols_meta, get_feature_cols  # type: ignore
    except Exception:
        # last resort: minimal fallback (sector off)
        def read_feature_cols_meta():
            return ([], False)

        def get_feature_cols(sector_enabled: bool = False):
            base = [
                "Drawdown_252", "Drawdown_60", "ATR_ratio", "Z_score",
                "MACD_hist", "MA20_slope", "Market_Drawdown", "Market_ATR_ratio",
                "ret_score",
                "ret_5", "ret_10", "ret_20",
                "breakout_20", "vol_surge", "trend_align", "beta_60",
            ]
            if sector_enabled:
                base += ["Sector_Ret_20", "RelStrength"]
            return base


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
LBL_DIR = DATA_DIR / "labels"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

FEATS_PARQ = FEAT_DIR / "features_model.parquet"
FEATS_CSV = FEAT_DIR / "features_model.csv"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_tag(pt: float, H: int, sl: float, ex: int) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{H}_sl{sl_tag}_ex{ex}"


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


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def resolve_feature_cols(args_feature_cols: str) -> tuple[list[str], str]:
    """
    Priority:
      1) --feature-cols explicit
      2) data/meta/feature_cols.json (SSOT)
      3) fallback SSOT default (sector disabled)
    """
    override = [x.strip() for x in str(args_feature_cols or "").split(",") if x.strip()]
    if override:
        return override, "--feature-cols"

    cols_meta, _sector_enabled = read_feature_cols_meta()
    if cols_meta:
        return cols_meta, "data/meta/feature_cols.json"

    return get_feature_cols(sector_enabled=False), "feature_spec.py (fallback)"


def clamp_invest_by_leverage(seed: float, entry_seed: float, desired: float, max_leverage_pct: float) -> float:
    """
    Enforce: seed_after >= - entry_seed * max_leverage_pct
    desired : amount we'd like to spend today (>=0)
    """
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


def simulate_tail_A_for_ticker(
    dates: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    H: int,
    ex: int,
    profit_target: float,
    tail_threshold: float,
    tp1_frac: float,
    max_leverage_pct: float,
    near_zone_mult: float,
) -> np.ndarray:
    """
    A안 TailTarget 정의 (엔진 친화):
      - 시작일 t에 entry close로 첫 매수(단위=entry_seed/H)
      - 이후 t+1..t+(H+ex)까지:
          * (TP1 전) low <= avg*(1+tail_thr) 발생하면 TailTarget=1
          * high >= avg*(1+PT) 발생하면 TP1(부분매도) 발생으로 간주하고 TailTarget=0로 종료
          * DCA는 엔진 룰(<=avg full, <=avg*near_zone half, else 0)
      - H 도달 이후(ex 구간 포함)엔 DCA STOP(엔진의 pending_reval/extend 느낌 반영)
      - 같은 날 low와 high가 같이 조건 충족해도 "리스크 사건" 보수적으로 Tail=1 우선
    """
    n = len(close)
    out = np.zeros(n, dtype=np.int8)

    pt_mult = 1.0 + float(profit_target)
    tail_mult = 1.0 + float(tail_threshold)

    tp1_frac = float(tp1_frac)
    tp1_frac = float(min(1.0, max(0.0, tp1_frac)))

    H = int(max(1, H))
    ex = int(max(0, ex))
    tail_h = int(H + ex)

    # simulate per start index i
    for i in range(n):
        entry_px = close[i]
        if not np.isfinite(entry_px) or entry_px <= 0:
            out[i] = 0
            continue

        # "capital" scale is arbitrary; avg dynamics invariant to scale
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
            out[i] = 0
            continue

        avg = invested / shares

        # forward days: i+1 .. i+tail_h (inclusive bound by data)
        end = min(n - 1, i + tail_h)

        holding_days = 1  # entry day counted as 1 in engine
        tail_hit = False

        for j in range(i + 1, end + 1):
            holding_days += 1

            lo = low[j]
            hi = high[j]
            cl = close[j]

            # (A) Tail check first (conservative on same-day TP1)
            if (not tp1_done) and np.isfinite(lo) and np.isfinite(avg) and avg > 0:
                if lo <= avg * tail_mult:
                    tail_hit = True
                    break

            # (B) TP1 check (stop label scan if TP1 happens)
            if (not tp1_done) and np.isfinite(hi) and np.isfinite(avg) and avg > 0:
                if hi >= avg * pt_mult:
                    # execute partial sell at tp_px (avg*(1+PT))
                    tp_px = avg * pt_mult
                    sell_shares = shares * tp1_frac
                    sell_shares = float(min(shares, max(0.0, sell_shares)))
                    if sell_shares > 0 and tp_px > 0:
                        proceeds = sell_shares * tp_px
                        shares_after = shares - sell_shares

                        # invested scales down proportional to remaining shares (same as engine)
                        if shares_after > 0:
                            invested *= (shares_after / shares)
                        else:
                            invested = 0.0

                        shares = shares_after
                        seed += proceeds

                    tp1_done = True
                    # TP1 occurred before any tail => label 0
                    break

            # (C) DCA rule only BEFORE we reach H
            # 엔진: (not extending) & (not pending_reval)에서만 매수
            # 여기서는 "holding_days <= H"에서만 DCA 허용 (H 이후 DCA stop)
            if (not tp1_done) and (holding_days <= H):
                if not (np.isfinite(cl) and cl > 0):
                    continue

                desired = unit

                # conditional buy rule (engine same)
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

        out[i] = 1 if tail_hit else 0

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build p_tail training labels (TailTarget) using engine-friendly A-definition."
    )
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)  # tag consistency only
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--tail-threshold", type=float, default=-0.30, help="e.g. -0.30 means -30% vs avg")
    ap.add_argument("--tp1-frac", type=float, default=0.50, help="engine TP1 fraction (for stop scanning)")
    ap.add_argument("--max-leverage-pct", type=float, default=1.0, help="engine leverage cap (scale-in limiter)")
    ap.add_argument("--near-zone-mult", type=float, default=1.05, help="engine near-zone for half-buy (avg*1.05)")

    ap.add_argument("--features-path", type=str, default=None, help="optional features file override")
    ap.add_argument("--feature-cols", type=str, default="", help="comma-separated; empty => SSOT/meta")

    args = ap.parse_args()

    pt = float(args.profit_target)
    H = int(args.max_days)
    sl = float(args.stop_level)
    ex = int(args.max_extend_days)
    tail_thr = float(args.tail_threshold)

    tag = fmt_tag(pt, H, sl, ex)

    # load prices
    prices = read_table(PRICES_PARQ, PRICES_CSV).copy()
    need_px = ["Date", "Ticker", "High", "Low", "Close"]
    miss = [c for c in need_px if c not in prices.columns]
    if miss:
        raise ValueError(f"prices missing required cols: {miss}")

    prices["Date"] = _norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    for c in ["High", "Low", "Close"]:
        prices[c] = pd.to_numeric(prices[c], errors="coerce")
    prices = (
        prices.dropna(subset=["Date", "Ticker", "High", "Low", "Close"])
        .sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    # load features (base frame) - keep SSOT feature cols + Date/Ticker
    if args.features_path:
        fp = Path(args.features_path)
        if not fp.exists():
            raise FileNotFoundError(f"features-path not found: {fp}")
        feats = pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
        feats_src = str(fp)
    else:
        feats = read_table(FEATS_PARQ, FEATS_CSV)
        feats_src = str(FEATS_PARQ if FEATS_PARQ.exists() else FEATS_CSV)

    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_model must include Date,Ticker")

    feats = feats.copy()
    feats["Date"] = _norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = (
        feats.dropna(subset=["Date", "Ticker"])
        .sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    # feature cols resolution
    feat_cols, feat_cols_src = resolve_feature_cols(args.feature_cols)
    feat_cols = [c for c in feat_cols if c in feats.columns]  # only those present in feats

    # base key frame (we'll join TailTarget onto this)
    base = feats[["Date", "Ticker"] + feat_cols].copy()

    # join prices (for label simulation)
    base = base.merge(
        prices[["Date", "Ticker", "High", "Low", "Close"]],
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
    )
    base = base.dropna(subset=["High", "Low", "Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if base.empty:
        raise RuntimeError("No rows after merging features with prices. Check date ranges.")

    # compute TailTarget per ticker
    tails = []
    for tkr, g in base.groupby("Ticker", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)

        dates = g["Date"].to_numpy()
        close = g["Close"].to_numpy(dtype=float)
        high = g["High"].to_numpy(dtype=float)
        low = g["Low"].to_numpy(dtype=float)

        tail_arr = simulate_tail_A_for_ticker(
            dates=dates,
            close=close,
            high=high,
            low=low,
            H=H,
            ex=ex,
            profit_target=pt,
            tail_threshold=tail_thr,
            tp1_frac=float(args.tp1_frac),
            max_leverage_pct=float(args.max_leverage_pct),
            near_zone_mult=float(args.near_zone_mult),
        )

        out_g = g[["Date", "Ticker"] + feat_cols].copy()
        out_g["TailTarget"] = tail_arr.astype(int)
        tails.append(out_g)

    labeled = pd.concat(tails, ignore_index=True) if tails else pd.DataFrame()
    if labeled.empty:
        raise RuntimeError("No TailTarget rows produced.")

    # outputs
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    out_parq = LBL_DIR / f"labels_tail_{tag}.parquet"
    out_csv = LBL_DIR / f"labels_tail_{tag}.csv"
    saved_to = save_table(labeled, out_parq, out_csv)

    meta = {
        "updated_at_utc": now_utc_iso(),
        "tag": tag,
        "profit_target": pt,
        "max_days": H,
        "stop_level": sl,
        "max_extend_days": ex,
        "tail_threshold": tail_thr,
        "tp1_frac": float(args.tp1_frac),
        "max_leverage_pct": float(args.max_leverage_pct),
        "near_zone_mult": float(args.near_zone_mult),
        "rows": int(len(labeled)),
        "tail_rate": float(labeled["TailTarget"].mean()) if len(labeled) else None,
        "features_source": feats_src,
        "feature_cols_source": feat_cols_src,
        "feature_cols": feat_cols,
        "saved_to": saved_to,
    }
    (LBL_DIR / f"labels_tail_{tag}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] wrote: {saved_to} rows={len(labeled)} tail_rate={meta['tail_rate']}")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()