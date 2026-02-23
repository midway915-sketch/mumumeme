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

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"
META_DIR = DATA_DIR / "meta"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

# ✅ 강제: tail labels는 features_model 기준으로만 align
FEATURES_MODEL_PARQ = FEAT_DIR / "features_model.parquet"
FEATURES_MODEL_CSV = FEAT_DIR / "features_model.csv"


# ------------------------------------------------------------
# SSOT feature cols resolution (forced 18 cols, consistent with feature_spec)
# ------------------------------------------------------------
def _validate_18_cols(cols: list[str], src: str) -> None:
    if len(cols) != 18:
        raise RuntimeError(f"SSOT feature cols must be 18, got {len(cols)} from {src}: {cols}")
    need = ["RelStrength"]
    miss = [c for c in need if c not in cols]
    if miss:
        raise RuntimeError(
            f"SSOT feature cols must include required features {need}. Missing={miss}. src={src}"
        )


def resolve_feature_cols_forced(args_feature_cols: str) -> tuple[list[str], str]:
    if str(args_feature_cols or "").strip():
        raise ValueError(
            "--feature-cols override is not allowed when SSOT 18 features are enforced.\n"
            "Remove --feature-cols and rely on SSOT/meta + feature_spec."
        )

    try:
        from scripts.feature_spec import read_feature_cols_meta, get_feature_cols  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import scripts.feature_spec: {e}")

    cols, _sector_enabled = read_feature_cols_meta()
    if cols:
        src = "data/meta/feature_cols.json"
        _validate_18_cols(cols, src)
        return cols, src

    cols = get_feature_cols(sector_enabled=False)
    src = "scripts/feature_spec.py:get_feature_cols(sector_enabled=False)"
    _validate_18_cols(cols, src)
    return cols, src


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------
def read_prices() -> pd.DataFrame:
    if PRICES_PARQ.exists():
        df = pd.read_parquet(PRICES_PARQ)
    elif PRICES_CSV.exists():
        df = pd.read_csv(PRICES_CSV)
    else:
        raise FileNotFoundError(f"Missing prices: {PRICES_PARQ} (or {PRICES_CSV})")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Ticker", "Close", "High", "Low"])
        .sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )
    return df


def read_features_model() -> tuple[pd.DataFrame, str]:
    if FEATURES_MODEL_PARQ.exists():
        df = pd.read_parquet(FEATURES_MODEL_PARQ).copy()
        src = str(FEATURES_MODEL_PARQ)
    elif FEATURES_MODEL_CSV.exists():
        df = pd.read_csv(FEATURES_MODEL_CSV).copy()
        src = str(FEATURES_MODEL_CSV)
    else:
        raise FileNotFoundError(
            "Missing features_model. Expected one of:\n"
            f"- {FEATURES_MODEL_PARQ}\n"
            f"- {FEATURES_MODEL_CSV}\n"
        )

    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError(f"features_model missing Date/Ticker: {src}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["Date", "Ticker"]).reset_index(drop=True)
    return df, src


# ------------------------------------------------------------
# Tail label primitives (DCA avg_price based)
# ------------------------------------------------------------
def _dca_desired_multiplier(close_px: float, avg_px: float) -> float:
    if not np.isfinite(close_px) or close_px <= 0:
        return 0.0
    if not np.isfinite(avg_px) or avg_px <= 0:
        return 1.0
    if close_px <= avg_px:
        return 1.0
    if close_px <= avg_px * 1.05:
        return 0.5
    return 0.0


def compute_tail_primitives_dca_for_ticker(
    g: pd.DataFrame,
    profit_target: float,
    hmax: int,
    stop_level: float,
    exmax: int,
) -> pd.DataFrame:
    """
    ✅ 평단(DCA) 기반 tail 원천 라벨

    원천 컬럼:
      TauSLDays               : Hmax 내 SL 최초 터치까지 일수(1..Hmax), 없으면 NaN
      TauTPDays               : Hmax 내 TP 최초 터치까지 일수(1..Hmax), 없으면 NaN
      TauRecoverAfterSLDays   : SL 이후(다음날부터) TP 회복 최초까지 일수(1..EXmax), 없으면 NaN
      TailPmaxEXmax           : (진입~Hmax+EXmax) 구간에서 close 기준 (close/avg - 1)의 최대값(진단용)

    주의:
      - DCA는 entry day 포함 Hmax 슬롯 동안만 수행 (unit=1/Hmax, normalized capital)
      - SL/TP 판정은 intraday High/Low가 '직전 평단' 임계치 터치 여부로 판단 (동시 터치 시 STOP 우선)
      - SL이 먼저인 케이스에서 "회복"은 SL 다음날부터 EXmax 구간에서 TP 터치로 정의
    """
    g = g.sort_values("Date").reset_index(drop=True).copy()
    close = pd.to_numeric(g["Close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(g["High"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(g["Low"], errors="coerce").to_numpy(dtype=float)

    n = len(g)
    H = int(max(1, hmax))
    EX = int(max(0, exmax))

    pt_mult = 1.0 + float(profit_target)
    sl_mult = 1.0 + float(stop_level)

    tau_sl = np.full(n, np.nan, dtype=float)
    tau_tp = np.full(n, np.nan, dtype=float)
    tau_rec = np.full(n, np.nan, dtype=float)
    pmax = np.full(n, np.nan, dtype=float)

    unit = 1.0 / float(H)

    for i in range(n):
        entry_px = close[i]
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        invested = 0.0
        shares = 0.0

        # entry buy
        shares += unit / entry_px
        invested += unit

        sl_day: int | None = None
        tp_day: int | None = None

        last_main = min(n - 1, i + H)          # inclusive
        last_ext = min(n - 1, i + H + EX)      # inclusive

        # main window scan (i+1..i+H)
        for day in range(i + 1, last_main + 1):
            avg = (invested / shares) if (shares > 0 and invested > 0) else np.nan
            if not np.isfinite(avg) or avg <= 0:
                break

            tp_px = avg * pt_mult
            sl_px = avg * sl_mult

            hit_stop = np.isfinite(low[day]) and (low[day] <= sl_px)
            hit_profit = np.isfinite(high[day]) and (high[day] >= tp_px)

            if hit_stop and hit_profit:
                sl_day = day
                break
            if hit_stop:
                sl_day = day
                break
            if hit_profit:
                tp_day = day
                break

            # DCA buy (slot # = day-i+1)
            slot_k = (day - i) + 1
            if slot_k <= H:
                cpx = close[day]
                if np.isfinite(cpx) and cpx > 0:
                    mult = _dca_desired_multiplier(cpx, avg)
                    invest = unit * mult
                    if invest > 0:
                        shares += invest / cpx
                        invested += invest

        if sl_day is not None:
            tau_sl[i] = float(sl_day - i)
        if tp_day is not None:
            tau_tp[i] = float(tp_day - i)

        # compute TailPmaxEXmax: max close/avg -1 over [i+1..i+H+EX], with avg evolving until H then fixed
        # We'll re-simulate avg path day by day and track max
        invested2 = unit
        shares2 = unit / entry_px
        best = -np.inf
        for day in range(i + 1, last_ext + 1):
            avg2 = (invested2 / shares2) if (shares2 > 0 and invested2 > 0) else np.nan
            if np.isfinite(avg2) and avg2 > 0 and np.isfinite(close[day]) and close[day] > 0:
                best = max(best, float((close[day] / avg2) - 1.0))

            # DCA only within H slots
            slot_k = (day - i) + 1
            if slot_k <= H:
                cpx = close[day]
                if np.isfinite(cpx) and cpx > 0 and np.isfinite(avg2) and avg2 > 0:
                    mult = _dca_desired_multiplier(cpx, avg2)
                    invest = unit * mult
                    if invest > 0:
                        shares2 += invest / cpx
                        invested2 += invest

        if np.isfinite(best):
            pmax[i] = float(best)

        # recovery after SL: only when SL occurred first (or TP not within H before SL)
        if sl_day is not None:
            # If TP happened within H at/before SL, we treat as NOT tail path (no recovery label needed)
            if tp_day is not None and tp_day <= sl_day:
                continue

            # simulate avg up to sl_day (so recovery uses correct avg trajectory)
            invested3 = unit
            shares3 = unit / entry_px
            for day in range(i + 1, sl_day + 1):
                avg3 = (invested3 / shares3) if (shares3 > 0 and invested3 > 0) else np.nan
                slot_k = (day - i) + 1
                if slot_k <= H:
                    cpx = close[day]
                    if np.isfinite(cpx) and cpx > 0 and np.isfinite(avg3) and avg3 > 0:
                        mult = _dca_desired_multiplier(cpx, avg3)
                        invest = unit * mult
                        if invest > 0:
                            shares3 += invest / cpx
                            invested3 += invest

            # recovery scan from sl_day+1 .. last_ext
            for day in range(sl_day + 1, last_ext + 1):
                avg3 = (invested3 / shares3) if (shares3 > 0 and invested3 > 0) else np.nan
                if not (np.isfinite(avg3) and avg3 > 0):
                    break
                tp_px = avg3 * pt_mult
                if np.isfinite(high[day]) and high[day] >= tp_px:
                    tau_rec[i] = float(day - sl_day)
                    break

                # continue DCA until H slots end
                slot_k = (day - i) + 1
                if slot_k <= H:
                    cpx = close[day]
                    if np.isfinite(cpx) and cpx > 0:
                        mult = _dca_desired_multiplier(cpx, avg3)
                        invest = unit * mult
                        if invest > 0:
                            shares3 += invest / cpx
                            invested3 += invest

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(g["Date"], errors="coerce").dt.tz_localize(None).to_numpy(),
            "Ticker": g["Ticker"].astype(str).str.upper().str.strip().to_numpy(),
            "TauSLDays": tau_sl.astype(float),
            "TauTPDays": tau_tp.astype(float),
            "TauRecoverAfterSLDays": tau_rec.astype(float),
            "TailPmaxEXmax": pmax.astype(float),
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build tail label primitives (DCA avg_price based) aligned to features_model only.")
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--hmax", type=int, required=True, help="max holding days for primitive scan (e.g. 50)")
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--exmax", type=int, required=True, help="max extension days for recovery scan (e.g. 30)")

    ap.add_argument("--feature-cols", type=str, default="", help="(DISALLOWED) override; kept for compatibility")
    args = ap.parse_args()

    pt = float(args.profit_target)
    Hmax = int(args.hmax)
    sl = float(args.stop_level)
    Exmax = int(args.exmax)

    cols, src = resolve_feature_cols_forced(args.feature_cols)
    print(f"[INFO] SSOT feature cols source={src} cols={cols}")

    prices = read_prices()
    feats, feats_src = read_features_model()
    print(f"[INFO] features source={feats_src} rows={len(feats)}")

    pt100 = int(round(pt * 100))
    sl100 = int(round(abs(sl) * 100))
    tag = f"pt{pt100}_sl{sl100}_hmax{Hmax}_exmax{Exmax}"

    out_parq = LABEL_DIR / f"labels_tail_base_{tag}.parquet"
    out_csv = LABEL_DIR / f"labels_tail_base_{tag}.csv"
    meta_json = LABEL_DIR / f"labels_tail_base_{tag}_meta.json"

    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    prim_list: list[pd.DataFrame] = []
    for tkr, g in prices.groupby("Ticker", sort=False):
        prim_list.append(
            compute_tail_primitives_dca_for_ticker(
                g,
                profit_target=pt,
                hmax=Hmax,
                stop_level=sl,
                exmax=Exmax,
            )
        )

    prim = pd.concat(prim_list, ignore_index=True) if prim_list else pd.DataFrame()
    if prim.empty:
        raise RuntimeError("No tail primitives produced. Check prices input.")

    prim["Date"] = pd.to_datetime(prim["Date"], errors="coerce").dt.tz_localize(None)
    prim["Ticker"] = prim["Ticker"].astype(str).str.upper().str.strip()
    prim = prim.dropna(subset=["Date", "Ticker"]).reset_index(drop=True)

    # ✅ Align to features_model keys only
    merged = feats[["Date", "Ticker"]].merge(prim, on=["Date", "Ticker"], how="inner")
    if merged.empty:
        raise RuntimeError(
            "No overlap between features_model and computed tail primitives. "
            "Check universe tickers & date ranges."
        )

    out = merged[
        ["Date", "Ticker", "TauSLDays", "TauTPDays", "TauRecoverAfterSLDays", "TailPmaxEXmax"]
    ].copy()
    out = out.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last").reset_index(drop=True)

    try:
        out.to_parquet(out_parq, index=False)
        print(f"[DONE] wrote parquet: {out_parq} rows={len(out)}")
    except Exception as e:
        print(f"[WARN] parquet save failed ({e}) -> writing csv only")

    out.to_csv(out_csv, index=False)
    print(f"[DONE] wrote csv: {out_csv} rows={len(out)}")

    meta = {
        "tag": tag,
        "profit_target": pt,
        "hmax": Hmax,
        "stop_level": sl,
        "exmax": Exmax,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "ssot_feature_cols_source": src,
        "ssot_feature_cols": cols,
        "features_source_used": feats_src,
        "label_cols": ["TauSLDays", "TauTPDays", "TauRecoverAfterSLDays", "TailPmaxEXmax"],
        "notes": "DCA(avg_price) primitives. Build p_tail(H,EX) downstream from these taus.",
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[DONE] wrote meta: {meta_json}")


if __name__ == "__main__":
    main()