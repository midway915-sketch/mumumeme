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
# SSOT feature cols resolution (forced 18 cols)
# ------------------------------------------------------------
def _validate_18_cols(cols: list[str], src: str) -> None:
    if len(cols) != 18:
        raise RuntimeError(f"SSOT feature cols must be 18, got {len(cols)} from {src}: {cols}")
    need = ["RelStrength"]
    miss = [c for c in need if c not in cols]
    if miss:
        raise RuntimeError(f"SSOT feature cols must include {need}. Missing={miss}. src={src}")


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
# DCA helper (avg price)
# ------------------------------------------------------------
def _dca_desired_multiplier(close_px: float, avg_px: float) -> float:
    """
    Engine-aligned simplified DCA rule:
      - if close <= avg: full buy
      - elif close <= avg*1.05: half buy
      - else: 0
    """
    if not np.isfinite(close_px) or close_px <= 0:
        return 0.0
    if not np.isfinite(avg_px) or avg_px <= 0:
        return 1.0
    if close_px <= avg_px:
        return 1.0
    if close_px <= avg_px * 1.05:
        return 0.5
    return 0.0


# ------------------------------------------------------------
# (A) legacy p_tail label (for training tail_model) - avg 기반
# ------------------------------------------------------------
def compute_p_tail_label_dca(
    g: pd.DataFrame,
    profit_target: float,
    max_days: int,
    stop_level: float,
    max_extend_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Legacy tail label (kept for train_tail_model compatibility):
      - If SL hit first within H, and then TP is reached within extension window => p_tail=1 else 0
    Also returns tail_pmax: max (close/avg -1) within (H+EX) window (diagnostic)

    ✅ avg_price(DCA) based:
      - entry day buy + DCA within H slots (unit=1/H normalized)
      - SL/TP check uses intraday High/Low vs current avg thresholds
      - same-day SL&TP => STOP first (conservative)
    """
    g = g.sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(g["Close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(g["High"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(g["Low"], errors="coerce").to_numpy(dtype=float)

    n = len(g)
    y_tail = np.zeros(n, dtype=int)
    y_pmax = np.full(n, np.nan, dtype=float)

    H = int(max(1, max_days))
    EX = int(max(0, max_extend_days))
    unit = 1.0 / float(H)

    pt_mult = 1.0 + float(profit_target)
    sl_mult = 1.0 + float(stop_level)

    for i in range(n):
        entry_px = close[i]
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        invested = unit
        shares = unit / entry_px

        end_main = min(n - 1, i + H)
        end_ext = min(n - 1, i + H + EX)

        tp_hit: int | None = None
        sl_hit: int | None = None

        # main scan
        for day in range(i + 1, end_main + 1):
            avg = invested / shares if (shares > 0 and invested > 0) else np.nan
            if not np.isfinite(avg) or avg <= 0:
                break

            tp_px = avg * pt_mult
            sl_px = avg * sl_mult

            hit_stop = np.isfinite(low[day]) and low[day] <= sl_px
            hit_profit = np.isfinite(high[day]) and high[day] >= tp_px

            if hit_stop and hit_profit:
                sl_hit = day
                break
            if hit_stop:
                sl_hit = day
                break
            if hit_profit:
                tp_hit = day
                break

            # DCA buy
            slot_k = (day - i) + 1
            if slot_k <= H:
                cpx = close[day]
                if np.isfinite(cpx) and cpx > 0:
                    mult = _dca_desired_multiplier(cpx, avg)
                    invest = unit * mult
                    if invest > 0:
                        invested += invest
                        shares += invest / cpx

        # tail_pmax over ext window
        invested2 = unit
        shares2 = unit / entry_px
        best = -np.inf
        for day in range(i + 1, end_ext + 1):
            avg2 = invested2 / shares2 if (shares2 > 0 and invested2 > 0) else np.nan
            if np.isfinite(avg2) and avg2 > 0 and np.isfinite(close[day]) and close[day] > 0:
                best = max(best, float((close[day] / avg2) - 1.0))

            slot_k = (day - i) + 1
            if slot_k <= H:
                cpx = close[day]
                if np.isfinite(cpx) and cpx > 0 and np.isfinite(avg2) and avg2 > 0:
                    mult = _dca_desired_multiplier(cpx, avg2)
                    invest = unit * mult
                    if invest > 0:
                        invested2 += invest
                        shares2 += invest / cpx

        if np.isfinite(best):
            y_pmax[i] = float(best)

        # decide tail
        if tp_hit is None and sl_hit is None:
            y_tail[i] = 0
        elif tp_hit is not None and (sl_hit is None or tp_hit <= sl_hit):
            y_tail[i] = 0
        else:
            # SL first -> recovery to TP within extension
            recovered = False

            # rebuild avg state through sl_hit
            invested3 = unit
            shares3 = unit / entry_px
            for day in range(i + 1, (sl_hit or end_main) + 1):
                avg3 = invested3 / shares3 if (shares3 > 0 and invested3 > 0) else np.nan
                slot_k = (day - i) + 1
                if slot_k <= H and np.isfinite(avg3) and avg3 > 0:
                    cpx = close[day]
                    if np.isfinite(cpx) and cpx > 0:
                        mult = _dca_desired_multiplier(cpx, avg3)
                        invest = unit * mult
                        if invest > 0:
                            invested3 += invest
                            shares3 += invest / cpx

            start = (sl_hit + 1) if sl_hit is not None else (end_main + 1)
            if start <= end_ext:
                for day in range(start, end_ext + 1):
                    avg3 = invested3 / shares3 if (shares3 > 0 and invested3 > 0) else np.nan
                    if not (np.isfinite(avg3) and avg3 > 0):
                        break
                    tp_px = avg3 * pt_mult
                    if np.isfinite(high[day]) and high[day] >= tp_px:
                        recovered = True
                        break

                    slot_k = (day - i) + 1
                    if slot_k <= H:
                        cpx = close[day]
                        if np.isfinite(cpx) and cpx > 0:
                            mult = _dca_desired_multiplier(cpx, avg3)
                            invest = unit * mult
                            if invest > 0:
                                invested3 += invest
                                shares3 += invest / cpx

            y_tail[i] = 1 if recovered else 0

    return y_tail, y_pmax


# ------------------------------------------------------------
# (B) tail primitives (for gate derivation) - avg 기반
# ------------------------------------------------------------
def compute_tail_primitives_dca(
    g: pd.DataFrame,
    profit_target: float,
    hmax: int,
    stop_level: float,
    exmax: int,
) -> pd.DataFrame:
    """
    Primitives for downstream p_tail(H,EX) derivation:
      TauSLDays               : first SL hit day within Hmax (1..Hmax)
      TauTPDays               : first TP hit day within Hmax (1..Hmax)
      TauRecoverAfterSLDays   : after SL (from next day), first TP hit within EXmax (1..EXmax)
      TailPmaxEXmax           : max(close/avg -1) in window [1..Hmax+EXmax]
    """
    g = g.sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(g["Close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(g["High"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(g["Low"], errors="coerce").to_numpy(dtype=float)

    n = len(g)
    H = int(max(1, hmax))
    EX = int(max(0, exmax))
    unit = 1.0 / float(H)

    pt_mult = 1.0 + float(profit_target)
    sl_mult = 1.0 + float(stop_level)

    tau_sl = np.full(n, np.nan, dtype=float)
    tau_tp = np.full(n, np.nan, dtype=float)
    tau_rec = np.full(n, np.nan, dtype=float)
    pmax = np.full(n, np.nan, dtype=float)

    for i in range(n):
        entry_px = close[i]
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        invested = unit
        shares = unit / entry_px

        last_main = min(n - 1, i + H)
        last_ext = min(n - 1, i + H + EX)

        sl_day: int | None = None
        tp_day: int | None = None

        # main scan
        for day in range(i + 1, last_main + 1):
            avg = invested / shares if (shares > 0 and invested > 0) else np.nan
            if not np.isfinite(avg) or avg <= 0:
                break

            tp_px = avg * pt_mult
            sl_px = avg * sl_mult

            hit_stop = np.isfinite(low[day]) and low[day] <= sl_px
            hit_profit = np.isfinite(high[day]) and high[day] >= tp_px

            if hit_stop and hit_profit:
                sl_day = day
                break
            if hit_stop:
                sl_day = day
                break
            if hit_profit:
                tp_day = day
                break

            # DCA buy
            slot_k = (day - i) + 1
            if slot_k <= H:
                cpx = close[day]
                if np.isfinite(cpx) and cpx > 0:
                    mult = _dca_desired_multiplier(cpx, avg)
                    invest = unit * mult
                    if invest > 0:
                        invested += invest
                        shares += invest / cpx

        if sl_day is not None:
            tau_sl[i] = float(sl_day - i)
        if tp_day is not None:
            tau_tp[i] = float(tp_day - i)

        # pmax over ext window
        invested2 = unit
        shares2 = unit / entry_px
        best = -np.inf
        for day in range(i + 1, last_ext + 1):
            avg2 = invested2 / shares2 if (shares2 > 0 and invested2 > 0) else np.nan
            if np.isfinite(avg2) and avg2 > 0 and np.isfinite(close[day]) and close[day] > 0:
                best = max(best, float((close[day] / avg2) - 1.0))

            slot_k = (day - i) + 1
            if slot_k <= H:
                cpx = close[day]
                if np.isfinite(cpx) and cpx > 0 and np.isfinite(avg2) and avg2 > 0:
                    mult = _dca_desired_multiplier(cpx, avg2)
                    invest = unit * mult
                    if invest > 0:
                        invested2 += invest
                        shares2 += invest / cpx

        if np.isfinite(best):
            pmax[i] = float(best)

        # recovery after SL (only if SL exists and TP not before/equal SL in main)
        if sl_day is not None and not (tp_day is not None and tp_day <= sl_day):
            # rebuild avg state through sl_day
            invested3 = unit
            shares3 = unit / entry_px
            for day in range(i + 1, sl_day + 1):
                avg3 = invested3 / shares3 if (shares3 > 0 and invested3 > 0) else np.nan
                slot_k = (day - i) + 1
                if slot_k <= H and np.isfinite(avg3) and avg3 > 0:
                    cpx = close[day]
                    if np.isfinite(cpx) and cpx > 0:
                        mult = _dca_desired_multiplier(cpx, avg3)
                        invest = unit * mult
                        if invest > 0:
                            invested3 += invest
                            shares3 += invest / cpx

            # scan sl_day+1 .. last_ext
            for day in range(sl_day + 1, last_ext + 1):
                avg3 = invested3 / shares3 if (shares3 > 0 and invested3 > 0) else np.nan
                if not (np.isfinite(avg3) and avg3 > 0):
                    break
                tp_px = avg3 * pt_mult
                if np.isfinite(high[day]) and high[day] >= tp_px:
                    tau_rec[i] = float(day - sl_day)
                    break

                slot_k = (day - i) + 1
                if slot_k <= H:
                    cpx = close[day]
                    if np.isfinite(cpx) and cpx > 0:
                        mult = _dca_desired_multiplier(cpx, avg3)
                        invest = unit * mult
                        if invest > 0:
                            invested3 += invest
                            shares3 += invest / cpx

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
    ap = argparse.ArgumentParser(description="Build tail labels (legacy) + tail primitives (for gate derivation).")

    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    # primitives window (optional; default: hmax=max(max-days, 50), exmax=max-extend-days)
    ap.add_argument("--hmax", type=int, default=0)
    ap.add_argument("--exmax", type=int, default=0)

    ap.add_argument("--feature-cols", type=str, default="", help="(DISALLOWED) override; kept for compatibility")
    args = ap.parse_args()

    pt = float(args.profit_target)
    H = int(args.max_days)
    sl = float(args.stop_level)
    ex = int(args.max_extend_days)

    # primitives range
    Hmax = int(args.hmax) if int(args.hmax) > 0 else int(max(H, 50))
    Exmax = int(args.exmax) if int(args.exmax) > 0 else int(ex)

    cols, src = resolve_feature_cols_forced(args.feature_cols)
    print(f"[INFO] SSOT feature cols source={src} cols={cols}")

    prices = read_prices()
    feats, feats_src = read_features_model()
    print(f"[INFO] features source={feats_src} rows={len(feats)}")

    pt100 = int(round(pt * 100))
    sl100 = int(round(abs(sl) * 100))
    tag = f"pt{pt100}_h{H}_sl{sl100}_ex{ex}"
    base_tag = f"pt{pt100}_sl{sl100}_hmax{Hmax}_exmax{Exmax}"

    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # (1) legacy labels_tail_{tag} (for training tail_model)
    # -------------------------
    out_parq = LABEL_DIR / f"labels_tail_{tag}.parquet"
    out_csv = LABEL_DIR / f"labels_tail_{tag}.csv"
    meta_json = LABEL_DIR / f"labels_tail_{tag}_meta.json"

    if (not out_parq.exists()) and (not out_csv.exists()):
        y_list = []
        pmax_list = []
        key_list = []

        for tkr, g in prices.groupby("Ticker", sort=False):
            y_tail, y_pmax = compute_p_tail_label_dca(
                g,
                profit_target=pt,
                max_days=H,
                stop_level=sl,
                max_extend_days=ex,
            )
            dts = pd.to_datetime(g["Date"], errors="coerce").dt.tz_localize(None).to_numpy()
            tkrs = np.array([tkr] * len(g))
            key_list.append(pd.DataFrame({"Date": dts, "Ticker": tkrs}))
            y_list.append(pd.Series(y_tail))
            pmax_list.append(pd.Series(y_pmax))

        keys = pd.concat(key_list, ignore_index=True)
        y_tail_all = pd.concat(y_list, ignore_index=True).to_numpy(dtype=int)
        y_pmax_all = pd.concat(pmax_list, ignore_index=True).to_numpy(dtype=float)

        labels = keys.copy()
        labels["p_tail"] = y_tail_all
        labels["tail_pmax"] = y_pmax_all

        labels["Date"] = pd.to_datetime(labels["Date"], errors="coerce").dt.tz_localize(None)
        labels["Ticker"] = labels["Ticker"].astype(str).str.upper().str.strip()
        labels = labels.dropna(subset=["Date", "Ticker"]).reset_index(drop=True)

        merged = feats[["Date", "Ticker"]].merge(labels, on=["Date", "Ticker"], how="inner")
        if merged.empty:
            raise RuntimeError("No overlap between features_model and computed tail labels.")

        out = merged[["Date", "Ticker", "p_tail", "tail_pmax"]].copy()
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
            "max_days": H,
            "stop_level": sl,
            "max_extend_days": ex,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "ssot_feature_cols_source": src,
            "ssot_feature_cols": cols,
            "features_source_used": feats_src,
            "label_cols": ["p_tail", "tail_pmax"],
            "notes": "legacy tail labels for training. avg_price(DCA) based.",
        }
        meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] wrote meta: {meta_json}")
    else:
        print(f"[INFO] labels_tail_{tag} exists -> reuse")

    # -------------------------
    # (2) primitives labels_tail_base_{base_tag} (for gate derivation)
    # -------------------------
    base_parq = LABEL_DIR / f"labels_tail_base_{base_tag}.parquet"
    base_csv = LABEL_DIR / f"labels_tail_base_{base_tag}.csv"
    base_meta = LABEL_DIR / f"labels_tail_base_{base_tag}_meta.json"

    if (not base_parq.exists()) and (not base_csv.exists()):
        prim_list: list[pd.DataFrame] = []
        for tkr, g in prices.groupby("Ticker", sort=False):
            prim_list.append(
                compute_tail_primitives_dca(
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

        merged = feats[["Date", "Ticker"]].merge(prim, on=["Date", "Ticker"], how="inner")
        if merged.empty:
            raise RuntimeError("No overlap between features_model and computed tail primitives.")

        outb = merged[
            ["Date", "Ticker", "TauSLDays", "TauTPDays", "TauRecoverAfterSLDays", "TailPmaxEXmax"]
        ].copy()
        outb = outb.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last").reset_index(drop=True)

        try:
            outb.to_parquet(base_parq, index=False)
            print(f"[DONE] wrote parquet: {base_parq} rows={len(outb)}")
        except Exception as e:
            print(f"[WARN] parquet save failed ({e}) -> writing csv only")

        outb.to_csv(base_csv, index=False)
        print(f"[DONE] wrote csv: {base_csv} rows={len(outb)}")

        meta = {
            "tag": base_tag,
            "profit_target": pt,
            "hmax": Hmax,
            "stop_level": sl,
            "exmax": Exmax,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "ssot_feature_cols_source": src,
            "ssot_feature_cols": cols,
            "features_source_used": feats_src,
            "label_cols": ["TauSLDays", "TauTPDays", "TauRecoverAfterSLDays", "TailPmaxEXmax"],
            "notes": "tail primitives for downstream p_tail(H=tau_H, EX) derivation. avg_price(DCA) based.",
        }
        base_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] wrote meta: {base_meta}")
    else:
        print(f"[INFO] labels_tail_base_{base_tag} exists -> reuse")


if __name__ == "__main__":
    main()