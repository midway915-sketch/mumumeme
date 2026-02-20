# scripts/build_tail_labels.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scripts.feature_spec import read_feature_cols_meta, get_feature_cols


DATA_DIR = Path("data")
LABEL_DIR = DATA_DIR / "labels"
META_DIR = DATA_DIR / "meta"


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


def coerce_float_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build tail labels (labels_tail_{tag}) from strategy_raw_data_{tag}.")
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument(
        "--feature-cols",
        type=str,
        default="",
        help="콤마로 feature 컬럼 지정(비우면 meta/SSOT 사용)",
    )
    args = ap.parse_args()

    tag = fmt_tag(float(args.profit_target), int(args.max_days), float(args.stop_level), int(args.max_extend_days))

    # input: strategy_raw_data_{tag}
    in_parq = LABEL_DIR / f"strategy_raw_data_{tag}.parquet"
    in_csv = LABEL_DIR / f"strategy_raw_data_{tag}.csv"
    df = read_table(in_parq, in_csv).copy()
    src = str(in_parq if in_parq.exists() else in_csv)

    # dates
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)

    # feature cols policy:
    # 1) args.feature_cols > 2) data/meta/feature_cols.json > 3) SSOT default (sector off)
    if str(args.feature_cols).strip():
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
        sector_enabled = None
    else:
        cols_meta, sector_enabled = read_feature_cols_meta()
        if cols_meta:
            feature_cols = cols_meta
        else:
            feature_cols = get_feature_cols(sector_enabled=False)

    # required columns
    required_base = ["Date", "Ticker"]
    missing_base = [c for c in required_base if c not in df.columns]
    if missing_base:
        raise ValueError(f"strategy_raw_data missing required columns: {missing_base} (src={src})")

    # Tail -> TailTarget (A안 핵심)
    if "TailTarget" not in df.columns:
        if "Tail" not in df.columns:
            raise ValueError(f"strategy_raw_data missing Tail column (need Tail -> TailTarget). src={src}")
        df["TailTarget"] = pd.to_numeric(df["Tail"], errors="coerce").fillna(0).astype(int)
    else:
        df["TailTarget"] = pd.to_numeric(df["TailTarget"], errors="coerce").fillna(0).astype(int)

    # make sure features exist
    missing_feat = [c for c in feature_cols if c not in df.columns]
    if missing_feat:
        raise ValueError(f"strategy_raw_data missing feature columns: {missing_feat} (src={src})")

    # clean numeric features
    df = coerce_float_cols(df, feature_cols)
    df = df.dropna(subset=required_base + feature_cols).copy()

    # output: labels_tail_{tag}
    out_parq = LABEL_DIR / f"labels_tail_{tag}.parquet"
    out_csv = LABEL_DIR / f"labels_tail_{tag}.csv"

    saved_df = df[required_base + feature_cols + ["TailTarget"]].copy()
    saved_to = save_table(saved_df, out_parq, out_csv)

    # optional: friendly fixed csv under labels (helps debugging)
    also_csv = LABEL_DIR / "labels_tail.csv"
    try:
        saved_df.to_csv(also_csv, index=False)
    except Exception:
        pass

    # meta
    META_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "updated_at_utc": now_utc_iso(),
        "tag": tag,
        "labels_src": src,
        "rows": int(len(saved_df)),
        "tail_rate": float(saved_df["TailTarget"].mean()) if len(saved_df) else None,
        "feature_cols": feature_cols,
        "sector_enabled": sector_enabled,
        "saved_to": saved_to,
        "also_written": str(also_csv),
    }
    (LABEL_DIR / f"labels_tail_{tag}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] wrote: {saved_to} rows={len(saved_df)}")
    print(f"[DONE] also wrote -> {also_csv}")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()