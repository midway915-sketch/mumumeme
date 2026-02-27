#!/usr/bin/env python3
# scripts/diagnose_distributions.py
"""Distribution diagnostics for walkforward picks.

Goal:
  - Compare picked rows vs universe rows over the same date window
  - Surface degenerate situations (e.g., p_success ~ 0, p_tail extreme, p_badexit filter too tight)

Outputs:
  - JSON: rich summary (counts, quantiles, histograms)
  - CSV: long-form table metric/stat/value (easy to diff across runs)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _read_any(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"missing {parq} (or {csv})")


def _quantiles(x: pd.Series, qs: List[float]) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {f"q{int(q*100):02d}": float("nan") for q in qs}
    out = {}
    for q in qs:
        out[f"q{int(q*100):02d}"] = float(x.quantile(q))
    return out


def _hist_01(x: pd.Series, bins: int = 20) -> Dict[str, Any]:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {"bins": [], "counts": []}
    lo, hi = 0.0, 1.0
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(np.clip(x.values.astype(float), lo, hi), bins=edges)
    return {"bins": [float(e) for e in edges.tolist()], "counts": [int(c) for c in counts.tolist()]}


def _basic_stats(x: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) >= 2 else 0.0,
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _summarize_block(df: pd.DataFrame, label: str, cols: List[str]) -> Dict[str, Any]:
    qs = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    out: Dict[str, Any] = {"label": label, "n_rows": int(len(df))}
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        block = {"basic": _basic_stats(s), "quantiles": _quantiles(s, qs)}
        if c in ("p_success", "p_tail", "p_badexit"):
            block["hist01"] = _hist_01(s, bins=20)
        out[c] = block
    # tau_H distribution
    if "tau_H" in df.columns:
        t = pd.to_numeric(df["tau_H"], errors="coerce").dropna()
        if not t.empty:
            vc = t.astype(int).value_counts().sort_index()
            out["tau_H_counts"] = {str(int(k)): int(v) for k, v in vc.items()}
    if "tau_class" in df.columns:
        t = pd.to_numeric(df["tau_class"], errors="coerce").dropna()
        if not t.empty:
            vc = t.astype(int).value_counts().sort_index()
            out["tau_class_counts"] = {str(int(k)): int(v) for k, v in vc.items()}
    return out


def _longform(stats: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    label = stats.get("label", "")

    # global
    rows.append({"block": prefix, "label": label, "metric": "n_rows", "stat": "value", "value": stats.get("n_rows", 0)})

    for k, v in stats.items():
        if k in ("label", "n_rows", "tau_H_counts", "tau_class_counts"):
            continue
        if not isinstance(v, dict):
            continue

        basic = v.get("basic", {})
        for bstat, bval in basic.items():
            rows.append({"block": prefix, "label": label, "metric": k, "stat": f"basic_{bstat}", "value": bval})

        qd = v.get("quantiles", {})
        for qk, qv in qd.items():
            rows.append({"block": prefix, "label": label, "metric": k, "stat": qk, "value": qv})

    # tau counts as separate rows
    if "tau_H_counts" in stats:
        for hh, cc in stats["tau_H_counts"].items():
            rows.append({"block": prefix, "label": label, "metric": "tau_H", "stat": f"count_{hh}", "value": cc})
    if "tau_class_counts" in stats:
        for hh, cc in stats["tau_class_counts"].items():
            rows.append({"block": prefix, "label": label, "metric": "tau_class", "stat": f"count_{hh}", "value": cc})

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Distribution diagnostics for picks vs universe")
    ap.add_argument("--picks-path", required=True, type=str)
    ap.add_argument("--features-parq", default="data/features/features_scored.parquet", type=str)
    ap.add_argument("--features-csv", default="data/features/features_scored.csv", type=str)
    ap.add_argument("--features-path", default="", type=str, help="Optional explicit features path (.parquet/.csv)")
    ap.add_argument("--date-from", default="", type=str)
    ap.add_argument("--date-to", default="", type=str)
    ap.add_argument("--out-json", required=True, type=str)
    ap.add_argument("--out-csv", required=True, type=str)
    args = ap.parse_args()

    picks_path = Path(args.picks_path)
    if not picks_path.exists():
        raise SystemExit(f"Missing picks: {picks_path}")

    picks = pd.read_csv(picks_path)
    if "Date" not in picks.columns or "Ticker" not in picks.columns:
        raise SystemExit("picks must have Date,Ticker")

    picks = picks.copy()
    picks["Date"] = _norm_date(picks["Date"])
    picks["Ticker"] = picks["Ticker"].astype(str).str.upper().str.strip()
    picks = picks.dropna(subset=["Date", "Ticker"]).drop_duplicates(subset=["Date", "Ticker"]).reset_index(drop=True)

    # Date window
    d_from = pd.to_datetime(args.date_from, errors="coerce") if str(args.date_from).strip() else None
    d_to = pd.to_datetime(args.date_to, errors="coerce") if str(args.date_to).strip() else None
    if d_from is None:
        d_from = picks["Date"].min() if not picks.empty else None
    if d_to is None:
        d_to = picks["Date"].max() if not picks.empty else None

    # Features
    if args.features_path:
        fp = Path(args.features_path)
        if not fp.exists():
            raise SystemExit(f"--features-path not found: {fp}")
        feats = pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
    else:
        feats = _read_any(Path(args.features_parq), Path(args.features_csv))

    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise SystemExit("features_scored must have Date,Ticker")

    feats = feats.copy()
    feats["Date"] = _norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).reset_index(drop=True)

    if d_from is not None and pd.notna(d_from):
        feats = feats[feats["Date"] >= d_from]
    if d_to is not None and pd.notna(d_to):
        feats = feats[feats["Date"] <= d_to]
        picks = picks[picks["Date"] <= d_to]
    if d_from is not None and pd.notna(d_from):
        picks = picks[picks["Date"] >= d_from]

    # Minimal feature columns we care about (if present)
    want = ["Date", "Ticker", "p_success", "p_tail", "p_badexit", "tau_H", "tau_class", "ret_score", "utility"]
    keep = [c for c in want if c in feats.columns]
    feats = feats[keep].drop_duplicates(subset=["Date", "Ticker"]).reset_index(drop=True)

    picked = picks.merge(feats, on=["Date", "Ticker"], how="left")

    cols = [c for c in ["p_success", "p_tail", "p_badexit", "ret_score", "utility"] if c in feats.columns]
    # add tau columns if present
    if "tau_H" in feats.columns:
        cols.append("tau_H")
    if "tau_class" in feats.columns:
        cols.append("tau_class")

    uni_stats = _summarize_block(feats, label="universe", cols=cols)
    pick_stats = _summarize_block(picked, label="picked", cols=cols)

    out: Dict[str, Any] = {
        "meta": {
            "picks_path": str(picks_path),
            "features_path": str(Path(args.features_path)) if args.features_path else "(features_parq/csv default)",
            "date_from": str(d_from.date()) if d_from is not None and pd.notna(d_from) else None,
            "date_to": str(d_to.date()) if d_to is not None and pd.notna(d_to) else None,
        },
        "universe": uni_stats,
        "picked": pick_stats,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    rows += _longform(uni_stats, prefix="universe")
    rows += _longform(pick_stats, prefix="picked")
    df_out = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    print(f"[DONE] wrote {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
