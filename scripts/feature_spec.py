#!/usr/bin/env python3
# scripts/feature_spec.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

# SSOT location (build_features.py / train_* / score_features.py가 공유)
DATA_DIR = Path("data")
META_DIR = DATA_DIR / "meta"
FEATURE_COLS_JSON = META_DIR / "feature_cols.json"


def get_feature_cols(sector_enabled: bool = False) -> List[str]:
    """
    SSOT default feature columns.
    - base: 16
    - sector: +2 (Sector_Ret_20, RelStrength)
    """
    base = [
        "Drawdown_252",
        "Drawdown_60",
        "ATR_ratio",
        "Z_score",
        "MACD_hist",
        "MA20_slope",
        "Market_Drawdown",
        "Market_ATR_ratio",
        "ret_score",
        "ret_5",
        "ret_10",
        "ret_20",
        "breakout_20",
        "vol_surge",
        "trend_align",
        "beta_60",
    ]
    if sector_enabled:
        base += ["Sector_Ret_20", "RelStrength"]
    return base


def write_feature_cols_meta(cols: List[str], sector_enabled: bool) -> str:
    """
    Write SSOT meta file: data/meta/feature_cols.json
    Format:
      {
        "feature_cols": [...],
        "sector_enabled": true/false
      }
    Returns: path string
    """
    META_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "feature_cols": [str(c).strip() for c in (cols or []) if str(c).strip()],
        "sector_enabled": bool(sector_enabled),
    }

    FEATURE_COLS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(FEATURE_COLS_JSON)


def read_feature_cols_meta() -> Tuple[List[str], bool]:
    """
    Read SSOT meta file.
    Returns: (feature_cols, sector_enabled)
    If missing/invalid -> ([], False)
    """
    if not FEATURE_COLS_JSON.exists():
        return ([], False)

    try:
        j = json.loads(FEATURE_COLS_JSON.read_text(encoding="utf-8"))
        cols = j.get("feature_cols", [])
        sector_enabled = bool(j.get("sector_enabled", False))
        if isinstance(cols, list) and cols:
            cols_out = [str(c).strip() for c in cols if str(c).strip()]
            return (cols_out, sector_enabled)
    except Exception:
        pass

    return ([], False)