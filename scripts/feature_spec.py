# scripts/feature_spec.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any


DATA_DIR = Path("data")
META_DIR = DATA_DIR / "meta"
FEATURE_COLS_JSON = META_DIR / "feature_cols.json"


# ✅ 이 리스트가 "단일 진실(SSOT)"이다.
BASE_COLS: List[str] = [
    "Drawdown_252",
    "Drawdown_60",
    "ATR_ratio",
    "Z_score",
    "MACD_hist",
    "MA20_slope",
    "Market_Drawdown",
    "Market_ATR_ratio",
    "ret_score",
]

NEW_CORE_COLS: List[str] = [
    "ret_5",
    "ret_10",
    "ret_20",
    "breakout_20",
    "vol_surge",
    "trend_align",
    "beta_60",
]

SECTOR_COLS: List[str] = [
    "Sector_Ret_20",
    "RelStrength",
]


def get_feature_cols(sector_enabled: bool) -> List[str]:
    cols = BASE_COLS + NEW_CORE_COLS
    if sector_enabled:
        cols += SECTOR_COLS
    return cols


def write_feature_cols_meta(cols: List[str], sector_enabled: bool) -> str:
    META_DIR.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "feature_cols": cols,
        "sector_enabled": bool(sector_enabled),
    }
    FEATURE_COLS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(FEATURE_COLS_JSON)


def read_feature_cols_meta() -> tuple[List[str], bool]:
    """
    Returns (cols, sector_enabled).
    If meta missing -> returns ([], False).
    """
    if not FEATURE_COLS_JSON.exists():
        return [], False
    try:
        obj = json.loads(FEATURE_COLS_JSON.read_text(encoding="utf-8"))
        cols = [str(x) for x in obj.get("feature_cols", []) if str(x).strip()]
        sector_enabled = bool(obj.get("sector_enabled", False))
        return cols, sector_enabled
    except Exception:
        return [], False