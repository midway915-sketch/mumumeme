#!/usr/bin/env bash
set -euo pipefail

# Runs ONE walkforward half-year slice.
# Responsibilities:
#  - Ensure data/raw/prices.* exists (workflow downloads raw-data artifact)
#  - Ensure data/universe.csv exists (auto-generate from prices if missing)
#  - Ensure data/features/features_scored.* exists (build_features + score_features)
#  - Run your grid runner into period OUT_DIR

: "${WF_PERIOD:?}"
: "${TRAIN_START:?}"
: "${TRAIN_END:?}"
: "${VALID_START:?}"
: "${VALID_END:?}"
: "${TEST_START:?}"
: "${TEST_END:?}"
: "${CUT_DATE:?}"

: "${OUT_DIR_BASE:=data/signals/walkforward}"
: "${WF_TAG_BASE:=wf}"

# existing envs (treat these as required)
: "${PROFIT_TARGET:?}"
: "${MAX_DAYS:?}"
: "${STOP_LEVEL:?}"
: "${P_TAIL_THRESHOLDS:?}"
: "${UTILITY_QUANTILES:?}"
: "${RANK_METRICS:?}"
: "${LAMBDA_TAIL:?}"
: "${GATE_MODES:?}"
: "${TRAIL_STOPS:?}"
: "${TP1_FRAC:?}"
: "${ENABLE_TRAILING:?}"
: "${TOPK_CONFIGS:?}"
: "${PS_MINS:?}"
: "${BADEXIT_MAXES:?}"
: "${MAX_LEVERAGE_PCT:?}"

# OPTIONAL (must NOT hard-fail if empty)
EXCLUDE_TICKERS="${EXCLUDE_TICKERS:-}"
REQUIRE_FILES="${REQUIRE_FILES:-}"

# optional envs (regime + tau/dca)
REGIME_MODE="${REGIME_MODE:-off}"
REGIME_DD_MAX="${REGIME_DD_MAX:-0.20}"
REGIME_RET20_MIN="${REGIME_RET20_MIN:-0.00}"
REGIME_ATR_MAX="${REGIME_ATR_MAX:-1.30}"
REGIME_LEVERAGE_MULT="${REGIME_LEVERAGE_MULT:-3.0}"

TAU_SPLIT="${TAU_SPLIT:-}"
USE_TAU_H="${USE_TAU_H:-false}"
ENABLE_DCA="${ENABLE_DCA:-false}"

# ---- period partitioning
export LABEL_KEY="${WF_TAG_BASE}_${WF_PERIOD}"
export OUT_DIR="${OUT_DIR_BASE}/${WF_PERIOD}"
mkdir -p "$OUT_DIR"

echo "[WF] period=$WF_PERIOD CUT_DATE=$CUT_DATE"
echo "[WF] TRAIN $TRAIN_START -> $TRAIN_END"
echo "[WF] VALID $VALID_START -> $VALID_END"
echo "[WF] TEST  $TEST_START -> $TEST_END"
echo "[WF] LABEL_KEY=$LABEL_KEY OUT_DIR=$OUT_DIR"
echo "[WF] EXCLUDE_TICKERS='${EXCLUDE_TICKERS}' REQUIRE_FILES='${REQUIRE_FILES}'"
echo "[WF] REGIME_MODE=$REGIME_MODE DD_MAX=$REGIME_DD_MAX RET20_MIN=$REGIME_RET20_MIN ATR_MAX=$REGIME_ATR_MAX LEV_MULT=$REGIME_LEVERAGE_MULT"
echo "[WF] TAU_SPLIT=${TAU_SPLIT:-<none>} USE_TAU_H=$USE_TAU_H ENABLE_DCA=$ENABLE_DCA"

# ---- 0) raw prices must exist (artifact download is done by workflow)
if [ ! -f data/raw/prices.parquet ] && [ ! -f data/raw/prices.csv ]; then
  echo "[ERROR] missing raw prices: data/raw/prices.parquet (or .csv). Run raw-data workflow first."
  exit 2
fi

# ---- 1) ensure universe.csv exists
if [ ! -f data/universe.csv ]; then
  echo "[INFO] data/universe.csv missing -> generating from raw prices"
  python - <<'PY'
from pathlib import Path
import pandas as pd

raw_parq = Path("data/raw/prices.parquet")
raw_csv  = Path("data/raw/prices.csv")

if raw_parq.exists():
    df = pd.read_parquet(raw_parq)
elif raw_csv.exists():
    df = pd.read_csv(raw_csv)
else:
    raise SystemExit("[ERROR] raw prices missing")

if "Ticker" not in df.columns:
    raise SystemExit("[ERROR] raw prices missing Ticker column")

ticks = (
    df["Ticker"].astype(str).str.upper().str.strip()
      .dropna().unique().tolist()
)
ticks = [t for t in ticks if t]
ticks = sorted(set(ticks))

out = pd.DataFrame({"Ticker": ticks, "Enabled": True})
Path("data").mkdir(parents=True, exist_ok=True)
out.to_csv("data/universe.csv", index=False)
print(f"[DONE] wrote data/universe.csv tickers={len(out)}")
PY
fi

# ---- 2) ensure features_scored exists (build_features -> score_features)
if [ ! -f data/features/features_scored.parquet ] && [ ! -f data/features/features_scored.csv ]; then
  echo "[INFO] features_scored missing -> build_features + score_features (CUT_DATE=$CUT_DATE)"

  # build_features does NOT accept --cut-date (your log proved it)
  # So we just build features from raw; later pick generation is filtered by test range.
  python scripts/build_features.py

  # score_features needs a tag (uses it to locate model reports/SSOT).
  # Here we use LABEL_KEY so you get per-slice audit naming.
  python scripts/score_features.py --tag "$LABEL_KEY"
fi

# ---- 3) run grid (and restrict picks to TEST range via env that run_grid_workflow forwards)
export WF_DATE_FROM="$TEST_START"
export WF_DATE_TO="$TEST_END"

chmod +x scripts/run_grid_workflow.sh
./scripts/run_grid_workflow.sh