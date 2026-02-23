#!/usr/bin/env bash
set -euo pipefail

# This script runs ONE walkforward half-year slice.
# It sets period-specific TAG/OUT_DIR and then calls your existing grid runner.

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

# your existing required envs (same as run_grid_workflow.sh)
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

# ✅ OPTIONAL envs (set -u safe)
: "${EXCLUDE_TICKERS:=}"
: "${REQUIRE_FILES:=}"

# optional envs (regime + tau/dca)
: "${REGIME_MODE:=off}"
: "${REGIME_DD_MAX:=0.20}"
: "${REGIME_RET20_MIN:=0.00}"
: "${REGIME_ATR_MAX:=1.30}"
: "${REGIME_LEVERAGE_MULT:=3.0}"

: "${TAU_SPLIT:=}"
: "${USE_TAU_H:=false}"
: "${ENABLE_DCA:=false}"

# ---- period partitioning
export LABEL_KEY="${WF_TAG_BASE}_${WF_PERIOD}"
export OUT_DIR="${OUT_DIR_BASE}/${WF_PERIOD}"
mkdir -p "$OUT_DIR"

echo "[WF] period=$WF_PERIOD CUT_DATE=$CUT_DATE"
echo "[WF] LABEL_KEY=$LABEL_KEY OUT_DIR=$OUT_DIR"
echo "[WF] EXCLUDE_TICKERS='${EXCLUDE_TICKERS}' REQUIRE_FILES='${REQUIRE_FILES}'"
echo "[WF] REGIME_MODE=$REGIME_MODE"
echo "[WF] TAU_SPLIT=${TAU_SPLIT:-<none>} USE_TAU_H=$USE_TAU_H ENABLE_DCA=$ENABLE_DCA"

# ---- WF-lite: if features_scored missing, build+score once (no --cut-date; build_features.py doesn't support it)
FEATURES_PARQ="data/features/features_scored.parquet"
FEATURES_CSV="data/features/features_scored.csv"

if [ ! -f "$FEATURES_PARQ" ] && [ ! -f "$FEATURES_CSV" ]; then
  echo "[INFO] features_scored missing -> build_features + score_features (start-date=$TRAIN_START, CUT_DATE=$CUT_DATE ignored by builder)"

  # build_features.py supports --start-date (per your usage output)
  python scripts/build_features.py --start-date "$TRAIN_START"

  # score_features.py creates features_scored.(parquet/csv)
  # tag은 최소한 LABEL_KEY와 독립적으로 아무거나 넣어도 되는데, 통일 위해 WF tag 사용
  python scripts/score_features.py --tag "$LABEL_KEY"
fi

# ---- Run your existing grid runner (already env-driven)
chmod +x scripts/run_grid_workflow.sh
./scripts/run_grid_workflow.sh