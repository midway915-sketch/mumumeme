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

# ✅ optional-but-allowed-empty
EXCLUDE_TICKERS="${EXCLUDE_TICKERS:-}"
export EXCLUDE_TICKERS
REQUIRE_FILES="${REQUIRE_FILES:-}"
export REQUIRE_FILES

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

echo "[WF] period=$WF_PERIOD"
echo "[WF] TRAIN $TRAIN_START -> $TRAIN_END"
echo "[WF] VALID $VALID_START -> $VALID_END"
echo "[WF] TEST  $TEST_START -> $TEST_END"
echo "[WF] CUT_DATE=$CUT_DATE"
echo "[WF] LABEL_KEY=$LABEL_KEY"
echo "[WF] OUT_DIR=$OUT_DIR"
echo "[WF] EXCLUDE_TICKERS=${EXCLUDE_TICKERS:-<empty>}"
echo "[WF] TAU_SPLIT=${TAU_SPLIT:-<none>} USE_TAU_H=$USE_TAU_H ENABLE_DCA=$ENABLE_DCA"
echo "[WF] REGIME_MODE=$REGIME_MODE DD_MAX=$REGIME_DD_MAX RET20_MIN=$REGIME_RET20_MIN ATR_MAX=$REGIME_ATR_MAX LEV_MULT=$REGIME_LEVERAGE_MULT"

# ---- Run your existing grid runner (already env-driven)
chmod +x scripts/run_grid_workflow.sh
./scripts/run_grid_workflow.sh