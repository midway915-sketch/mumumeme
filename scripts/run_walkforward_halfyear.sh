#!/usr/bin/env bash
set -euo pipefail

# This script runs ONE walkforward half-year slice.
# It builds/loads features_scored for that CUT_DATE and then calls the grid runner.

: "${WF_PERIOD:?}"
: "${CUT_DATE:?}"

: "${OUT_DIR_BASE:=data/signals/walkforward}"
: "${WF_TAG_BASE:=wf}"

# ---- required core knobs (너가 “3개로 줄였다”던 그거)
: "${PROFIT_TARGET:?}"
: "${MAX_DAYS:?}"
: "${STOP_LEVEL:?}"

# ---- grid envs (필수/옵션 분리)
: "${P_TAIL_THRESHOLDS:=0.20}"
: "${UTILITY_QUANTILES:=0.75}"
: "${RANK_METRICS:=utility}"
: "${LAMBDA_TAIL:=1.0}"
: "${GATE_MODES:=tail_utility}"
: "${TRAIL_STOPS:=0.10}"
: "${TP1_FRAC:=0.50}"
: "${ENABLE_TRAILING:=true}"
: "${TOPK_CONFIGS:=1|1.0}"
: "${PS_MINS:=0.55}"
: "${BADEXIT_MAXES:=1.00}"
: "${MAX_LEVERAGE_PCT:=1.0}"

# ✅ optional (절대 :? 쓰지 말기)
: "${EXCLUDE_TICKERS:=}"
: "${REQUIRE_FILES:=}"

# ✅ regime optional
: "${REGIME_MODE:=off}"
: "${REGIME_DD_MAX:=0.20}"
: "${REGIME_RET20_MIN:=0.00}"
: "${REGIME_ATR_MAX:=1.30}"
: "${REGIME_LEVERAGE_MULT:=3.0}"

# ✅ tau/dca optional
: "${TAU_SPLIT:=}"
: "${USE_TAU_H:=false}"
: "${ENABLE_DCA:=false}"

# ---- partitioning
export LABEL_KEY="${WF_TAG_BASE}_${WF_PERIOD}"
export OUT_DIR="${OUT_DIR_BASE}/${WF_PERIOD}"
mkdir -p "$OUT_DIR"

echo "[WF] period=$WF_PERIOD CUT_DATE=$CUT_DATE"
echo "[WF] LABEL_KEY=$LABEL_KEY OUT_DIR=$OUT_DIR"
echo "[WF] EXCLUDE_TICKERS='${EXCLUDE_TICKERS}' REQUIRE_FILES='${REQUIRE_FILES}'"
echo "[WF] REGIME_MODE=$REGIME_MODE"
echo "[WF] TAU_SPLIT=${TAU_SPLIT:-<none>} USE_TAU_H=$USE_TAU_H ENABLE_DCA=$ENABLE_DCA"

# ------------------------------------------------------------
# ✅ PREP: ensure raw prices exist (fail with clear msg)
# ------------------------------------------------------------
if [ ! -f data/raw/prices.parquet ] && [ ! -f data/raw/prices.csv ]; then
  echo "[ERROR] Missing raw prices: data/raw/prices.parquet (or data/raw/prices.csv)"
  echo "[HINT] In workflow, add a step to download/upload raw-data artifact (or checkout LFS)."
  exit 3
fi

# ------------------------------------------------------------
# ✅ PREP: ensure features_scored exists (build + score)
# ------------------------------------------------------------
if [ ! -f data/features/features_scored.parquet ] && [ ! -f data/features/features_scored.csv ]; then
  echo "[INFO] features_scored missing -> build_features + score_features for CUT_DATE=$CUT_DATE"

  # build_features should accept --cut-date in your repo (이미 너 로그에 CUT_DATE로 돌렸음)
  python scripts/build_features.py --cut-date "$CUT_DATE"

  # score_features needs tag; 여기서는 “현재 레포의 기본 태그”를 쓴다고 가정
  # (원하면 WF_PERIOD 기반으로 tag 분리 가능)
  python scripts/score_features.py --tag "pt10_h40_sl10"
fi

# ------------------------------------------------------------
# Run your existing grid runner
# ------------------------------------------------------------
chmod +x scripts/run_grid_workflow.sh
./scripts/run_grid_workflow.sh