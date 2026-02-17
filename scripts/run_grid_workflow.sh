#!/usr/bin/env bash
set -euo pipefail

# =========================================
# run_grid_workflow.sh
# - Reads env inputs (set by GitHub Actions)
# - Iterates PT/H/SL combos
# - Runs baseline + gate grid using gate_grid_lib.sh
# =========================================

# ---- locate repo root (when run from anywhere) ----
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- require lib ----
LIB="scripts/gate_grid_lib.sh"
if [ ! -f "$LIB" ]; then
  echo "[ERROR] $LIB not found"
  echo "[DEBUG] pwd=$(pwd)"
  echo "[DEBUG] scripts/:"
  ls -la scripts | sed -n '1,200p' || true
  exit 1
fi

# shellcheck source=/dev/null
source "$LIB"

# ---- required env vars ----
: "${PROFIT_TARGET_INPUT:?Missing env PROFIT_TARGET_INPUT (e.g. 0.10 or 0.08,0.10)}"
: "${HOLDING_DAYS_INPUT:?Missing env HOLDING_DAYS_INPUT (e.g. 40 or 35,40,45)}"
: "${STOP_LEVEL_INPUT:?Missing env STOP_LEVEL_INPUT (e.g. -0.10 or -0.08,-0.10)}"
: "${MAX_EXTEND_DAYS:?Missing env MAX_EXTEND_DAYS (e.g. 30)}"

# gate grid vars (have defaults for convenience)
TAIL_MAX_LIST="${TAIL_MAX_LIST:-0.20,0.30}"
U_QUANTILE_LIST="${U_QUANTILE_LIST:-0.75,0.90}"
RANK_BY_LIST="${RANK_BY_LIST:-utility}"
LAMBDA_TAIL="${LAMBDA_TAIL:-0.05}"
TAU_GAMMA_LIST="${TAU_GAMMA_LIST:-0.05}"

echo "===================================================="
echo "[GRID] PROFIT_TARGET_INPUT=${PROFIT_TARGET_INPUT}"
echo "[GRID] HOLDING_DAYS_INPUT=${HOLDING_DAYS_INPUT}"
echo "[GRID] STOP_LEVEL_INPUT=${STOP_LEVEL_INPUT}"
echo "[GRID] MAX_EXTEND_DAYS=${MAX_EXTEND_DAYS}"
echo "[GRID] TAIL_MAX_LIST=${TAIL_MAX_LIST}"
echo "[GRID] U_QUANTILE_LIST=${U_QUANTILE_LIST}"
echo "[GRID] RANK_BY_LIST=${RANK_BY_LIST}"
echo "[GRID] LAMBDA_TAIL=${LAMBDA_TAIL}"
echo "[GRID] TAU_GAMMA_LIST=${TAU_GAMMA_LIST}"
echo "===================================================="

mkdir -p data/signals

# =========================================
# Main loops: PT x H x SL
# =========================================
for pt in $(split_csv "${PROFIT_TARGET_INPUT}"); do
  for h in $(split_csv "${HOLDING_DAYS_INPUT}"); do
    for sl in $(split_csv "${STOP_LEVEL_INPUT}"); do

      # ---------------------------
      # 1) Baseline (No Gate)
      # - IMPORTANT: downstream expects a specific file name pattern
      # - keep these defaults stable for baseline
      # ---------------------------
      # baseline suffix should match what downstream expects when mode=none
      run_one_gate \
        "$pt" "$h" "$sl" "${MAX_EXTEND_DAYS}" \
        "none" \
        "0.20" \
        "0.75" \
        "utility" \
        "${LAMBDA_TAIL}" \
        "0.05"

      # ---------------------------
      # 2) Gate grid runs
      # modes: tail / utility / tail_utility
      # ---------------------------
      for tail in $(split_csv "${TAIL_MAX_LIST}"); do
        for uq in $(split_csv "${U_QUANTILE_LIST}"); do
          for rank in $(split_csv "${RANK_BY_LIST}"); do
            for tg in $(split_csv "${TAU_GAMMA_LIST}"); do

              run_one_gate \
                "$pt" "$h" "$sl" "${MAX_EXTEND_DAYS}" \
                "tail" \
                "$tail" \
                "$uq" \
                "$rank" \
                "${LAMBDA_TAIL}" \
                "$tg"

              run_one_gate \
                "$pt" "$h" "$sl" "${MAX_EXTEND_DAYS}" \
                "utility" \
                "$tail" \
                "$uq" \
                "$rank" \
                "${LAMBDA_TAIL}" \
                "$tg"

              run_one_gate \
                "$pt" "$h" "$sl" "${MAX_EXTEND_DAYS}" \
                "tail_utility" \
                "$tail" \
                "$uq" \
                "$rank" \
                "${LAMBDA_TAIL}" \
                "$tg"

            done
          done
        done
      done

    done
  done
done

echo "[DONE] grid finished."
echo "[INFO] listing data/signals (tail 200):"
ls -la data/signals | tail -n 200 || true