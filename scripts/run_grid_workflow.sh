#!/usr/bin/env bash
set -euo pipefail

# This script orchestrates the gate grid runs using env vars.
# It relies on scripts/gate_grid_lib.sh and the Python scripts it calls.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source scripts/gate_grid_lib.sh

# -------- required env (fail fast)
: "${PROFIT_TARGET:?missing env PROFIT_TARGET}"
: "${MAX_DAYS:?missing env MAX_DAYS}"
: "${STOP_LEVEL:?missing env STOP_LEVEL}"
: "${MAX_EXTEND_DAYS:?missing env MAX_EXTEND_DAYS}"
: "${LABEL_KEY:?missing env LABEL_KEY}"
: "${OUT_DIR:?missing env OUT_DIR}"

# optional env
GATE_MODES="${GATE_MODES:-none,tail,utility,tail_utility}"
P_TAIL_THRESHOLDS="${P_TAIL_THRESHOLDS:-0.10,0.20,0.30}"
UTILITY_QUANTILES="${UTILITY_QUANTILES:-0.60,0.75,0.90}"
RANK_METRICS="${RANK_METRICS:-utility,ret_score}"
PS_MINS="${PS_MINS:-0.00,0.50,0.60,0.70}"
TOPK_CONFIGS="${TOPK_CONFIGS:-1|1.0}"
LAMBDA_TAIL="${LAMBDA_TAIL:-0.05}"

echo "[INFO] PROFIT_TARGET=${PROFIT_TARGET} MAX_DAYS=${MAX_DAYS} STOP_LEVEL=${STOP_LEVEL} MAX_EXTEND_DAYS=${MAX_EXTEND_DAYS}"
echo "[INFO] LABEL_KEY=${LABEL_KEY}"
echo "[INFO] MODES=${GATE_MODES}"
echo "[INFO] P_TAIL_THRESHOLDS=${P_TAIL_THRESHOLDS}"
echo "[INFO] UTILITY_QUANTILES=${UTILITY_QUANTILES}"
echo "[INFO] RANK_METRICS=${RANK_METRICS}"
echo "[INFO] PS_MINS=${PS_MINS}"
echo "[INFO] TOPK_CONFIGS=${TOPK_CONFIGS}"
echo "[INFO] LAMBDA_TAIL=${LAMBDA_TAIL}"
echo "[INFO] OUT_DIR=${OUT_DIR}"

# helpers: split CSV into bash arrays
IFS=',' read -r -a MODES_ARR <<< "${GATE_MODES}"
IFS=',' read -r -a TAIL_ARR  <<< "${P_TAIL_THRESHOLDS}"
IFS=',' read -r -a UQ_ARR    <<< "${UTILITY_QUANTILES}"
IFS=',' read -r -a RANK_ARR  <<< "${RANK_METRICS}"
IFS=',' read -r -a PSMIN_ARR <<< "${PS_MINS}"

# TOPK_CONFIGS is ';' separated
IFS=';' read -r -a TOPK_ARR <<< "${TOPK_CONFIGS}"

# suffix maker (safe)
mk_suffix () {
  local mode="$1" tail="$2" uq="$3" rank="$4" topk="$5" psmin="$6"
  # normalize dots
  local tail_s="${tail//./p}"
  local uq_s="${uq//./p}"
  local ps_s="${psmin//./p}"
  echo "${mode}_t${tail_s}_u${uq_s}_r${rank}_k${topk//|/_}_ps${ps_s}_ex${MAX_EXTEND_DAYS}"
}

run_mode_none () {
  local rank="${RANK_ARR[0]:-ret_score}"
  for topk in "${TOPK_ARR[@]}"; do
    for psmin in "${PSMIN_ARR[@]}"; do
      local suffix
      suffix="$(mk_suffix "none" "0" "0" "${rank}" "${topk}" "${psmin}")"
      echo "[RUN] mode=none rank=${rank} topk=${topk} psmin=${psmin} suffix=${suffix}"
      PS_MIN="${psmin}" TOPK="${topk}" \
      run_one_gate "none" "${LABEL_KEY}" "${suffix}" "0" "0" "${rank}" "${LAMBDA_TAIL}" "${OUT_DIR}"
    done
  done
}

run_mode_tail () {
  local rank="${RANK_ARR[0]:-ret_score}"
  for tail in "${TAIL_ARR[@]}"; do
    for topk in "${TOPK_ARR[@]}"; do
      for psmin in "${PSMIN_ARR[@]}"; do
        local suffix
        suffix="$(mk_suffix "tail" "${tail}" "0" "${rank}" "${topk}" "${psmin}")"
        echo "[RUN] mode=tail tail=${tail} rank=${rank} topk=${topk} psmin=${psmin} suffix=${suffix}"
        PS_MIN="${psmin}" TOPK="${topk}" \
        run_one_gate "tail" "${LABEL_KEY}" "${suffix}" "${tail}" "0" "${rank}" "${LAMBDA_TAIL}" "${OUT_DIR}"
      done
    done
  done
}

run_mode_utility () {
  # utility mode: vary utility quantile and rank metric
  for uq in "${UQ_ARR[@]}"; do
    for rank in "${RANK_ARR[@]}"; do
      for topk in "${TOPK_ARR[@]}"; do
        for psmin in "${PSMIN_ARR[@]}"; do
          local suffix
          suffix="$(mk_suffix "utility" "0" "${uq}" "${rank}" "${topk}" "${psmin}")"
          echo "[RUN] mode=utility uq=${uq} rank=${rank} topk=${topk} psmin=${psmin} suffix=${suffix}"
          PS_MIN="${psmin}" TOPK="${topk}" \
          run_one_gate "utility" "${LABEL_KEY}" "${suffix}" "0" "${uq}" "${rank}" "${LAMBDA_TAIL}" "${OUT_DIR}"
        done
      done
    done
  done
}

run_mode_tail_utility () {
  for tail in "${TAIL_ARR[@]}"; do
    for uq in "${UQ_ARR[@]}"; do
      for rank in "${RANK_ARR[@]}"; do
        for topk in "${TOPK_ARR[@]}"; do
          for psmin in "${PSMIN_ARR[@]}"; do
            local suffix
            suffix="$(mk_suffix "tail_utility" "${tail}" "${uq}" "${rank}" "${topk}" "${psmin}")"
            echo "[RUN] mode=tail_utility tail=${tail} uq=${uq} rank=${rank} topk=${topk} psmin=${psmin} suffix=${suffix}"
            PS_MIN="${psmin}" TOPK="${topk}" \
            run_one_gate "tail_utility" "${LABEL_KEY}" "${suffix}" "${tail}" "${uq}" "${rank}" "${LAMBDA_TAIL}" "${OUT_DIR}"
          done
        done
      done
    done
  done
}

# main
for mode in "${MODES_ARR[@]}"; do
  case "${mode}" in
    none)         run_mode_none ;;
    tail)         run_mode_tail ;;
    utility)      run_mode_utility ;;
    tail_utility) run_mode_tail_utility ;;
    *)
      echo "[WARN] unknown mode: ${mode} (skip)"
      ;;
  esac
done

echo "[DONE] run_grid_workflow.sh finished"