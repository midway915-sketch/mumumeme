#!/usr/bin/env bash
set -euo pipefail

gg_trim() { echo "$1" | xargs; }
gg_safe() { echo "$1" | sed 's/\./p/g' | sed 's/-/m/g'; }

gg_find_one() {
  # usage: gg_find_one "predict_gate.py"
  local name="$1"
  local found
  found="$(find . -maxdepth 8 -type f -iname "${name}" | head -n 1 || true)"
  echo "${found}"
}

gg_require_env() {
  local v
  for v in TAG PROFIT_TARGET HOLDING_DAYS STOP_LEVEL MAX_EXTEND_DAYS TAIL_MAX_LIST U_QUANTILE_LIST RANK_BY_LIST LAMBDA_TAIL; do
    if [ -z "${!v:-}" ]; then
      echo "[ERROR] missing env: ${v}"
      exit 1
    fi
  done
}

gg_run_one() {
  local MODE="$1"
  local TMAX="$2"
  local UQ="$3"
  local RANK="$4"
  local PRED="$5"
  local SIM="$6"

  local TMAX_T UQ_T RANK_T SUFFIX
  TMAX_T="$(gg_safe "$(gg_trim "$TMAX")")"
  UQ_T="$(gg_safe "$(gg_trim "$UQ")")"
  RANK_T="$(gg_trim "$RANK")"
  SUFFIX="${MODE}_t${TMAX_T}_q${UQ_T}_r${RANK_T}"

  echo ""
  echo "=============================="
  echo "[RUN] TAG=${TAG} mode=${MODE} tail_max=${TMAX} u_q=${UQ} rank_by=${RANK_T} suffix=${SUFFIX}"
  echo "=============================="

  python "${PRED}" \
    --profit-target "${PROFIT_TARGET}" \
    --max-days "${HOLDING_DAYS}" \
    --stop-level "${STOP_LEVEL}" \
    --max-extend-days "${MAX_EXTEND_DAYS}" \
    --gate-mode "${MODE}" \
    --tail-max "$(gg_trim "$TMAX")" \
    --u-quantile "$(gg_trim "$UQ")" \
    --rank-by "${RANK_T}" \
    --lambda-tail "${LAMBDA_TAIL}" \
    --out-suffix "${SUFFIX}"

  python "${SIM}" \
    --profit-target "${PROFIT_TARGET}" \
    --max-days "${HOLDING_DAYS}" \
    --stop-level "${STOP_LEVEL}" \
    --max-extend-days "${MAX_EXTEND_DAYS}" \
    --method custom \
    --pick-col pick_custom \
    --picks-path "data/signals/picks_${TAG}_gate_${SUFFIX}.csv" \
    --label "${SUFFIX}" \
    --out-suffix "${SUFFIX}" \
    --initial-seed 40000000
}

run_gate_grid() {
  gg_require_env

  # 워크스페이스 루트 기준 실행
  cd "${GITHUB_WORKSPACE:-.}" || true

  echo "PWD=$(pwd)"
  echo "=== find scripts ==="
  local PRED SIM
  PRED="$(gg_find_one "predict_gate.py")"
  SIM="$(gg_find_one "simulate_single_position_engine.py")"

  echo "[INFO] PRED=${PRED}"
  echo "[INFO] SIM=${SIM}"

  if [ -z "${PRED}" ]; then
    echo "[ERROR] predict_gate.py not found anywhere."
    git ls-files | grep -i "predict_gate.py" || true
    exit 1
  fi
  if [ -z "${SIM}" ]; then
    echo "[ERROR] simulate_single_position_engine.py not found anywhere."
    git ls-files | grep -i "simulate_single_position_engine.py" || true
    exit 1
  fi

  # 리스트 파싱
  IFS=',' read -ra TAILS <<< "${TAIL_MAX_LIST}"
  IFS=',' read -ra UQS   <<< "${U_QUANTILE_LIST}"
  IFS=',' read -ra RANKS <<< "${RANK_BY_LIST}"

  local BASE_T BASE_Q
  BASE_T="$(gg_trim "${TAILS[0]}")"
  BASE_Q="$(gg_trim "${UQS[0]}")"

  local TAIL_OK_LOCAL="${TAIL_OK:-0}"

  # 1) baseline: none (기준 없음)
  for R in "${RANKS[@]}"; do
    gg_run_one "none" "${BASE_T}" "${BASE_Q}" "${R}" "${PRED}" "${SIM}"
  done

  # 2) tail gate (tail 모델 있을 때만)
  if [ "${TAIL_OK_LOCAL}" = "1" ]; then
    for T in "${TAILS[@]}"; do
      for R in "${RANKS[@]}"; do
        gg_run_one "tail" "${T}" "${BASE_Q}" "${R}" "${PRED}" "${SIM}"
      done
    done
  else
    echo "[SKIP] tail gates skipped (TAIL_OK=0)"
  fi

  # 3) utility gate
  for Q in "${UQS[@]}"; do
    for R in "${RANKS[@]}"; do
      gg_run_one "utility" "${BASE_T}" "${Q}" "${R}" "${PRED}" "${SIM}"
    done
  done

  # 4) tail + utility gate (tail 모델 있을 때만)
  if [ "${TAIL_OK_LOCAL}" = "1" ]; then
    for T in "${TAILS[@]}"; do
      for Q in "${UQS[@]}"; do
        for R in "${RANKS[@]}"; do
          gg_run_one "tail_utility" "${T}" "${Q}" "${R}" "${PRED}" "${SIM}"
        done
      done
    done
  else
    echo "[SKIP] tail_utility gates skipped (TAIL_OK=0)"
  fi

  echo ""
  echo "[DONE] gate grid finished for TAG=${TAG}"
  ls -la data/signals | tail -n 200 || true
}
