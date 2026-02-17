#!/usr/bin/env bash
set -euo pipefail

gg_trim() { echo "$1" | xargs; }
gg_safe() { echo "$1" | sed 's/\./p/g' | sed 's/-/m/g'; }

gg_find_one() {
  local name="$1"
  find . -maxdepth 8 -type f -iname "${name}" | head -n 1 || true
}

gg_require_env() {
  local v
  for v in TAG PROFIT_TARGET HOLDING_DAYS STOP_LEVEL MAX_EXTEND_DAYS TAIL_MAX_LIST U_QUANTILE_LIST RANK_BY_LIST LAMBDA_TAIL; do
    if [ -z "${!v:-}" ]; then
      echo "[ERROR] missing env: ${v}"
      exit 1
    fi
  done
  # tau gamma는 옵션(없으면 기본 0.05만 사용)
  if [ -z "${TAU_GAMMA_LIST:-}" ]; then
    export TAU_GAMMA_LIST="0.05"
  fi
}

gg_run_one() {
  local MODE="$1" TMAX="$2" UQ="$3" RANK="$4" GAMMA="$5" PRED="$6" SIM="$7"

  local TMAX_T UQ_T RANK_T GAMMA_T SUFFIX
  TMAX_T="$(gg_safe "$(gg_trim "$TMAX")")"
  UQ_T="$(gg_safe "$(gg_trim "$UQ")")"
  RANK_T="$(gg_trim "$RANK")"
  GAMMA_T="$(gg_safe "$(gg_trim "$GAMMA")")"

  if [ "${RANK_T}" = "utility_time" ]; then
    SUFFIX="${MODE}_t${TMAX_T}_q${UQ_T}_r${RANK_T}_g${GAMMA_T}"
  else
    SUFFIX="${MODE}_t${TMAX_T}_q${UQ_T}_r${RANK_T}"
  fi

  echo ""
  echo "=============================="
  echo "[RUN] TAG=${TAG} mode=${MODE} tail_max=${TMAX} u_q=${UQ} rank_by=${RANK_T} gamma=${GAMMA} suffix=${SUFFIX}"
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
    --tau-gamma "$(gg_trim "$GAMMA")" \
    --out-suffix "${SUFFIX}"

  local PICKS="data/signals/picks_${TAG}_gate_${SUFFIX}.csv"

  # --- BASE
  python "${SIM}" \
    --profit-target "${PROFIT_TARGET}" \
    --max-days "${HOLDING_DAYS}" \
    --stop-level "${STOP_LEVEL}" \
    --max-extend-days "${MAX_EXTEND_DAYS}" \
    --method custom \
    --pick-col pick_custom \
    --picks-path "${PICKS}" \
    --label "${SUFFIX}" \
    --out-suffix "${SUFFIX}" \
    --variant "BASE" \
    --initial-seed 40000000

  # --- A: extend soft brake
  python "${SIM}" \
    --profit-target "${PROFIT_TARGET}" \
    --max-days "${HOLDING_DAYS}" \
    --stop-level "${STOP_LEVEL}" \
    --max-extend-days "${MAX_EXTEND_DAYS}" \
    --method custom \
    --pick-col pick_custom \
    --picks-path "${PICKS}" \
    --label "${SUFFIX}" \
    --out-suffix "${SUFFIX}" \
    --variant "A" \
    --extend-lev-cap 0.80 \
    --extend-min-buy-frac 0.10 \
    --extend-buy-every 1 \
    --initial-seed 40000000

  # --- B: extend buy every 2 days
  python "${SIM}" \
    --profit-target "${PROFIT_TARGET}" \
    --max-days "${HOLDING_DAYS}" \
    --stop-level "${STOP_LEVEL}" \
    --max-extend-days "${MAX_EXTEND_DAYS}" \
    --method custom \
    --pick-col pick_custom \
    --picks-path "${PICKS}" \
    --label "${SUFFIX}" \
    --out-suffix "${SUFFIX}" \
    --variant "B" \
    --extend-lev-cap 1.00 \
    --extend-min-buy-frac 0.25 \
    --extend-buy-every 2 \
    --initial-seed 40000000
}

run_gate_grid() {
  gg_require_env
  cd "${GITHUB_WORKSPACE:-.}" || true

  local PRED SIM
  PRED="$(gg_find_one "predict_gate.py")"
  SIM="$(gg_find_one "simulate_single_position_engine.py")"

  echo "[INFO] PRED=${PRED}"
  echo "[INFO] SIM=${SIM}"

  if [ -z "${PRED}" ]; then
    echo "[ERROR] predict_gate.py not found"
    git ls-files | grep -i "predict_gate.py" || true
    exit 1
  fi
  if [ -z "${SIM}" ]; then
    echo "[ERROR] simulate_single_position_engine.py not found"
    git ls-files | grep -i "simulate_single_position_engine.py" || true
    exit 1
  fi

  IFS=',' read -ra TAILS <<< "${TAIL_MAX_LIST}"
  IFS=',' read -ra UQS   <<< "${U_QUANTILE_LIST}"
  IFS=',' read -ra RANKS <<< "${RANK_BY_LIST}"
  IFS=',' read -ra GAMS  <<< "${TAU_GAMMA_LIST}"

  local BASE_T BASE_Q
  BASE_T="$(gg_trim "${TAILS[0]}")"
  BASE_Q="$(gg_trim "${UQS[0]}")"

  for R in "${RANKS[@]}"; do
    R="$(gg_trim "$R")"
    if [ "${R}" = "utility_time" ]; then
      for G in "${GAMS[@]}"; do
        gg_run_one "none" "${BASE_T}" "${BASE_Q}" "${R}" "${G}" "${PRED}" "${SIM}"
      done
    else
      gg_run_one "none" "${BASE_T}" "${BASE_Q}" "${R}" "${GAMS[0]}" "${PRED}" "${SIM}"
    fi
  done

  if [ "${TAIL_OK:-0}" = "1" ]; then
    for T in "${TAILS[@]}"; do
      for R in "${RANKS[@]}"; do
        R="$(gg_trim "$R")"
        if [ "${R}" = "utility_time" ]; then
          for G in "${GAMS[@]}"; do
            gg_run_one "tail" "${T}" "${BASE_Q}" "${R}" "${G}" "${PRED}" "${SIM}"
          done
        else
          gg_run_one "tail" "${T}" "${BASE_Q}" "${R}" "${GAMS[0]}" "${PRED}" "${SIM}"
        fi
      done
    done
  fi

  for Q in "${UQS[@]}"; do
    for R in "${RANKS[@]}"; do
      R="$(gg_trim "$R")"
      if [ "${R}" = "utility_time" ]; then
        for G in "${GAMS[@]}"; do
          gg_run_one "utility" "${BASE_T}" "${Q}" "${R}" "${G}" "${PRED}" "${SIM}"
        done
      else
        gg_run_one "utility" "${BASE_T}" "${Q}" "${R}" "${GAMS[0]}" "${PRED}" "${SIM}"
      fi
    done
  done

  if [ "${TAIL_OK:-0}" = "1" ]; then
    for T in "${TAILS[@]}"; do
      for Q in "${UQS[@]}"; do
        for R in "${RANKS[@]}"; do
          R="$(gg_trim "$R")"
          if [ "${R}" = "utility_time" ]; then
            for G in "${GAMS[@]}"; do
              gg_run_one "tail_utility" "${T}" "${Q}" "${R}" "${G}" "${PRED}" "${SIM}"
            done
          else
            gg_run_one "tail_utility" "${T}" "${Q}" "${R}" "${GAMS[0]}" "${PRED}" "${SIM}"
          fi
        done
      done
    done
  fi

  echo "[DONE] gate grid finished for TAG=${TAG}"
}