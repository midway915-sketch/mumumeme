#!/usr/bin/env bash
set -euo pipefail

# scripts/run_grid_workflow.sh
# Requires env:
#   PROFIT_TARGET MAX_DAYS STOP_LEVEL MAX_EXTEND_DAYS
#   P_TAIL_THRESHOLDS UTILITY_QUANTILES RANK_METRICS LAMBDA_TAIL
#   (optional) P_SUCCESS_MINS
#   GATE_MODES (default: none,tail,utility,tail_utility)
#   TAU_THR (default: 0.45)

PROFIT_TARGET="${PROFIT_TARGET:?}"
MAX_DAYS="${MAX_DAYS:?}"
STOP_LEVEL="${STOP_LEVEL:?}"
MAX_EXTEND_DAYS="${MAX_EXTEND_DAYS:?}"

P_TAIL_THRESHOLDS="${P_TAIL_THRESHOLDS:-0.20,0.30}"
UTILITY_QUANTILES="${UTILITY_QUANTILES:-0.75,0.90}"
RANK_METRICS="${RANK_METRICS:-utility}"
LAMBDA_TAIL="${LAMBDA_TAIL:-0.05}"
P_SUCCESS_MINS="${P_SUCCESS_MINS:--1}"  # -1 means disabled
GATE_MODES="${GATE_MODES:-none,tail,utility,tail_utility}"
TAU_THR="${TAU_THR:-0.45}"

TAG="pt$(python - <<PY
pt=float("${PROFIT_TARGET}")
print(int(round(pt*100)))
PY)_h${MAX_DAYS}_sl$(python - <<PY
sl=float("${STOP_LEVEL}")
print(int(round(abs(sl)*100)))
PY)_ex${MAX_EXTEND_DAYS}"

mkdir -p data/signals data/labels data/features app

echo "[INFO] TAG=${TAG}"

# 1) tau labels
if [ ! -f data/labels/labels_tau.parquet ] && [ ! -f data/labels/labels_tau.csv ]; then
  echo "[RUN] build_tau_labels.py"
  python scripts/build_tau_labels.py \
    --profit-target "${PROFIT_TARGET}" \
    --max-days "${MAX_DAYS}" \
    --stop-level "${STOP_LEVEL}" \
    --max-extend-days "${MAX_EXTEND_DAYS}" \
    --k1 10 --k2 20
else
  echo "[INFO] labels_tau exists -> reuse"
fi

# 2) tau model
if [ ! -f app/tau_model.pkl ] || [ ! -f app/tau_scaler.pkl ]; then
  echo "[RUN] train_tau_model.py"
  python scripts/train_tau_model.py
else
  echo "[INFO] tau model exists -> reuse"
fi

# 3) tau predictions
if [ ! -f data/features/features_tau.parquet ] && [ ! -f data/features/features_tau.csv ]; then
  echo "[RUN] predict_tau.py"
  python scripts/predict_tau.py --thr "${TAU_THR}"
else
  echo "[INFO] features_tau exists -> reuse"
fi

# helpers: split CSV lists
IFS=',' read -r -a arr_tail <<< "${P_TAIL_THRESHOLDS}"
IFS=',' read -r -a arr_uq <<< "${UTILITY_QUANTILES}"
IFS=',' read -r -a arr_rank <<< "${RANK_METRICS}"
IFS=',' read -r -a arr_modes <<< "${GATE_MODES}"
IFS=',' read -r -a arr_psmin <<< "${P_SUCCESS_MINS}"

for mode in "${arr_modes[@]}"; do
  mode="$(echo "$mode" | xargs)"
  for tail in "${arr_tail[@]}"; do
    tail="$(echo "$tail" | xargs)"
    for uq in "${arr_uq[@]}"; do
      uq="$(echo "$uq" | xargs)"
      for rank in "${arr_rank[@]}"; do
        rank="$(echo "$rank" | xargs)"
        for psmin in "${arr_psmin[@]}"; do
          psmin="$(echo "$psmin" | xargs)"

          suffix="${mode}_t${tail}_q${uq}_r${rank}"
          if [ "${psmin}" != "-1" ] && [ "${psmin}" != "" ]; then
            suffix="${suffix}_ps${psmin}"
          fi

          echo "=============================="
          echo "[RUN] mode=${mode} tail_max=${tail} u_q=${uq} rank_by=${rank} ps_min=${psmin} suffix=${suffix}"
          echo "=============================="

          python scripts/predict_gate.py \
            --profit-target "${PROFIT_TARGET}" \
            --max-days "${MAX_DAYS}" \
            --stop-level "${STOP_LEVEL}" \
            --max-extend-days "${MAX_EXTEND_DAYS}" \
            --mode "${mode}" \
            --tag "${TAG}" \
            --suffix "${suffix}" \
            --tail-threshold "${tail}" \
            --utility-quantile "${uq}" \
            --rank-by "${rank}" \
            --lambda-tail "${LAMBDA_TAIL}" \
            --p-success-min "${psmin}" \
            --tau-parq "data/features/features_tau.parquet" \
            --tau-csv "data/features/features_tau.csv"

          PICKS="data/signals/picks_${TAG}_gate_${suffix}.csv"

          python scripts/simulate_single_position_engine.py \
            --picks-path "${PICKS}" \
            --profit-target "${PROFIT_TARGET}" \
            --max-days "${MAX_DAYS}" \
            --stop-level "${STOP_LEVEL}" \
            --max-extend-days "${MAX_EXTEND_DAYS}" \
            --max-leverage-pct "1.0" \
            --tag "${TAG}" \
            --suffix "${suffix}" \
            --out-dir "data/signals"

          # summarize trades (you already have summarize_sim_trades.py; keep using it)
          if [ -f scripts/summarize_sim_trades.py ]; then
            python scripts/summarize_sim_trades.py \
              --trades-path "data/signals/sim_engine_trades_${TAG}_gate_${suffix}.parquet" \
              --tag "${TAG}" \
              --suffix "${suffix}" \
              --profit-target "${PROFIT_TARGET}" \
              --max-days "${MAX_DAYS}" \
              --stop-level "${STOP_LEVEL}" \
              --max-extend-days "${MAX_EXTEND_DAYS}" \
              --out-dir "data/signals"
          fi

        done
      done
    done
  done
done

echo "[DONE] grid completed"