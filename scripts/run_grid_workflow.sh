#!/usr/bin/env bash
set -euo pipefail

# required env:
# PROFIT_TARGET, MAX_DAYS, STOP_LEVEL, MAX_EXTEND_DAYS
# P_TAIL_THRESHOLDS, UTILITY_QUANTILES, RANK_METRICS
# LAMBDA_TAIL, TAU_GAMMA
# optional: GATE_MODES (default none,tail,utility,tail_utility)

PT="${PROFIT_TARGET}"
H="${MAX_DAYS}"
SL="${STOP_LEVEL}"
EX="${MAX_EXTEND_DAYS}"

TAILS="${P_TAIL_THRESHOLDS}"
UQS="${UTILITY_QUANTILES}"
RANKS="${RANK_METRICS}"

LAMBDA="${LAMBDA_TAIL:-0.05}"
TAU_GAMMA="${TAU_GAMMA:-0.0}"

MODES="${GATE_MODES:-none,tail,utility,tail_utility}"

split_csv() {
  echo "$1" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | awk 'NF>0{print}'
}

if [ ! -f scripts/gate_grid_lib.sh ]; then
  echo "[ERROR] scripts/gate_grid_lib.sh not found"
  exit 1
fi
source scripts/gate_grid_lib.sh

echo "=============================="
echo "[INFO] PT=$PT H=$H SL=$SL EX=$EX"
echo "[INFO] MODES=$MODES"
echo "[INFO] TAILS=$TAILS"
echo "[INFO] UQS=$UQS"
echo "[INFO] RANKS=$RANKS"
echo "[INFO] LAMBDA=$LAMBDA TAU_GAMMA=$TAU_GAMMA"
echo "=============================="

mkdir -p data/signals

for mode in $(split_csv "$MODES"); do
  for tail in $(split_csv "$TAILS"); do
    for uq in $(split_csv "$UQS"); do
      for rank in $(split_csv "$RANKS"); do
        run_one_gate "$PT" "$H" "$SL" "$EX" "$mode" "$tail" "$uq" "$rank" "$LAMBDA" "$TAU_GAMMA"
      done
    done
  done
done

echo "=============================="
echo "[INFO] signals outputs:"
ls -la data/signals | sed -n '1,200p' || true
echo "=============================="