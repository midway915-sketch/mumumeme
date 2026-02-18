#!/usr/bin/env bash
# scripts/run_grid_workflow.sh
set -euo pipefail

OUT_DIR="${OUT_DIR:-data/signals}"
mkdir -p "$OUT_DIR"

TAG="${LABEL_KEY:-pt10_h40_sl10_ex30}"

# Required base params
PROFIT_TARGET="${PROFIT_TARGET:-0.10}"
MAX_DAYS="${MAX_DAYS:-40}"
STOP_LEVEL="${STOP_LEVEL:--0.10}"

P_TAIL_THRESHOLDS="${P_TAIL_THRESHOLDS:-0.10,0.20,0.30}"
UTILITY_QUANTILES="${UTILITY_QUANTILES:-0.60,0.75,0.90}"
RANK_METRICS="${RANK_METRICS:-utility,ret_score}"
LAMBDA_TAIL="${LAMBDA_TAIL:-0.05}"
GATE_MODES="${GATE_MODES:-none,tail,utility,tail_utility}"

# NEW: p_success min grid
PS_MIN_THRESHOLDS="${PS_MIN_THRESHOLDS:-0.0,0.05,0.10}"

# topk configs: "topk|w1,w2;topk|w1,w2"
TOPK_CONFIGS="${TOPK_CONFIGS:-1|1.0}"

# trailing config (engine controls this)
ENABLE_TRAILING="${ENABLE_TRAILING:-true}"
TP1_FRAC="${TP1_FRAC:-0.50}"
TRAIL_STOPS="${TRAIL_STOPS:-0.10}"

EXCLUDE_TICKERS="${EXCLUDE_TICKERS:-SPY,^VIX}"

echo "[INFO] TAG=$TAG"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] TOPK_CONFIGS=$TOPK_CONFIGS"
echo "[INFO] PS_MIN_THRESHOLDS=$PS_MIN_THRESHOLDS"
echo "[INFO] TRAIL_STOPS=$TRAIL_STOPS TP1_FRAC=$TP1_FRAC ENABLE_TRAILING=$ENABLE_TRAILING"

# split csv helper (bash-safe)
csv_to_array() {
  local s="$1"
  s="${s// /}"
  IFS=',' read -r -a _ARR <<< "$s"
  for x in "${_ARR[@]}"; do
    [[ -n "$x" ]] && echo "$x"
  done
}

# split semicolon helper
sc_to_array() {
  local s="$1"
  s="${s// /}"
  IFS=';' read -r -a _ARR <<< "$s"
  for x in "${_ARR[@]}"; do
    [[ -n "$x" ]] && echo "$x"
  done
}

# ensure scripts exist
if [[ ! -f "scripts/predict_gate.py" ]]; then
  echo "[ERROR] scripts/predict_gate.py not found" >&2
  exit 1
fi
if [[ ! -f "scripts/simulate_single_position_engine.py" ]]; then
  echo "[ERROR] scripts/simulate_single_position_engine.py not found" >&2
  exit 1
fi
if [[ ! -f "scripts/summarize_sim_trades.py" ]]; then
  echo "[ERROR] scripts/summarize_sim_trades.py not found" >&2
  exit 1
fi

# loops
for mode in $(csv_to_array "$GATE_MODES"); do
  for tail in $(csv_to_array "$P_TAIL_THRESHOLDS"); do
    for uq in $(csv_to_array "$UTILITY_QUANTILES"); do
      for rank_by in $(csv_to_array "$RANK_METRICS"); do
        for ps_min in $(csv_to_array "$PS_MIN_THRESHOLDS"); do
          for topk_cfg in $(sc_to_array "$TOPK_CONFIGS"); do
            topk="${topk_cfg%%|*}"
            weights="${topk_cfg#*|}"

            for tr in $(csv_to_array "$TRAIL_STOPS"); do
              suffix="${mode}_t${tail}_q${uq}_r${rank_by}_lam${LAMBDA_TAIL}_ps${ps_min}_k${topk}_w${weights}_tp${TP1_FRAC}_tr${tr}"
              # sanitize for filenames
              suffix="${suffix//./p}"
              suffix="${suffix//,/}"
              suffix="${suffix//|/}"

              echo "=============================="
              echo "[RUN] mode=$mode tail_max=$tail u_q=$uq rank_by=$rank_by ps_min=$ps_min topk=$topk weights=$weights trail=$tr suffix=$suffix"
              echo "=============================="

              python scripts/predict_gate.py \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --mode "$mode" \
                --tag "$TAG" \
                --suffix "$suffix" \
                --out-dir "$OUT_DIR" \
                --tail-threshold "$tail" \
                --utility-quantile "$uq" \
                --rank-by "$rank_by" \
                --lambda-tail "$LAMBDA_TAIL" \
                --ps-min "$ps_min" \
                --topk "$topk" \
                --exclude-tickers "$EXCLUDE_TICKERS"

              picks_path="$OUT_DIR/picks_${TAG}_gate_${suffix}.csv"

              python scripts/simulate_single_position_engine.py \
                --picks-path "$picks_path" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-leverage-pct "${MAX_LEVERAGE_PCT:-1.0}" \
                --enable-trailing "$ENABLE_TRAILING" \
                --tp1-frac "$TP1_FRAC" \
                --trail-stop "$tr" \
                --topk "$topk" \
                --weights "$weights" \
                --tag "$TAG" \
                --suffix "$suffix" \
                --out-dir "$OUT_DIR"

              trades_parq="$OUT_DIR/sim_engine_trades_${TAG}_gate_${suffix}.parquet"

              python scripts/summarize_sim_trades.py \
                --trades-path "$trades_parq" \
                --tag "$TAG" \
                --suffix "$suffix" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --out-dir "$OUT_DIR"
            done
          done
        done
      done
    done
  done
done

echo "[DONE] grid run complete"