#!/usr/bin/env bash
# scripts/run_grid_workflow.sh
set -euo pipefail

split_csv() {
  echo "${1:-}" | tr ',' '\n' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | awk 'NF>0'
}
need_file() {
  local p="$1"
  if [ ! -f "$p" ]; then
    echo "[ERROR] required file not found: $p"
    exit 1
  fi
}
py() { python "$@"; }

PRED=""
SIM=""
SUM=""

if [ -f scripts/predict_gate.py ]; then PRED="scripts/predict_gate.py"; elif [ -f predict_gate.py ]; then PRED="predict_gate.py"; fi
if [ -f scripts/simulate_single_position_engine.py ]; then SIM="scripts/simulate_single_position_engine.py"; elif [ -f simulate_single_position_engine.py ]; then SIM="simulate_single_position_engine.py"; fi
if [ -f scripts/summarize_sim_trades.py ]; then SUM="scripts/summarize_sim_trades.py"; elif [ -f summarize_sim_trades.py ]; then SUM="summarize_sim_trades.py"; fi

need_file "$PRED"
need_file "$SIM"
need_file "$SUM"

: "${TAG:?TAG env required}"
: "${PROFIT_TARGET:?}"
: "${MAX_DAYS:?}"
: "${STOP_LEVEL:?}"
: "${MAX_EXTEND_DAYS:?}"
: "${P_TAIL_THRESHOLDS:?}"
: "${UTILITY_QUANTILES:?}"
: "${RANK_METRICS:?}"
: "${LAMBDA_TAIL:?}"     # single float

OUT_DIR="${OUT_DIR:-data/signals}"
GATE_MODES="${GATE_MODES:-none,tail,utility,tail_utility}"
MAX_LEVERAGE_PCT="${MAX_LEVERAGE_PCT:-1.0}"

# TopK compare (default: run both)
TOPK_LIST="${TOPK_LIST:-1,2}"
TOP2_WEIGHTS="${TOP2_WEIGHTS:-0.7,0.3}"

# 2-stage TP (default enabled as 추천)
TP1_FRAC="${TP1_FRAC:-0.50}"
TRAIL_STOP="${TRAIL_STOP:-0.10}"

mkdir -p "$OUT_DIR"

f2s() {
  python - <<PY
x=float("$1")
s=f"{x:.4f}".rstrip("0").rstrip(".")
print(s.replace(".","p").replace("-","m"))
PY
}

echo "============================================================"
echo "[INFO] TAG=$TAG"
echo "[INFO] PT=$PROFIT_TARGET H=$MAX_DAYS SL=$STOP_LEVEL EX=$MAX_EXTEND_DAYS"
echo "[INFO] TOPK_LIST=$TOPK_LIST TOP2_WEIGHTS=$TOP2_WEIGHTS"
echo "[INFO] tp1_frac=$TP1_FRAC trail_stop=$TRAIL_STOP"
echo "============================================================"

modes="$(split_csv "$GATE_MODES")"
tails="$(split_csv "$P_TAIL_THRESHOLDS")"
uqs="$(split_csv "$UTILITY_QUANTILES")"
ranks="$(split_csv "$RANK_METRICS")"
topks="$(split_csv "$TOPK_LIST")"

for topk in $topks; do
  if [ "$topk" = "1" ]; then
    topk_suffix="k1"
    weights="1.0"
  else
    wtok="$(echo "$TOP2_WEIGHTS" | sed 's/[[:space:]]//g')"
    topk_suffix="k2w$(echo "$wtok" | tr '.' 'p' | tr ',' '_' )"
    weights="$TOP2_WEIGHTS"
  fi

  for mode in $modes; do
    for t in $tails; do
      for uq in $uqs; do
        for rb in $ranks; do

          t_tok="$(f2s "$t")"
          uq_tok="$(f2s "$uq")"
          lam_tok="$(f2s "$LAMBDA_TAIL")"

          suffix="${mode}_t${t_tok}_q${uq_tok}_r${rb}_l${lam_tok}_${topk_suffix}"

          echo "=============================="
          echo "[RUN] topk=$topk weights=$weights mode=$mode tail_max=$t u_q=$uq rank_by=$rb suffix=$suffix"
          echo "=============================="

          # 1) Predict (Top-K)
          py "$PRED" \
            --profit-target "$PROFIT_TARGET" \
            --max-days "$MAX_DAYS" \
            --stop-level "$STOP_LEVEL" \
            --max-extend-days "$MAX_EXTEND_DAYS" \
            --mode "$mode" \
            --tag "$TAG" \
            --suffix "$suffix" \
            --out-dir "$OUT_DIR" \
            --tail-threshold "$t" \
            --utility-quantile "$uq" \
            --rank-by "$rb" \
            --lambda-tail "$LAMBDA_TAIL" \
            --topk "$topk"

          picks_path="$OUT_DIR/picks_${TAG}_gate_${suffix}.csv"
          if [ ! -f "$picks_path" ]; then
            echo "[ERROR] picks not created: $picks_path"
            exit 1
          fi

          # 2) Simulate (Top-1 or Top-2 split)
          python "$SIM" \
            --picks-path "$picks_path" \
            --profit-target "$PROFIT_TARGET" \
            --max-days "$MAX_DAYS" \
            --stop-level "$STOP_LEVEL" \
            --max-extend-days "$MAX_EXTEND_DAYS" \
            --max-leverage-pct "$MAX_LEVERAGE_PCT" \
            --topk "$topk" \
            --topk-weights "$weights" \
            --tp1-frac "$TP1_FRAC" \
            --trail-stop "$TRAIL_STOP" \
            --tag "$TAG" \
            --suffix "$suffix" \
            --out-dir "$OUT_DIR"

          trades_path="$OUT_DIR/sim_engine_trades_${TAG}_gate_${suffix}.parquet"
          if [ ! -f "$trades_path" ]; then
            echo "[ERROR] trades not created: $trades_path"
            exit 1
          fi

          # 3) Summarize
          py "$SUM" \
            --trades-path "$trades_path" \
            --tag "$TAG" \
            --suffix "$suffix" \
            --profit-target "$PROFIT_TARGET" \
            --max-days "$MAX_DAYS" \
            --stop-level "$STOP_LEVEL" \
            --max-extend-days "$MAX_EXTEND_DAYS" \
            --out-dir "$OUT_DIR"

        done
      done
    done
  done
done

echo "============================================================"
echo "[DONE] grid completed"
ls -la "$OUT_DIR" | sed -n '1,200p'
echo "============================================================"