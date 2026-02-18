#!/usr/bin/env bash
# scripts/run_grid_workflow.sh
set -euo pipefail

# -------------------------
# Helpers
# -------------------------
split_csv() {
  # Split comma-separated values -> one per line (trim spaces)
  echo "${1:-}" | tr ',' '\n' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | awk 'NF>0'
}

need_file() {
  local p="$1"
  if [ ! -f "$p" ]; then
    echo "[ERROR] required file not found: $p"
    exit 1
  fi
}

py() {
  python "$@"
}

# -------------------------
# Resolve script paths
# -------------------------
PRED=""
SIM=""
SUM=""

if [ -f scripts/predict_gate.py ]; then PRED="scripts/predict_gate.py"; elif [ -f predict_gate.py ]; then PRED="predict_gate.py"; fi
if [ -f scripts/simulate_single_position_engine.py ]; then SIM="scripts/simulate_single_position_engine.py"; elif [ -f simulate_single_position_engine.py ]; then SIM="simulate_single_position_engine.py"; fi
if [ -f scripts/summarize_sim_trades.py ]; then SUM="scripts/summarize_sim_trades.py"; elif [ -f summarize_sim_trades.py ]; then SUM="summarize_sim_trades.py"; fi

need_file "$PRED"
need_file "$SIM"
need_file "$SUM"

# -------------------------
# Read env (from workflow)
# -------------------------
: "${TAG:?TAG env is required (e.g. pt10_h40_sl10_ex30)}"

: "${PROFIT_TARGET:?PROFIT_TARGET env required}"
: "${MAX_DAYS:?MAX_DAYS env required}"
: "${STOP_LEVEL:?STOP_LEVEL env required}"
: "${MAX_EXTEND_DAYS:?MAX_EXTEND_DAYS env required}"

: "${P_TAIL_THRESHOLDS:?P_TAIL_THRESHOLDS env required (comma-separated)}"
: "${UTILITY_QUANTILES:?UTILITY_QUANTILES env required (comma-separated)}"
: "${RANK_METRICS:?RANK_METRICS env required (comma-separated)}"

# LAMBDA_TAIL is single float (NOT a list)
: "${LAMBDA_TAIL:?LAMBDA_TAIL env required (single float)}"

# Optional envs
OUT_DIR="${OUT_DIR:-data/signals}"
GATE_MODES="${GATE_MODES:-none,tail,utility,tail_utility}"
MAX_LEVERAGE_PCT="${MAX_LEVERAGE_PCT:-1.0}"

# 2-stage TP optional (will be passed only if simulator supports)
TP1_FRAC="${TP1_FRAC:-0.50}"
TRAIL_STOP="${TRAIL_STOP:-0.10}"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "[INFO] TAG=$TAG"
echo "[INFO] PT=$PROFIT_TARGET H=$MAX_DAYS SL=$STOP_LEVEL EX=$MAX_EXTEND_DAYS"
echo "[INFO] MODES=$GATE_MODES"
echo "[INFO] p_tail thresholds=$P_TAIL_THRESHOLDS"
echo "[INFO] utility quantiles=$UTILITY_QUANTILES"
echo "[INFO] rank metrics=$RANK_METRICS"
echo "[INFO] lambda_tail=$LAMBDA_TAIL"
echo "[INFO] max_leverage_pct=$MAX_LEVERAGE_PCT"
echo "[INFO] tp1_frac=$TP1_FRAC trail_stop=$TRAIL_STOP (only if SIM supports)"
echo "============================================================"

# Detect SIM optional args
SIM_SUPPORTS_TP1="0"
SIM_SUPPORTS_TRAIL="0"
if grep -q -- "--tp1-frac" "$SIM"; then SIM_SUPPORTS_TP1="1"; fi
if grep -q -- "--trail-stop" "$SIM"; then SIM_SUPPORTS_TRAIL="1"; fi

echo "[INFO] SIM supports --tp1-frac?  $SIM_SUPPORTS_TP1"
echo "[INFO] SIM supports --trail-stop? $SIM_SUPPORTS_TRAIL"

# -------------------------
# Grid loop
# -------------------------
modes="$(split_csv "$GATE_MODES")"
tails="$(split_csv "$P_TAIL_THRESHOLDS")"
uqs="$(split_csv "$UTILITY_QUANTILES")"
ranks="$(split_csv "$RANK_METRICS")"

# Normalize floats for suffix (0.10 -> 0p10)
f2s() {
  # float to suffix token
  python - <<PY
x=float("$1")
s=f"{x:.4f}".rstrip("0").rstrip(".")
print(s.replace(".","p").replace("-","m"))
PY
}

for mode in $modes; do
  for t in $tails; do
    for uq in $uqs; do
      for rb in $ranks; do

        t_tok="$(f2s "$t")"
        uq_tok="$(f2s "$uq")"
        lam_tok="$(f2s "$LAMBDA_TAIL")"

        suffix="${mode}_t${t_tok}_q${uq_tok}_r${rb}_l${lam_tok}"

        echo "=============================="
        echo "[RUN] mode=$mode tail_max=$t utility_q=$uq rank_by=$rb lambda=$LAMBDA_TAIL suffix=$suffix"
        echo "=============================="

        # 1) Predict picks
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
          --lambda-tail "$LAMBDA_TAIL"

        picks_path="$OUT_DIR/picks_${TAG}_gate_${suffix}.csv"
        if [ ! -f "$picks_path" ]; then
          echo "[ERROR] picks file not created: $picks_path"
          exit 1
        fi

        # 2) Simulate
        sim_cmd=( python "$SIM"
          --picks-path "$picks_path"
          --profit-target "$PROFIT_TARGET"
          --max-days "$MAX_DAYS"
          --stop-level "$STOP_LEVEL"
          --max-extend-days "$MAX_EXTEND_DAYS"
          --max-leverage-pct "$MAX_LEVERAGE_PCT"
          --tag "$TAG"
          --suffix "$suffix"
          --out-dir "$OUT_DIR"
        )

        if [ "$SIM_SUPPORTS_TP1" = "1" ]; then
          sim_cmd+=( --tp1-frac "$TP1_FRAC" )
        fi
        if [ "$SIM_SUPPORTS_TRAIL" = "1" ]; then
          sim_cmd+=( --trail-stop "$TRAIL_STOP" )
        fi

        "${sim_cmd[@]}"

        trades_path="$OUT_DIR/sim_engine_trades_${TAG}_gate_${suffix}.parquet"
        if [ ! -f "$trades_path" ]; then
          echo "[ERROR] trades parquet not created: $trades_path"
          exit 1
        fi

        # 3) Summarize (gate_summary_*.csv)
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

echo "============================================================"
echo "[DONE] grid completed"
echo "[INFO] outputs:"
ls -la "$OUT_DIR" | sed -n '1,200p'
echo "============================================================"