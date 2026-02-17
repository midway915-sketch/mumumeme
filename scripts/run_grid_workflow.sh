#!/usr/bin/env bash
set -euo pipefail

# Required envs
: "${PROFIT_TARGET:?}"
: "${MAX_DAYS:?}"
: "${STOP_LEVEL:?}"
: "${MAX_EXTEND_DAYS:?}"
: "${P_TAIL_THRESHOLDS:?}"
: "${UTILITY_QUANTILES:?}"
: "${RANK_METRICS:?}"
: "${LAMBDA_TAIL:?}"
: "${GATE_MODES:?}"

OUT_DIR="data/signals"
FEATURES_PARQ="data/features/features_model.parquet"
FEATURES_CSV="data/features/features_model.csv"

# tools
PRED="scripts/predict_gate.py"
SIM="scripts/simulate_single_position_engine.py"
SUM="scripts/summarize_sim_trades.py"

if [ ! -f "$PRED" ]; then echo "[ERROR] missing $PRED"; exit 1; fi
if [ ! -f "$SIM" ]; then echo "[ERROR] missing $SIM"; exit 1; fi
if [ ! -f "$SUM" ]; then echo "[ERROR] missing $SUM"; exit 1; fi

mkdir -p "$OUT_DIR"

has_tail_model="0"
if [ -f "app/tail_model.pkl" ] && [ -f "app/tail_scaler.pkl" ]; then
  has_tail_model="1"
fi

# helper: trim spaces
trim() { echo "$1" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'; }

# helper: normalize floats for suffix
fkey() {
  python - <<PY
x=float("$1")
s=str(x)
s=s.replace("-","m").replace(".","p")
print(s)
PY
}

# TAG from PT/H/SL/EX (match your yml logic)
TAG="pt$(python - <<PY
print(int(round(float("$PROFIT_TARGET")*100)))
PY
)_h${MAX_DAYS}_sl$(python - <<PY
print(int(round(abs(float("$STOP_LEVEL"))*100)))
PY
)_ex${MAX_EXTEND_DAYS}"

echo "[INFO] TAG=$TAG"

# Parse lists
IFS=',' read -r -a MODE_ARR <<< "$GATE_MODES"
IFS=',' read -r -a TAIL_ARR <<< "$P_TAIL_THRESHOLDS"
IFS=',' read -r -a UQ_ARR <<< "$UTILITY_QUANTILES"
IFS=',' read -r -a RM_ARR <<< "$RANK_METRICS"

for m0 in "${MODE_ARR[@]}"; do
  mode="$(trim "$m0")"
  if [ -z "$mode" ]; then continue; fi

  if [[ "$mode" == "tail" || "$mode" == "tail_utility" ]]; then
    if [ "$has_tail_model" != "1" ]; then
      echo "[SKIP] mode=$mode (tail model missing)"
      continue
    fi
  fi

  for t0 in "${TAIL_ARR[@]}"; do
    tail_thr="$(trim "$t0")"
    [ -z "$tail_thr" ] && continue

    for uq0 in "${UQ_ARR[@]}"; do
      uq="$(trim "$uq0")"
      [ -z "$uq" ] && continue

      for rm0 in "${RM_ARR[@]}"; do
        rank_by="$(trim "$rm0")"
        [ -z "$rank_by" ] && continue

        # suffix (stable)
        tkey="$(fkey "$tail_thr")"
        uqkey="$(fkey "$uq")"
        lamkey="$(fkey "$LAMBDA_TAIL")"
        SUFFIX="${mode}_t${tkey}_q${uqkey}_r${rank_by}_lam${lamkey}"

        echo "=============================="
        echo "[RUN] mode=${mode} tail_max=${tail_thr} u_q=${uq} rank_by=${rank_by} suffix=${SUFFIX}"
        echo "=============================="

        # 1) predict -> picks
        python "$PRED" \
          --profit-target "$PROFIT_TARGET" \
          --max-days "$MAX_DAYS" \
          --stop-level "$STOP_LEVEL" \
          --max-extend-days "$MAX_EXTEND_DAYS" \
          --mode "$mode" \
          --tag "$TAG" \
          --suffix "$SUFFIX" \
          --out-dir "$OUT_DIR" \
          --features-parq "$FEATURES_PARQ" \
          --features-csv "$FEATURES_CSV" \
          --tail-threshold "$tail_thr" \
          --utility-quantile "$uq" \
          --rank-by "$rank_by" \
          --lambda-tail "$LAMBDA_TAIL"

        # picks path (predict_gate writes standard name)
        PICKS_CSV="${OUT_DIR}/picks_${TAG}_gate_${SUFFIX}.csv"
        if [ ! -f "$PICKS_CSV" ]; then
          echo "[ERROR] picks missing: $PICKS_CSV"
          ls -la "$OUT_DIR" | sed -n '1,120p'
          exit 1
        fi

        # 2) simulate -> trades + curve
        python "$SIM" \
          --picks-path "$PICKS_CSV" \
          --profit-target "$PROFIT_TARGET" \
          --max-days "$MAX_DAYS" \
          --stop-level "$STOP_LEVEL" \
          --max-extend-days "$MAX_EXTEND_DAYS" \
          --tag "$TAG" \
          --suffix "$SUFFIX" \
          --out-dir "$OUT_DIR"

        # locate trades/curve (parquet)
        TRADES_PQ="${OUT_DIR}/sim_engine_trades_${TAG}_gate_${SUFFIX}.parquet"
        CURVE_PQ="${OUT_DIR}/sim_engine_curve_${TAG}_gate_${SUFFIX}.parquet"

        if [ ! -f "$TRADES_PQ" ]; then
          echo "[ERROR] trades parquet missing: $TRADES_PQ"
          echo "[DEBUG] listing sim_engine_trades*:"
          ls -la "$OUT_DIR"/sim_engine_trades* 2>/dev/null || true
          exit 1
        fi

        if [ ! -f "$CURVE_PQ" ]; then
          echo "[WARN] curve parquet missing: $CURVE_PQ (summary will fallback)"
          CURVE_PQ=""
        fi

        # 3) summarize -> gate_summary csv
        if [ -n "$CURVE_PQ" ]; then
          python "$SUM" \
            --trades-path "$TRADES_PQ" \
            --curve-path "$CURVE_PQ" \
            --tag "$TAG" \
            --suffix "$SUFFIX" \
            --profit-target "$PROFIT_TARGET" \
            --max-days "$MAX_DAYS" \
            --stop-level "$STOP_LEVEL" \
            --max-extend-days "$MAX_EXTEND_DAYS" \
            --out-dir "$OUT_DIR"
        else
          python "$SUM" \
            --trades-path "$TRADES_PQ" \
            --tag "$TAG" \
            --suffix "$SUFFIX" \
            --profit-target "$PROFIT_TARGET" \
            --max-days "$MAX_DAYS" \
            --stop-level "$STOP_LEVEL" \
            --max-extend-days "$MAX_EXTEND_DAYS" \
            --out-dir "$OUT_DIR"
        fi

      done
    done
  done
done

echo "[DONE] grid finished. summaries in $OUT_DIR/gate_summary_*.csv"