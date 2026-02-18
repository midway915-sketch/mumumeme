#!/usr/bin/env bash
set -euo pipefail

# Required envs (from workflow)
: "${PROFIT_TARGET:?}"
: "${MAX_DAYS:?}"
: "${STOP_LEVEL:?}"
: "${MAX_EXTEND_DAYS:?}"

: "${P_TAIL_THRESHOLDS:?}"
: "${UTILITY_QUANTILES:?}"
: "${RANK_METRICS:?}"
: "${LAMBDA_TAIL:?}"
: "${GATE_MODES:?}"

: "${TP1_FRAC:?}"
: "${TRAIL_STOPS:?}"
: "${ENABLE_TRAILING:?}"
: "${TOPK_CONFIGS:?}"

: "${OUT_DIR:=data/signals}"
: "${EXCLUDE_TICKERS:=SPY,^VIX}"

# Optional
: "${PS_MIN_VALUES:=0}"

TAG="pt$(python - <<PY
pt=float("$PROFIT_TARGET")
print(int(round(abs(pt)*100)))
PY)_h${MAX_DAYS}_sl$(python - <<PY
sl=float("$STOP_LEVEL")
print(int(round(abs(sl)*100)))
PY)_ex${MAX_EXTEND_DAYS}"

echo "[INFO] TAG=$TAG"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] TOPK_CONFIGS=$TOPK_CONFIGS"
echo "[INFO] TRAIL_STOPS=$TRAIL_STOPS TP1_FRAC=$TP1_FRAC ENABLE_TRAILING=$ENABLE_TRAILING"
echo "[INFO] PS_MIN_VALUES=$PS_MIN_VALUES"

mkdir -p "$OUT_DIR"

# Split helpers
IFS=',' read -r -a TAILS <<< "$P_TAIL_THRESHOLDS"
IFS=',' read -r -a QS <<< "$UTILITY_QUANTILES"
IFS=',' read -r -a RANKS <<< "$RANK_METRICS"
IFS=',' read -r -a MODES <<< "$GATE_MODES"
IFS=',' read -r -a TRS <<< "$TRAIL_STOPS"
IFS=',' read -r -a PSMINS <<< "$PS_MIN_VALUES"

IFS=';' read -r -a TOPKS <<< "$TOPK_CONFIGS"

# NOTE: predict_gate.py in your repo reads features_model by default,
# but we WANT p_tail/p_success => use features_scored explicitly.
FEATURES_PARQ="data/features/features_scored.parquet"
FEATURES_CSV="data/features/features_scored.csv"

if [ ! -f "$FEATURES_PARQ" ] && [ ! -f "$FEATURES_CSV" ]; then
  echo "[FATAL] missing features_scored. Did you run scripts/score_models.py?"
  exit 2
fi

for mode in "${MODES[@]}"; do
  mode="$(echo "$mode" | xargs)"
  for tail in "${TAILS[@]}"; do
    tail="$(echo "$tail" | xargs)"
    for q in "${QS[@]}"; do
      q="$(echo "$q" | xargs)"
      for rank in "${RANKS[@]}"; do
        rank="$(echo "$rank" | xargs)"
        for psmin in "${PSMINS[@]}"; do
          psmin="$(echo "$psmin" | xargs)"
          for topk_cfg in "${TOPKS[@]}"; do
            # topk_cfg = "1|1.0" or "2|0.7,0.3"
            topk="$(echo "$topk_cfg" | cut -d'|' -f1)"
            weights="$(echo "$topk_cfg" | cut -d'|' -f2)"

            for tr in "${TRS[@]}"; do
              tr="$(echo "$tr" | xargs)"

              suffix="${mode}_t${tail}_q${q}_r${rank}_lam${LAMBDA_TAIL}_ps${psmin}_k${topk}_w$(echo "$weights" | tr ',' '_')_tp$(echo "$TP1_FRAC" | sed 's/0\.//')_tr$(echo "$tr" | sed 's/0\.//')"
              suffix="${suffix//./p}"
              suffix="${suffix//-/_}"

              echo "=============================="
              echo "[RUN] mode=$mode tail_max=$tail u_q=$q rank_by=$rank lambda=$LAMBDA_TAIL ps_min=$psmin topk=$topk weights=$weights trail=$tr suffix=$suffix"
              echo "=============================="

              # 1) predict picks (topk rows per day will be sliced inside simulate)
              python scripts/predict_gate.py \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-extend-days "$MAX_EXTEND_DAYS" \
                --mode "$mode" \
                --tag "$TAG" \
                --suffix "$suffix" \
                --out-dir "$OUT_DIR" \
                --features-parq "$FEATURES_PARQ" \
                --features-csv "$FEATURES_CSV" \
                --exclude-tickers "$EXCLUDE_TICKERS" \
                --tail-threshold "$tail" \
                --utility-quantile "$q" \
                --rank-by "$rank" \
                --lambda-tail "$LAMBDA_TAIL" \
                --ps-min "$psmin" \
                --topk "$topk"

              picks_path="$OUT_DIR/picks_${TAG}_gate_${suffix}.csv"

              # 2) simulate
              python scripts/simulate_single_position_engine.py \
                --picks-path "$picks_path" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-extend-days "$MAX_EXTEND_DAYS" \
                --enable-trailing "$ENABLE_TRAILING" \
                --tp1-frac "$TP1_FRAC" \
                --trail-stop "$tr" \
                --topk "$topk" \
                --weights "$weights" \
                --tag "$TAG" \
                --suffix "$suffix" \
                --out-dir "$OUT_DIR"

              # 3) summarize
              python scripts/summarize_gate_run.py \
                --curve-parq "$OUT_DIR/sim_engine_curve_${TAG}_gate_${suffix}.parquet" \
                --trades-parq "$OUT_DIR/sim_engine_trades_${TAG}_gate_${suffix}.parquet" \
                --out-csv "$OUT_DIR/gate_summary_${TAG}_gate_${suffix}.csv"

            done
          done
        done
      done
    done
  done
done

echo "[DONE] grid finished."