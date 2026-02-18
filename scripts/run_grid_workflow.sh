#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Required env (from workflow)
# ----------------------------
: "${PROFIT_TARGET:?missing}"
: "${MAX_DAYS:?missing}"
: "${STOP_LEVEL:?missing}"
: "${MAX_EXTEND_DAYS:?missing}"
: "${P_TAIL_THRESHOLDS:?missing}"
: "${UTILITY_QUANTILES:?missing}"
: "${RANK_METRICS:?missing}"
: "${LAMBDA_TAIL:?missing}"
: "${GATE_MODES:?missing}"

# optional env
OUT_DIR="${OUT_DIR:-data/signals}"
TP1_FRAC="${TP1_FRAC:-0.50}"                  # 50% partial take profit at PT
TRAIL_STOPS="${TRAIL_STOPS:-0.08,0.10,0.12}"  # trailing stop list (8/10/12%)
ENABLE_TRAILING="${ENABLE_TRAILING:-true}"

# Compare Top-1 vs Top-2 split configs (hardcoded: 안정 + 수익률 우선)
# format: "topk|weights"
TOPK_CONFIGS="${TOPK_CONFIGS:-1|1.0;2|0.7,0.3;2|0.6,0.4}"

EXCLUDE_TICKERS="${EXCLUDE_TICKERS:-SPY,^VIX}"
REQUIRE_FILES="${REQUIRE_FILES:-data/features/features_model.parquet,app/model.pkl,app/scaler.pkl}"

mkdir -p "$OUT_DIR"

# ----------------------------
# Helpers
# ----------------------------
to_tag() {
  # pt=0.10,h=40,sl=-0.10,ex=30 -> pt10_h40_sl10_ex30
  python - <<'PY'
import os
pt=float(os.environ["PROFIT_TARGET"])
h=int(os.environ["MAX_DAYS"])
sl=float(os.environ["STOP_LEVEL"])
ex=int(os.environ["MAX_EXTEND_DAYS"])
def pct(x): return int(round(abs(x)*100))
print(f"pt{pct(pt)}_h{h}_sl{pct(sl)}_ex{ex}")
PY
}

fmt_num() {
  # 0.2 -> 0p20, -0.1 -> m0p10
  python - <<'PY'
import sys
x=float(sys.argv[1])
s=("m" if x<0 else "") + f"{abs(x):.4f}".rstrip("0").rstrip(".")
print(s.replace(".","p"))
PY "$1"
}

# split "a,b,c" into lines
split_csv() {
  python - <<'PY'
import os,sys
s=sys.argv[1].strip()
parts=[p.strip() for p in s.split(",") if p.strip()]
for p in parts:
  print(p)
PY "$1"
}

# ----------------------------
# Locate scripts
# ----------------------------
PRED=""
SIM=""
SUM=""

if [ -f scripts/predict_gate.py ]; then PRED="scripts/predict_gate.py"; else echo "[ERROR] scripts/predict_gate.py not found"; exit 1; fi
if [ -f scripts/simulate_single_position_engine.py ]; then SIM="scripts/simulate_single_position_engine.py"; else echo "[ERROR] scripts/simulate_single_position_engine.py not found"; exit 1; fi
if [ -f scripts/summarize_sim_trades.py ]; then SUM="scripts/summarize_sim_trades.py"; else echo "[ERROR] scripts/summarize_sim_trades.py not found"; exit 1; fi

TAG="$(to_tag)"
echo "[INFO] TAG=$TAG"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] TOPK_CONFIGS=$TOPK_CONFIGS"
echo "[INFO] TRAIL_STOPS=$TRAIL_STOPS TP1_FRAC=$TP1_FRAC ENABLE_TRAILING=$ENABLE_TRAILING"

# ----------------------------
# Grid loops
# ----------------------------
# allow LAMBDA_TAIL to be single or comma list
LAMBDA_LIST="$(split_csv "$LAMBDA_TAIL" || true)"
if [ -z "${LAMBDA_LIST:-}" ]; then
  LAMBDA_LIST="0.05"
fi

# iterate configs
while IFS= read -r mode; do
  [ -z "$mode" ] && continue

  while IFS= read -r tmax; do
    [ -z "$tmax" ] && continue

    while IFS= read -r uq; do
      [ -z "$uq" ] && continue

      while IFS= read -r rnk; do
        [ -z "$rnk" ] && continue

        while IFS= read -r lam; do
          [ -z "$lam" ] && continue

          # TopK configs (Top-1, Top-2 splits)
          IFS=';' read -r -a cfgs <<< "$TOPK_CONFIGS"
          for cfg in "${cfgs[@]}"; do
            cfg="$(echo "$cfg" | xargs)"
            [ -z "$cfg" ] && continue
            topk="${cfg%%|*}"
            weights="${cfg#*|}"

            # Trailing stop list
            while IFS= read -r tr; do
              [ -z "$tr" ] && continue

              # Suffix: safe chars only
              s_mode="$mode"
              s_t="t$(fmt_num "$tmax")"
              s_q="q$(fmt_num "$uq")"
              s_r="r${rnk}"
              s_l="lam$(fmt_num "$lam")"
              s_k="k${topk}"
              w_safe="$(echo "$weights" | tr ',' '_' | tr '.' 'p')"
              s_w="w${w_safe}"
              s_tp="tp$(python - <<PY
import os
print(str(int(round(float(os.environ["TP1_FRAC"])*100))))
PY
)"
              s_tr="tr$(fmt_num "$tr")"

              SUFFIX="${s_mode}_${s_t}_${s_q}_${s_r}_${s_l}_${s_k}_${s_w}_${s_tp}_${s_tr}"

              echo "=============================="
              echo "[RUN] mode=$mode tail_max=$tmax u_q=$uq rank_by=$rnk lambda=$lam topk=$topk weights=$weights trail=$tr suffix=$SUFFIX"
              echo "=============================="

              # 1) Predict picks (TopK)
              python "$PRED" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-extend-days "$MAX_EXTEND_DAYS" \
                --mode "$mode" \
                --tag "$TAG" \
                --suffix "$SUFFIX" \
                --out-dir "$OUT_DIR" \
                --exclude-tickers "$EXCLUDE_TICKERS" \
                --tail-threshold "$tmax" \
                --utility-quantile "$uq" \
                --rank-by "$rnk" \
                --lambda-tail "$lam" \
                --topk "$topk" \
                --topk-weights "$weights" \
                --require-files "$REQUIRE_FILES"

              PICKS_PATH="$OUT_DIR/picks_${TAG}_gate_${SUFFIX}.csv"
              if [ ! -f "$PICKS_PATH" ]; then
                echo "[WARN] picks missing -> skip sim: $PICKS_PATH"
                continue
              fi

              # 2) Simulate (Top-1 or Top-2 auto by Weight column)
              python "$SIM" \
                --picks-path "$PICKS_PATH" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-extend-days "$MAX_EXTEND_DAYS" \
                --max-leverage-pct "1.0" \
                --tp1-frac "$TP1_FRAC" \
                --trail-stop "$tr" \
                --enable-trailing "$ENABLE_TRAILING" \
                --out-dir "$OUT_DIR" \
                --tag "$TAG" \
                --suffix "$SUFFIX"

              TRADES_PATH="$OUT_DIR/sim_engine_trades_${TAG}_gate_${SUFFIX}.parquet"
              if [ ! -f "$TRADES_PATH" ]; then
                echo "[WARN] trades missing -> skip summary: $TRADES_PATH"
                continue
              fi

              # 3) Summarize
              python "$SUM" \
                --trades-path "$TRADES_PATH" \
                --tag "$TAG" \
                --suffix "$SUFFIX" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-extend-days "$MAX_EXTEND_DAYS" \
                --out-dir "$OUT_DIR"

            done < <(split_csv "$TRAIL_STOPS")

          done

        done < <(printf "%s\n" "$LAMBDA_LIST")

      done < <(split_csv "$RANK_METRICS")

    done < <(split_csv "$UTILITY_QUANTILES")

  done < <(split_csv "$P_TAIL_THRESHOLDS")

done < <(split_csv "$GATE_MODES")

echo "[DONE] grid finished."