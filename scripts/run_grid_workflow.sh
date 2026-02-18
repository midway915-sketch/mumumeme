#!/usr/bin/env bash
set -euo pipefail

# Required env:
#   PROFIT_TARGET, MAX_DAYS, STOP_LEVEL, MAX_EXTEND_DAYS
#   P_TAIL_THRESHOLDS (csv), UTILITY_QUANTILES (csv), RANK_METRICS (csv)
#   LAMBDA_TAIL
#   GATE_MODES (csv): none,tail,utility,tail_utility
#   P_SUCCESS_MINS (csv): e.g. 0.0,0.55,0.60,0.65

PRED="scripts/predict_gate.py"
SIM="scripts/simulate_single_position_engine.py"
SUM="scripts/summarize_sim_trades.py"

if [ ! -f "$PRED" ]; then echo "[ERROR] $PRED not found"; exit 1; fi
if [ ! -f "$SIM" ]; then echo "[ERROR] $SIM not found"; exit 1; fi
if [ ! -f "$SUM" ]; then echo "[ERROR] $SUM not found"; exit 1; fi

mkdir -p data/signals

pt_tag=$(python - <<PY
pt=float("${PROFIT_TARGET}")
h=int("${MAX_DAYS}")
sl=float("${STOP_LEVEL}")
ex=int("${MAX_EXTEND_DAYS}")
def fmt_pct(x): return str(int(round(x*100)))
def fmt_negpct(x): return str(int(round(abs(x)*100)))
print(f"pt{fmt_pct(pt)}_h{h}_sl{fmt_negpct(sl)}_ex{ex}")
PY
)

f2tag() {
  python - <<PY
s="${1}".strip()
if s.startswith("."): s="0"+s
if s.startswith("-."): s="-0"+s[1:]
s=s.replace("-","m").replace(".","p")
print(s)
PY
}

csv_to_array() {
  local s="${1}"
  python - <<PY
s="${s}"
parts=[p.strip() for p in s.split(",") if p.strip()!=""]
print("\n".join(parts))
PY
}

echo "[RUN] TAG=$pt_tag"
echo "[RUN] PT=$PROFIT_TARGET H=$MAX_DAYS SL=$STOP_LEVEL EX=$MAX_EXTEND_DAYS"

modes="$(csv_to_array "${GATE_MODES:-none,tail,utility,tail_utility}")"
tails="$(csv_to_array "${P_TAIL_THRESHOLDS:-0.20,0.30}")"
uqs="$(csv_to_array "${UTILITY_QUANTILES:-0.75,0.90}")"
ranks="$(csv_to_array "${RANK_METRICS:-utility}")"
psmins="$(csv_to_array "${P_SUCCESS_MINS:-0.0}")"

lambda_tail="${LAMBDA_TAIL:-0.05}"

for mode in $modes; do
  for t in $tails; do
    for q in $uqs; do
      for r in $ranks; do
        for ps in $psmins; do

          ttag="$(f2tag "$t")"
          qtag="$(f2tag "$q")"
          pstag="$(f2tag "$ps")"
          suffix="${mode}_t${ttag}_q${qtag}_r${r}_lam$(f2tag "$lambda_tail")_ps${pstag}"

          echo "=============================="
          echo "[RUN] mode=$mode tail_max=$t u_q=$q rank_by=$r lambda=$lambda_tail p_success_min=$ps"
          echo "      suffix=$suffix"
          echo "=============================="

          python "$PRED" \
            --profit-target "$PROFIT_TARGET" \
            --max-days "$MAX_DAYS" \
            --stop-level "$STOP_LEVEL" \
            --max-extend-days "$MAX_EXTEND_DAYS" \
            --mode "$mode" \
            --tag "$pt_tag" \
            --suffix "$suffix" \
            --tail-threshold "$t" \
            --utility-quantile "$q" \
            --rank-by "$r" \
            --lambda-tail "$lambda_tail" \
            --p-success-min "$ps" \
            --require-files "data/features/features_model.parquet,app/model.pkl,app/scaler.pkl"

          picks_path="data/signals/picks_${pt_tag}_gate_${suffix}.csv"

          python "$SIM" \
            --picks-path "$picks_path" \
            --profit-target "$PROFIT_TARGET" \
            --max-days "$MAX_DAYS" \
            --stop-level "$STOP_LEVEL" \
            --max-extend-days "$MAX_EXTEND_DAYS" \
            --max-leverage-pct "1.0" \
            --out-dir "data/signals"

          trades_path="data/signals/sim_engine_trades_${pt_tag}_gate_${suffix}.parquet"

          python "$SUM" \
            --trades-path "$trades_path" \
            --tag "$pt_tag" \
            --suffix "$suffix" \
            --profit-target "$PROFIT_TARGET" \
            --max-days "$MAX_DAYS" \
            --stop-level "$STOP_LEVEL" \
            --max-extend-days "$MAX_EXTEND_DAYS" \
            --out-dir "data/signals"

        done
      done
    done
  done
done

echo "[DONE] grid finished. outputs in data/signals"