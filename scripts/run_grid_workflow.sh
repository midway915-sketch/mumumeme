#!/usr/bin/env bash
set -euo pipefail

# scripts/run_grid_workflow.sh

PT="${PROFIT_TARGET:?}"
H="${MAX_DAYS:?}"
SL="${STOP_LEVEL:?}"
EX="${MAX_EXTEND_DAYS:?}"

P_TAILS="${P_TAIL_THRESHOLDS:?}"          # e.g. "0.20,0.30"
U_QS="${UTILITY_QUANTILES:?}"             # e.g. "0.75,0.60,0.50"
RANKS="${RANK_METRICS:?}"                 # e.g. "utility,ret_score,p_success"
LAM="${LAMBDA_TAIL:?}"                    # e.g. "0.05"

GATE_MODES="${GATE_MODES:-none,tail,utility,tail_utility}"

TAG="pt$(python - <<PY
pt=float("$PT"); print(int(round(pt*100)))
PY
)_h${H}_sl$(python - <<PY
sl=float("$SL"); print(int(round(abs(sl)*100)))
PY
)_ex${EX}"

PRED="scripts/predict_gate.py"
SIM="scripts/simulate_single_position_engine.py"
SUM="scripts/summarize_sim_trades.py"

if [ ! -f "$PRED" ]; then echo "[ERROR] $PRED not found"; exit 1; fi
if [ ! -f "$SIM" ]; then echo "[ERROR] $SIM not found"; exit 1; fi
if [ ! -f "$SUM" ]; then echo "[ERROR] $SUM not found"; exit 1; fi

mkdir -p data/signals

# tail model 존재 확인 -> 없으면 tail 계열 모드 스킵
TAIL_OK=0
if [ -f "app/tail_model_${TAG}.pkl" ] && [ -f "app/tail_scaler_${TAG}.pkl" ]; then
  TAIL_OK=1
elif [ -f "app/tail_model.pkl" ] && [ -f "app/tail_scaler.pkl" ]; then
  TAIL_OK=1
fi

if [ "$TAIL_OK" = "0" ]; then
  echo "[WARN] tail model missing -> will skip tail / tail_utility modes"
fi

# CSV split helper
split_csv () {
  local s="$1"
  python - <<PY
s="$s"
print("\n".join([x.strip() for x in s.split(",") if x.strip()]))
PY
}

# sanitize suffix for file name
sanitize () {
  python - <<PY
import re
s=r"""$1"""
print(re.sub(r"[^A-Za-z0-9_.-]+","_",s))
PY
}

for mode in $(split_csv "$GATE_MODES"); do
  if [ "$TAIL_OK" = "0" ] && { [ "$mode" = "tail" ] || [ "$mode" = "tail_utility" ]; }; then
    continue
  fi

  for rank_by in $(split_csv "$RANKS"); do
    for tail_max in $(split_csv "$P_TAILS"); do

      # utility 모드가 아니면 u_q는 의미 없지만, suffix 통일 위해 그대로 루프
      # (원하면 none/tail에서는 u_q 루프를 1회로 줄일 수도 있음)
      for u_qs in "$U_QS"; do
        # u_qs는 "0.75,0.60,0.50" 전체 문자열을 그대로 predict_gate에 넘김
        # suffix는 보기 좋게 q0p75_0p60_0p50 로 표기
        U_TAG="$(python - <<PY
s="$u_qs"
parts=[p.strip() for p in s.split(",") if p.strip()]
def qtag(x):
  x=float(x)
  return "q"+str(int(round(x*100))).replace("-", "m")
print("_".join(qtag(p) for p in parts))
PY
)"
        base_suffix="${mode}_t$(python - <<PY
x=float("$tail_max"); print(str(x).replace(".","p"))
PY
)_${U_TAG}_r${rank_by}_lam$(python - <<PY
x=float("$LAM"); print(str(x).replace(".","p"))
PY
)"
        suffix="$(sanitize "$base_suffix")"

        echo "=============================="
        echo "[RUN] mode=$mode tail_max=$tail_max u_qs=$u_qs rank_by=$rank_by suffix=$suffix"
        echo "=============================="

        python "$PRED" \
          --profit-target "$PT" --max-days "$H" --stop-level "$SL" --max-extend-days "$EX" \
          --mode "$mode" --tag "$TAG" --suffix "$suffix" --out-dir "data/signals" \
          --tail-threshold "$tail_max" --utility-quantile "$u_qs" \
          --rank-by "$rank_by" --lambda-tail "$LAM"

        PICKS="data/signals/picks_${TAG}_gate_${suffix}.csv"
        if [ ! -f "$PICKS" ]; then
          echo "[ERROR] picks file missing: $PICKS"
          exit 1
        fi

        # simulate engine: picks-path 필수인 너 환경 기준으로 호출
        python "$SIM" \
          --profit-target "$PT" --max-days "$H" --stop-level "$SL" --max-extend-days "$EX" \
          --picks-path "$PICKS" \
          --out-dir "data/signals" \
          --tag "$TAG" \
          --suffix "$suffix"

        TRADES="data/signals/sim_engine_trades_${TAG}_gate_${suffix}.parquet"
        if [ ! -f "$TRADES" ]; then
          echo "[ERROR] trades parquet missing: $TRADES"
          exit 1
        fi

        python "$SUM" \
          --trades-path "$TRADES" \
          --tag "$TAG" \
          --suffix "$suffix" \
          --profit-target "$PT" --max-days "$H" --stop-level "$SL" --max-extend-days "$EX" \
          --out-dir "data/signals"
      done
    done
  done
done

echo "[DONE] run_grid_workflow.sh"