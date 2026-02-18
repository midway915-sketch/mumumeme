#!/usr/bin/env bash
# scripts/run_grid_workflow.sh
set -euo pipefail

# ----------------------------
# Helpers
# ----------------------------
csv_to_array() {
  local s="${1:-}"
  s="${s// /}"
  IFS=',' read -r -a ARR <<< "$s"
  echo "${ARR[@]}"
}

# suffix-safe float like 0.10 -> 0p10
fslug() {
  local x="$1"
  x="${x#-}"
  echo "$x" | sed 's/\./p/g'
}

# ----------------------------
# Env (must be provided by workflow)
# ----------------------------
: "${LABEL_KEY:?missing LABEL_KEY}"
: "${PROFIT_TARGET:?missing PROFIT_TARGET}"
: "${MAX_DAYS:?missing MAX_DAYS}"
: "${STOP_LEVEL:?missing STOP_LEVEL}"
: "${MAX_EXTEND_DAYS:?missing MAX_EXTEND_DAYS}"

: "${P_TAIL_THRESHOLDS:?missing P_TAIL_THRESHOLDS}"
: "${UTILITY_QUANTILES:?missing UTILITY_QUANTILES}"
: "${RANK_METRICS:?missing RANK_METRICS}"

: "${LAMBDA_TAIL:?missing LAMBDA_TAIL}"         # MUST be a single float
: "${GATE_MODES:?missing GATE_MODES}"

: "${TOPK_CONFIGS:?missing TOPK_CONFIGS}"       # ex: "1|1.0;2|0.7,0.3;2|0.6,0.4"

: "${ENABLE_TRAILING:?missing ENABLE_TRAILING}" # true/false
: "${TP1_FRAC:?missing TP1_FRAC}"               # ex 0.50
: "${TRAIL_STOPS:?missing TRAIL_STOPS}"         # ex 0.08,0.10,0.12

: "${OUT_DIR:=data/signals}"
: "${EXCLUDE_TICKERS:=SPY,^VIX}"
: "${REQUIRE_FILES:=}"                          # optional. if set, must be a comma-separated list

TAG="${LABEL_KEY}"

echo "[INFO] TAG=$TAG"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] TOPK_CONFIGS=$TOPK_CONFIGS"
echo "[INFO] TRAIL_STOPS=$TRAIL_STOPS TP1_FRAC=$TP1_FRAC ENABLE_TRAILING=$ENABLE_TRAILING"

mkdir -p "$OUT_DIR"

# Arrays
TAILS=( $(csv_to_array "$P_TAIL_THRESHOLDS") )
UQS=( $(csv_to_array "$UTILITY_QUANTILES") )
RANKS=( $(csv_to_array "$RANK_METRICS") )
MODES=( $(csv_to_array "$GATE_MODES") )
TRS=( $(csv_to_array "$TRAIL_STOPS") )

# Parse TOPK_CONFIGS: "1|1.0;2|0.7,0.3"
IFS=';' read -r -a TOPK_ITEMS <<< "${TOPK_CONFIGS}"

# Quick sanity: lambda-tail must be float
python - <<'PY'
import os
x=os.environ.get("LAMBDA_TAIL","")
try:
  float(x)
except Exception as e:
  raise SystemExit(f"[ERROR] LAMBDA_TAIL must be a single float like 0.05 (got {x!r})")
print("[OK] lambda_tail=", float(x))
PY

run_one() {
  local mode="$1"
  local tmax="$2"
  local uq="$3"
  local rank_by="$4"
  local topk="$5"
  local weights="$6"
  local tr="$7"

  local tslug="t0p$(echo "$tmax" | sed 's/0\.//; s/\.//g')"
  local uslug="q0p$(echo "$uq" | sed 's/0\.//; s/\.//g')"
  local rslug="r${rank_by}"
  local lslug="lam$(echo "$LAMBDA_TAIL" | sed 's/\./p/g')"
  local kslug="k${topk}"
  local wslug="w$(echo "$weights" | sed 's/\./p/g; s/,/_/g')"
  local tpslug="tp$(python - <<PY
x=float("$TP1_FRAC"); print(int(round(x*100)))
PY
)"
  local trslug="tr$(echo "$tr" | sed 's/\./p/g')"

  local suffix="${mode}_${tslug}_${uslug}_${rslug}_${lslug}_${kslug}_${wslug}_${tpslug}_${trslug}"

  echo "=============================="
  echo "[RUN] mode=$mode tail_max=$tmax u_q=$uq rank_by=$rank_by lambda=$LAMBDA_TAIL topk=$topk weights=$weights trail=$tr suffix=$suffix"
  echo "=============================="

  # 1) predict picks (TopK)
  python scripts/predict_gate.py \
    --profit-target "$PROFIT_TARGET" \
    --max-days "$MAX_DAYS" \
    --stop-level "$STOP_LEVEL" \
    --max-extend-days "$MAX_EXTEND_DAYS" \
    --mode "$mode" \
    --tag "$TAG" \
    --suffix "$suffix" \
    --out-dir "$OUT_DIR" \
    --exclude-tickers "$EXCLUDE_TICKERS" \
    --tail-threshold "$tmax" \
    --utility-quantile "$uq" \
    --rank-by "$rank_by" \
    --lambda-tail "$LAMBDA_TAIL" \
    --topk "$topk" \
    --require-files "${REQUIRE_FILES:-}"

  local picks_path="$OUT_DIR/picks_${TAG}_gate_${suffix}.csv"
  if [ ! -f "$picks_path" ]; then
    echo "[ERROR] picks not found: $picks_path"
    exit 1
  fi

  # 2) simulate (TopK compare)
  python scripts/simulate_single_position_engine.py \
    --picks-path "$picks_path" \
    --profit-target "$PROFIT_TARGET" \
    --max-days "$MAX_DAYS" \
    --stop-level "$STOP_LEVEL" \
    --max-extend-days "$MAX_EXTEND_DAYS" \
    --max-leverage-pct "1.0" \
    --topk "$topk" \
    --weights "$weights" \
    --enable-trailing "$ENABLE_TRAILING" \
    --tp1-frac "$TP1_FRAC" \
    --trail-stop "$tr" \
    --out-dir "$OUT_DIR"

  # 3) summarize (per run)
  local trades_path="$OUT_DIR/sim_engine_trades_${TAG}_gate_${suffix}.parquet"
  python scripts/summarize_sim_trades.py \
    --trades-path "$trades_path" \
    --tag "$TAG" \
    --suffix "$suffix" \
    --profit-target "$PROFIT_TARGET" \
    --max-days "$MAX_DAYS" \
    --stop-level "$STOP_LEVEL" \
    --max-extend-days "$MAX_EXTEND_DAYS" \
    --out-dir "$OUT_DIR"
}

# ----------------------------
# Main loops
# ----------------------------
for mode in "${MODES[@]}"; do
  for tmax in "${TAILS[@]}"; do
    for uq in "${UQS[@]}"; do
      for rank_by in "${RANKS[@]}"; do
        for item in "${TOPK_ITEMS[@]}"; do
          k="${item%%|*}"
          w="${item#*|}"
          for tr in "${TRS[@]}"; do
            run_one "$mode" "$tmax" "$uq" "$rank_by" "$k" "$w" "$tr"
          done
        done
      done
    done
  done
done

echo "[DONE] grid finished. outputs in $OUT_DIR"