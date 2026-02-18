#!/usr/bin/env bash
# scripts/run_grid_workflow.sh
set -euo pipefail

# -----------------------------
# Required env (from workflow)
# -----------------------------
: "${LABEL_KEY:?missing LABEL_KEY}"
: "${PROFIT_TARGET:?missing PROFIT_TARGET}"
: "${MAX_DAYS:?missing MAX_DAYS}"
: "${STOP_LEVEL:?missing STOP_LEVEL}"
: "${MAX_EXTEND_DAYS:?missing MAX_EXTEND_DAYS}"

: "${P_TAIL_THRESHOLDS:=0.10,0.20,0.30}"
: "${UTILITY_QUANTILES:=0.60,0.75,0.90}"
: "${RANK_METRICS:=utility}"
: "${LAMBDA_TAIL:=0.05}"
: "${GATE_MODES:=none,tail,utility,tail_utility}"

: "${TP1_FRAC:=0.50}"
: "${TRAIL_STOPS:=0.08,0.10,0.12}"
: "${ENABLE_TRAILING:=true}"

# TopK configs format:
#   "1|1.0;2|0.7,0.3;2|0.6,0.4"
: "${TOPK_CONFIGS:=1|1.0}"

: "${OUT_DIR:=data/signals}"
: "${EXCLUDE_TICKERS:=SPY,^VIX}"

# p_success min thresholds (grid)
: "${PS_MIN_THRESHOLDS:=0.0,0.4,0.5}"

# optional strict required files list (comma separated)
: "${REQUIRE_FILES:=}"

mkdir -p "$OUT_DIR"

echo "[INFO] TAG=$LABEL_KEY"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] TOPK_CONFIGS=$TOPK_CONFIGS"
echo "[INFO] TRAIL_STOPS=$TRAIL_STOPS TP1_FRAC=$TP1_FRAC ENABLE_TRAILING=$ENABLE_TRAILING"
echo "[INFO] PS_MIN_THRESHOLDS=$PS_MIN_THRESHOLDS"

# -----------------------------
# Helpers
# -----------------------------
split_csv () {
  # usage: split_csv "a,b,c" -> prints lines
  local s="$1"
  python - <<PY
s="${s}"
for x in [p.strip() for p in s.split(",") if p.strip()]:
    print(x)
PY
}

have_arg () {
  # check if python script supports a specific CLI flag
  # usage: have_arg scripts/predict_gate.py --ps-min
  local script="$1"
  local flag="$2"
  python "$script" -h 2>&1 | grep -q -- "$flag"
}

# -----------------------------
# Optional strict file existence
# -----------------------------
if [ -n "$REQUIRE_FILES" ]; then
  python - <<PY
from pathlib import Path
spec="${REQUIRE_FILES}"
missing=[p.strip() for p in spec.split(",") if p.strip() and not Path(p.strip()).exists()]
if missing:
    raise FileNotFoundError(f"Missing required files: {missing}")
print("[INFO] REQUIRE_FILES OK")
PY
fi

# -----------------------------
# Must have scored features with p_success/p_tail
# -----------------------------
python - <<'PY'
import pandas as pd
from pathlib import Path

p = Path("data/features/features_scored.parquet")
c = Path("data/features/features_scored.csv")
if not p.exists() and not c.exists():
    raise FileNotFoundError("features_scored missing (need data/features/features_scored.parquet or .csv)")

df = pd.read_parquet(p) if p.exists() else pd.read_csv(c)
need = ["Date","Ticker","p_success","p_tail"]
miss = [x for x in need if x not in df.columns]
if miss:
    raise ValueError(f"features_scored missing required columns: {miss}")

# hard-fail if both constant zeros (common silent bug)
u1 = df["p_success"].dropna().unique()
u2 = df["p_tail"].dropna().unique()
if len(u1)==1 and float(u1[0])==0.0 and len(u2)==1 and float(u2[0])==0.0:
    raise RuntimeError("p_success and p_tail are BOTH constant 0.0 -> scoring failed or wrong input.")
print("[INFO] features_scored sanity OK. rows=", len(df))
PY

# -----------------------------
# Grid loops
# -----------------------------
for MODE in $(split_csv "$GATE_MODES"); do
  for T in $(split_csv "$P_TAIL_THRESHOLDS"); do
    for Q in $(split_csv "$UTILITY_QUANTILES"); do
      for RANK_BY in $(split_csv "$RANK_METRICS"); do
        for PS_MIN in $(split_csv "$PS_MIN_THRESHOLDS"); do
          for TOPK_SPEC in ${TOPK_CONFIGS//;/ }; do
            TOPK="${TOPK_SPEC%%|*}"
            WEIGHTS="${TOPK_SPEC##*|}"

            for TR in $(split_csv "$TRAIL_STOPS"); do
              # suffix strings: avoid commas/unsafe chars
              T_TAG=$(python - <<PY
x=float("${T}")
print(str(x).replace(".","p"))
PY
)
              Q_TAG=$(python - <<PY
x=float("${Q}")
print(str(x).replace(".","p"))
PY
)
              L_TAG=$(python - <<PY
x=float("${LAMBDA_TAIL}")
print(str(x).replace(".","p"))
PY
)
              PS_TAG=$(python - <<PY
x=float("${PS_MIN}")
print(str(x).replace(".","p"))
PY
)
              W_TAG=$(echo "$WEIGHTS" | tr ',' '_' | tr '.' 'p')
              TR_TAG=$(python - <<PY
x=float("${TR}")
print(str(x).replace(".","p"))
PY
)
              TP_TAG=$(python - <<PY
x=float("${TP1_FRAC}")
print(int(round(x*100)))
PY
)

              SUFFIX="${MODE}_t${T_TAG}_q${Q_TAG}_r${RANK_BY}_lam${L_TAG}_ps${PS_TAG}_k${TOPK}_w${W_TAG}_tp${TP_TAG}_tr${TR_TAG}"

              echo "=============================="
              echo "[RUN] mode=$MODE tail_max=$T u_q=$Q rank_by=$RANK_BY lambda=$LAMBDA_TAIL ps_min=$PS_MIN topk=$TOPK weights=$WEIGHTS trail=$TR suffix=$SUFFIX"
              echo "=============================="

              # ---- predict_gate
              PRED_ARGS=(
                --profit-target "$PROFIT_TARGET"
                --max-days "$MAX_DAYS"
                --stop-level "$STOP_LEVEL"
                --max-extend-days "$MAX_EXTEND_DAYS"
                --mode "$MODE"
                --tag "$LABEL_KEY"
                --suffix "$SUFFIX"
                --out-dir "$OUT_DIR"
                --features-parq data/features/features_scored.parquet
                --features-csv  data/features/features_scored.csv
                --tail-threshold "$T"
                --utility-quantile "$Q"
                --rank-by "$RANK_BY"
                --lambda-tail "$LAMBDA_TAIL"
                --exclude-tickers "$EXCLUDE_TICKERS"
              )

              # optional: ps-min / topk if predict_gate supports them
              if have_arg scripts/predict_gate.py "--ps-min"; then
                PRED_ARGS+=( --ps-min "$PS_MIN" )
              fi
              if have_arg scripts/predict_gate.py "--topk"; then
                PRED_ARGS+=( --topk "$TOPK" )
              fi

              python scripts/predict_gate.py "${PRED_ARGS[@]}"

              PICKS_PATH="${OUT_DIR}/picks_${LABEL_KEY}_gate_${SUFFIX}.csv"

              # ---- simulate (TopK, trailing, tp1)
              python scripts/simulate_single_position_engine.py \
                --picks-path "$PICKS_PATH" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-extend-days "$MAX_EXTEND_DAYS" \
                --enable-trailing "$ENABLE_TRAILING" \
                --tp1-frac "$TP1_FRAC" \
                --trail-stop "$TR" \
                --topk "$TOPK" \
                --weights "$WEIGHTS" \
                --tag "$LABEL_KEY" \
                --suffix "$SUFFIX" \
                --out-dir "$OUT_DIR"

            done
          done
        done
      done
    done
  done
done

echo "[DONE] grid finished"