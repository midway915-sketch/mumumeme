#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# required envs
# ----------------------------
: "${PROFIT_TARGET:?missing PROFIT_TARGET}"
: "${MAX_DAYS:?missing MAX_DAYS}"
: "${STOP_LEVEL:?missing STOP_LEVEL}"
: "${MAX_EXTEND_DAYS:?missing MAX_EXTEND_DAYS}"

: "${P_TAIL_THRESHOLDS:?missing P_TAIL_THRESHOLDS}"   # e.g. "0.20,0.30"
: "${UTILITY_QUANTILES:?missing UTILITY_QUANTILES}"   # e.g. "0.75,0.90"
: "${RANK_METRICS:?missing RANK_METRICS}"             # e.g. "utility,ret_score"
: "${LAMBDA_TAIL:?missing LAMBDA_TAIL}"               # e.g. "0.05"

: "${GATE_MODES:=none,tail,utility,tail_utility}"     # default
: "${MAX_LEVERAGE_PCT:=1.0}"                          # 1.0 => 100%
: "${P_SUCCESS_MINS:=}"                               # optional. e.g. "0.45,0.55" or ""

OUT_DIR="data/signals"
mkdir -p "$OUT_DIR"

PRED="scripts/predict_gate.py"
SIM="scripts/simulate_single_position_engine.py"
SUM="scripts/summarize_sim_trades.py"

if [ ! -f "$PRED" ]; then echo "[ERROR] $PRED not found"; exit 1; fi
if [ ! -f "$SIM" ]; then echo "[ERROR] $SIM not found"; exit 1; fi
if [ ! -f "$SUM" ]; then echo "[ERROR] $SUM not found"; exit 1; fi

# ----------------------------
# TAG e.g. pt10_h40_sl10_ex30
# IMPORTANT: heredoc must allow bash var expansion -> <<PY (NO quotes)
# ----------------------------
pt_tag="$(python - <<PY
pt=float("${PROFIT_TARGET}")
md=int("${MAX_DAYS}")
sl=float("${STOP_LEVEL}")
ex=int("${MAX_EXTEND_DAYS}")
print(f"pt{int(round(pt*100)):02d}_h{md}_sl{int(round(abs(sl)*100)):02d}_ex{ex}")
PY
)"
TAG="$pt_tag"
echo "[INFO] TAG=$TAG"

# parse lists
IFS=',' read -r -a tail_list <<< "${P_TAIL_THRESHOLDS}"
IFS=',' read -r -a uq_list <<< "${UTILITY_QUANTILES}"
IFS=',' read -r -a rank_list <<< "${RANK_METRICS}"
IFS=',' read -r -a mode_list <<< "${GATE_MODES}"

# p_success mins list (optional)
if [ -n "${P_SUCCESS_MINS}" ]; then
  IFS=',' read -r -a ps_list <<< "${P_SUCCESS_MINS}"
else
  ps_list=("NA")
fi

# helper: normalize number suffix e.g. 0.30 -> 0p30
to_suffix_num () {
  python - <<PY
x=float("$1")
s=f"{x:.4f}".rstrip("0").rstrip(".")
print(s.replace(".","p").replace("-","m"))
PY
}

for mode in "${mode_list[@]}"; do
  mode="$(echo "$mode" | xargs)"
  for t in "${tail_list[@]}"; do
    t="$(echo "$t" | xargs)"
    t_suf="$(to_suffix_num "$t")"

    for uq in "${uq_list[@]}"; do
      uq="$(echo "$uq" | xargs)"
      uq_suf="$(to_suffix_num "$uq")"

      for rank_by in "${rank_list[@]}"; do
        rank_by="$(echo "$rank_by" | xargs)"

        for ps_min in "${ps_list[@]}"; do
          ps_min="$(echo "$ps_min" | xargs)"

          # suffix: include p_success min only if enabled
          if [ "$ps_min" = "NA" ]; then
            SUFFIX="${mode}_t${t_suf}_q${uq_suf}_r${rank_by}"
          else
            ps_suf="$(to_suffix_num "$ps_min")"
            SUFFIX="${mode}_t${t_suf}_q${uq_suf}_ps${ps_suf}_r${rank_by}"
          fi

          echo "=============================="
          echo "[RUN] mode=$mode tail_max=$t u_q=$uq rank_by=$rank_by p_success_min=$ps_min suffix=$SUFFIX"
          echo "=============================="

          PICKS="${OUT_DIR}/picks_${TAG}_gate_${SUFFIX}.csv"
          TRADES="${OUT_DIR}/sim_engine_trades_${TAG}_gate_${SUFFIX}.parquet"
          CURVE="${OUT_DIR}/sim_engine_curve_${TAG}_gate_${SUFFIX}.parquet"

          # ---- 1) predict_gate -> picks
          python "$PRED" \
            --profit-target "$PROFIT_TARGET" \
            --max-days "$MAX_DAYS" \
            --stop-level "$STOP_LEVEL" \
            --max-extend-days "$MAX_EXTEND_DAYS" \
            --mode "$mode" \
            --tag "$TAG" \
            --suffix "$SUFFIX" \
            --out-dir "$OUT_DIR" \
            --tail-threshold "$t" \
            --utility-quantile "$uq" \
            --rank-by "$rank_by" \
            --lambda-tail "$LAMBDA_TAIL" \
            --require-files "data/features/features_model.parquet"

          if [ ! -f "$PICKS" ]; then
            echo "[ERROR] picks not created: $PICKS"
            exit 1
          fi

          # ---- 1.5) apply p_success min filter (optional)
          if [ "$ps_min" != "NA" ]; then
            PICKS_PATH="$PICKS" P_SUCCESS_MIN="$ps_min" python - <<'PY'
import os
import pandas as pd
from pathlib import Path

p = Path(os.environ["PICKS_PATH"])
ps_min = float(os.environ["P_SUCCESS_MIN"])

df = pd.read_csv(p)

if "p_success" not in df.columns:
    print("[WARN] p_success not found in picks -> skip p_success_min filtering")
else:
    before = len(df)
    df["p_success"] = pd.to_numeric(df["p_success"], errors="coerce").fillna(0.0)
    df = df[df["p_success"] >= ps_min].copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values(["Date", "p_success"], ascending=[True, False]).drop_duplicates(["Date"], keep="first")

    after = len(df)
    print(f"[INFO] p_success_min applied: {before} -> {after} (min={ps_min})")
    df.to_csv(p, index=False)
PY
          fi

          # ---- 2) simulate -> trades/curve
          python "$SIM" \
            --picks-path "$PICKS" \
            --profit-target "$PROFIT_TARGET" \
            --max-days "$MAX_DAYS" \
            --stop-level "$STOP_LEVEL" \
            --max-extend-days "$MAX_EXTEND_DAYS" \
            --max-leverage-pct "$MAX_LEVERAGE_PCT" \
            --tag "$TAG" \
            --suffix "$SUFFIX" \
            --out-dir "$OUT_DIR"

          if [ ! -f "$TRADES" ] || [ ! -f "$CURVE" ]; then
            echo "[ERROR] trades/curve not created for $SUFFIX"
            echo "  trades=$TRADES exists? $(test -f "$TRADES" && echo yes || echo no)"
            echo "  curve =$CURVE exists? $(test -f "$CURVE" && echo yes || echo no)"
            exit 1
          fi

          # ---- 3) summarize (curve-path explicitly)
          python "$SUM" \
            --trades-path "$TRADES" \
            --curve-path "$CURVE" \
            --tag "$TAG" \
            --suffix "$SUFFIX" \
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

echo "[DONE] grid runs complete."
ls -la "$OUT_DIR" | sed -n '1,200p' || true