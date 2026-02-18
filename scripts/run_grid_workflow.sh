#!/usr/bin/env bash
# scripts/run_grid_workflow.sh
set -euo pipefail

# ----------------------------
# Required env (from workflow)
# ----------------------------
: "${LABEL_KEY:?}"
: "${PROFIT_TARGET:?}"
: "${MAX_DAYS:?}"
: "${STOP_LEVEL:?}"
: "${MAX_EXTEND_DAYS:?}"
: "${P_TAIL_THRESHOLDS:?}"
: "${UTILITY_QUANTILES:?}"
: "${RANK_METRICS:?}"
: "${LAMBDA_TAIL:?}"
: "${GATE_MODES:?}"
: "${OUT_DIR:=data/signals}"
: "${EXCLUDE_TICKERS:=SPY,^VIX}"

# trailing / tp
: "${ENABLE_TRAILING:=true}"
: "${TP1_FRAC:=0.50}"
: "${TRAIL_STOPS:=0.08,0.10,0.12}"

# Top-k configs
# format: "K|w1,w2;K|w1,w2;..."
: "${TOPK_CONFIGS:=1|1.0;2|0.7,0.3;2|0.6,0.4}"

# file require check (optional)
: "${REQUIRE_FILES:=}"

TAG="${LABEL_KEY}"
mkdir -p "${OUT_DIR}"

echo "[INFO] TAG=${TAG}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] TOPK_CONFIGS=${TOPK_CONFIGS}"
echo "[INFO] TRAIL_STOPS=${TRAIL_STOPS} TP1_FRAC=${TP1_FRAC} ENABLE_TRAILING=${ENABLE_TRAILING}"

# script paths
PRED="scripts/predict_gate.py"
SIM="scripts/simulate_single_position_engine.py"
SUM="scripts/summarize_sim_trades.py"

if [ ! -f "${PRED}" ]; then echo "[ERROR] missing ${PRED}" ; exit 1; fi
if [ ! -f "${SIM}" ]; then echo "[ERROR] missing ${SIM}" ; exit 1; fi
if [ ! -f "${SUM}" ]; then echo "[ERROR] missing ${SUM}" ; exit 1; fi

# strict required files check (if set)
if [ -n "${REQUIRE_FILES}" ]; then
  IFS=',' read -r -a reqs <<< "${REQUIRE_FILES}"
  missing=0
  for f in "${reqs[@]}"; do
    ff="$(echo "$f" | xargs)"
    [ -z "$ff" ] && continue
    if [ ! -f "$ff" ]; then
      echo "[ERROR] required file missing: $ff"
      missing=1
    fi
  done
  [ "$missing" -eq 1 ] && exit 1
fi

# ----------------------------
# helpers (NO heredoc)
# ----------------------------
to_suffix_num() {
  # 0.10 -> 0p10, -0.10 -> m0p10
  python -c "import sys; x=float(sys.argv[1]); s=('m' if x<0 else ''); x=abs(x); out=f'{s}{x:.4f}'.rstrip('0').rstrip('.').replace('.','p'); print(out)" "$1"
}

tp_frac_pct() {
  # 0.50 -> 50
  python -c "import sys; x=float(sys.argv[1]); print(str(int(round(x*100))))" "$1"
}

# ----------------------------
# parse lists
# ----------------------------
IFS=',' read -r -a MODES <<< "${GATE_MODES}"
IFS=',' read -r -a TAILS <<< "${P_TAIL_THRESHOLDS}"
IFS=',' read -r -a UQS   <<< "${UTILITY_QUANTILES}"
IFS=',' read -r -a RANKS <<< "${RANK_METRICS}"
IFS=',' read -r -a TRS   <<< "${TRAIL_STOPS}"
IFS=';' read -r -a TOPKS <<< "${TOPK_CONFIGS}"

TPP="$(tp_frac_pct "${TP1_FRAC}")"

# ----------------------------
# run grid
# ----------------------------
for mode in "${MODES[@]}"; do
  mode="$(echo "$mode" | xargs)"
  [ -z "$mode" ] && continue

  for tmax in "${TAILS[@]}"; do
    tmax="$(echo "$tmax" | xargs)"
    [ -z "$tmax" ] && continue

    for uq in "${UQS[@]}"; do
      uq="$(echo "$uq" | xargs)"
      [ -z "$uq" ] && continue

      for rnk in "${RANKS[@]}"; do
        rnk="$(echo "$rnk" | xargs)"
        [ -z "$rnk" ] && continue

        for topk_spec in "${TOPKS[@]}"; do
          topk_spec="$(echo "$topk_spec" | xargs)"
          [ -z "$topk_spec" ] && continue

          # parse "K|w1,w2"
          if [[ "$topk_spec" != *"|"* ]]; then
            echo "[ERROR] TOPK_CONFIGS element must contain '|': $topk_spec"
            exit 1
          fi
          K="${topk_spec%%|*}"
          W="${topk_spec#*|}"

          for tr in "${TRS[@]}"; do
            tr="$(echo "$tr" | xargs)"
            [ -z "$tr" ] && continue

            # suffix build
            sm_t="$(to_suffix_num "${tmax}")"
            sm_q="$(to_suffix_num "${uq}")"
            sm_l="$(to_suffix_num "${LAMBDA_TAIL}")"
            sm_tr="$(to_suffix_num "${tr}")"

            # weights suffix
            w_sfx="$(echo "${W}" | tr ',' '_' | tr '.' 'p')"

            SUFFIX="${mode}_t${sm_t}_q${sm_q}_r${rnk}_lam${sm_l}_k${K}_w${w_sfx}_tp${TPP}_tr${sm_tr}"

            echo "=============================="
            echo "[RUN] mode=${mode} tail_max=${tmax} u_q=${uq} rank_by=${rnk} lambda=${LAMBDA_TAIL} topk=${K} weights=${W} trail=${tr} suffix=${SUFFIX}"
            echo "=============================="

            # 1) picks
            python "${PRED}" \
              --profit-target "${PROFIT_TARGET}" \
              --max-days "${MAX_DAYS}" \
              --stop-level "${STOP_LEVEL}" \
              --max-extend-days "${MAX_EXTEND_DAYS}" \
              --mode "${mode}" \
              --tag "${TAG}" \
              --suffix "${SUFFIX}" \
              --out-dir "${OUT_DIR}" \
              --tail-threshold "${tmax}" \
              --utility-quantile "${uq}" \
              --rank-by "${rnk}" \
              --lambda-tail "${LAMBDA_TAIL}" \
              --exclude-tickers "${EXCLUDE_TICKERS}"

            PICKS_PATH="${OUT_DIR}/picks_${TAG}_gate_${SUFFIX}.csv"

            # 2) simulate (Top-k bundle + trailing + leverage cap)
            python "${SIM}" \
              --picks-path "${PICKS_PATH}" \
              --profit-target "${PROFIT_TARGET}" \
              --max-days "${MAX_DAYS}" \
              --stop-level "${STOP_LEVEL}" \
              --max-extend-days "${MAX_EXTEND_DAYS}" \
              --max-leverage-pct "1.0" \
              --topk "${K}" \
              --weights "${W}" \
              --enable-trailing "${ENABLE_TRAILING}" \
              --tp1-frac "${TP1_FRAC}" \
              --trail-stop "${tr}" \
              --tag "${TAG}" \
              --suffix "${SUFFIX}" \
              --out-dir "${OUT_DIR}"

            TRADES_PATH="${OUT_DIR}/sim_engine_trades_${TAG}_gate_${SUFFIX}.parquet"

            # 3) summarize -> gate_summary_*.csv
            python "${SUM}" \
              --trades-path "${TRADES_PATH}" \
              --tag "${TAG}" \
              --suffix "${SUFFIX}" \
              --profit-target "${PROFIT_TARGET}" \
              --max-days "${MAX_DAYS}" \
              --stop-level "${STOP_LEVEL}" \
              --max-extend-days "${MAX_EXTEND_DAYS}" \
              --out-dir "${OUT_DIR}"

          done
        done
      done
    done
  done
done

echo "[DONE] grid finished"