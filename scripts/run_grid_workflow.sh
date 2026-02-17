#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------
# Env required (set by workflow)
# --------------------------------------------
: "${PROFIT_TARGET:?missing PROFIT_TARGET}"
: "${MAX_DAYS:?missing MAX_DAYS}"
: "${STOP_LEVEL:?missing STOP_LEVEL}"
: "${MAX_EXTEND_DAYS:?missing MAX_EXTEND_DAYS}"

: "${P_TAIL_THRESHOLDS:?missing P_TAIL_THRESHOLDS}"   # e.g. "0.20,0.30"
: "${UTILITY_QUANTILES:?missing UTILITY_QUANTILES}"   # e.g. "0.75,0.90"
: "${RANK_METRICS:?missing RANK_METRICS}"             # e.g. "utility,ret_score"
: "${LAMBDA_TAIL:?missing LAMBDA_TAIL}"               # e.g. "0.05"

# optional
GATE_MODES="${GATE_MODES:-none,tail,utility,tail_utility}"
TAU_GAMMA="${TAU_GAMMA:-0.0}"                         # reserved (not required now)
MAX_LEVERAGE_PCT="${MAX_LEVERAGE_PCT:-1.0}"           # 1.0 => 100% cap
LEV_PENALTY_K="${LEV_PENALTY_K:-1.0}"                 # for summarize leverage-adjusted metric

SIGNALS_DIR="${SIGNALS_DIR:-data/signals}"
FEATURES_MODEL="${FEATURES_MODEL:-data/features/features_model.parquet}"
FEATURES_MODEL_CSV="${FEATURES_MODEL_CSV:-data/features/features_model.csv}"
PRICES_PARQ="${PRICES_PARQ:-data/raw/prices.parquet}"
PRICES_CSV="${PRICES_CSV:-data/raw/prices.csv}"

mkdir -p "${SIGNALS_DIR}"

# --------------------------------------------
# Resolve python scripts paths (scripts/ or root)
# --------------------------------------------
resolve_py() {
  local name="$1"
  if [[ -f "scripts/${name}" ]]; then
    echo "scripts/${name}"
  elif [[ -f "${name}" ]]; then
    echo "${name}"
  else
    echo ""
  fi
}

PRED="$(resolve_py predict_gate.py)"
SIM="$(resolve_py simulate_single_position_engine.py)"
SUM="$(resolve_py summarize_sim_trades.py)"

if [[ -z "${PRED}" ]]; then
  echo "[ERROR] predict_gate.py not found (checked ./scripts and ./)"
  exit 1
fi
if [[ -z "${SIM}" ]]; then
  echo "[ERROR] simulate_single_position_engine.py not found (checked ./scripts and ./)"
  exit 1
fi
if [[ -z "${SUM}" ]]; then
  echo "[ERROR] summarize_sim_trades.py not found (checked ./scripts and ./)"
  exit 1
fi

# --------------------------------------------
# Build tag from PT/H/SL/EX inputs (canonical)
# --------------------------------------------
# pt10_h40_sl10_ex30
pt_tag="$(python - <<PY
pt=float("${PROFIT_TARGET}")
print(f"pt{int(round(pt*100)):02d}")
PY
)"
sl_tag="$(python - <<PY
sl=float("${STOP_LEVEL}")
print(f"sl{int(round(abs(sl)*100)):02d}")
PY
)"
TAG="${pt_tag}_h${MAX_DAYS}_${sl_tag}_ex${MAX_EXTEND_DAYS}"
echo "[INFO] TAG=${TAG}"

# --------------------------------------------
# Helpers
# --------------------------------------------
trim() { sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }

to_list() {
  # split comma string into lines
  echo "$1" | tr ',' '\n' | trim | sed '/^$/d'
}

fmt_q() {
  # float -> q0p75 style
  python - <<PY
x=float("$1")
s=f"{x:.2f}".rstrip("0").rstrip(".")
print(s.replace(".","p"))
PY
}

fmt_t() {
  # float -> t0p20 style
  python - <<PY
x=float("$1")
s=f"{x:.2f}".rstrip("0").rstrip(".")
print("t"+s.replace(".","p"))
PY
}

# --------------------------------------------
# Grid loops
# --------------------------------------------
echo "[INFO] modes=${GATE_MODES}"
echo "[INFO] p_tail_thresholds=${P_TAIL_THRESHOLDS}"
echo "[INFO] utility_quantiles=${UTILITY_QUANTILES}"
echo "[INFO] rank_metrics=${RANK_METRICS}"
echo "[INFO] lambda_tail=${LAMBDA_TAIL}"
echo "[INFO] max_leverage_pct=${MAX_LEVERAGE_PCT} lev_penalty_k=${LEV_PENALTY_K}"

modes="$(to_list "${GATE_MODES}")"
tails="$(to_list "${P_TAIL_THRESHOLDS}")"
uqnts="$(to_list "${UTILITY_QUANTILES}")"
ranks="$(to_list "${RANK_METRICS}")"

run_one() {
  local mode="$1"
  local tail="$2"
  local uq="$3"
  local rank="$4"

  local ttag qtag
  ttag="$(fmt_t "${tail}")"
  qtag="$(fmt_q "${uq}")"

  # suffix format: <mode>_<t>_<q>_r<rank>
  local suffix="${mode}_${ttag}_q${qtag}_r${rank}"

  echo "=============================="
  echo "[RUN] mode=${mode} tail_max=${tail} u_q=${uq} rank_by=${rank} suffix=${suffix}"
  echo "=============================="

  local picks_path="${SIGNALS_DIR}/picks_${TAG}_gate_${suffix}.csv"
  local trades_parq="${SIGNALS_DIR}/sim_engine_trades_${TAG}_gate_${suffix}.parquet"
  local curve_parq="${SIGNALS_DIR}/sim_engine_curve_${TAG}_gate_${suffix}.parquet"
  local summary_csv="${SIGNALS_DIR}/gate_summary_${TAG}_gate_${suffix}.csv"

  # 1) predict + pick (writes picks_*.csv)
  python "${PRED}" \
    --mode "${mode}" \
    --tag "${TAG}" \
    --suffix "${suffix}" \
    --out-dir "${SIGNALS_DIR}" \
    --features-parq "${FEATURES_MODEL}" \
    --features-csv "${FEATURES_MODEL_CSV}" \
    --tail-threshold "${tail}" \
    --utility-quantile "${uq}" \
    --rank-by "${rank}" \
    --lambda-tail "${LAMBDA_TAIL}" \
    --require-files

  if [[ ! -f "${picks_path}" ]]; then
    echo "[ERROR] picks not created: ${picks_path}"
    ls -la "${SIGNALS_DIR}" | sed -n '1,200p' || true
    exit 1
  fi

  # 2) simulate (writes trades/curve parquet)
  python "${SIM}" \
    --picks-path "${picks_path}" \
    --prices-parq "${PRICES_PARQ}" \
    --prices-csv "${PRICES_CSV}" \
    --profit-target "${PROFIT_TARGET}" \
    --max-days "${MAX_DAYS}" \
    --stop-level "${STOP_LEVEL}" \
    --max-extend-days "${MAX_EXTEND_DAYS}" \
    --max-leverage-pct "${MAX_LEVERAGE_PCT}" \
    --tag "${TAG}" \
    --suffix "${suffix}" \
    --out-dir "${SIGNALS_DIR}"

  if [[ ! -f "${trades_parq}" ]]; then
    echo "[ERROR] trades parquet not created: ${trades_parq}"
    exit 1
  fi
  if [[ ! -f "${curve_parq}" ]]; then
    echo "[ERROR] curve parquet not created: ${curve_parq}"
    exit 1
  fi

  # 3) summarize -> gate_summary_*.csv (includes leverage-adjusted metrics)
  python "${SUM}" \
    --trades-path "${trades_parq}" \
    --curve-path "${curve_parq}" \
    --tag "${TAG}" \
    --suffix "${suffix}" \
    --profit-target "${PROFIT_TARGET}" \
    --max-days "${MAX_DAYS}" \
    --stop-level "${STOP_LEVEL}" \
    --max-extend-days "${MAX_EXTEND_DAYS}" \
    --max-leverage-pct "${MAX_LEVERAGE_PCT}" \
    --lev-penalty-k "${LEV_PENALTY_K}" \
    --out-dir "${SIGNALS_DIR}"

  if [[ ! -f "${summary_csv}" ]]; then
    echo "[ERROR] summary csv not created: ${summary_csv}"
    exit 1
  fi

  echo "[OK] summary=${summary_csv}"
}

# Modes determine what axes to run:
# - none      : ignores tail/uq but still iterates for consistent suffix grid (use given values)
# - tail      : iter tail x ranks (uq fixed)
# - utility   : iter uq x ranks (tail fixed)
# - tail_utility : full tail x uq x ranks
for mode in ${modes}; do
  case "${mode}" in
    none)
      # pick first tail/uq values (still loop ranks)
      first_tail="$(echo "${tails}" | head -n 1)"
      first_uq="$(echo "${uqnts}" | head -n 1)"
      for rank in ${ranks}; do
        run_one "none" "${first_tail}" "${first_uq}" "${rank}"
      done
      ;;
    tail)
      first_uq="$(echo "${uqnts}" | head -n 1)"
      for tail in ${tails}; do
        for rank in ${ranks}; do
          run_one "tail" "${tail}" "${first_uq}" "${rank}"
        done
      done
      ;;
    utility)
      first_tail="$(echo "${tails}" | head -n 1)"
      for uq in ${uqnts}; do
        for rank in ${ranks}; do
          run_one "utility" "${first_tail}" "${uq}" "${rank}"
        done
      done
      ;;
    tail_utility)
      for tail in ${tails}; do
        for uq in ${uqnts}; do
          for rank in ${ranks}; do
            run_one "tail_utility" "${tail}" "${uq}" "${rank}"
          done
        done
      done
      ;;
    *)
      echo "[WARN] unknown mode=${mode} (skip)"
      ;;
  esac
done

echo "--------------------------------------------"
echo "[DONE] gate grid runs completed"
echo "[DEBUG] summaries:"
ls -la "${SIGNALS_DIR}"/gate_summary_"${TAG}"_gate_*.csv | sed -n '1,200p' || true
echo "[DEBUG] count summaries:"
ls -1 "${SIGNALS_DIR}"/gate_summary_"${TAG}"_gate_*.csv 2>/dev/null | wc -l || true
echo "--------------------------------------------"