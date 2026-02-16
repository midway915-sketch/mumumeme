#!/usr/bin/env bash
set -euo pipefail

cd "${GITHUB_WORKSPACE:-.}"

trim(){ echo "$1" | xargs; }

# --- 필수 env 체크
need_env() {
  local v
  for v in "$@"; do
    if [ -z "${!v:-}" ]; then
      echo "[ERROR] missing env: $v"
      exit 1
    fi
  done
}

need_env PROFIT_TARGET_INPUT HOLDING_DAYS_INPUT STOP_LEVEL_INPUT MAX_EXTEND_DAYS
need_env TAIL_MAX_LIST U_QUANTILE_LIST RANK_BY_LIST LAMBDA_TAIL

# --- gate lib 존재 확인
if [ ! -f scripts/gate_grid_lib.sh ]; then
  echo "[ERROR] scripts/gate_grid_lib.sh not found (must be committed)"
  ls -la scripts || true
  exit 1
fi
# shellcheck disable=SC1091
source scripts/gate_grid_lib.sh

# --- 배열 파싱 (콤마 없으면 1개짜리 배열로 들어감)
IFS=',' read -r -a PT_LIST <<< "${PROFIT_TARGET_INPUT}"
IFS=',' read -r -a H_LIST  <<< "${HOLDING_DAYS_INPUT}"
IFS=',' read -r -a SL_LIST <<< "${STOP_LEVEL_INPUT}"

make_tag () {
python - <<'PY'
import os
pt=float(os.environ["PT"])
h=int(os.environ["H"])
sl=float(os.environ["SL"])
ex=int(os.environ["EX"])
pt_tag=f"pt{int(round(pt*100))}"
sl_tag=f"sl{int(round(abs(sl)*100))}"
print(f"{pt_tag}_h{h}_{sl_tag}_ex{ex}")
PY
}

echo "[INFO] PT_LIST=${PROFIT_TARGET_INPUT}"
echo "[INFO] H_LIST=${HOLDING_DAYS_INPUT}"
echo "[INFO] SL_LIST=${STOP_LEVEL_INPUT}"
echo "[INFO] EX=${MAX_EXTEND_DAYS}"
echo "[INFO] tail_max_list=${TAIL_MAX_LIST}"
echo "[INFO] u_quantile_list=${U_QUANTILE_LIST}"
echo "[INFO] rank_by_list=${RANK_BY_LIST}"
echo "[INFO] lambda_tail=${LAMBDA_TAIL}"

# --- base prices/features는 (job에서 이미 1번 만들었고) 여기서는 반복 X

for PT0 in "${PT_LIST[@]}"; do
  PT0="$(trim "$PT0")"
  for H0 in "${H_LIST[@]}"; do
    H0="$(trim "$H0")"
    for SL0 in "${SL_LIST[@]}"; do
      SL0="$(trim "$SL0")"

      export PROFIT_TARGET="${PT0}"
      export HOLDING_DAYS="${H0}"
      export STOP_LEVEL="${SL0}"
      export MAX_EXTEND_DAYS="${MAX_EXTEND_DAYS}"

      export PT="${PT0}" H="${H0}" SL="${SL0}" EX="${MAX_EXTEND_DAYS}"
      export TAG
      TAG="$(make_tag)"
      export TAG

      echo ""
      echo "===================================================="
      echo "[GRID] TAG=${TAG} (PT=${PT0}, H=${H0}, SL=${SL0}, EX=${MAX_EXTEND_DAYS})"
      echo "===================================================="

      # --- 라벨/모델은 조합마다 재생성/재학습
      python scripts/build_labels.py \
        --profit-target "${PT0}" \
        --max-days "${H0}" \
        --stop-level "${SL0}" \
        --max-extend-days "${MAX_EXTEND_DAYS}"

      python scripts/train_model.py

      # --- tail/strategy 스크립트가 있으면 수행
      if [ -f scripts/build_strategy_labels.py ] && [ -f scripts/train_strategy_models.py ]; then
        python scripts/build_strategy_labels.py \
          --profit-target "${PT0}" \
          --max-days "${H0}" \
          --stop-level "${SL0}" \
          --max-extend-days "${MAX_EXTEND_DAYS}"

        python scripts/train_strategy_models.py \
          --profit-target "${PT0}" \
          --max-days "${H0}" \
          --stop-level "${SL0}" \
          --max-extend-days "${MAX_EXTEND_DAYS}"
      else
        echo "[WARN] strategy label/model scripts missing -> skipping"
      fi

      # --- tail 모델 있으면 tail gate 활성화
      if [ -f app/tail_model.pkl ] && [ -f app/tail_scaler.pkl ]; then
        export TAIL_OK=1
      else
        export TAIL_OK=0
      fi

      # --- gate grid + simulate
      run_gate_grid

      # --- sim_engine_summary_{TAG}_GATES_ALL.csv 생성(요약 파일들 합치기)
      python scripts/merge_sim_summaries.py --tag "${TAG}"

      # --- 최근10년 배수/최대홀딩 포함 gate_summary_{TAG}.csv 생성
      python scripts/make_gate_summary.py --tag "${TAG}" --max-days "${H0}" --recent-years 10

    done
  done
done

# --- 전체 집계(PT/H/SL × tail/u × rank)
python scripts/aggregate_gate_grid.py

echo "[DONE] grid run finished"
ls -la data/signals | tail -n 200 || true
