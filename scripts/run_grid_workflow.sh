#!/usr/bin/env bash
set -euo pipefail

cd "${GITHUB_WORKSPACE:-.}"

trim(){ echo "$1" | xargs; }

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

# tau gamma grid(원하면 workflow input으로 뺄 수도 있지만, 스키마 유지 위해 env 기본값으로)
if [ -z "${TAU_GAMMA_LIST:-}" ]; then
  export TAU_GAMMA_LIST="0.03,0.05,0.08"
fi

if [ ! -f scripts/gate_grid_lib.sh ]; then
  echo "[ERROR] scripts/gate_grid_lib.sh not found"
  ls -la scripts || true
  exit 1
fi
# shellcheck disable=SC1091
source scripts/gate_grid_lib.sh

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

      # (1) 기존 성공/꼬리 라벨 + 성공확률 모델
      python scripts/build_labels.py \
        --profit-target "${PT0}" \
        --max-days "${H0}" \
        --stop-level "${SL0}" \
        --max-extend-days "${MAX_EXTEND_DAYS}"

      python scripts/train_model.py

      # tail 모델 존재 여부
      if [ -f app/tail_model.pkl ] && [ -f app/tail_scaler.pkl ]; then
        export TAIL_OK=1
      else
        export TAIL_OK=0
      fi

      # (2) τ 라벨 → τ 모델 → tau_pred 생성 (권장 루트)
      python scripts/build_tau_labels.py \
        --profit-target "${PT0}" \
        --max-days "${H0}" \
        --max-extend-days "${MAX_EXTEND_DAYS}"

      python scripts/train_tau_model.py
      python scripts/predict_tau.py

      # (3) gate grid 실행 (utility_time면 TAU_GAMMA_LIST로 내부 그리드 자동)
      run_gate_grid

      # (4) tag별 요약 생성 (BASE/A/B 포함)
      python scripts/merge_sim_summaries.py --tag "${TAG}" --max-days "${H0}" --recent-years 10
    done
  done
done

python scripts/aggregate_gate_grid.py

echo "[DONE] grid finished. data/signals:"
ls -la data/signals | tail -n 200 || true