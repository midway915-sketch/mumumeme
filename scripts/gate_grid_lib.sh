#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Helpers
# -------------------------
trim() { awk '{$1=$1;print}'; }

split_csv() {
  # usage: split_csv "0.1,0.2, 0.3"
  echo "$1" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | awk 'NF>0{print}'
}

tok() {
  # float -> token (0.75 -> 0p75, -0.10 -> m0p10)
  python - <<'PY' "$1"
import sys
x=float(sys.argv[1])
s=f"{x:.4f}".rstrip("0").rstrip(".")
s=s.replace(".","p").replace("-","m")
print(s)
PY
}

build_tag() {
  # profit_target max_days stop_level max_extend_days
  python - <<'PY' "$1" "$2" "$3" "$4"
import sys
pt=float(sys.argv[1])
h=int(float(sys.argv[2]))
sl=float(sys.argv[3])
ex=int(float(sys.argv[4]))
pt_i=int(round(pt*100))
sl_i=int(round(abs(sl)*100))
print(f"pt{pt_i}_h{h}_sl{sl_i}_ex{ex}")
PY
}

# -------------------------
# Core runner: one gate config
# -------------------------
run_one_gate() {
  # args:
  #   pt h sl ex mode tail_max u_quantile rank_by lambda_tail tau_gamma
  local pt="$1"; local h="$2"; local sl="$3"; local ex="$4"
  local mode="$5"; local tail_max="$6"; local u_q="$7"; local rank_by="$8"
  local lambda_tail="$9"; local tau_gamma="${10}"

  local tag; tag="$(build_tag "$pt" "$h" "$sl" "$ex")"

  # ✅ 다운스트림이 찾는 suffix 규칙을 "고정"한다 (gamma 같은 거 파일명에 절대 안 넣음)
  local suffix="${mode}_t$(tok "$tail_max")_q$(tok "$u_q")_r${rank_by}"

  # ✅ 파일명도 downstream 기대값으로 강제
  local picks="data/signals/picks_${tag}_gate_${suffix}.csv"

  echo "=============================="
  echo "[RUN] tag=${tag}"
  echo "[RUN] mode=${mode} tail_max=${tail_max} u_q=${u_q} rank_by=${rank_by} lambda=${lambda_tail} tau_gamma=${tau_gamma}"
  echo "[RUN] suffix=${suffix}"
  echo "[RUN] picks=${picks}"
  echo "=============================="

  mkdir -p data/signals

  # ---- predict_gate: 반드시 picks 파일을 기대 이름으로 생성하게 강제 ----
  python scripts/predict_gate.py \
    --profit-target "$pt" \
    --max-days "$h" \
    --stop-level "$sl" \
    --max-extend-days "$ex" \
    --mode "$mode" \
    --tail-threshold "$tail_max" \
    --utility-quantile "$u_q" \
    --rank-by "$rank_by" \
    --lambda-tail "$lambda_tail" \
    --tau-gamma "$tau_gamma" \
    --suffix "$suffix" \
    --out-csv "$picks"

  # ---- hard check (여기서 없으면 predict_gate 저장이 안 된 것) ----
  if [ ! -f "$picks" ]; then
    echo "[ERROR] predict_gate did not create picks: $picks"
    echo "[DEBUG] list data/signals (picks*):"
    ls -la data/signals | sed -n '1,200p' || true
    exit 1
  fi

  # ---- simulate (너 repo의 simulate CLI에 맞게 호출) ----
  # 아래는 '일반적인' 형태고, 네 simulate 파일의 인자명과 다르면 여기만 맞추면 됨.
  python scripts/simulate_single_position_engine.py \
    --profit-target "$pt" \
    --max-days "$h" \
    --stop-level "$sl" \
    --max-extend-days "$ex" \
    --picks-csv "$picks" \
    --suffix "$suffix" \
    --tag "$tag"

  echo "[OK] finished: ${tag} ${suffix}"
}