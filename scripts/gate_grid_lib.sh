#!/usr/bin/env bash
set -euo pipefail

split_csv() {
  echo "$1" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | awk 'NF>0{print}'
}

tok() {
  python - <<'PY' "$1"
import sys
x=float(sys.argv[1])
s=f"{x:.4f}".rstrip("0").rstrip(".")
s=s.replace(".","p").replace("-","m")
print(s)
PY
}

build_tag() {
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

find_latest_trades_parquet() {
  python - <<'PY'
import glob, os
paths = glob.glob("data/**/sim_engine_trades*.parquet", recursive=True)
paths = [p for p in paths if os.path.isfile(p)]
if not paths:
    raise SystemExit("")
paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
print(paths[0])
PY
}

run_one_gate() {
  local pt="$1"; local h="$2"; local sl="$3"; local ex="$4"
  local mode="$5"; local tail_max="$6"; local u_q="$7"; local rank_by="$8"
  local lambda_tail="$9"; local tau_gamma="${10}"

  local tag; tag="$(build_tag "$pt" "$h" "$sl" "$ex")"
  local suffix="${mode}_t$(tok "$tail_max")_q$(tok "$u_q")_r${rank_by}"
  local picks="data/signals/picks_${tag}_gate_${suffix}.csv"

  echo "=============================="
  echo "[RUN] tag=${tag}"
  echo "[RUN] mode=${mode} tail_max=${tail_max} u_q=${u_q} rank_by=${rank_by} lambda=${lambda_tail} tau_gamma=${tau_gamma}"
  echo "[RUN] suffix=${suffix}"
  echo "[RUN] picks=${picks}"
  echo "=============================="

  mkdir -p data/signals

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

  if [ ! -f "$picks" ]; then
    echo "[ERROR] predict_gate did not create picks: $picks"
    ls -la data/signals | sed -n '1,200p' || true
    exit 1
  fi

  python scripts/simulate_single_position_engine.py \
    --profit-target "$pt" \
    --max-days "$h" \
    --stop-level "$sl" \
    --max-extend-days "$ex" \
    --picks-path "$picks"

  local latest_trades
  latest_trades="$(find_latest_trades_parquet || true)"
  if [ -z "${latest_trades}" ]; then
    echo "[ERROR] No sim_engine_trades parquet found under data/** after simulate."
    find data -maxdepth 4 -type f | sed -n '1,200p' || true
    exit 1
  fi

  # =========================
  # ✅ HARD DEBUG (원인 추적용)
  # =========================
  echo "------------------------------"
  echo "[DEBUG] summarize_sim_trades.py head:"
  python - <<'PY'
from pathlib import Path
p = Path("scripts/summarize_sim_trades.py")
print("exists:", p.exists(), "path:", p.resolve())
if p.exists():
    lines = p.read_text(encoding="utf-8").splitlines()
    for i in range(min(40, len(lines))):
        print(f"{i+1:04d}: {lines[i]}")
PY

  echo "------------------------------"
  echo "[DEBUG] grep fillna in scripts/:"
  grep -R --line-number "fillna(" scripts | sed -n '1,200p' || true
  echo "------------------------------"

  # summarize (faulthandler로 traceback 강제)
  python -X faulthandler scripts/summarize_sim_trades.py \
    --trades-path "${latest_trades}" \
    --tag "${tag}" \
    --suffix "${suffix}" \
    --profit-target "$pt" \
    --max-days "$h" \
    --stop-level "$sl" \
    --max-extend-days "$ex" \
    --out-dir "data/signals"

  echo "[OK] finished: ${tag} ${suffix}"
}