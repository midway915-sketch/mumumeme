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

list_trades_parquets() {
  python - <<'PY'
import glob, os
paths = glob.glob("data/**/sim_engine_trades*.parquet", recursive=True)
paths = [p for p in paths if os.path.isfile(p)]
paths.sort()
for p in paths:
    print(p)
PY
}

pick_newest_from_set() {
  # stdin: file paths, pick newest mtime
  python - <<'PY'
import sys, os
paths=[line.strip() for line in sys.stdin if line.strip()]
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

  # ✅ BEFORE list
  local before_list
  before_list="$(mktemp)"
  list_trades_parquets > "$before_list" || true

  # simulate
  python scripts/simulate_single_position_engine.py \
    --profit-target "$pt" \
    --max-days "$h" \
    --stop-level "$sl" \
    --max-extend-days "$ex" \
    --picks-path "$picks"

  # ✅ AFTER list
  local after_list
  after_list="$(mktemp)"
  list_trades_parquets > "$after_list" || true

  # ✅ DIFF = files that appear only after
  local new_list
  new_list="$(mktemp)"
  comm -13 <(sort "$before_list") <(sort "$after_list") > "$new_list" || true

  local trades_path=""
  trades_path="$(cat "$new_list" | pick_newest_from_set || true)"

  if [ -z "$trades_path" ]; then
    echo "[ERROR] No NEW sim_engine_trades parquet produced by simulate for this run."
    echo "[DEBUG] before:"
    sed -n '1,200p' "$before_list" || true
    echo "[DEBUG] after:"
    sed -n '1,200p' "$after_list" || true
    echo "[DEBUG] if simulate overwrites same filename, we must detect by mtime instead."
    # fallback: choose newest matching this tag
    trades_path="$(python - <<'PY' "$tag"
import sys, glob, os
tag=sys.argv[1]
paths=glob.glob(f"data/**/sim_engine_trades*{tag}*.parquet", recursive=True)
paths=[p for p in paths if os.path.isfile(p)]
paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
print(paths[0] if paths else "")
PY
)"
  fi

  if [ -z "$trades_path" ]; then
    echo "[ERROR] Could not find trades parquet for tag=${tag}"
    find data -type f -name "sim_engine_trades*.parquet" | sed -n '1,200p' || true
    exit 1
  fi

  echo "[INFO] using trades parquet: $trades_path"

  python scripts/summarize_sim_trades.py \
    --trades-path "${trades_path}" \
    --tag "${tag}" \
    --suffix "${suffix}" \
    --profit-target "$pt" \
    --max-days "$h" \
    --stop-level "$sl" \
    --max-extend-days "$ex" \
    --out-dir "data/signals"

  echo "[OK] finished: ${tag} ${suffix}"
}