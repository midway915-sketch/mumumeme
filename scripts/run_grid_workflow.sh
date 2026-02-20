#!/usr/bin/env bash
set -euo pipefail

source scripts/gate_grid_lib.sh

# Required envs
: "${PROFIT_TARGET:?}"
: "${MAX_DAYS:?}"
: "${STOP_LEVEL:?}"
: "${P_TAIL_THRESHOLDS:?}"
: "${UTILITY_QUANTILES:?}"
: "${RANK_METRICS:?}"
: "${LAMBDA_TAIL:?}"
: "${GATE_MODES:?}"
: "${TRAIL_STOPS:?}"
: "${TP1_FRAC:?}"
: "${ENABLE_TRAILING:?}"
: "${TOPK_CONFIGS:?}"
: "${PS_MINS:?}"
: "${MAX_LEVERAGE_PCT:?}"
: "${EXCLUDE_TICKERS:?}"
: "${REQUIRE_FILES:?}"
: "${LABEL_KEY:?}"

# ✅ MAX_EXTEND_DAYS 없으면 H//2
if [ -z "${MAX_EXTEND_DAYS:-}" ]; then
  MAX_EXTEND_DAYS="$(python - <<PY
H=int("${MAX_DAYS}")
print(max(1, H//2))
PY
)"
  export MAX_EXTEND_DAYS
fi

# tau_gamma는 지금 파이프라인에서 고정값으로라도 세팅(없으면 0.0)
TAU_GAMMA="${TAU_GAMMA:-0.0}"

OUT_DIR="${OUT_DIR:-data/signals}"
mkdir -p "$OUT_DIR"

split_csv() {
  local s="$1"
  python - <<PY
s = """$s"""
parts=[p.strip() for p in s.split(",") if p.strip()]
for p in parts:
  print(p)
PY
}

split_scsv() {
  local s="$1"
  python - <<PY
s = """$s"""
parts=[p.strip() for p in s.split(";") if p.strip()]
for p in parts:
  print(p)
PY
}

first_csv_item() {
  local s="$1"
  python - <<PY
s = """$s"""
parts=[p.strip() for p in s.split(",") if p.strip()]
print(parts[0] if parts else "")
PY
}

DEFAULT_TMAX="$(first_csv_item "$P_TAIL_THRESHOLDS")"
DEFAULT_UQ="$(first_csv_item "$UTILITY_QUANTILES")"

while read -r mode; do
  mode="$(echo "$mode" | tr '[:upper:]' '[:lower:]' | xargs)"
  [ -z "$mode" ] && continue

  if [ "$mode" = "none" ]; then
    TMAX_LIST="$DEFAULT_TMAX"
    UQ_LIST="$DEFAULT_UQ"
  elif [ "$mode" = "tail" ]; then
    TMAX_LIST="$P_TAIL_THRESHOLDS"
    UQ_LIST="$DEFAULT_UQ"
  elif [ "$mode" = "utility" ]; then
    TMAX_LIST="$DEFAULT_TMAX"
    UQ_LIST="$UTILITY_QUANTILES"
  elif [ "$mode" = "tail_utility" ]; then
    TMAX_LIST="$P_TAIL_THRESHOLDS"
    UQ_LIST="$UTILITY_QUANTILES"
  else
    echo "[ERROR] Unknown mode: $mode"
    exit 1
  fi

  while read -r tmax; do
    while read -r uq; do
      while read -r rank_by; do
        while read -r psmin; do
          while read -r topk_line; do
            K="${topk_line%%|*}"
            W="${topk_line#*|}"

            # ✅ 여기서 run_one_gate 호출을 “항상 10개 인자”로 통일
            # (pt, h, sl, ex, mode, tail_max, u_q, rank_by, lambda_tail, tau_gamma)
            run_one_gate \
              "${PROFIT_TARGET}" \
              "${MAX_DAYS}" \
              "${STOP_LEVEL}" \
              "${MAX_EXTEND_DAYS}" \
              "${mode}" \
              "${tmax}" \
              "${uq}" \
              "${rank_by}" \
              "${LAMBDA_TAIL}" \
              "${TAU_GAMMA}"

          done < <(split_scsv "$TOPK_CONFIGS")
        done < <(split_csv "$PS_MINS")
      done < <(split_csv "$RANK_METRICS")
    done < <(split_csv "$UQ_LIST")
  done < <(split_csv "$TMAX_LIST")

done < <(split_csv "$GATE_MODES")

echo "[DONE] grid finished"