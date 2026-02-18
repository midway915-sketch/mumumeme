#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Grid runner for:
# - gate modes (none/tail/utility/tail_utility)
# - tail thresholds
# - utility quantiles
# - rank metrics
# - p_success minimum thresholds (ps_min)
# - TopK configs (1 or 2 with weights)
# - TP1 + trailing stops
# Produces:
# - picks_*.csv / picks_meta_*.json
# - sim_engine_trades_*.parquet / sim_engine_curve_*.parquet
# - gate_summary_*.csv  (for aggregate_gate_grid.py)
# ============================================================

TAG="${LABEL_KEY:-${TAG:-run}}"
OUT_DIR="${OUT_DIR:-data/signals}"
mkdir -p "$OUT_DIR"

PROFIT_TARGET="${PROFIT_TARGET:-0.10}"
MAX_DAYS="${MAX_DAYS:-40}"
STOP_LEVEL="${STOP_LEVEL:--0.10}"
MAX_EXTEND_DAYS="${MAX_EXTEND_DAYS:-30}"

P_TAIL_THRESHOLDS="${P_TAIL_THRESHOLDS:-0.10,0.20,0.30}"
UTILITY_QUANTILES="${UTILITY_QUANTILES:-0.60,0.75,0.90}"
RANK_METRICS="${RANK_METRICS:-utility,ret_score}"
LAMBDA_TAIL="${LAMBDA_TAIL:-0.05}"

# ✅ NEW: p_success min thresholds grid
PS_MIN_THRESHOLDS="${PS_MIN_THRESHOLDS:-0.00,0.05,0.10}"

GATE_MODES="${GATE_MODES:-none,tail,utility,tail_utility}"

# Exit / portfolio
ENABLE_TRAILING="${ENABLE_TRAILING:-true}"   # true/false
TP1_FRAC="${TP1_FRAC:-0.50}"                 # 0~1
TRAIL_STOPS="${TRAIL_STOPS:-0.08,0.10,0.12}" # comma list
TOPK_CONFIGS="${TOPK_CONFIGS:-1|1.0;2|0.7,0.3;2|0.6,0.4}"  # semicolon list

EXCLUDE_TICKERS="${EXCLUDE_TICKERS:-SPY,^VIX}"

# Strict file checks (predict_gate.py expects ONE argument string if provided)
REQUIRE_FILES="${REQUIRE_FILES:-}"

echo "[INFO] TAG=$TAG"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] TOPK_CONFIGS=$TOPK_CONFIGS"
echo "[INFO] TRAIL_STOPS=$TRAIL_STOPS TP1_FRAC=$TP1_FRAC ENABLE_TRAILING=$ENABLE_TRAILING"
echo "[INFO] PS_MIN_THRESHOLDS=$PS_MIN_THRESHOLDS"
echo "[INFO] REQUIRE_FILES=${REQUIRE_FILES:-<empty>}"

# --------------------------
# helpers
# --------------------------
norm_num() {
  # 0.05 -> 0p05, 1.2 -> 1p2, -0.1 -> m0p1
  local x="$1"
  x="${x//-/m}"
  x="${x//./p}"
  echo "$x"
}

trim() {
  local s="$1"
  # shellcheck disable=SC2001
  echo "$s" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

split_csv() {
  # split_csv "a,b,c" -> echoes lines
  local s="$1"
  s="$(trim "$s")"
  if [[ -z "$s" ]]; then
    return 0
  fi
  echo "$s" | tr ',' '\n' | sed '/^[[:space:]]*$/d'
}

# --------------------------
# loops
# --------------------------
enable_trailing_lc="$(echo "$ENABLE_TRAILING" | tr '[:upper:]' '[:lower:]')"
if [[ "$enable_trailing_lc" == "true" || "$enable_trailing_lc" == "1" || "$enable_trailing_lc" == "yes" || "$enable_trailing_lc" == "y" ]]; then
  TRAIL_LIST="$(split_csv "$TRAIL_STOPS")"
else
  TRAIL_LIST="NA"
fi

# TOPK_CONFIGS parsing:
# "1|1.0;2|0.7,0.3;2|0.6,0.4"
IFS=';' read -r -a TOPK_ARR <<< "$TOPK_CONFIGS"

for mode in $(split_csv "$GATE_MODES"); do
  for tail_max in $(split_csv "$P_TAIL_THRESHOLDS"); do
    for uq in $(split_csv "$UTILITY_QUANTILES"); do
      for rank_by in $(split_csv "$RANK_METRICS"); do
        for ps_min in $(split_csv "$PS_MIN_THRESHOLDS"); do
          for tkconf in "${TOPK_ARR[@]}"; do
            tkconf="$(trim "$tkconf")"
            [[ -z "$tkconf" ]] && continue

            # tkconf format: TOPK|w1,w2  (or 1|1.0)
            TOPK="$(echo "$tkconf" | cut -d'|' -f1)"
            WEIGHTS="$(echo "$tkconf" | cut -d'|' -f2-)"
            TOPK="$(trim "$TOPK")"
            WEIGHTS="$(trim "$WEIGHTS")"

            # weights suffix friendly
            w_sfx="$(echo "$WEIGHTS" | tr ',' '_' | sed 's/\./p/g')"

            for tr in $TRAIL_LIST; do
              if [[ "$tr" == "NA" ]]; then
                tr_sfx="notrail"
                tr_arg="0.10"  # unused when enable_trailing=false in engine, but engine wants a float
              else
                tr_sfx="tr$(norm_num "$tr")"
                tr_arg="$tr"
              fi

              # suffix build
              # Example:
              # tail_utility_t0p1_q0p6_rutility_lam0p05_ps0p05_k2_w0p7_0p3_tp50_tr0p08
              suf_mode="$mode"
              suf_tail="t$(norm_num "$tail_max")"
              suf_uq="q$(norm_num "$uq")"
              suf_rank="r${rank_by}"
              suf_lam="lam$(norm_num "$LAMBDA_TAIL")"
              suf_ps="ps$(norm_num "$ps_min")"
              suf_k="k${TOPK}"
              suf_w="w${w_sfx}"
              tp_pct="$(python - <<PY
v=float("$TP1_FRAC")
print(int(round(v*100)))
PY
)"
              suf_tp="tp${tp_pct}"
              suffix="${suf_mode}_${suf_tail}_${suf_uq}_${suf_rank}_${suf_lam}_${suf_ps}_${suf_k}_${suf_w}_${suf_tp}_${tr_sfx}"

              echo "=============================="
              echo "[RUN] mode=$mode tail_max=$tail_max u_q=$uq rank_by=$rank_by lambda=$LAMBDA_TAIL ps_min=$ps_min topk=$TOPK weights=$WEIGHTS trail=$tr_sfx suffix=$suffix"
              echo "=============================="

              # 1) predict picks (TopK rows/day)
              if [[ -n "${REQUIRE_FILES:-}" ]]; then
                python scripts/predict_gate.py \
                  --profit-target "$PROFIT_TARGET" \
                  --max-days "$MAX_DAYS" \
                  --stop-level "$STOP_LEVEL" \
                  --max-extend-days "$MAX_EXTEND_DAYS" \
                  --mode "$mode" \
                  --tag "$TAG" \
                  --suffix "$suffix" \
                  --out-dir "$OUT_DIR" \
                  --tail-threshold "$tail_max" \
                  --utility-quantile "$uq" \
                  --rank-by "$rank_by" \
                  --lambda-tail "$LAMBDA_TAIL" \
                  --topk "$TOPK" \
                  --ps-min "$ps_min" \
                  --exclude-tickers "$EXCLUDE_TICKERS" \
                  --require-files "$REQUIRE_FILES"
              else
                python scripts/predict_gate.py \
                  --profit-target "$PROFIT_TARGET" \
                  --max-days "$MAX_DAYS" \
                  --stop-level "$STOP_LEVEL" \
                  --max-extend-days "$MAX_EXTEND_DAYS" \
                  --mode "$mode" \
                  --tag "$TAG" \
                  --suffix "$suffix" \
                  --out-dir "$OUT_DIR" \
                  --tail-threshold "$tail_max" \
                  --utility-quantile "$uq" \
                  --rank-by "$rank_by" \
                  --lambda-tail "$LAMBDA_TAIL" \
                  --topk "$TOPK" \
                  --ps-min "$ps_min" \
                  --exclude-tickers "$EXCLUDE_TICKERS"
              fi

              picks_path="$OUT_DIR/picks_${TAG}_gate_${suffix}.csv"
              if [[ ! -f "$picks_path" ]]; then
                echo "[ERROR] picks not found: $picks_path"
                exit 1
              fi

              # 2) simulate
              python scripts/simulate_single_position_engine.py \
                --picks-path "$picks_path" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-extend-days "$MAX_EXTEND_DAYS" \
                --max-leverage-pct "1.0" \
                --enable-trailing "$ENABLE_TRAILING" \
                --tp1-frac "$TP1_FRAC" \
                --trail-stop "$tr_arg" \
                --topk "$TOPK" \
                --weights "$WEIGHTS" \
                --tag "$TAG" \
                --suffix "$suffix" \
                --out-dir "$OUT_DIR"

              curve_path="$OUT_DIR/sim_engine_curve_${TAG}_gate_${suffix}.parquet"
              trades_path="$OUT_DIR/sim_engine_trades_${TAG}_gate_${suffix}.parquet"

              # 3) write summary CSV (gate_summary_*.csv) for aggregator
              python - <<PY
import math
import pandas as pd
from pathlib import Path

out_dir = Path("$OUT_DIR")
tag = "$TAG"
suffix = "$suffix"
curve_path = out_dir / f"sim_engine_curve_{tag}_gate_{suffix}.parquet"
trades_path = out_dir / f"sim_engine_trades_{tag}_gate_{suffix}.parquet"
summary_path = out_dir / f"gate_summary_{tag}_gate_{suffix}.csv"

curve = pd.read_parquet(curve_path)
trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()

# detect columns (engine variants)
date_col = "Date"
eq_col = "Equity"
in_col = "InCycle" if "InCycle" in curve.columns else ("InPosition" if "InPosition" in curve.columns else None)

curve[date_col] = pd.to_datetime(curve[date_col], errors="coerce")
curve = curve.dropna(subset=[date_col, eq_col]).sort_values(date_col).reset_index(drop=True)

seed_mult = float(curve[eq_col].iloc[-1]) / float(curve[eq_col].iloc[0]) if len(curve) else float("nan")

# warmup 제거: 첫 진입 이후부터 CAGR 계산
cagr = float("nan")
idle_days = float("nan")
active_days = float("nan")
total_days_post = float("nan")

if in_col is not None and len(curve):
    mask = curve[in_col].astype(int) == 1
    if mask.any():
        first_idx = mask.idxmax()
        post = curve.loc[first_idx:].copy()
        total_days_post = float(len(post))
        active_days = float((post[in_col].astype(int) == 1).sum())
        idle_days = float(total_days_post - active_days)

        start_eq = float(post[eq_col].iloc[0])
        end_eq = float(post[eq_col].iloc[-1])
        start_dt = post[date_col].iloc[0]
        end_dt = post[date_col].iloc[-1]
        years = (end_dt - start_dt).days / 365.25
        if years > 0 and start_eq > 0:
            cagr = (end_eq / start_eq) ** (1.0 / years) - 1.0

maxdd = float(curve["MaxDrawdownPortfolio"].min()) if "MaxDrawdownPortfolio" in curve.columns else float("nan")
n_trades = int(len(trades)) if len(trades) else 0
winrate = float(trades["Win"].mean()) if "Win" in trades.columns and len(trades) else float("nan")

row = {
    "tag": tag,
    "suffix": suffix,
    "final_seed_multiple": seed_mult,
    "cagr_post_first_entry": cagr,
    "max_drawdown": maxdd,
    "num_trades": n_trades,
    "winrate": winrate,
    "idle_days_post_first_entry": idle_days,
    "active_days_post_first_entry": active_days,
    "total_days_post_first_entry": total_days_post,
}

pd.DataFrame([row]).to_csv(summary_path, index=False)
print(f"[DONE] wrote summary: {summary_path}")
PY

            done
          done
        done
      done
    done
  done
done

echo "[DONE] run_grid_workflow.sh complete"