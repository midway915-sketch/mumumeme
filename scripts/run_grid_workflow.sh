#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Required env (from workflow)
# ----------------------------
: "${PROFIT_TARGET:?missing}"
: "${MAX_DAYS:?missing}"
: "${STOP_LEVEL:?missing}"
: "${MAX_EXTEND_DAYS:?missing}"
: "${P_TAIL_THRESHOLDS:?missing}"
: "${UTILITY_QUANTILES:?missing}"
: "${RANK_METRICS:?missing}"
: "${LAMBDA_TAIL:?missing}"
: "${GATE_MODES:?missing}"

# optional env
OUT_DIR="${OUT_DIR:-data/signals}"
TP1_FRAC="${TP1_FRAC:-0.50}"                  # 50% partial take profit at PT
TRAIL_STOPS="${TRAIL_STOPS:-0.08,0.10,0.12}"  # trailing stop list
ENABLE_TRAILING="${ENABLE_TRAILING:-true}"

# Compare Top-1 vs Top-2 split configs
# format: "topk|weights;topk|weights;..."
TOPK_CONFIGS="${TOPK_CONFIGS:-1|1.0;2|0.7,0.3;2|0.6,0.4}"

EXCLUDE_TICKERS="${EXCLUDE_TICKERS:-SPY,^VIX}"
REQUIRE_FILES="${REQUIRE_FILES:-data/features/features_model.parquet,app/model.pkl,app/scaler.pkl}"

mkdir -p "$OUT_DIR"

# ----------------------------
# Locate scripts
# ----------------------------
if [ ! -f scripts/predict_gate.py ]; then echo "[ERROR] scripts/predict_gate.py not found"; exit 1; fi
if [ ! -f scripts/simulate_single_position_engine.py ]; then echo "[ERROR] scripts/simulate_single_position_engine.py not found"; exit 1; fi
if [ ! -f scripts/summarize_sim_trades.py ]; then echo "[ERROR] scripts/summarize_sim_trades.py not found"; exit 1; fi

PRED="scripts/predict_gate.py"
SIM="scripts/simulate_single_position_engine.py"
SUM="scripts/summarize_sim_trades.py"

# ----------------------------
# Tag builder
# ----------------------------
TAG="$(
python - <<'PY'
import os
pt=float(os.environ["PROFIT_TARGET"])
h=int(os.environ["MAX_DAYS"])
sl=float(os.environ["STOP_LEVEL"])
ex=int(os.environ["MAX_EXTEND_DAYS"])
def pct(x): return int(round(abs(x)*100))
print(f"pt{pct(pt)}_h{h}_sl{pct(sl)}_ex{ex}")
PY
)"
echo "[INFO] TAG=$TAG"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] TOPK_CONFIGS=$TOPK_CONFIGS"
echo "[INFO] TRAIL_STOPS=$TRAIL_STOPS TP1_FRAC=$TP1_FRAC ENABLE_TRAILING=$ENABLE_TRAILING"

# ----------------------------
# Build combos using Python (robust; avoids bash parsing bugs)
# Output TSV columns:
# mode  tail  uq  rank  lambda  topk  weights  trail  suffix
# ----------------------------
python - <<'PY' > /tmp/gate_grid_combos.tsv
import os

def split_csv(s: str):
    return [x.strip() for x in (s or "").split(",") if x.strip()]

def fmt_num(x: float):
    # 0.2 -> 0p20, -0.1 -> m0p10
    s = ("m" if x < 0 else "") + f"{abs(x):.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")

modes = split_csv(os.environ["GATE_MODES"])
tails = [float(x) for x in split_csv(os.environ["P_TAIL_THRESHOLDS"])]
uqs   = [float(x) for x in split_csv(os.environ["UTILITY_QUANTILES"])]
ranks = split_csv(os.environ["RANK_METRICS"])
lams  = [float(x) for x in split_csv(os.environ["LAMBDA_TAIL"])] or [0.05]
trails= [float(x) for x in split_csv(os.environ["TRAIL_STOPS"])]

topk_cfgs_raw = os.environ.get("TOPK_CONFIGS","").strip()
cfgs=[]
for part in [p.strip() for p in topk_cfgs_raw.split(";") if p.strip()]:
    if "|" not in part:
        continue
    k, w = part.split("|",1)
    k=int(k.strip())
    w=w.strip()
    cfgs.append((k,w))

tp1_frac = float(os.environ.get("TP1_FRAC","0.50"))

def safe_w(w: str):
    return w.replace(",", "_").replace(".", "p")

for mode in modes:
    for t in tails:
        for uq in uqs:
            for rnk in ranks:
                for lam in lams:
                    for topk, weights in cfgs:
                        for tr in trails:
                            suffix = "_".join([
                                mode,
                                f"t{fmt_num(t)}",
                                f"q{fmt_num(uq)}",
                                f"r{rnk}",
                                f"lam{fmt_num(lam)}",
                                f"k{topk}",
                                f"w{safe_w(weights)}",
                                f"tp{int(round(tp1_frac*100))}",
                                f"tr{fmt_num(tr)}",
                            ])
                            print(
                                mode, t, uq, rnk, lam, topk, weights, tr, suffix,
                                sep="\t"
                            )
PY

# ----------------------------
# Run combos
# ----------------------------
while IFS=$'\t' read -r mode tmax uq rnk lam topk weights tr suffix; do
  [ -z "${mode:-}" ] && continue

  echo "=============================="
  echo "[RUN] mode=$mode tail_max=$tmax u_q=$uq rank_by=$rnk lambda=$lam topk=$topk weights=$weights trail=$tr suffix=$suffix"
  echo "=============================="

  python "$PRED" \
    --profit-target "$PROFIT_TARGET" \
    --max-days "$MAX_DAYS" \
    --stop-level "$STOP_LEVEL" \
    --max-extend-days "$MAX_EXTEND_DAYS" \
    --mode "$mode" \
    --tag "$TAG" \
    --suffix "$suffix" \
    --out-dir "$OUT_DIR" \
    --exclude-tickers "$EXCLUDE_TICKERS" \
    --tail-threshold "$tmax" \
    --utility-quantile "$uq" \
    --rank-by "$rnk" \
    --lambda-tail "$lam" \
    --topk "$topk" \
    --topk-weights "$weights" \
    --require-files "$REQUIRE_FILES"

  PICKS_PATH="$OUT_DIR/picks_${TAG}_gate_${suffix}.csv"
  if [ ! -f "$PICKS_PATH" ]; then
    echo "[WARN] picks missing -> skip sim: $PICKS_PATH"
    continue
  fi

  python "$SIM" \
    --picks-path "$PICKS_PATH" \
    --profit-target "$PROFIT_TARGET" \
    --max-days "$MAX_DAYS" \
    --stop-level "$STOP_LEVEL" \
    --max-extend-days "$MAX_EXTEND_DAYS" \
    --max-leverage-pct "1.0" \
    --tp1-frac "$TP1_FRAC" \
    --trail-stop "$tr" \
    --enable-trailing "$ENABLE_TRAILING" \
    --out-dir "$OUT_DIR" \
    --tag "$TAG" \
    --suffix "$suffix"

  TRADES_PATH="$OUT_DIR/sim_engine_trades_${TAG}_gate_${suffix}.parquet"
  if [ ! -f "$TRADES_PATH" ]; then
    echo "[WARN] trades missing -> skip summary: $TRADES_PATH"
    continue
  fi

  python "$SUM" \
    --trades-path "$TRADES_PATH" \
    --tag "$TAG" \
    --suffix "$suffix" \
    --profit-target "$PROFIT_TARGET" \
    --max-days "$MAX_DAYS" \
    --stop-level "$STOP_LEVEL" \
    --max-extend-days "$MAX_EXTEND_DAYS" \
    --out-dir "$OUT_DIR"

done < /tmp/gate_grid_combos.tsv

echo "[DONE] grid finished."