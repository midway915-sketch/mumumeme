# scripts/aggregate_gate_grid.py
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")
SIG_DIR = DATA_DIR / "signals"


TAG_RE = re.compile(r"pt(?P<pt>\d+)_h(?P<h>\d+)_sl(?P<sl>\d+)_ex(?P<ex>\d+)")
LABEL_RE = re.compile(
    r"^(?P<mode>tail_utility|tail|utility|none)_t(?P<t>[\dp]+)_q(?P<q>[\dp]+)_r(?P<r>.+)$"
)

def p_to_float(s: str) -> float:
    # "0p30" -> 0.30, "0p2" -> 0.2
    return float(s.replace("p", "."))

def parse_tag(tag: str) -> dict:
    m = TAG_RE.search(tag)
    if not m:
        return {}
    pt = int(m.group("pt")) / 100.0
    h = int(m.group("h"))
    sl = -int(m.group("sl")) / 100.0  # tag는 sl10처럼 절댓값이라 -로 복원
    ex = int(m.group("ex"))
    return {"PT": pt, "H": h, "SL": sl, "EX": ex}

def parse_label(label: str) -> dict:
    m = LABEL_RE.match(label)
    if not m:
        return {"gate_mode": None, "tail_max": None, "u_q": None, "rank_by": None}
    return {
        "gate_mode": m.group("mode"),
        "tail_max": p_to_float(m.group("t")),
        "u_q": p_to_float(m.group("q")),
        "rank_by": m.group("r"),
    }

def main() -> None:
    files = sorted(SIG_DIR.glob("gate_summary_*.csv"))
    if not files:
        raise FileNotFoundError(f"No gate_summary_*.csv found in {SIG_DIR}. Run make_gate_summary.py first.")

    rows = []
    for fp in files:
        # gate_summary_{tag}.csv
        tag = fp.stem.replace("gate_summary_", "")
        tag_meta = parse_tag(tag)

        df = pd.read_csv(fp)
        if "label" not in df.columns:
            continue

        # tag 파라미터 붙이기
        for k, v in tag_meta.items():
            df[k] = v
        df["TAG"] = tag

        # label 파라미터 풀기
        parsed = df["label"].astype(str).apply(parse_label).apply(pd.Series)
        df = pd.concat([df, parsed], axis=1)

        rows.append(df)

    out = pd.concat(rows, ignore_index=True)

    # 핵심 컬럼 정리 (없으면 자동 제외)
    preferred = [
        "TAG","PT","H","SL","EX",
        "label","gate_mode","tail_max","u_q","rank_by",
        "Seed_Multiple_All","recent_seed_multiple",
        "max_holding_days","max_extend_days_over_maxday",
        "Cycle_Count_Closed","Success_Rate_Closed","Max_Drawdown",
        "Avg_LeveragePct_Closed","Max_LeveragePct_Closed",
        "Skipped_Days","Entered_Days",
        "last_date","recent_start_date","Final_Equity"
    ]
    cols = [c for c in preferred if c in out.columns]
    out = out[cols].copy()

    # 보기 좋게 정렬
    sort_cols = [c for c in ["recent_seed_multiple","Seed_Multiple_All"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    out_path = SIG_DIR / "gate_grid_aggregate.csv"
    out.to_csv(out_path, index=False)

    # Top 파일도 하나 더
    top_path = SIG_DIR / "gate_grid_top_by_recent10y.csv"
    if "recent_seed_multiple" in out.columns:
        out.sort_values("recent_seed_multiple", ascending=False).head(50).to_csv(top_path, index=False)

    print(f"[DONE] wrote {out_path} rows={len(out)}")
    print(f"[DONE] wrote {top_path}")

if __name__ == "__main__":
    main()
