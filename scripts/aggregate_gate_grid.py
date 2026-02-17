from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd


SIGNALS_DIR = Path("data/signals")


def parse_tag_parts(tag: str) -> dict:
    # 예: pt10_h40_sl10_ex30
    out = {"TAG": tag}
    try:
        parts = tag.split("_")
        for p in parts:
            if p.startswith("pt"):
                out["PT"] = float(p.replace("pt", "")) / 100.0
            elif p.startswith("h"):
                out["H"] = int(p.replace("h", ""))
            elif p.startswith("sl"):
                out["SL"] = -float(p.replace("sl", "")) / 100.0
            elif p.startswith("ex"):
                out["EX"] = int(p.replace("ex", ""))
    except Exception:
        pass
    return out


def main() -> None:
    paths = sorted(glob.glob(str(SIGNALS_DIR / "gate_summary_*.csv")))
    if not paths:
        raise SystemExit("[ERROR] no gate_summary_*.csv found in data/signals")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        tag = Path(p).stem.replace("gate_summary_", "")
        meta = parse_tag_parts(tag)
        for k, v in meta.items():
            df[k] = v
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # 정렬: 교정배수(선형) 우선, 그 다음 최근10년 배수
    sort_cols = [c for c in ["adj_recent_multiple_linear", "recent_seed_multiple"] if c in all_df.columns]
    if sort_cols:
        all_df = all_df.sort_values(sort_cols, ascending=False)

    out_path = SIGNALS_DIR / "gate_grid_aggregate.csv"
    all_df.to_csv(out_path, index=False)

    # “최근10년 교정배수 Top”도 따로
    top_path = SIGNALS_DIR / "gate_grid_top_by_adj_recent10y.csv"
    all_df.head(200).to_csv(top_path, index=False)

    print("[DONE] wrote:")
    print(" -", out_path)
    print(" -", top_path)
    print("rows=", len(all_df))


if __name__ == "__main__":
    main()