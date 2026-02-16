from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    tag = args.tag

    # 보통은 data/signals 아래에 있음
    candidates = sorted(
        p for p in glob.glob(f"data/**/sim_engine_summary*{tag}*.csv", recursive=True)
        if "GATES_ALL" not in p
    )

    if not candidates:
        print(f"[ERROR] No sim_engine_summary files found for tag={tag} under data/**")
        print("[DEBUG] listing data/**/sim_engine_summary*.csv (first 200):")
        all_summ = sorted(glob.glob("data/**/sim_engine_summary*.csv", recursive=True))[:200]
        for p in all_summ:
            print("  -", p)
        raise SystemExit(1)

    dfs = []
    for p in candidates:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print("[WARN] failed reading", p, "err=", e)

    if not dfs:
        raise SystemExit(f"[ERROR] Found summary paths but none readable (tag={tag}).")

    df = pd.concat(dfs, ignore_index=True)
    if "label" in df.columns:
        df = df.drop_duplicates(subset=["label"], keep="last")

    out_dir = Path("data/signals")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sim_engine_summary_{tag}_GATES_ALL.csv"
    df.to_csv(out_path, index=False)
    print(f"[DONE] wrote {out_path} rows={len(df)} from {len(candidates)} files")


if __name__ == "__main__":
    main()
