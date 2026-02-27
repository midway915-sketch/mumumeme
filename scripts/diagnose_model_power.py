# scripts/diagnose_model_power.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


def ks_stat(y_true, y_score):
    df = pd.DataFrame({"y": y_true, "s": y_score}).sort_values("s")
    pos = (df["y"] == 1).sum()
    neg = (df["y"] == 0).sum()
    df["cum_pos"] = (df["y"] == 1).cumsum() / max(pos, 1)
    df["cum_neg"] = (df["y"] == 0).cumsum() / max(neg, 1)
    return float(np.max(np.abs(df["cum_pos"] - df["cum_neg"])))


def pick_first_existing(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    return None


def bucket_analysis(df, score_col, ret_col, buckets=(0.1, 0.2, 0.3, 0.5)):
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(df)
    rows = []
    for b in buckets:
        k = int(n * b)
        if k < 10:
            continue
        top = df.iloc[:k]
        bot = df.iloc[-k:]
        rows.append(
            {
                "bucket": b,
                "n": n,
                "k": k,
                "top_mean_return": float(top[ret_col].mean()),
                "bottom_mean_return": float(bot[ret_col].mean()),
                "spread": float(top[ret_col].mean() - bot[ret_col].mean()),
                "top_winrate": float((top[ret_col] > 0).mean()),
                "bottom_winrate": float((bot[ret_col] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", required=True, help="csv/parquet file path")
    ap.add_argument("--score-col", default="p_success")
    ap.add_argument("--ret-col", default="Return")
    ap.add_argument("--label-col", default="y_success")
    ap.add_argument("--out", required=True, help="output csv path (power)")
    args = ap.parse_args()

    p = Path(args.picks)
    if not p.exists():
        raise SystemExit(f"[ERROR] picks not found: {p}")

    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)

    # --- Auto-fallback columns (robust) ---
    cols = set(df.columns)

    score_col = args.score_col if args.score_col in cols else pick_first_existing(
        cols, ["p_success", "psuccess", "p_succ", "score", "utility", "p_tail"]
    )
    ret_col = args.ret_col if args.ret_col in cols else pick_first_existing(
        cols, ["Return", "ret", "pnl", "trade_return", "r", "net_return"]
    )
    label_col = args.label_col if args.label_col in cols else pick_first_existing(
        cols, ["y_success", "y", "label", "target", "is_win", "win"]
    )

    if score_col is None:
        raise SystemExit(f"[ERROR] score col missing (tried {args.score_col} + fallbacks). cols={list(df.columns)[:50]}")
    if ret_col is None:
        raise SystemExit(f"[ERROR] return col missing (tried {args.ret_col} + fallbacks). cols={list(df.columns)[:50]}")

    # --- Clean numeric ---
    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    df = df.dropna(subset=[score_col, ret_col])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Bucket analysis ---
    buckets_df = bucket_analysis(df, score_col, ret_col)
    buckets_out = out_path.with_name(out_path.stem + "_buckets.csv")
    buckets_df.to_csv(buckets_out, index=False)

    # --- AUC / KS ---
    auc = np.nan
    ks = np.nan
    n_label = 0

    if label_col is not None and label_col in df.columns:
        y_true = pd.to_numeric(df[label_col], errors="coerce").dropna()
        # align
        aligned = df.loc[y_true.index, [score_col]].copy()
        y_score = aligned[score_col].values
        y_true = y_true.values.astype(int)
        n_label = int(len(y_true))

        if n_label >= 50 and len(np.unique(y_true)) >= 2:
            if roc_auc_score is not None:
                try:
                    auc = float(roc_auc_score(y_true, y_score))
                except Exception:
                    auc = np.nan
            try:
                ks = ks_stat(y_true, y_score)
            except Exception:
                ks = np.nan

    # --- Summary row ---
    report = pd.DataFrame(
        [
            {
                "file": str(p),
                "rows": int(len(df)),
                "score_col": score_col,
                "ret_col": ret_col,
                "label_col": label_col if label_col is not None else "",
                "labeled_rows": n_label,
                "AUC": auc,
                "KS": ks,
                "bucket_0.2_spread": float(buckets_df.loc[buckets_df["bucket"] == 0.2, "spread"].iloc[0])
                if (buckets_df is not None and (buckets_df["bucket"] == 0.2).any())
                else np.nan,
            }
        ]
    )
    report.to_csv(out_path, index=False)

    print("[DONE]", out_path)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()