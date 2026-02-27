# scripts/diagnose_model_power.py

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

def ks_stat(y_true, y_score):
    df = pd.DataFrame({"y": y_true, "s": y_score}).sort_values("s")
    df["cum_pos"] = (df["y"] == 1).cumsum() / max((df["y"] == 1).sum(), 1)
    df["cum_neg"] = (df["y"] == 0).cumsum() / max((df["y"] == 0).sum(), 1)
    return np.max(np.abs(df["cum_pos"] - df["cum_neg"]))

def bucket_analysis(df, score_col, ret_col, buckets=[0.1,0.2,0.3,0.5]):
    results = []
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(df)

    for b in buckets:
        k = int(n * b)
        if k < 5:
            continue

        top = df.iloc[:k]
        bottom = df.iloc[-k:]

        results.append({
            "bucket": b,
            "top_mean_return": top[ret_col].mean(),
            "bottom_mean_return": bottom[ret_col].mean(),
            "spread": top[ret_col].mean() - bottom[ret_col].mean(),
            "top_winrate": (top[ret_col] > 0).mean(),
            "bottom_winrate": (bottom[ret_col] > 0).mean(),
        })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--picks", required=True)
    parser.add_argument("--score-col", default="p_success")
    parser.add_argument("--ret-col", default="Return")
    parser.add_argument("--label-col", default="y_success")
    parser.add_argument("--out", default="model_power_report.csv")
    args = parser.parse_args()

    df = pd.read_parquet(args.picks) if args.picks.endswith(".parquet") else pd.read_csv(args.picks)

    required_cols = [args.score_col, args.ret_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    report = {}

    # === Bucket Analysis ===
    bucket_df = bucket_analysis(df, args.score_col, args.ret_col)
    bucket_df.to_csv(args.out.replace(".csv", "_buckets.csv"), index=False)

    # === AUC / KS ===
    if args.label_col in df.columns:
        y_true = df[args.label_col].values
        y_score = df[args.score_col].values

        try:
            auc = roc_auc_score(y_true, y_score)
        except:
            auc = np.nan

        ks = ks_stat(y_true, y_score)

        report["AUC"] = auc
        report["KS"] = ks
    else:
        report["AUC"] = np.nan
        report["KS"] = np.nan

    power_df = pd.DataFrame([report])
    power_df.to_csv(args.out, index=False)

    print("=== Model Power Summary ===")
    print(power_df)
    print("\n=== Bucket Analysis ===")
    print(bucket_df)

if __name__ == "__main__":
    main()