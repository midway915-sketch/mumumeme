from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
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


LABEL_RE = re.compile(
    r"^(?P<mode>none|tail|utility|tail_utility)"
    r"_t(?P<tmax>[-0-9p]+)"
    r"_q(?P<uq>[-0-9p]+)"
    r"_r(?P<rank>[A-Za-z0-9_]+)"
    r"(?:_g(?P<gamma>[-0-9p]+))?$"
)

def _unpack_float_token(tok: str) -> float:
    # 0p75 -> 0.75 , m0p10 -> -0.10 (혹시 대비)
    s = str(tok).strip().lower()
    sign = -1.0 if s.startswith("m") else 1.0
    s = s[1:] if s.startswith("m") else s
    s = s.replace("p", ".")
    try:
        return sign * float(s)
    except Exception:
        return float("nan")


def parse_label(label: str) -> Dict[str, Any]:
    s = str(label)
    m = LABEL_RE.match(s)
    if not m:
        return {
            "gate_mode": None,
            "tail_max": np.nan,
            "u_quantile": np.nan,
            "rank_by": None,
            "tau_gamma": np.nan,
        }

    mode = m.group("mode")
    tmax = _unpack_float_token(m.group("tmax"))
    uq = _unpack_float_token(m.group("uq"))
    rank = m.group("rank")
    gamma_raw = m.group("gamma")
    gamma = _unpack_float_token(gamma_raw) if gamma_raw is not None else np.nan

    return {
        "gate_mode": mode,
        "tail_max": float(tmax),
        "u_quantile": float(uq),
        "rank_by": rank,
        "tau_gamma": float(gamma),
    }


def pareto_frontier(df: pd.DataFrame, maximize_cols: list[str], minimize_cols: list[str]) -> pd.DataFrame:
    """
    Return subset of df that is Pareto-optimal:
      - maximize maximize_cols
      - minimize minimize_cols
    """
    if df.empty:
        return df

    # numeric coercion
    work = df.copy()
    for c in maximize_cols + minimize_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=maximize_cols + minimize_cols)
    if work.empty:
        return work

    # Convert to "all minimize" space by negating maximize columns
    X = []
    cols = []
    for c in maximize_cols:
        X.append(-work[c].to_numpy(dtype=float))
        cols.append(c)
    for c in minimize_cols:
        X.append(work[c].to_numpy(dtype=float))
        cols.append(c)
    X = np.vstack(X).T

    n = X.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        # any point j dominates i if all <= and at least one <
        xi = X[i]
        # vectorized dominance check
        le = np.all(X <= xi, axis=1)
        lt = np.any(X < xi, axis=1)
        dominates = le & lt
        dominates[i] = False
        if np.any(dominates):
            is_pareto[i] = False

    out = work.loc[is_pareto].copy()
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

        # label 파싱(= predict_gate suffix)
        parsed = df["label"].apply(parse_label) if "label" in df.columns else pd.Series([parse_label("")] * len(df))
        parsed_df = pd.DataFrame(parsed.tolist())
        df = pd.concat([df, parsed_df], axis=1)

        # tau_gamma가 NaN인데 rank_by가 utility_time이면 0.05로 보정(기본값)
        if "rank_by" in df.columns:
            mask = (df["rank_by"].astype(str) == "utility_time") & (pd.to_numeric(df["tau_gamma"], errors="coerce").isna())
            df.loc[mask, "tau_gamma"] = 0.05

        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # 추천 정렬 우선순위:
    # 1) 레버 교정 최근10년 배수(선형)
    # 2) 최근10년 배수
    # 3) max leverage 낮을수록
    sort_cols = []
    if "adj_recent_multiple_linear" in all_df.columns:
        sort_cols.append("adj_recent_multiple_linear")
    if "recent_seed_multiple" in all_df.columns:
        sort_cols.append("recent_seed_multiple")

    # 보조로 레버(낮을수록) / 연장(낮을수록)
    if "Max_LeveragePct_Closed" in all_df.columns:
        all_df["Max_LeveragePct_Closed"] = pd.to_numeric(all_df["Max_LeveragePct_Closed"], errors="coerce")
    if "Max_Extend_Over_MaxDays" in all_df.columns:
        all_df["Max_Extend_Over_MaxDays"] = pd.to_numeric(all_df["Max_Extend_Over_MaxDays"], errors="coerce")

    if sort_cols:
        all_df = all_df.sort_values(sort_cols, ascending=False)

    # ---- 출력 1) 전체 aggregate
    out_path = SIGNALS_DIR / "gate_grid_aggregate.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_path, index=False)

    # ---- 출력 2) 레버 교정 최근10년 Top
    top_path = SIGNALS_DIR / "gate_grid_top_by_adj_recent10y.csv"
    if "adj_recent_multiple_linear" in all_df.columns:
        top = all_df.sort_values("adj_recent_multiple_linear", ascending=False).head(300)
    else:
        top = all_df.head(300)
    top.to_csv(top_path, index=False)

    # ---- 출력 3) gamma별 Top-N (utility_time만 대상)
    gamma_path = SIGNALS_DIR / "gate_grid_top_by_gamma.csv"
    gamma_df = all_df.copy()
    gamma_df["tau_gamma"] = pd.to_numeric(gamma_df["tau_gamma"], errors="coerce")
    gamma_df = gamma_df[(gamma_df["rank_by"].astype(str) == "utility_time") & (gamma_df["tau_gamma"].notna())].copy()

    if not gamma_df.empty:
        # gamma별 top 50
        metric = "adj_recent_multiple_linear" if "adj_recent_multiple_linear" in gamma_df.columns else "recent_seed_multiple"
        gamma_df[metric] = pd.to_numeric(gamma_df[metric], errors="coerce")
        gamma_df = gamma_df.dropna(subset=[metric])

        gamma_top = (
            gamma_df.sort_values(metric, ascending=False)
                    .groupby("tau_gamma", sort=True)
                    .head(50)
                    .reset_index(drop=True)
        )
        gamma_top.to_csv(gamma_path, index=False)
    else:
        pd.DataFrame().to_csv(gamma_path, index=False)

    # ---- 출력 4) Pareto frontier (수익↑ / 레버↓ / 연장↓)
    # 기본: maximize adj_recent_multiple_linear, minimize Max_LeveragePct_Closed, Max_Extend_Over_MaxDays
    pareto_metric = "adj_recent_multiple_linear" if "adj_recent_multiple_linear" in all_df.columns else "recent_seed_multiple"
    maximize_cols = [pareto_metric]
    minimize_cols = []
    if "Max_LeveragePct_Closed" in all_df.columns:
        minimize_cols.append("Max_LeveragePct_Closed")
    if "Max_Extend_Over_MaxDays" in all_df.columns:
        minimize_cols.append("Max_Extend_Over_MaxDays")

    pareto = pareto_frontier(all_df, maximize_cols=maximize_cols, minimize_cols=minimize_cols)

    # 보기 좋게 "효율점수"도 하나 추가(참고용)
    # score = adj_recent_multiple_linear / (1+lev/100) / (1+extend/10)
    pareto = pareto.copy()
    lev = pd.to_numeric(pareto.get("Max_LeveragePct_Closed", np.nan), errors="coerce")
    ext = pd.to_numeric(pareto.get("Max_Extend_Over_MaxDays", np.nan), errors="coerce")
    base = pd.to_numeric(pareto.get(pareto_metric, np.nan), errors="coerce")
    pareto["eff_score"] = base / (1.0 + lev.fillna(0.0)/100.0) / (1.0 + ext.fillna(0.0)/10.0)

    pareto = pareto.sort_values(["eff_score", pareto_metric], ascending=False)

    pareto_path = SIGNALS_DIR / "gate_grid_pareto.csv"
    pareto.to_csv(pareto_path, index=False)

    print("[DONE] wrote:")
    print(" -", out_path)
    print(" -", top_path)
    print(" -", gamma_path)
    print(" -", pareto_path)
    print("rows=", len(all_df), "pareto_rows=", len(pareto))


if __name__ == "__main__":
    main()