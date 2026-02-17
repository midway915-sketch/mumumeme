from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Dict, Any, List

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
    # 0p75 -> 0.75 , m0p10 -> -0.10
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


def _coerce_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pareto_frontier(df: pd.DataFrame, maximize_cols: list[str], minimize_cols: list[str]) -> pd.DataFrame:
    """
    Return subset of df that is Pareto-optimal:
      - maximize maximize_cols
      - minimize minimize_cols
    """
    if df.empty:
        return df

    work = df.copy()

    # numeric coercion
    for c in maximize_cols + minimize_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=maximize_cols + minimize_cols)
    if work.empty:
        return work

    # Convert to "all minimize" space by negating maximize columns
    X = []
    for c in maximize_cols:
        X.append(-work[c].to_numpy(dtype=float))
    for c in minimize_cols:
        X.append(work[c].to_numpy(dtype=float))
    X = np.vstack(X).T

    n = X.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        xi = X[i]
        le = np.all(X <= xi, axis=1)
        lt = np.any(X < xi, axis=1)
        dominates = le & lt
        dominates[i] = False
        if np.any(dominates):
            is_pareto[i] = False

    return work.loc[is_pareto].copy()


def add_eff_scores(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = df.copy()
    base = pd.to_numeric(out.get(metric, np.nan), errors="coerce")

    lev = pd.to_numeric(out.get("Max_LeveragePct_Closed", np.nan), errors="coerce").fillna(0.0)
    ext = pd.to_numeric(out.get("Max_Extend_Over_MaxDays", np.nan), errors="coerce").fillna(0.0)
    succ = pd.to_numeric(out.get("SuccessRate", np.nan), errors="coerce").fillna(0.0)
    cyc = pd.to_numeric(out.get("CycleCount", np.nan), errors="coerce").fillna(0.0)

    # 3목표용 효율점수(참고용)
    out["eff_score"] = base / (1.0 + lev/100.0) / (1.0 + ext/10.0)

    # 4목표용 참고점수: (수익 * (1+성공률) * log(1+cycle)) / (1+레버) / (1+연장)
    out["eff_score4"] = (
        base
        * (1.0 + succ)
        * np.log1p(cyc.clip(lower=0.0))
        / (1.0 + lev/100.0)
        / (1.0 + ext/10.0)
    )
    return out


def write_pareto_bundle(df: pd.DataFrame, metric: str, name_prefix: str) -> None:
    # 3목표 파레토: 수익↑, 레버↓, 연장↓
    maximize_cols = [metric]
    minimize_cols = [c for c in ["Max_LeveragePct_Closed", "Max_Extend_Over_MaxDays"] if c in df.columns]

    p3 = pareto_frontier(df, maximize_cols=maximize_cols, minimize_cols=minimize_cols)
    p3 = add_eff_scores(p3, metric).sort_values(["eff_score", metric], ascending=False)
    (SIGNALS_DIR / f"{name_prefix}.csv").parent.mkdir(parents=True, exist_ok=True)
    p3.to_csv(SIGNALS_DIR / f"{name_prefix}.csv", index=False)

    # 4목표 파레토: 수익↑, 성공률↑, 사이클↑, 레버↓, 연장↓
    # (성공률/사이클이 없으면 자동으로 3목표와 동일하게 나옴)
    maximize4 = [metric]
    if "SuccessRate" in df.columns:
        maximize4.append("SuccessRate")
    if "CycleCount" in df.columns:
        maximize4.append("CycleCount")

    p4 = pareto_frontier(df, maximize_cols=maximize4, minimize_cols=minimize_cols)
    p4 = add_eff_scores(p4, metric).sort_values(["eff_score4", "eff_score", metric], ascending=False)
    p4.to_csv(SIGNALS_DIR / f"{name_prefix}4.csv", index=False)


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

        # label 파싱
        if "label" in df.columns:
            parsed = df["label"].apply(parse_label)
        else:
            parsed = pd.Series([parse_label("")] * len(df))
        parsed_df = pd.DataFrame(parsed.tolist())
        df = pd.concat([df, parsed_df], axis=1)

        # utility_time인데 gamma NaN이면 기본값으로 보정
        if "rank_by" in df.columns:
            mask = (df["rank_by"].astype(str) == "utility_time") & (pd.to_numeric(df["tau_gamma"], errors="coerce").isna())
            df.loc[mask, "tau_gamma"] = 0.05

        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # 숫자형 정리
    all_df = _coerce_num(all_df, [
        "recent_seed_multiple",
        "adj_recent_multiple_linear",
        "Max_LeveragePct_Closed",
        "Max_Extend_Over_MaxDays",
        "SuccessRate",
        "CycleCount",
        "MaxHoldingDays",
        "MaxDays",
    ])

    # 기본 metric 선택
    metric = "adj_recent_multiple_linear" if "adj_recent_multiple_linear" in all_df.columns else "recent_seed_multiple"
    all_df = add_eff_scores(all_df, metric)

    # ---- 출력 1) 전체 aggregate
    out_path = SIGNALS_DIR / "gate_grid_aggregate.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_path, index=False)

    # ---- 출력 2) 전체 Top
    top_path = SIGNALS_DIR / "gate_grid_top_by_adj_recent10y.csv"
    top = all_df.sort_values(metric, ascending=False).head(500)
    top.to_csv(top_path, index=False)

    # ---- 출력 3) gamma별 Top-N (utility_time만)
    gamma_path = SIGNALS_DIR / "gate_grid_top_by_gamma.csv"
    gdf = all_df.copy()
    gdf["tau_gamma"] = pd.to_numeric(gdf.get("tau_gamma", np.nan), errors="coerce")
    gdf = gdf[(gdf.get("rank_by", "").astype(str) == "utility_time") & (gdf["tau_gamma"].notna())].copy()
    if not gdf.empty:
        gdf = gdf.dropna(subset=[metric])
        gamma_top = (
            gdf.sort_values(metric, ascending=False)
               .groupby("tau_gamma", sort=True)
               .head(80)
               .reset_index(drop=True)
        )
        gamma_top.to_csv(gamma_path, index=False)
    else:
        pd.DataFrame().to_csv(gamma_path, index=False)

    # ---- Pareto bundles (전체 + variant별)
    # 전체
    write_pareto_bundle(all_df, metric, "gate_grid_pareto")

    # variant별
    if "Variant" in all_df.columns:
        for v in ["BASE", "A", "B"]:
            sub = all_df[all_df["Variant"].astype(str) == v].copy()
            if not sub.empty:
                write_pareto_bundle(sub, metric, f"gate_grid_pareto_{v}")

    print("[DONE] wrote:")
    print(" -", out_path)
    print(" -", top_path)
    print(" -", gamma_path)
    print(" -", SIGNALS_DIR / "gate_grid_pareto.csv / gate_grid_pareto4.csv (and per-variant)")
    print("rows=", len(all_df))


if __name__ == "__main__":
    main()