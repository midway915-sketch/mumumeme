# scripts/build_ev_features.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
FEATURE_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"

FEATURES_PARQUET = FEATURE_DIR / "features.parquet"
FEATURES_CSV = FEATURE_DIR / "features.csv"

OUT_MODEL_PARQUET = FEATURE_DIR / "features_model.parquet"
OUT_MODEL_CSV = FEATURE_DIR / "features_model.csv"
META_JSON = FEATURE_DIR / "features_model_meta.json"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_features() -> pd.DataFrame:
    if FEATURES_PARQUET.exists():
        f = pd.read_parquet(FEATURES_PARQUET)
    elif FEATURES_CSV.exists():
        f = pd.read_csv(FEATURES_CSV)
    else:
        raise FileNotFoundError("features not found. Run scripts/build_features.py first.")
    f = f.copy()
    f["Date"] = pd.to_datetime(f["Date"])
    f["Ticker"] = f["Ticker"].astype(str).str.upper().str.strip()
    return f.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def load_strategy_labels(tag: str) -> pd.DataFrame:
    p = LABEL_DIR / f"strategy_labels_{tag}.parquet"
    c = LABEL_DIR / f"strategy_labels_{tag}.csv"
    if p.exists():
        df = pd.read_parquet(p)
    elif c.exists():
        df = pd.read_csv(c)
    else:
        raise FileNotFoundError(f"strategy labels not found for tag={tag}. Run build_strategy_labels.py first.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["ExitDate"] = pd.to_datetime(df["ExitDate"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df


def compute_event_ev(events: pd.DataFrame, win_events: int) -> pd.DataFrame:
    """
    events: ExitDate 기준으로 정렬된 이벤트(각 entry가 exit한 시점에 결과가 확정되는 샘플)
    win_events: 최근 N개 이벤트 기준 롤링 통계(이벤트 수 기준)
    반환: ExitDate, Ticker, EV_* 컬럼
    """
    e = events.sort_values(["Ticker", "ExitDate"]).reset_index(drop=True).copy()

    # 이벤트 시계열(ExitDate가 시간축)
    grp = e.groupby("Ticker", group_keys=False)

    # rolling은 "이벤트 수" 기준
    e[f"EV_ret_mean_{win_events}e"] = grp["CycleReturn"].transform(
        lambda x: x.shift(1).rolling(win_events, min_periods=max(5, win_events // 5)).mean()
    )
    e[f"EV_winrate_{win_events}e"] = grp["CycleReturn"].transform(
        lambda x: (x.shift(1) > 0).rolling(win_events, min_periods=max(5, win_events // 5)).mean()
    )
    e[f"EV_minret_mean_{win_events}e"] = grp["MinCycleRet"].transform(
        lambda x: x.shift(1).rolling(win_events, min_periods=max(5, win_events // 5)).mean()
    )
    e[f"EV_extend_rate_{win_events}e"] = grp["ExtendedFlag"].transform(
        lambda x: x.shift(1).rolling(win_events, min_periods=max(5, win_events // 5)).mean()
    )

    # 이벤트당 1행이므로 그대로 사용
    cols = ["ExitDate", "Ticker"] + [c for c in e.columns if c.startswith("EV_")]
    return e[cols]


def merge_ev_to_daily(features: pd.DataFrame, ev_events: pd.DataFrame) -> pd.DataFrame:
    """
    Date별로 '이전(<= Date-1)'의 마지막 EV 상태를 붙이기 위해 merge_asof 사용.
    """
    out = []

    for t, f_t in features.groupby("Ticker", sort=False):
        f_t = f_t.sort_values("Date").reset_index(drop=True)

        e_t = ev_events[ev_events["Ticker"] == t].sort_values("ExitDate").reset_index(drop=True)
        if e_t.empty:
            # EV 컬럼 NaN으로
            for c in ev_events.columns:
                if c.startswith("EV_"):
                    f_t[c] = np.nan
            out.append(f_t)
            continue

        # merge_asof: Date에 대해 "가장 최근 ExitDate <= Date-1"를 붙임
        # (ExitDate 당일은 결과를 알 수 있다고 가정하면 <= Date로 바꿔도 됨. 여기선 보수적으로 Date-1.)
        f_key = f_t[["Date"]].copy()
        f_key["KeyDate"] = f_key["Date"] - pd.Timedelta(days=1)

        merged = pd.merge_asof(
            left=f_key.sort_values("KeyDate"),
            right=e_t.rename(columns={"ExitDate": "KeyDate"}).sort_values("KeyDate"),
            on="KeyDate",
            direction="backward",
        )

        # EV 컬럼만 붙이기
        ev_cols = [c for c in merged.columns if c.startswith("EV_")]
        for c in ev_cols:
            f_t[c] = merged[c].to_numpy()

        out.append(f_t)

    return pd.concat(out, ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)


def add_cross_sectional_ranks(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    일자별 cross-sectional rank(0~1) 추가: 모델이 '상대적 우위'를 더 잘 잡게 됨.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[f"{c}_cs_rank"] = out.groupby("Date")[c].rank(pct=True)
    return out


def save(df: pd.DataFrame) -> str:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(OUT_MODEL_PARQUET, index=False)
        return f"parquet:{OUT_MODEL_PARQUET}"
    except Exception as e:
        print(f"[WARN] parquet save failed ({e}), saving csv: {OUT_MODEL_CSV}")
        df.to_csv(OUT_MODEL_CSV, index=False)
        return f"csv:{OUT_MODEL_CSV}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-tag", type=str, required=True, help="strategy_labels_{tag} 의 tag")
    ap.add_argument("--ev-windows", type=str, default="50,200", help="event window sizes, comma-separated")
    ap.add_argument("--add-cs-rank", action="store_true")
    args = ap.parse_args()

    features = load_features()
    labels = load_strategy_labels(args.label_tag)

    # 이벤트(ExitDate 기준 결과 확정) 테이블
    events = labels.dropna(subset=["CycleReturn", "ExitDate"]).copy()
    events = events[["Ticker", "ExitDate", "CycleReturn", "MinCycleRet", "ExtendedFlag"]]

    win_list = [int(x.strip()) for x in args.ev_windows.split(",") if x.strip()]
    ev_all = []
    for w in win_list:
        ev_all.append(compute_event_ev(events, w))
    ev_events = ev_all[0]
    for x in ev_all[1:]:
        ev_events = ev_events.merge(x, on=["ExitDate", "Ticker"], how="outer")

    merged = merge_ev_to_daily(features, ev_events)

    ev_cols = [c for c in merged.columns if c.startswith("EV_")]
    if args.add_cs_rank:
        merged = add_cross_sectional_ranks(merged, ev_cols)

    saved_to = save(merged)

    meta = {
        "updated_at_utc": now_utc_iso(),
        "saved_to": saved_to,
        "label_tag": args.label_tag,
        "ev_windows": win_list,
        "rows": int(len(merged)),
        "min_date": str(merged["Date"].min().date()),
        "max_date": str(merged["Date"].max().date()),
        "ev_cols": ev_cols,
        "cs_rank_added": bool(args.add_cs_rank),
    }
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] features_model saved={saved_to} rows={len(merged)} ev_cols={len(ev_cols)}")
    print(merged.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
