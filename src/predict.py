# -*- coding: utf-8 -*-

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"
ART_DIR = BASE_DIR / "artifacts" / "lgbm"
OUT_DIR.mkdir(exist_ok=True, parents=True)


KINDS = ["delta"]            
TAGS  = ["3m", "6m"]
GROUP_KEYS = ["customer", "market", "industry", "sales", "macro"]


DEBUG_SIGNAL_COVERAGE = True


COMPOSITE_COLS = {
    "customer_3m": "CustomerRiskIndex_3m",
    "customer_6m": "CustomerRiskIndex_6m",
    "industry_3m": "IndustryRiskIndex_3m",
    "industry_6m": "IndustryRiskIndex_6m",
    "market_close_rate": "market_close_rate",
    "market_density": "market_density",
    "sales_index": "sales_index",
    "CPI_yoy": "CPI_yoy",
    "PPI_yoy": "PPI_yoy",
    "rent_index": "rent_index",
}


SIGNAL_SPEC = {

    "youth_share": {
        "value": ["M12_MAL_1020_RAT","M12_FME_1020_RAT"],
        "higher_is_better": True, "label": "20대 이하 고객 비중"
    },
    "male30_50_share": {
        "value": ["M12_MAL_30_RAT","M12_MAL_40_RAT","M12_MAL_50_RAT"],
        "higher_is_better": True, "label": "남성 30~50대 비중"
    },
    "female30_50_share": {
        "value": ["M12_FME_30_RAT","M12_FME_40_RAT","M12_FME_50_RAT"],
        "higher_is_better": True, "label": "여성 30~50대 비중"
    },
    "revisit_share":   {"value": ["MCT_UE_CLN_REU_RAT"], "higher_is_better": True, "label": "재방문 고객 비중"},
    "new_share":       {"value": ["MCT_UE_CLN_NEW_RAT"], "higher_is_better": True, "label": "신규 고객 비중"},

    "resident_share":  {"value": ["RC_M1_SHC_RSD_UE_CLN_RAT"], "higher_is_better": True, "label": "거주 고객 비율"},
    "worker_share":    {"value": ["RC_M1_SHC_WP_UE_CLN_RAT"],  "higher_is_better": True, "label": "직장 고객 비율"},
    "floating_share":  {"value": ["RC_M1_SHC_FLP_UE_CLN_RAT"], "higher_is_better": True, "label": "유동 고객 비율"},
  
    "delivery_ratio":   {"value": ["DLV_SAA_RAT"], "higher_is_better": True, "label": "배달 매출 비중"},
    "peer_sales_ratio": {"value": ["M1_SME_RY_SAA_RAT"], "higher_is_better": True, "label": "동일 업종 매출 비율(평균=100)"},
    "peer_trx_ratio":   {"value": ["M1_SME_RY_CNT_RAT"], "higher_is_better": True, "label": "동일 업종 매출건수 비율(평균=100)"},
   
    "industry_rank_pct": {"value": ["M12_SME_RY_SAA_PCE_RT"], "higher_is_better": False, "label": "업종 내 매출 순위%"},
    "zone_rank_pct":     {"value": ["M12_SME_BZN_SAA_PCE_RT"], "higher_is_better": False, "label": "상권 내 매출 순위%"},
  
    "industry_close_ratio": {"value": ["M12_SME_RY_ME_MCT_RAT"], "higher_is_better": False, "label": "업종 내 해지 가맹점 비중"},
    "zone_close_ratio":     {"value": ["M12_SME_BZN_ME_MCT_RAT"], "higher_is_better": False, "label": "상권 내 해지 가맹점 비중"},
}

def _load_df() -> pd.DataFrame:
    """파이프라인 산출물을 Parquet 우선으로 로드 + 표준 키 안전 매핑(중복 방지)"""
    cand = [
        OUT_DIR / "merged_indices_monthly.parquet",
        OUT_DIR / "merged_indices_monthly.csv",
        OUT_DIR / "merged_indices.csv",
    ]
    path = next((p for p in cand if p.exists()), None)
    if path is None:
        raise FileNotFoundError("merged_indices_monthly.(parquet/csv) 를 찾을 수 없습니다.")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    def _coalesce_into(target: str, source: str, cast: str | None = None):
        """source가 있으면 target으로 안전 병합하고 source는 제거"""
        nonlocal df
        if source not in df.columns:
            return
        if target in df.columns:
            if cast == "str":
                a = df[target].astype(str)
                b = df[source].astype(str)
            else:
                a = df[target]; b = df[source]
            df[target] = a.where(~a.isna(), b)
            df.drop(columns=[source], inplace=True)
        else:
            df.rename(columns={source: target}, inplace=True)

    _coalesce_into("store_id", "ENCODED_MCT", cast="str")
    _coalesce_into("district", "MCT_SIGUNGU_NM")
    _coalesce_into("category", "HPSN_MCT_ZCD_NM")
    _coalesce_into("month", "TA_YM")

    if "store_id" in df.columns:
        df["store_id"] = df["store_id"].astype(str)
    if "month" in df.columns:
        m = pd.to_datetime(df["month"].astype(str), errors="coerce")
        df["month"] = m.dt.to_period("M").dt.to_timestamp()

    return df

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _debug_signal_coverage(df: pd.DataFrame):
    print("---- SIGNAL COVERAGE ----")
    for key, spec in SIGNAL_SPEC.items():
        have = [c for c in spec.get("value", []) if c in df.columns]
        print(f"{key:20s}: {' | '.join(have) if have else 'NO COLUMN FOUND'}")
    print("-------------------------")

def _load_feature_list(kind: str, tag: str) -> list[str]:
    p = ART_DIR / f"feature_list_{kind}_{tag}.json"
    if not p.exists():
        raise FileNotFoundError(f"피처리스트 없음: {p}")
    data = _read_json(p)
    return list(data["features"]) if isinstance(data, dict) and "features" in data else list(data)

def _best_threshold(kind: str, tag: str) -> tuple[float, float, float]:
    p = ART_DIR / f"threshold_summary_{kind}_{tag}.json"
    if p.exists():
        d = _read_json(p)
        return float(d.get("best_thr", 0.5)), float(d.get("f1", np.nan)), float(d.get("auc", np.nan))
    return 0.5, np.nan, np.nan

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

def _minmax_series(x: pd.Series, min: float, max: float) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if not np.isfinite(min) or not np.isfinite(max) or max <= min:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - min) / (max - min)

def _renorm(weights: dict, available_keys: list[str]) -> dict:
    sub = {k: float(weights.get(k, 0.0)) for k in available_keys}
    s = sum(sub.values())
    if s <= 0:
        n = len(available_keys)
        return {k: (1.0 / n) for k in available_keys}
    return {k: v / s for k, v in sub.items()}

def _load_weights_and_scalers(kind: str, tag: str, df_for_fallback: pd.DataFrame) -> tuple[dict, dict, bool]:
    used_fallback = False
    w_path = ART_DIR / f"group_weights_{kind}_{tag}.json"
    s_path = ART_DIR / f"group_scalers_{kind}_{tag}.json"
    if w_path.exists():
        weights = _read_json(w_path)
    else:
        used_fallback = True
        weights = {k: 1.0 / len(GROUP_KEYS) for k in GROUP_KEYS}
    if s_path.exists():
        scalers = _read_json(s_path)
    else:
        used_fallback = True
        scalers = {}
        for k, c in {
            "customer_3m":"CustomerRiskIndex_3m","customer_6m":"CustomerRiskIndex_6m",
            "industry_3m":"IndustryRiskIndex_3m","industry_6m":"IndustryRiskIndex_6m",
            "market_close_rate":"market_close_rate","market_density":"market_density",
            "sales_index":"sales_index","CPI_yoy":"CPI_yoy","PPI_yoy":"PPI_yoy","rent_index":"rent_index",
        }.items():
            s = pd.to_numeric(df_for_fallback.get(c, pd.Series(dtype=float)), errors="coerce")
            if s.notna().any():
                mn, mx = float(np.nanmin(s)), float(np.nanmax(s))
                if (not np.isfinite(mn)) or (not np.isfinite(mx)) or mx <= mn:
                    mn, mx = 0.0, 1.0
            else:
                mn, mx = 0.0, 1.0
            scalers[k] = {"min": mn, "max": mx}
    return weights, scalers, used_fallback

def _composite_and_contrib(df: pd.DataFrame, kind: str, tag: str) -> tuple[pd.Series, dict[str, pd.Series], dict]:
    weights, scalers, used_fallback = _load_weights_and_scalers(kind, tag, df)
    cust_key = f"customer_{tag}"; indu_key = f"industry_{tag}"
    cust = _minmax_series(_safe_col(df, COMPOSITE_COLS.get(cust_key, "")),
                          **scalers.get(cust_key, {"min": 0.0, "max": 1.0}))
    indu = _minmax_series(_safe_col(df, COMPOSITE_COLS.get(indu_key, "")),
                          **scalers.get(indu_key, {"min": 0.0, "max": 1.0}))
    m_close = _minmax_series(_safe_col(df, COMPOSITE_COLS["market_close_rate"]),
                             **scalers.get("market_close_rate", {"min": 0.0, "max": 1.0}))
    m_dens  = _minmax_series(_safe_col(df, COMPOSITE_COLS["market_density"]),
                             **scalers.get("market_density", {"min": 0.0, "max": 1.0}))
    market = 0.5 * (m_close + m_dens)
    sales  = _minmax_series(_safe_col(df, COMPOSITE_COLS["sales_index"]),
                            **scalers.get("sales_index", {"min": 0.0, "max": 1.0}))
    cpi = _minmax_series(_safe_col(df, COMPOSITE_COLS["CPI_yoy"]),
                         **scalers.get("CPI_yoy", {"min": 0.0, "max": 1.0}))
    ppi = _minmax_series(_safe_col(df, COMPOSITE_COLS["PPI_yoy"]),
                         **scalers.get("PPI_yoy", {"min": 0.0, "max": 1.0}))
    rent = _minmax_series(_safe_col(df, COMPOSITE_COLS["rent_index"]),
                          **scalers.get("rent_index", {"min": 0.0, "max": 1.0}))
    macro = (cpi + ppi + rent) / 3.0

    group_vals = {"customer": cust, "industry": indu, "market": market, "sales": sales, "macro": macro}
    avail = [k for k in GROUP_KEYS if k in group_vals]
    w = _renorm(weights, avail)
    contrib = {k: w[k] * group_vals[k] for k in avail}
    comp = sum(contrib.values()).fillna(0.0).clip(0, 1)

    meta_w = {"used_fallback": used_fallback}
    meta_w.update({f"weight_{k}": float(w.get(k, np.nan)) for k in GROUP_KEYS})
    return comp, contrib, meta_w

def _make_risk_tier(label_3m: np.ndarray, label_6m: np.ndarray):
    tier_code = np.where(label_3m == 1, 2, np.where(label_6m == 1, 1, 0))
    mapper = {0: "안정", 1: "위험", 2: "고위험"}
    tier_name = np.vectorize(mapper.get)(tier_code)
    return tier_code, tier_name

def _topk_groups(contrib_dict: dict, tag: str, k: int = 3) -> dict[str, np.ndarray]:
    items = sorted(((g, contrib_dict[g].values) for g in contrib_dict.keys()),
                   key=lambda x: np.nanmean(x[1]), reverse=True)
    tops = {}
    for i, (g, vals) in enumerate(items[:k], 1):
        tops[f"top{i}_group_{tag}"] = np.array([g]*len(vals), dtype=object)
        tops[f"top{i}_value_{tag}"] = vals
    return tops

def _load_and_predict(df: pd.DataFrame, kind: str, tag: str):
    feat_cols = _load_feature_list(kind, tag)
    model_p = ART_DIR / f"risk_lgbm_{kind}_{tag}.joblib"
    if not model_p.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_p}")
    model = joblib.load(model_p)
    thr, f1, auc = _best_threshold(kind, tag)

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        gap_path = ART_DIR / f"prediction_feature_gaps_{kind}_{tag}.csv"
        pd.DataFrame({"missing_feature": missing}).to_csv(gap_path, index=False, encoding="utf-8-sig")
        print(f"ℹ️ feature gaps 기록: {gap_path}")

    X = df.reindex(columns=feat_cols).astype(np.float32).fillna(0.0)
    proba = model.predict_proba(X)[:, 1].astype(np.float32)
    label = (proba >= thr).astype(int)
    comp, contrib, meta_w = _composite_and_contrib(df, kind, tag)
    return proba, label, comp, contrib, thr, f1, auc, meta_w

def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def _build_signal_values(df: pd.DataFrame) -> pd.DataFrame:
    """SIGNAL_SPEC을 이용해 단일 값 열(여러 원천 평균 등) 생성"""
    sdf = df[["store_id","district","category","month"]].copy()
    for key, spec in SIGNAL_SPEC.items():
        cols = _ensure_columns(df, spec["value"])
        if len(cols) == 0:
            continue
        sdf[key] = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in cols], axis=1).mean(axis=1)
    return sdf

def _recent_delta(s: pd.Series, w: int = 3):

    s = pd.to_numeric(s, errors="coerce")
    return s.rolling(w).mean() - s.rolling(w).mean().shift(w)

def _make_signals(df: pd.DataFrame, out_df: pd.DataFrame, kind: str):
    """
    signals_recent_{kind}.csv & signals_alerts_{kind}.csv 생성
    - set_index/loc 반복을 피하고 사전 매핑으로 최적화
    - 모든 점포(안정 포함) 저장
    """
    sdf = _build_signal_values(df)
    if sdf.shape[1] <= 4: 
        print("SIGNAL_SPEC에 해당하는 열이 없어 signals 파일을 생성하지 않습니다.")
        return

    sdf.sort_values(["store_id","month"], inplace=True)
    signal_keys = [c for c in sdf.columns if c not in ["store_id","district","category","month"]]


    for key in signal_keys:
        sdf[f"{key}_delta3m"] = sdf.groupby("store_id", group_keys=False)[key].apply(lambda x: _recent_delta(x, 3))

    for key in signal_keys:
        peer_mean = sdf.groupby(["district","category","month"], observed=False, group_keys=False)[key].transform("mean")
        sdf[f"{key}_peer_gap"] = pd.to_numeric(sdf[key], errors="coerce") - peer_mean

    rt_map = dict(zip(
        out_df["store_id"].astype(str) + "|" + out_df["month"].astype(str),
        out_df["risk_tier"].astype(str)
    ))
    sdf["__key"] = sdf["store_id"].astype(str) + "|" + sdf["month"].astype(str)
    sdf["risk_tier"] = sdf["__key"].map(rt_map)
    sdf.drop(columns="__key", inplace=True)

    keep_cols = ["store_id","district","category","month","risk_tier"]
    for key in signal_keys:
        keep_cols += [f"{key}_delta3m", f"{key}_peer_gap"]
    spath = OUT_DIR / f"signals_recent_{kind}.csv"
    sdf[keep_cols].to_csv(spath, index=False, encoding="utf-8-sig")
    print(f"saved: {spath}")

    alerts_rows = []
    for _, row in sdf.iterrows():
        scores = []
        for key, spec in SIGNAL_SPEC.items():
            dcol = f"{key}_delta3m"; gcol = f"{key}_peer_gap"
            if dcol not in sdf.columns or gcol not in sdf.columns:
                continue
            delta_raw = row.get(dcol, np.nan)
            gap_raw   = row.get(gcol, np.nan)
            if not (np.isfinite(delta_raw) or np.isfinite(gap_raw)):
                continue

            sign  = -1.0 if spec["higher_is_better"] else +1.0
            delta = sign * (delta_raw if np.isfinite(delta_raw) else 0.0)
            gap   = sign * (gap_raw   if np.isfinite(gap_raw)   else 0.0)
            score = np.nanmean([delta, gap])
            scores.append((key, spec["label"], score, delta_raw, gap_raw))

        if not scores:
            continue
        scores = sorted(scores, key=lambda x: (x[2] if np.isfinite(x[2]) else -1e18), reverse=True)[:3]

        reasons = []
        for key, lab, _, dval, gval in scores:
            if SIGNAL_SPEC[key]["higher_is_better"]:
                txt = f"{lab} 하락(최근 3개월 Δ {dval:+.2f}), 행정동·업종 동월 평균 대비 {gval:+.2f}%p"
            else:
                txt = f"{lab} 상승(최근 3개월 Δ {dval:+.2f}), 행정동·업종 동월 평균 대비 {gval:+.2f}%p"
            reasons.append(txt)

        alerts_rows.append({
            "store_id": row["store_id"],
            "district": row["district"],
            "category": row["category"],
            "month": row["month"],
            "risk_tier": row.get("risk_tier", np.nan),
            "reason_1": reasons[0] if len(reasons) > 0 else "",
            "reason_2": reasons[1] if len(reasons) > 1 else "",
            "reason_3": reasons[2] if len(reasons) > 2 else "",
        })

    if alerts_rows:
        apath = OUT_DIR / f"signals_alerts_{kind}.csv"
        pd.DataFrame(alerts_rows).to_csv(apath, index=False, encoding="utf-8-sig")
        print(f"saved: {apath}")
    else:
        print("alerts 생성 대상 없음(신호 부족).")

def main():
    df = _load_df()

    if DEBUG_SIGNAL_COVERAGE:
        _debug_signal_coverage(df)

    keys = [c for c in ["store_id", "district", "category", "month"] if c in df.columns]
    out_frames = {}

    for kind in KINDS:
        proba_cols, label_cols, comp_cols, meta_cols, explain_cols = {}, {}, {}, {}, {}

        for tag in TAGS:
            proba, label, comp, contrib, thr, f1, auc, meta_w = _load_and_predict(df, kind, tag)
            print(f"loaded model: risk_lgbm_{kind}_{tag}.joblib")
            print(f"threshold from summary: {thr:.3f} (F1={np.nan if not np.isfinite(f1) else f1:.3f} | AUC={np.nan if not np.isfinite(auc) else auc:.3f})")

            proba_cols[f"risk_proba_{tag}"] = proba
            label_cols[f"risk_label_{tag}"] = label
            comp_cols[f"comp_index_{tag}"] = comp.astype(np.float32)
            meta_cols[f"thr_{tag}"] = np.full(len(df), thr, dtype=np.float32)
            meta_cols[f"f1_{tag}"]  = np.full(len(df), f1, dtype=np.float32)
            meta_cols[f"auc_{tag}"] = np.full(len(df), auc, dtype=np.float32)

            for gk, s in contrib.items():
                explain_cols[f"contrib_{gk}_{tag}"] = s.values
            for gk in GROUP_KEYS:
                explain_cols[f"weight_{gk}_{tag}"] = np.full(len(df), meta_w.get(f"weight_{gk}", np.nan))
            explain_cols[f"weights_used_fallback_{tag}"] = np.full(len(df), bool(meta_w.get("used_fallback", False)))

            tops = _topk_groups(contrib, tag, k=3)
            for k2, v in tops.items():
                explain_cols[k2] = v

        label_3m = label_cols.get("risk_label_3m", np.zeros(len(df), dtype=int))
        label_6m = label_cols.get("risk_label_6m", np.zeros(len(df), dtype=int))
        tier_code, tier_name = _make_risk_tier(label_3m, label_6m)

        out = pd.DataFrame({
            **{k: df[k] for k in keys},
            **proba_cols, **label_cols, **comp_cols, **meta_cols,
            "risk_tier_code": tier_code.astype(np.int8),
            "risk_tier": tier_name.astype(object),
        })

        for k, v in explain_cols.items():
            if k.startswith("contrib_"):
                out[k] = v

        path_parq = OUT_DIR / f"predictions_latest_both_{kind}.parquet"
        out.to_parquet(path_parq, index=False)
        print(f"saved: {path_parq}")


        keep_drivers = [
            *keys, "risk_tier", "risk_tier_code",
            "risk_label_3m","risk_label_6m",
            "comp_index_3m","comp_index_6m",
            "top1_group_3m","top1_value_3m","top2_group_3m","top2_value_3m","top3_group_3m","top3_value_3m",
            "top1_group_6m","top1_value_6m","top2_group_6m","top2_value_6m","top3_group_6m","top3_value_6m",
        ]
        exp_drivers = pd.DataFrame({
            **{k: df[k] for k in keys},
            **{k: out[k] for k in ["risk_tier","risk_tier_code","risk_label_3m","risk_label_6m","comp_index_3m","comp_index_6m"] if k in out.columns},
            **{k: explain_cols[k] for k in explain_cols.keys() if k.startswith("top")},
        })
        dpath = OUT_DIR / f"explain_topdrivers_{kind}.csv"
        exp_drivers.reindex(columns=[c for c in keep_drivers if c in exp_drivers.columns])\
                   .to_csv(dpath, index=False, encoding="utf-8-sig")
        print(f"saved: {dpath}")

        _make_signals(df, out, kind)

        out_frames[kind] = out

    if len(out_frames) == 2 and keys:
        merged = out_frames["delta"].merge(
            out_frames["trend"], on=keys, how="outer", suffixes=("_delta", "_trend")
        )
        mpath = OUT_DIR / "predictions_latest_both_merged.parquet"
        merged.to_parquet(mpath, index=False)
        print(f"saved: {mpath}")

if __name__ == "__main__":
    main()
