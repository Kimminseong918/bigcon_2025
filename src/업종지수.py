# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


INDU_CANDIDATES = [
    "세분류", "서비스_업종_코드", "서비스_업종_코드_명", "industry", "category"
]

def _pick_industry_col(df: pd.DataFrame) -> str:
    for c in INDU_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError("[industry] 업종 식별 컬럼을 찾을 수 없습니다.")

def _safe_to_cat(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.replace({"None": "미상", "nan": "미상", "": "미상"})
    return x.fillna("미상")

def build_industry_index(df: pd.DataFrame, y3_col="y_risk_3m", y6_col="y_risk_6m") -> pd.DataFrame:
 
    need = ["store_id", "month", y3_col, y6_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"[industry] 필요한 컬럼 누락: {miss}")

    indu_col = _pick_industry_col(df)
    g = df[["store_id", "month", indu_col, y3_col, y6_col]].copy()

 
    g["month"] = pd.to_datetime(g["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    g[indu_col] = _safe_to_cat(g[indu_col])


    def _agg_rate(label_col: str) -> pd.DataFrame:

        out = (
            g.groupby([indu_col, "month"], as_index=False)[label_col]
             .mean(numeric_only=True)
             .rename(columns={label_col: f"{label_col}_rate"})
        )

        out[f"{label_col}_rate"] = (
            out.sort_values("month")
               .groupby(indu_col)[f"{label_col}_rate"]
               .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )
        return out

    r3 = _agg_rate(y3_col)
    r6 = _agg_rate(y6_col)


    def _minmax_by_group(df_rate: pd.DataFrame, col: str) -> pd.DataFrame:
        mm = df_rate.copy()
        key = [indu_col]
        gmin = mm.groupby(key)[col].transform("min")
        gmax = mm.groupby(key)[col].transform("max")
        mm[col + "_scaled"] = (mm[col] - gmin) / (gmax - gmin + 1e-12)

        same = (gmax - gmin).abs() < 1e-12
        mm.loc[same, col + "_scaled"] = 0.5
        return mm

    r3 = _minmax_by_group(r3, f"{y3_col}_rate")
    r6 = _minmax_by_group(r6, f"{y6_col}_rate")

    key = ["store_id", "month", indu_col]
    base = g[key].drop_duplicates()

    out = (
        base.merge(r3[[indu_col, "month", f"{y3_col}_rate_scaled"]],
                   on=[indu_col, "month"], how="left")
            .merge(r6[[indu_col, "month", f"{y6_col}_rate_scaled"]],
                   on=[indu_col, "month"], how="left")
    )

    for c in [f"{y3_col}_rate_scaled", f"{y6_col}_rate_scaled"]:
        med = out[c].median()
        out[c] = out[c].fillna(med if np.isfinite(med) else 0.5)

    out = out.rename(columns={
        f"{y3_col}_rate_scaled": "IndustryRiskIndex_3m",
        f"{y6_col}_rate_scaled": "IndustryRiskIndex_6m",
    })

    return out[["store_id", "month", "IndustryRiskIndex_3m", "IndustryRiskIndex_6m"]].drop_duplicates()
