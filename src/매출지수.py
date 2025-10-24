# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from typing import Optional
from .io_utils import read_csv_safe, to_numeric, normalize_district

def _parse_TA_YM_any(s: pd.Series) -> pd.Series:

    x = s.astype(str).str.replace(r"[^0-9]", "", regex=True)
    x6 = np.where(x.str.len() >= 6, x.str.slice(0, 6), np.nan)
    out = pd.to_numeric(x6, errors="coerce").astype("int64")

    return out

def build_sales_index_monthly(path_sales: str,
                              key_df: Optional[pd.DataFrame]=None,
                              master_for_fallback: Optional[pd.DataFrame]=None,
                              base_year: int=2023,
                              min_year: int=2019,
                              max_year: int=2025) -> pd.DataFrame:

    df = read_csv_safe(path_sales, low_memory=False).copy()

    if "TA_YM" in df.columns:
        df["TA_YM"] = _parse_TA_YM_any(df["TA_YM"])
    else:
        raise KeyError("매출 파일에 TA_YM 컬럼이 없습니다.")

    before = len(df)
    df = df.dropna(subset=["TA_YM"]).copy()
    if len(df) < before:
        print(f"sales: dropped {before-len(df)} rows invalid TA_YM")

    df["TA_YM"] = df["TA_YM"].astype(int)
    df["year"]  = (df["TA_YM"]//100).astype(int)
    df = df[(df["year"]>=min_year) & (df["year"]<=max_year)].copy()


    if "행정동_코드_명" in df.columns:
        df["district"] = df["행정동_코드_명"].map(normalize_district)
    elif "행정동_명" in df.columns:
        df["district"] = df["행정동_명"].map(normalize_district)
    else:
        df["district"] = pd.NA

    if "세분류" in df.columns:
        df["category"] = df["세분류"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
    elif "카테고리" in df.columns:
        df["category"] = df["카테고리"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
    else:
        df["category"] = pd.NA

    val_col = None
    for c in ["당월_매출_금액","매출금액","sales","amt"]:
        if c in df.columns:
            val_col = c; break
    if not val_col:
        raise KeyError("매출 금액 컬럼(당월_매출_금액 등)을 찾을 수 없습니다.")
    df["sales"] = to_numeric(df[val_col], allow_percent=False)

    if key_df is not None and len(key_df):
        key = key_df.copy()
        key["TA_YM"] = key["TA_YM"].astype(int)
        key["district"] = key["district"].astype(str)
        key["category"] = key["category"].astype(str)
        months = key["TA_YM"].unique()
        df = df[df["TA_YM"].isin(months)].copy()
        print(f"sales: filtered by master months → rows {before} -> {len(df)}")


    m_all = (df.groupby("TA_YM", as_index=False)["sales"]
               .mean().rename(columns={"sales":"avg_sales"}))
    base = m_all.loc[(m_all["TA_YM"]//100)==base_year, "avg_sales"].mean()
    if not np.isfinite(base):
        base = float(m_all["avg_sales"].mean())
    m_all["sales_index"] = (m_all["avg_sales"]/base)*100.0

 
    m_d = (df.groupby(["district","TA_YM"], as_index=False)["sales"]
             .mean().rename(columns={"sales":"avg_sales_d"}))
    m_c = (df.groupby(["category","TA_YM"], as_index=False)["sales"]
             .mean().rename(columns={"sales":"avg_sales_c"}))
    m_dc = (df.groupby(["district","category","TA_YM"], as_index=False)["sales"]
              .mean().rename(columns={"sales":"avg_sales_dc"}))

    m_d = m_d.merge(m_all[["TA_YM","avg_sales"]], on="TA_YM", how="left")
    m_c = m_c.merge(m_all[["TA_YM","avg_sales"]], on="TA_YM", how="left")
    m_dc = m_dc.merge(m_all[["TA_YM","avg_sales"]], on="TA_YM", how="left")

    m_d["sales_index_district"]  = (m_d["avg_sales_d"]/m_d["avg_sales"]).replace([np.inf,-np.inf], np.nan)
    m_c["sales_index_category"]  = (m_c["avg_sales_c"]/m_c["avg_sales"]).replace([np.inf,-np.inf], np.nan)
    m_dc["sales_index_dc"]       = (m_dc["avg_sales_dc"]/m_dc["avg_sales"]).replace([np.inf,-np.inf], np.nan)

    out = m_all[["TA_YM","sales_index"]].copy()

    if "district" in m_d.columns and m_d["district"].notna().any():
        out = out.merge(m_d[["district","TA_YM","sales_index_district"]],
                        on=["TA_YM"], how="left")
   
    if "category" in m_c.columns and m_c["category"].notna().any():
        out = out.merge(m_c[["category","TA_YM","sales_index_category"]],
                        on=["TA_YM"], how="left")

    if {"district","category"}.issubset(m_dc.columns) and m_dc[["district","category"]].notna().any(axis=None):
        out = out.merge(m_dc[["district","category","TA_YM","sales_index_dc"]],
                        on=["TA_YM","district","category"], how="left")

    return out
