# -*- coding: utf-8 -*-
import pandas as pd
from .io_utils import read_csv_safe, to_numeric, normalize_district

def _melt_mm(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    mm = [c for c in df.columns if "-" in c]
    for c in mm:
        df[c] = to_numeric(df[c])
    long = df.melt(value_vars=mm, var_name="월", value_name=value_col)
    yy = 2000 + long["월"].str.split("-").str[1].astype(int)
    mm_ = long["월"].str.split("-").str[0].astype(int)
    long["TA_YM"] = (yy * 100 + mm_).astype(int)
    long["month"] = pd.to_datetime(dict(year=yy, month=mm_, day=1))
    return long[["TA_YM", "month", value_col]]

def build_cpi_ppi_monthly(path_cpi: str, path_ppi: str) -> pd.DataFrame:

    cpi = read_csv_safe(path_cpi)
    cpi = cpi[cpi["계정항목"].astype(str).str.contains("총지수")]
    cpi_m = _melt_mm(cpi, "CPI")

    ppi = read_csv_safe(path_ppi)
    ppi = ppi[ppi["계정코드"].astype(str).str.strip() == "총지수"]
    ppi_m = _melt_mm(ppi, "PPI")

    df = cpi_m.merge(ppi_m, on=["TA_YM", "month"], how="outer").sort_values("TA_YM")

    for col in ("CPI", "PPI"):
        df[f"{col}_mom"] = df[col].pct_change() * 100.0
        df[f"{col}_yoy"] = df[col].pct_change(12) * 100.0
        df[f"{col}_shock3"] = df[f"{col}_yoy"] - df[f"{col}_yoy"].rolling(3, min_periods=2).mean().shift(1)

    for c in ["CPI", "PPI", "CPI_mom", "CPI_yoy", "CPI_shock3", "PPI_mom", "PPI_yoy", "PPI_shock3"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def build_rent_index_monthly(path_seoul: str, path_sd: str, latest_col="2024년 4분기") -> pd.DataFrame:

    seoul = read_csv_safe(path_seoul)
    sd = read_csv_safe(path_sd)
    for d in (seoul, sd):
        for c in d.columns[1:]:
            d[c] = to_numeric(d[c])

    seoul_avg = seoul.loc[seoul["행정구역"].astype(str).str.contains("서울"), latest_col].iloc[0]
    sd["district"] = sd["행정구역"].map(normalize_district)
    sd["rent_index"] = sd[latest_col] / seoul_avg

    out = sd[["district", "rent_index"]].copy()
    out["rent_index"] = pd.to_numeric(out["rent_index"], errors="coerce")
    return out
