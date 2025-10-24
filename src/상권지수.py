# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
import pandas as pd
from .io_utils import read_csv_safe, to_numeric, normalize_district, parse_month_safely

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c

    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        lc = c.lower()
        if lc in lower:
            return lower[lc]
    return None

def _safe_mean(g: pd.Series) -> float:
    g = pd.to_numeric(g, errors="coerce")
    if g.notna().any():
        return float(g.mean())
    return np.nan

def build_market_indices_monthly(path_master: str) -> pd.DataFrame:

    raw = read_csv_safe(path_master)
    df = parse_month_safely(raw, out_col="month")
    df = df[df["month"].notna()].copy()
    df["TA_YM"] = (df["month"].dt.year * 100 + df["month"].dt.month).astype(int)

    dist_col = _pick_col(df, ["행정동_코드_명","행정동_명","행정구역","행정동","district"])
    if dist_col is None:
        raise KeyError("행정동 컬럼을 찾지 못했습니다.")
    df["district"] = df[dist_col].map(normalize_district)

    close_col = _pick_col(df, ["폐업률","폐업_비율","close_rate","closed_rate","폐업비율"])
    dens_col  = _pick_col(df, ["점포밀도","밀도","density","store_density"])

    if close_col:
        df["market_close_rate"] = to_numeric(df[close_col], allow_percent=True)
    if dens_col:
        df["market_density"] = to_numeric(df[dens_col], allow_percent=False)

    
    if "market_close_rate" not in df.columns or df["market_close_rate"].isna().all():
        
        sid_col = _pick_col(df, ["ENCODED_MCT","store_id","가맹점ID","점포ID"])
        if sid_col is None:
            
            cnt = df.groupby(["district","TA_YM"], as_index=False).size().rename(columns={"size":"n_store"})
        else:
            cnt = df.groupby(["district","TA_YM"], as_index=False)[sid_col].nunique().rename(columns={sid_col:"n_store"})
        cnt = cnt.sort_values(["district","TA_YM"])
        cnt["n_store_prev3"] = cnt.groupby("district")["n_store"].shift(3)
        
        close_rate = (cnt["n_store_prev3"] - cnt["n_store"]) / (cnt["n_store_prev3"] + 1e-9)
        close_rate = close_rate.clip(lower=0)  
        df = df.merge(cnt[["district","TA_YM","n_store"]], on=["district","TA_YM"], how="left")
        df["market_close_rate"] = df["market_close_rate"] if "market_close_rate" in df.columns else np.nan
        df["market_close_rate"] = df["market_close_rate"].fillna(close_rate)

    if "market_density" not in df.columns or df["market_density"].isna().all():
        
        if "n_store" not in df.columns:
            sid_col = _pick_col(df, ["ENCODED_MCT","store_id","가맹점ID","점포ID"])
            if sid_col is None:
                cnt = df.groupby(["district","TA_YM"], as_index=False).size().rename(columns={"size":"n_store"})
            else:
                cnt = df.groupby(["district","TA_YM"], as_index=False)[sid_col].nunique().rename(columns={sid_col:"n_store"})
            df = df.merge(cnt, on=["district","TA_YM"], how="left")
        
        base = df.groupby("TA_YM", as_index=False)["n_store"].mean().rename(columns={"n_store":"n_store_all"})
        tmp = df.merge(base, on="TA_YM", how="left")
        dens = tmp["n_store"] / (tmp["n_store_all"] + 1e-9)
        df["market_density"] = df["market_density"] if "market_density" in df.columns else np.nan
        df["market_density"] = df["market_density"].fillna(dens)

    out = df.groupby(["district","TA_YM"], as_index=False).agg({
        "market_close_rate": _safe_mean,
        "market_density":    _safe_mean
    })
    
    for c in ["market_close_rate","market_density"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf,-np.inf], np.nan)
    return out
