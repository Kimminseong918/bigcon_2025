# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd

from src.io_utils import (
    read_csv_safe,
    normalize_district,
    parse_month_safely,
    month_features,
)
from src.매출지수 import build_sales_index_monthly
from src.상권지수 import build_market_indices_monthly
from src.거시지수 import build_cpi_ppi_monthly, build_rent_index_monthly
from src.고객지수 import build_customer_index
from src.업종지수 import build_industry_index

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)


SAVE_PREVIEW_CSV = True
PREVIEW_ROWS = 100

PATH_MASTER = DATA / "master_df.csv"
PATH_SALES  = DATA / "추정매출_2019_2024.csv"
PATH_SEOUL  = DATA / "임대시세_서울시_251017.csv"
PATH_SD     = DATA / "임대시세_성동구_251017.csv"
PATH_CPI    = DATA / "소비자물가지수_17211723.csv"
PATH_PPI    = DATA / "생산자물가지수(기본분류)_17212514.csv"

def _norm_category(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.replace({"": np.nan, "nan": np.nan, "None": np.nan}).fillna("미상")

def _to_month_start(s):
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()

def _optimize_types(df: pd.DataFrame) -> pd.DataFrame:
    """메모리 절약형 다운캐스트 + 카테고리화(문자열)"""
    out = df.copy()

    if "month" in out.columns:
        out["month"] = _to_month_start(out["month"])
 
    for c in out.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns:
        out[c] = pd.to_numeric(out[c], downcast="integer")
    for c in out.select_dtypes(include=["float64", "float32"]).columns:
        out[c] = pd.to_numeric(out[c], downcast="float")

    cat_cols = [c for c in ["district", "category"] if c in out.columns]
    for c in cat_cols:
        out[c] = out[c].astype("category")

    if "store_id" in out.columns:
        out["store_id"] = out["store_id"].astype(str)
 
    if "TA_YM" in out.columns:
        out["TA_YM"] = pd.to_numeric(out["TA_YM"], errors="coerce").fillna(0).astype("int32")
    return out

def _save_parquet(df: pd.DataFrame, path: Path):
    """Parquet(gzip)로 저장"""
    df.to_parquet(path, compression="gzip", index=False)
    print(f"saved (parquet:gzip): {path} | size ≈ {path.stat().st_size/1024:.1f} KB")

def _save_preview_csv(df: pd.DataFrame, path: Path, n=100):
    """검수용 미니 CSV(UTF-8-SIG)"""
    head = df.head(n).copy()
    head.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"saved preview CSV ({n} rows): {path}")

def _adapt_master(df_raw: pd.DataFrame) -> pd.DataFrame:
    """master_df → 표준 키: store_id, district, category, TA_YM(int32), month(월초 Timestamp)"""
    df = parse_month_safely(df_raw, out_col="month")
    df = df[df["month"].notna()].copy()

    if "ENCODED_MCT" in df.columns:
        df = df.rename(columns={"ENCODED_MCT": "store_id"})
    if "store_id" not in df.columns:
        raise KeyError("ENCODED_MCT / store_id 없음")
 
    for cand in ["행정동_명", "행정동_코드_명", "행정구역", "행정동", "MCT_SIGUNGU_NM"]:
        if cand in df.columns:
            df["district"] = df[cand].map(normalize_district) if cand != "MCT_SIGUNGU_NM" else df[cand]
            break
    if "district" not in df.columns:
        raise KeyError("district 미검출")

    df["category"] = _norm_category(df.get("세분류", None) if "세분류" in df.columns else df.get("HPSN_MCT_ZCD_NM", None))
  
    df["month"] = _to_month_start(df["month"])
    df["TA_YM"] = (df["month"].dt.year * 100 + df["month"].dt.month).astype("int32")
    keep = ["store_id", "district", "category", "TA_YM", "month"]
    out = (
        df[keep]
        .dropna(subset=["store_id", "month"])
        .drop_duplicates()
        .copy()
    )
    return _optimize_types(out)

def _attach_targets(df_raw: pd.DataFrame) -> pd.DataFrame:

    if "M12_SME_RY_SAA_PCE_RT" not in df_raw.columns:
        raise KeyError("M12_SME_RY_SAA_PCE_RT 없음")

    key = _adapt_master(df_raw)
    y = pd.concat(
        [key.reset_index(drop=True),
         df_raw["M12_SME_RY_SAA_PCE_RT"].reset_index(drop=True)],
        axis=1,
    )
    y["sales_score"] = -pd.to_numeric(y["M12_SME_RY_SAA_PCE_RT"], errors="coerce")
    y = y.sort_values(["store_id", "month"])
    y["sales_score_3m_future"] = y.groupby("store_id")["sales_score"].shift(-3)
    y["sales_score_6m_future"] = y.groupby("store_id")["sales_score"].shift(-6)
    y["sales_delta_3m"] = y["sales_score_3m_future"] - y["sales_score"]
    y["sales_delta_6m"] = y["sales_score_6m_future"] - y["sales_score"]
    thr3 = y["sales_delta_3m"].dropna().quantile(0.25)
    thr6 = y["sales_delta_6m"].dropna().quantile(0.25)
    print(f"Label thresholds: 3M≤{thr3:.3f}, 6M≤{thr6:.3f}")
    y["y_risk_3m"] = np.where(
        y["sales_delta_3m"] <= thr3, 1, np.where(y["sales_delta_3m"].isna(), np.nan, 0)
    )
    y["y_risk_6m"] = np.where(
        y["sales_delta_6m"] <= thr6, 1, np.where(y["sales_delta_6m"].isna(), np.nan, 0)
    )

    def _future_slope(s: pd.Series, h: int) -> pd.Series:
        n = len(s)
        x = np.arange(h + 1)
        out = np.full(n, np.nan)
        for i in range(n):
            j = i + h
            if j < n and s.iloc[i:j+1].notna().all():
                out[i] = np.polyfit(x, s.iloc[i:j+1].values, 1)[0]
        return pd.Series(out, index=s.index)
    y["slope_3m"] = y.groupby("store_id")["sales_score"].transform(lambda s: _future_slope(s, 3))
    y["slope_6m"] = y.groupby("store_id")["sales_score"].transform(lambda s: _future_slope(s, 6))
    y["y_risk_trend_3m"] = np.where(y["slope_3m"] < 0, 1, np.where(y["slope_3m"].isna(), np.nan, 0))
    y["y_risk_trend_6m"] = np.where(y["slope_6m"] < 0, 1, np.where(y["slope_6m"].isna(), np.nan, 0))
    y["month"] = _to_month_start(y["month"])
    y["TA_YM"] = (y["month"].dt.year * 100 + y["month"].dt.month).astype("int32")
    return _optimize_types(y)

def _merge(left, right, on, how="left", validate=None):
    if right is None or (isinstance(right, pd.DataFrame) and right.empty):
        return left
    try:
        return left.merge(right, on=on, how=how, validate=validate)
    except Exception as e:
        print(f"merge(validate={validate}) 실패 → validate 없이 재시도 | on={on} | 이유: {e}")
        return left.merge(right, on=on, how=how)


PASSTHRU_SIGNAL_COLS = [
 
    "M12_MAL_1020_RAT","M12_MAL_30_RAT","M12_MAL_40_RAT","M12_MAL_50_RAT","M12_MAL_60_RAT",
    "M12_FME_1020_RAT","M12_FME_30_RAT","M12_FME_40_RAT","M12_FME_50_RAT","M12_FME_60_RAT",

    "MCT_UE_CLN_REU_RAT","MCT_UE_CLN_NEW_RAT",

    "RC_M1_SHC_RSD_UE_CLN_RAT","RC_M1_SHC_WP_UE_CLN_RAT","RC_M1_SHC_FLP_UE_CLN_RAT",

    "DLV_SAA_RAT","M1_SME_RY_SAA_RAT","M1_SME_RY_CNT_RAT",

    "M12_SME_RY_SAA_PCE_RT","M12_SME_BZN_SAA_PCE_RT",
    "M12_SME_RY_ME_MCT_RAT","M12_SME_BZN_ME_MCT_RAT",
]

def _build_passthru_signals(df_raw: pd.DataFrame) -> pd.DataFrame:
    if "ENCODED_MCT" in df_raw.columns and "store_id" not in df_raw.columns:
        df_raw = df_raw.rename(columns={"ENCODED_MCT":"store_id"})
    if "store_id" not in df_raw.columns:
        return pd.DataFrame()
    tmp = parse_month_safely(df_raw, out_col="month")
    tmp["month"] = _to_month_start(tmp["month"])
    tmp["TA_YM"] = (tmp["month"].dt.year * 100 + tmp["month"].dt.month).astype("int32")
    have = [c for c in PASSTHRU_SIGNAL_COLS if c in tmp.columns]
    if not have:
        return pd.DataFrame()
    keep = ["store_id","TA_YM","month"] + have
    sig = tmp[keep].copy()

    for c in have:
        sig[c] = pd.to_numeric(sig[c], errors="coerce")
 
    sig = sig.groupby(["store_id","TA_YM","month"], as_index=False).mean(numeric_only=True)
    return _optimize_types(sig)


def main():
    print("load master...")
    master_raw = read_csv_safe(str(PATH_MASTER))


    key  = _adapt_master(master_raw)      
    ydf  = _attach_targets(master_raw)    
    sigp = _build_passthru_signals(master_raw) 

   
    sales_m = build_sales_index_monthly(
        str(PATH_SALES), key_df=key, master_for_fallback=master_raw
    )
    if sales_m is None:
        sales_m = pd.DataFrame()
    if not isinstance(sales_m, pd.DataFrame):
        sales_m = pd.DataFrame(sales_m)
    sales_m = _optimize_types(sales_m)
    _save_parquet(sales_m, OUT / "Indices_Sales_monthly.parquet")

    
    market_m = build_market_indices_monthly(str(PATH_MASTER))  
    if market_m is None:
        market_m = pd.DataFrame()
    market_m = _optimize_types(market_m)
    _save_parquet(market_m, OUT / "Indices_Market_monthly.parquet")

    macro_m  = build_cpi_ppi_monthly(str(PATH_CPI), str(PATH_PPI))
    if macro_m is None:
        macro_m = pd.DataFrame()
    macro_m = macro_m.drop(columns=["month"], errors="ignore")
    macro_m = _optimize_types(macro_m)
    _save_parquet(macro_m, OUT / "Indices_Macro_CPI_PPI_monthly.parquet")

    rent_m   = build_rent_index_monthly(str(PATH_SEOUL), str(PATH_SD))  
    if rent_m is None:
        rent_m = pd.DataFrame()
    rent_m = _optimize_types(rent_m)
    _save_parquet(rent_m, OUT / "Indices_Rent_monthly.parquet")

 
    mr = parse_month_safely(master_raw, out_col="month")
    if "ENCODED_MCT" in mr.columns:
        mr = mr.rename(columns={"ENCODED_MCT": "store_id"})
    if "store_id" not in mr.columns:
        raise KeyError("store_id(ENCODED_MCT) 키가 없습니다.")
    slim = [c for c in [
        "store_id","month","TA_YM","MCT_FRAN_YN","서비스_업종_코드","서비스_업종_코드_명","세분류"
    ] if c in mr.columns]
    mr_slim = mr[slim].copy()
    mr_slim["month"] = _to_month_start(mr_slim["month"])

    y_need = _merge(
        key[["store_id","month"]].drop_duplicates(),
        ydf[["store_id","month","y_risk_3m","y_risk_6m"]].drop_duplicates(),
        on=["store_id","month"], how="left", validate="1:1"
    )
    mr_with_labels = _merge(
        mr_slim.drop_duplicates(["store_id","month"]),
        y_need, on=["store_id","month"], how="left", validate="1:1"
    )

    cust = build_customer_index(mr_with_labels, y3_col="y_risk_3m", y6_col="y_risk_6m")
    indu = build_industry_index(mr_with_labels, y3_col="y_risk_3m", y6_col="y_risk_6m")
    for df_ in (cust, indu):
        if isinstance(df_, pd.DataFrame) and "month" in df_.columns:
            df_["month"] = _to_month_start(df_["month"])
    cust = _optimize_types(cust)
    indu = _optimize_types(indu)
    _save_parquet(cust, OUT / "Index_Risk_Customer_v2.parquet")
    _save_parquet(indu, OUT / "Index_Risk_Industry_v2.parquet")

   
    base = _merge(
        key,
        ydf.drop(columns=["district","category"], errors="ignore"),
        on=["store_id","TA_YM","month"], how="left", validate="1:1"
    )

    merged = base
    if not market_m.empty:
        merged = _merge(merged, market_m, on=["district","TA_YM"], how="left", validate="m:1")

    
    if not sales_m.empty:
        if "sales_index_dc" in sales_m.columns:
            sales_dc = sales_m[["TA_YM","district","category","sales_index_dc"]].drop_duplicates()
            merged   = _merge(merged, sales_dc, on=["TA_YM","district","category"], how="left", validate="m:1")
        if "sales_index_district" in sales_m.columns:
            sales_d  = sales_m[["TA_YM","district","sales_index_district"]].drop_duplicates()
            merged   = _merge(merged, sales_d,  on=["TA_YM","district"], how="left", validate="m:1")
        if "sales_index_category" in sales_m.columns:
            sales_c  = sales_m[["TA_YM","category","sales_index_category"]].drop_duplicates()
            merged   = _merge(merged, sales_c,  on=["TA_YM","category"], how="left", validate="m:1")
        if "sales_index" in sales_m.columns:
            sales_g  = sales_m[["TA_YM","sales_index"]].drop_duplicates()
            merged   = _merge(merged, sales_g,  on=["TA_YM"], how="left", validate="m:1")
        merged["sales_index_final"] = (
            merged.get("sales_index_dc")
                  .combine_first(merged.get("sales_index_district"))
                  .combine_first(merged.get("sales_index_category"))
                  .combine_first(merged.get("sales_index"))
        )
        merged["sales_index"] = merged["sales_index_final"]
    else:
        merged["sales_index"] = np.nan

    
    if not macro_m.empty:
        merged = _merge(merged, macro_m, on=["TA_YM"], how="left", validate="m:1")
    if not rent_m.empty:
        merged = _merge(merged, rent_m,  on=["district"], how="left", validate="m:1")
    if isinstance(cust, pd.DataFrame) and not cust.empty:
        merged = _merge(merged, cust,    on=["store_id","month"], how="left", validate="m:1")
    if isinstance(indu, pd.DataFrame) and not indu.empty:
        merged = _merge(merged, indu,    on=["store_id","month"], how="left", validate="m:1")

    
    if isinstance(sigp, pd.DataFrame) and not sigp.empty:
        merged = _merge(merged, sigp, on=["store_id","TA_YM","month"], how="left", validate="1:1")

    
    for tag in ("3m","6m"):
        idx = f"IndustryRiskIndex_{tag}"
        merged[f"CPIyoyxIndu_{tag}"] = merged.get("CPI_yoy") * merged.get(idx)
        merged[f"PPIyoyxIndu_{tag}"] = merged.get("PPI_yoy") * merged.get(idx)
    merged["RentxMarket"] = merged.get("rent_index") * merged.get("market_density")

    mf = month_features(merged["month"])
    merged = pd.concat([merged, mf], axis=1)

    
    merged = merged.sort_values(["store_id","month"]).reset_index(drop=True)
    dup_cols = [c for c in set(merged.columns) if list(merged.columns).count(c) > 1]
    if dup_cols:
        print("duplicate columns detected:", dup_cols)
    merged = _optimize_types(merged)

    PARQUET_PATH = OUT / "merged_indices_monthly.parquet"
    _save_parquet(merged, PARQUET_PATH)
    if SAVE_PREVIEW_CSV:
        _save_preview_csv(merged, OUT / "merged_indices_monthly_preview.csv", n=PREVIEW_ROWS)
    print(f"final shape = {merged.shape}")

   
    must_cols = [
        "store_id","district","category","month","TA_YM","sales_index",
        "market_close_rate","market_density","CPI_yoy","PPI_yoy","rent_index",
        "CustomerRiskIndex_3m","CustomerRiskIndex_6m",
        "IndustryRiskIndex_3m","IndustryRiskIndex_6m",
    ]
    missing_schema = [c for c in must_cols if c not in merged.columns]
    print("schema check | missing:", missing_schema if missing_schema else "none")

    miss_cols = ["sales_index","market_close_rate","market_density","CPI_yoy","PPI_yoy","rent_index"]
    miss = {c: float(merged[c].isna().mean()) if c in merged.columns else 1.0 for c in miss_cols}
    print("missing-rate:", miss)

if __name__ == "__main__":
    main()
