# -*- coding: utf-8 -*-
"""
고객지수.py
- 점포별 과거 리스크 라벨의 롤링(누적) 평균 기반 고객 리스크 지수
- 누출 방지: t월 지수는 (t-1)월까지의 정보만 사용
- 출력: store_id, month, CustomerRiskIndex_3m, CustomerRiskIndex_6m
"""
import numpy as np
import pandas as pd
from .io_utils import read_csv_safe, parse_month_safely

def _rolling_te(s: pd.Series, window: int, min_periods: int = 3) -> pd.Series:
    x = s.shift(1)  
    return x.rolling(window=window, min_periods=min_periods).mean()

def build_customer_index(master_raw: pd.DataFrame | str,
                         y3_col: str = "y_risk_3m",
                         y6_col: str = "y_risk_6m",
                         win: int = 12) -> pd.DataFrame:
    if isinstance(master_raw, str):
        df = read_csv_safe(master_raw)
    else:
        df = master_raw.copy()

    df = parse_month_safely(df, out_col="month")
    if "ENCODED_MCT" in df.columns:
        df = df.rename(columns={"ENCODED_MCT":"store_id"})
    if "store_id" not in df.columns:
        raise KeyError("store_id(ENCODED_MCT) 컬럼 없음")

    for c in [y3_col, y6_col]:
        if c not in df.columns:
            raise KeyError(f"{c} 라벨 없음. pipeline의 _attach_targets 선행 필요")

    key = df[["store_id","month"]].copy()
    y = df[["store_id","month", y3_col, y6_col]].copy().sort_values(["store_id","month"])
    y[y3_col] = pd.to_numeric(y[y3_col], errors="coerce")
    y[y6_col] = pd.to_numeric(y[y6_col], errors="coerce")

    y["cust_te3"] = y.groupby("store_id")[y3_col].transform(lambda s: _rolling_te(s, win))
    y["cust_te6"] = y.groupby("store_id")[y6_col].transform(lambda s: _rolling_te(s, win))

    g3 = float(y[y3_col].mean(skipna=True))
    g6 = float(y[y6_col].mean(skipna=True))
    y["CustomerRiskIndex_3m"] = y["cust_te3"].fillna(g3)
    y["CustomerRiskIndex_6m"] = y["cust_te6"].fillna(g6)

    out = y[["store_id","month","CustomerRiskIndex_3m","CustomerRiskIndex_6m"]].copy()
    return out
