# -*- coding: utf-8 -*-
import re, unicodedata
from typing import Iterable, Sequence, Union
from pathlib import Path
import pandas as pd
import numpy as np
import io as _io

__all__ = [
    "read_csv_safe","to_numeric","parse_year_from_TA_YM","normalize_district",
    "ensure_month_col","parse_month_safely","ensure_keys","month_features",
    "to_month_from_TA_YM","to_TA_YM_from_month"
]

def read_csv_safe(path: str,
                  encodings: Iterable[str]=("utf-8-sig","utf-8","cp949","euc-kr"),
                  low_memory: bool=False,
                  **kwargs) -> pd.DataFrame:

    last = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=low_memory, **kwargs)
        except Exception as e:
            last = e

    try:
        raw = Path(path).read_bytes()
    except Exception as e:
        raise ValueError(f"CSV read fail (read_bytes): {path} ({e})")

    def _decode_try(bytes_obj, codec):
        try:
            return bytes_obj.decode(codec, errors="replace")
        except Exception:
            return None

    text = (
        _decode_try(raw, "utf-8")
        or _decode_try(raw, "cp949")
        or _decode_try(raw, "euc-kr")
        or _decode_try(raw, "latin-1")
    )
    if text is None:
        raise ValueError(f"CSV read fail: {path} ({last})")

    try:
        return pd.read_csv(_io.StringIO(text), low_memory=low_memory, **kwargs)
    except Exception as e:
        raise ValueError(f"CSV parse fail via StringIO: {path} ({e})")

def to_numeric(s: pd.Series, allow_percent: bool=True) -> pd.Series:
    x = s.astype(str).str.strip().str.replace(",","",regex=False)
    neg = x.str.match(r"^\(.*\)$")
    x = x.str.replace(r"[()]", "", regex=True)
    x = x.mask(neg, "-" + x)
    if allow_percent:
        pct = x.str.endswith("%")
        x = x.str.replace("%","",regex=False)
        out = pd.to_numeric(x, errors="coerce")
        return out.mask(pct, out/100.0)
    return pd.to_numeric(x, errors="coerce")

def parse_year_from_TA_YM(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d{4})")[0].astype(float).astype("Int64")

def normalize_district(v: Union[str,float]) -> str:
    if pd.isna(v): 
        return v
    s = unicodedata.normalize("NFKC", str(v)).strip()
    s = re.sub(r"\s+","", s).replace("ㆍ","·")

    for tok in ("(", "（"):
        if tok in s:
            s = s.split(tok)[0]

    s = s.replace("?", ",")
    return s

def parse_month_safely(df: pd.DataFrame, out_col: str="month") -> pd.DataFrame:
    g = df.copy()
    if "TA_YM" in g.columns:
        s = g["TA_YM"].astype(str).str.extract(r"(\d{6})")[0]
        g[out_col] = pd.to_datetime(s, format="%Y%m", errors="coerce")
    elif "month" in g.columns:
        s = g["month"].astype(str).str.replace(r"[^0-9]", "", regex=True)
        g[out_col] = pd.to_datetime(s, format="%Y%m", errors="coerce")
    else:
        for c in ("date","기준일자","거래일자","BaseYm"):
            if c in g.columns:
                g[out_col] = pd.to_datetime(g[c], errors="coerce")
                break
    g[out_col] = g[out_col].values.astype("datetime64[M]")
    return g

def ensure_month_col(df: pd.DataFrame, col: str="month") -> pd.DataFrame:
    g = df.copy()
    if col not in g.columns:
        raise KeyError(f"'{col}' column not found")
    g[col] = pd.to_datetime(g[col], errors="coerce").values.astype("datetime64[M]")
    return g

def ensure_keys(df: pd.DataFrame, keys: Sequence[str]) -> None:
    miss = [k for k in keys if k not in df.columns]
    if miss:
        raise KeyError(f"Missing key columns: {miss}")

def month_features(dt: pd.Series) -> pd.DataFrame:
    m = dt.dt.month.astype(int)
    return pd.DataFrame({
        "m": m,
        "m_sin": np.sin(2*np.pi*m/12),
        "m_cos": np.cos(2*np.pi*m/12),
    }, index=dt.index)

def to_month_from_TA_YM(ser: pd.Series) -> pd.Series:

    s = pd.Series(ser)  
    s = s.astype(str).str.extract(r"(\d{6})")[0]
    dt = pd.to_datetime(s, format="%Y%m", errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()

def to_TA_YM_from_month(ser: pd.Series) -> pd.Series:

    s = pd.Series(ser)  
    dt = pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()
    return (dt.dt.year * 100 + dt.dt.month).astype("Int64")
