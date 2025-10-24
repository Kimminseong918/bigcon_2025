# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import io, os, json, datetime, textwrap, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

# (선택) Gemini
try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

try:
    from streamlit.runtime.secrets import StreamlitSecretNotFoundError
except Exception:
    class StreamlitSecretNotFoundError(Exception): ...

# -------------------- 경로 --------------------
THIS = Path(__file__).resolve()
APP_DIR = THIS.parent
ROOT = APP_DIR.parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)  # outputs 폴더 보장

# -------------------- Google Drive URL (secrets 우선, fallback 존재) --------------------
try:
    PREDICTIONS_URL       = st.secrets["PREDICTIONS_URL"]
    PREDICTIONS_NAMED_URL = st.secrets["PREDICTIONS_NAMED_URL"]
    MERGED_URL            = st.secrets["MERGED_URL"]
    ALERTS_URL            = st.secrets["ALERTS_URL"]          # signals_alerts_delta.csv
    SIGREC_URL            = st.secrets.get("SIGREC_URL", "")  # (선택) signals_recent_delta.csv
except Exception:
    PREDICTIONS_URL       = "https://drive.google.com/uc?id=1qInDALlRx25MlShIL4yT4GTiqO-qmSWd&export=download"
    PREDICTIONS_NAMED_URL = "https://drive.google.com/uc?id=1oDGLLAtPhvweruKWq2x9DTHC_LSyLG34&export=download"
    MERGED_URL            = "https://drive.google.com/uc?id=1-iPvmfHz3mjhRe95XEoB17Ja0S_zulJm&export=download"
    ALERTS_URL            = "https://drive.google.com/uc?id=1_WdKGUzAK1xaXlxTbpkCfDpyonYQGWBx&export=download"
    SIGREC_URL            = ""

# -------------------- 공통: Google Drive에서 파일 다운로드 --------------------
def _extract_gdrive_id(url_or_id: str) -> str | None:
    """id=… 또는 /d/…/ 형태 모두에서 파일ID 추출"""
    if not url_or_id:
        return None
    if "/" not in url_or_id:  # 이미 id만 온 경우
        return url_or_id
    m = re.search(r"(?:id=|/d/)([A-Za-z0-9_-]{20,})", url_or_id)
    return m.group(1) if m else None

def _download_from_gdrive(url_or_id: str, out_path: Path) -> bool:
    try:
        file_id = _extract_gdrive_id(url_or_id)
        if not file_id:
            return False
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
        print(f"[OK] downloaded -> {out_path.name}")
        return True
    except Exception as e:
        print(f"[WARN] download failed: {out_path.name} ({e})")
        return False

def ensure_outputs_files(out_dir: Path) -> None:
    """필요 파일 없으면 Drive에서 자동 다운로드 (URL이 비었으면 건너뜀)"""
    targets = {
        "predictions_latest_both_delta.parquet":       PREDICTIONS_URL,
        "predictions_latest_both_delta_named.parquet": PREDICTIONS_NAMED_URL,
        "merged_indices_monthly.parquet":              MERGED_URL,
        "signals_alerts_delta.csv":                    ALERTS_URL,
        "signals_recent_delta.csv":                    SIGREC_URL,
    }
    for fname, url in targets.items():
        if not url:
            continue
        p = out_dir / fname
        if not p.exists() or p.stat().st_size == 0:
            _download_from_gdrive(url, p)

# 실제 다운로드 수행
ensure_outputs_files(OUT)

# -------------------- 정책 파일 탐색 보완 --------------------
def _resolve_policy_file() -> Path | None:
    # 기본 이름
    candidate = OUT / "정책지원관련매핑_251022.xlsx"
    if candidate.exists():
        return candidate
    # outputs 폴더에서 "정책"과 "매핑"이 들어간 아무 xlsx 찾기 (이름이 약간 달라도 로딩)
    xls = sorted([p for p in OUT.glob("*.xlsx") if ("정책" in p.name and "매핑" in p.name)])
    if xls:
        return xls[0]
    return None

POLICY_XLSX = _resolve_policy_file()

# -------------------- 파일 경로 (다운로드 이후) --------------------
named_candidate = OUT / "predictions_latest_both_delta_named.parquet"
FILE_PRED   = named_candidate if named_candidate.exists() else (OUT / "predictions_latest_both_delta.parquet")
FILE_MERGED = OUT / "merged_indices_monthly.parquet"
FILE_MAPCSV = OUT / "big_data_set1_f.csv"          # (선택) 가맹점명 매핑 CSV
FILE_ALERTS = OUT / "signals_alerts_delta.csv"     # (선택) 경고사유 텍스트
FILE_SIGREC = OUT / "signals_recent_delta.csv"     # (선택) 지표 delta/gap 기반 설명

LOG_DIR = OUT / "ai_logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "ai_explanation_log.jsonl"

# -------------------- UI/스타일 --------------------
st.set_page_config(page_title="AI 기반 폐업 조기경보 플랫폼", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.2rem !important; padding-bottom: 1.8rem !important; }
h1.app-title{ font-size: 32px; font-weight: 900; line-height: 1.36; margin: .1rem 0 .8rem 0; }
.badge{display:inline-block;padding:.2rem .5rem;border-radius:8px;font-size:12px;font-weight:800}
.badge-red{background:rgba(239,68,68,.18);color:#ef4444}
.badge-amber{background:rgba(245,158,11,.18);color:#f59e0b}
.badge-emerald{background:rgba(16,185,129,.18);color:#10b981}
.small{font-size:12px;opacity:.75}
.caption-note{font-size:13px; opacity:.92;}
.caption-list{font-size:13px; opacity:.92; padding-left:1.1rem}

/* 공동구매 & 리워드 카드 스타일 */
.card-stack { display: grid; row-gap: 12px; }
.card { border-radius: 14px; padding: 16px 18px; border: 1px solid rgba(0,0,0,.06);
        background: #fff; box-shadow: 0 2px 6px rgba(0,0,0,.04); }
.card-title { font-weight: 700; margin-bottom: 6px; }
.card-sub { font-size: 13.5px; opacity: .86; margin-bottom: 12px; }
.card-badge { font-size: 12.5px; opacity: .8; }
.btn { display:inline-block; padding:10px 14px; border-radius:10px; font-weight:800; text-align:center; }
.btn-primary { background:#ea580c1a; border:1px solid #ea580c55; color:#b45309; }
.btn-primary:hover{ background:#ea580c2a; }
.btn-magenta { background:#db27771a; border:1px solid #db277755; color:#9d174d; }
.btn-magenta:hover{ background:#db27772a; }
.section-chip{font-weight:800;opacity:.9}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1 class='app-title'>😀 AI 기반 폐업 조기경보 플랫폼</h1>", unsafe_allow_html=True)
st.caption("AI 기반 폐업위험 예측 및 맞춤형 지원 시스템")

# -------------------- 로딩 유틸 --------------------
def _try_read_csv(path: Path) -> pd.DataFrame:
    for enc in ("cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    try:
        txt = path.read_bytes().decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(txt))
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_pred() -> pd.DataFrame:
    p = FILE_PRED
    if p.suffix.lower()==".parquet":
        return pd.read_parquet(p)
    return _try_read_csv(p)

@st.cache_data(show_spinner=False)
def load_merged() -> pd.DataFrame:
    if FILE_MERGED.exists():
        return pd.read_parquet(FILE_MERGED) if FILE_MERGED.suffix.lower()==".parquet" else _try_read_csv(FILE_MERGED)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_mapping() -> pd.DataFrame:
    if not FILE_MAPCSV.exists(): return pd.DataFrame()
    df = _try_read_csv(FILE_MAPCSV)
    id_col  = next((c for c in df.columns if str(c).upper() in {"ENCODED_MCT","ENCODED","STORE_ID"}), None)
    nm_col  = next((c for c in df.columns if str(c).upper() in {"MCT_NM","STORE_NAME","가맹점명"}), None)
    bse_col = next((c for c in df.columns if "MCT_BSE" in str(c).upper()), None)
    if not id_col or not nm_col: return pd.DataFrame()
    m = df[[id_col, nm_col] + ([bse_col] if bse_col else [])].copy()
    m.columns = ["ENCODED_MCT","MCT_NM"] + (["MCT_BSE_AR"] if bse_col else [])
    m["ENCODED_MCT"] = m["ENCODED_MCT"].astype(str).str.strip()
    m["MCT_NM"] = m["MCT_NM"].astype(str).str.strip()
    if "MCT_BSE_AR" in m.columns: m["MCT_BSE_AR"] = m["MCT_BSE_AR"].astype(str).str.strip()
    return m.drop_duplicates()

@st.cache_data(show_spinner=False)
def load_alerts() -> pd.DataFrame:
    return _try_read_csv(FILE_ALERTS) if FILE_ALERTS.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_sigrec() -> pd.DataFrame:
    return _try_read_csv(FILE_SIGREC) if FILE_SIGREC.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_policy_map() -> pd.DataFrame:
    if POLICY_XLSX is None or not Path(POLICY_XLSX).exists(): 
        return pd.DataFrame()
    try: df = pd.read_excel(POLICY_XLSX)
    except Exception: 
        df = pd.read_excel(POLICY_XLSX, engine="openpyxl")
    colmap = {}
    for c in df.columns:
        cu = str(c).strip().lower()
        if cu in {"policy_id","id","정책id","pid"}: colmap[c] = "policy_id"
        elif cu in {"title","정책명","name"}: colmap[c] = "title"
        elif cu in {"owner","기관","주관기관"}: colmap[c] = "owner"
        elif cu in {"url","link","신청링크"}: colmap[c] = "url"
        elif cu in {"deadline","마감","마감일"}: colmap[c] = "deadline"
        elif cu in {"support_type","유형","type","지원유형"}: colmap[c] = "support_type"
        elif cu in {"region","지역","행정동","지자체"}: colmap[c] = "region"
        elif cu in {"industry","업종"}: colmap[c] = "industry"
        elif cu in {"min_risk","최소등급","위험등급"}: colmap[c] = "min_risk"
        elif cu in {"driver_tags","risk_tags","태그","tags"}: colmap[c] = "risk_tags"
        elif cu in {"risk_group","위험군","group"}: colmap[c] = "risk_group"
        elif cu in {"summary","설명","비고"}: colmap[c] = "summary"
    df = df.rename(columns=colmap)
    for k in ["policy_id","title","owner","url","deadline","support_type","region","industry","min_risk","risk_tags","risk_group","summary"]:
        if k not in df.columns: df[k] = ""
    df["policy_id"] = df["policy_id"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["owner"] = df["owner"].astype(str).str.strip()
    df["url"] = df["url"].astype(str).str.strip()
    df["support_type"] = df["support_type"].astype(str).str.strip().str.lower()
    df["region"] = df["region"].astype(str).str.strip()
    df["industry"] = df["industry"].astype(str).str.strip()
    df["min_risk"] = df["min_risk"].astype(str).str.strip()
    df["risk_group"] = df["risk_group"].astype(str).str.strip().str.lower()
    df["risk_tags"] = (df.get("risk_tags","").astype(str) if "risk_tags" in df else "")
    try: df["deadline"] = pd.to_datetime(df["deadline"], errors="coerce")
    except Exception: df["deadline"] = pd.NaT
    def _infer_support_type(row):
        t = (row.get("support_type") or "").strip().lower()
        if t: return t
        g = (row.get("risk_group") or "").lower()
        if any(k in g for k in ["대출","금융","부담","금리"]): return "loan"
        if any(k in g for k in ["보험"]): return "insurance"
        if any(k in g for k in ["고객","마케팅","홍보","판로"]): return "marketing"
        if any(k in g for k in ["공동구매","원가","임대","비용"]): return "sourcing"
        return "policy"
    df["support_type"] = df.apply(_infer_support_type, axis=1)
    return df

def _ensure_datetime(df: pd.DataFrame, col="month"):
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")

# -------------------- 데이터 적재/정리 --------------------
pred   = load_pred(); merged = load_merged(); mapping = load_mapping()
alerts = load_alerts(); sigrec  = load_sigrec(); policy_map = load_policy_map()
for _df in (pred, merged, alerts, sigrec):
    if not _df.empty: _ensure_datetime(_df, "month")

# 표준 키/이름 매핑
if "store_id" in pred.columns: pred["store_id"] = pred["store_id"].astype(str)
if "ENCODED_MCT" not in pred.columns and "store_id" in pred.columns:
    pred["ENCODED_MCT"] = pred["store_id"]
if not mapping.empty and {"ENCODED_MCT","MCT_NM"}.issubset(mapping.columns):
    pred = pred.merge(mapping, on="ENCODED_MCT", how="left")
name = pred.get("MCT_NM", pd.Series(index=pred.index, dtype=object))
name = name.astype(str).str.strip().replace({"nan":"","None":"","NULL":"","<NA>":""})
pred["MCT_NM"] = np.where(name.eq(""), pred["store_id"], name)

# -------------------- 공통 유틸/설명 --------------------
def latest_month(df: pd.DataFrame) -> pd.Timestamp | None:
    return pd.to_datetime(df["month"]).max() if "month" in df else None

def latest_per_store(df: pd.DataFrame) -> pd.DataFrame:
    if "month" not in df: return df.copy()
    idx = df.groupby("store_id", observed=False)["month"].idxmax()
    return df.loc[idx].copy()

def pct(n, d):
    try: return f"{100*n/d:.1f}%"
    except Exception: return "N/A"

def num(x):
    try: return f"{int(x):,}"
    except Exception: return "0"

def _kdict():
    return {"customer":"고객층 변화","market":"상권 혼잡·폐업률","industry":"업종 위험","sales":"매출 흐름","macro":"거시·임대료"}

def _intro_text(kind: str) -> str:
    msg = {
        "avg3":"<div class='callout'><b>전체 점포의 월별 평균 ‘3개월 후 폐업 위험 확률’</b>입니다.</div>",
        "avg6":"<div class='callout'><b>전체 점포의 월별 평균 ‘6개월 후 폐업 위험 확률’</b>입니다.</div>",
        "shop3":"<div class='callout'>선택한 점포의 <b>월별 ‘3개월 후 폐업 위험 확률’</b> 추세입니다.</div>",
        "shop6":"<div class='callout'>선택한 점포의 <b>월별 ‘6개월 후 폐업 위험 확률’</b> 추세입니다.</div>",
    }
    return msg.get(kind,"")

def _describe_ts(months: pd.Series, values: pd.Series, scope_label: str) -> str:
    try:
        m = pd.to_datetime(months); v = pd.to_numeric(values, errors="coerce")
        ok = m.notna() & v.notna(); m, v = m[ok], v[ok]
        if len(v) < 2: return f"<div class='caption-note'>· {scope_label}: 데이터가 충분하지 않습니다.</div>"
        last, last_m = float(v.iloc[-1]), m.iloc[-1]
        w = 3 if len(v) >= 4 else (len(v)-1)
        recent_delta = last - float(v.iloc[-(w+1)])
        idx_max, idx_min = int(v.argmax()), int(v.argmin())
        m_max, v_max = m.iloc[idx_max], float(v.iloc[idx_max]); m_min, v_min = m.iloc[idx_min], float(v.iloc[idx_min])
        def p(x): return f"{x*100:.1f}%"
        def sign_txt(x):
            return f"상승(+{p(x)})" if x>0.0001 else (f"하락({p(x)})" if x<-0.0001 else "큰 변화 없음(±0.0%p)")
        return "".join([
            f"<div class='caption-note'><b>그래프 요약</b> — {scope_label}</div>",
            "<ul class='caption-list'>",
            f"<li><b>현재</b>: {last_m.strftime('%Y-%m')} 기준 {p(last)} 입니다.</li>",
            f"<li><b>최근</b>: 최근 {w}개월 {sign_txt(recent_delta)}.</li>",
            f"<li><b>범위</b>: 최고 {p(v_max)}({m_max.strftime('%Y-%m')}), 최저 {p(v_min)}({m_min.strftime('%Y-%m')}).</li>",
            "</ul>"
        ])
    except Exception:
        return f"<div class='caption-note'>· {scope_label}: 해석 생성 중 오류가 발생했습니다.</div>"

# === build_ai_prompt ===
def build_ai_prompt(store_name: str, store_id: str, district: str, category: str,
                    top_groups: list[str], reasons: list[str], score_now: float,
                    extra_metrics: dict) -> str:
    lines = []
    lines.append(f"[점포] 이름: {store_name} / ID: {store_id}")
    if district or category:
        lines.append(f"[메타] 행정동: {district or '-'} / 업종: {category or '-'}")
    if np.isfinite(score_now): lines.append(f"[현재 위험확률(3M)] {score_now*100:.1f}%")
    if top_groups: lines.append("[영향 그룹 Top3] " + " · ".join(top_groups))
    if reasons: lines.append("[원인 신호] " + " / ".join([r.lstrip('- ').strip() for r in reasons]))
    if extra_metrics: lines.append("[요약 수치] " + " / ".join([f"{k}: {v}" for k,v in extra_metrics.items()]))
    system = textwrap.dedent("""
    작업: 주어진 점포 위험요약을 2~3줄로 설명하세요.
    - 왜 알림이 떴는지(핵심 원인)
    - 지금 점포주가 할 일(CTA, 행동 1~2개)
    문체: 간결, 실행지향, 숫자는 그대로 사용.
    """).strip()
    return system + "\n\n" + "\n".join(lines)

def _save_ai_log(store_id: str, prompt: str, response: str, metrics: dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.datetime.now().isoformat(),
            "store_id": store_id, "prompt": prompt, "response": response, "metrics": metrics
        }, ensure_ascii=False) + "\n")

# ------------- Gemini 키/모델 -------------
def _get_gemini_key_from_user() -> str | None:
    key = None
    try:
        if hasattr(st, "secrets"): key = st.secrets.get("GEMINI_API_KEY", None)
    except StreamlitSecretNotFoundError: key = None
    if not key: key = os.getenv("GEMINI_API_KEY")
    if not key: key = st.session_state.get("GEMINI_API_KEY")
    if not key:
        with st.popover("🔐 Gemini API 키 입력", use_container_width=True):
            st.caption("· 권장: .streamlit/secrets.toml 또는 환경변수 사용", unsafe_allow_html=True)
            _k = st.text_input("GEMINI_API_KEY", type="password", placeholder="AIza...", label_visibility="collapsed")
            if _k: st.session_state["GEMINI_API_KEY"] = key = _k.strip(); st.success("키가 세션에 저장되었습니다.")
    return key

def _list_models_safe():
    if not _HAS_GEMINI: return []
    try:
        models=[]
        for m in genai.list_models():
            name = getattr(m,"name",None) or getattr(m,"model",None)
            methods = set(getattr(m,"supported_generation_methods",[]) or [])
            if not methods:
                caps = getattr(m,"generation_capabilities",None)
                if isinstance(caps,dict): methods = set(caps.get("methods",[]))
            models.append((name,methods))
        return models
    except Exception:
        return []

def _pick_gemini_model() -> str:
    preferred = [
        "models/gemini-2.5-flash","models/gemini-2.5-pro",
        "models/gemini-1.5-flash-latest","models/gemini-1.5-flash",
        "models/gemini-1.5-pro","models/gemini-1.0-pro",
    ]
    avail = _list_models_safe()
    def ok(ms):
        s={str(x).lower() for x in (ms or set())}
        return any(k in s for k in ("generatecontent","generate_content"))
    text_models=[n for (n,ms) in avail if n and ok(ms)]
    for m in preferred:
        if m in text_models: return m
    return text_models[0] if text_models else "models/gemini-2.5-flash"

def _gemini_generate(prompt: str) -> str:
    key = _get_gemini_key_from_user()
    if not _HAS_GEMINI: return "[설치 필요] pip install google-generativeai"
    if not key: return "[키 필요] 오른쪽 상단 '🔐 Gemini API 키 입력'에서 키를 넣어주세요."
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name=_pick_gemini_model())
        resp = model.generate_content(prompt, generation_config={"temperature":0.4})
        return (resp.text or "").strip() or "[AI 설명 생성 실패] 응답이 비었습니다."
    except Exception as e:
        return f"[AI 설명 생성 오류] {e}"

# -------------------- 탭 --------------------
t_overview, t_map, t_store, t_policy = st.tabs(
    ["Overview", "Risk Map", "Store Explorer", "AI Policy Lab"]
)

# -------------------- Overview --------------------
with t_overview:
    st.markdown("### 🧭 상권 위험 개요")
    lm = latest_month(pred)
    latest = pred[pred["month"]==lm].copy() if lm is not None else pred.copy()
    if not pred.empty:
        any3 = latest.groupby("store_id", observed=False)["risk_label_3m"].max() if "risk_label_3m" in latest else pd.Series(dtype=int)
        any6 = latest.groupby("store_id", observed=False)["risk_label_6m"].max() if "risk_label_6m" in latest else pd.Series(dtype=int)
        total = any3.index.nunique() if len(any3) else pred["store_id"].nunique()
        high3 = int((any3==1).sum()) if len(any3) else 0
        warn6 = int(((any6==1) & ((any3!=1) if len(any3) else True)).sum()) if len(any6) else 0
    else:
        total = high3 = warn6 = 0
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("전체 점포 수", num(total))
    with c2: st.metric("3개월 내 고위험 점포", num(high3), pct(high3, total))
    with c3: st.metric("6개월 내 위험 점포", num(warn6), pct(warn6, total))
    with c4:
        f1 = pred.get("f1_3m", pd.Series([np.nan])).dropna(); auc = pred.get("auc_3m", pd.Series([np.nan])).dropna()
        txt = ("F1 " + (f"{f1.iloc[0]:.3f}" if len(f1) else "N/A")) + (f" · AUC {auc.iloc[0]:.3f}" if len(auc) else "")
        st.metric("모델 성능(3M)", txt)

    two = st.columns(2)
    if {"month","risk_proba_3m"}.issubset(pred.columns):
        t = pred.groupby("month", as_index=False, observed=False)["risk_proba_3m"].mean().dropna()
        two[0].markdown(_intro_text("avg3"), unsafe_allow_html=True)
        fig = px.line(t, x="month", y="risk_proba_3m", markers=True, height=360, title="월별 평균 위험 확률(3개월)")
        fig.update_layout(yaxis_title="위험 확률(평균)", xaxis_title="월")
        two[0].plotly_chart(fig, use_container_width=True)
        two[0].markdown(_describe_ts(t["month"], t["risk_proba_3m"], "전체 평균(3개월)"), unsafe_allow_html=True)
    if {"month","risk_proba_6m"}.issubset(pred.columns):
        t2 = pred.groupby("month", as_index=False, observed=False)["risk_proba_6m"].mean().dropna()
        two[1].markdown(_intro_text("avg6"), unsafe_allow_html=True)
        fig2 = px.line(t2, x="month", y="risk_proba_6m", markers=True, height=360, title="월별 평균 위험 확률(6개월)")
        fig2.update_layout(yaxis_title="위험 확률(평균)", xaxis_title="월")
        two[1].plotly_chart(fig2, use_container_width=True)
        two[1].markdown(_describe_ts(t2["month"], t2["risk_proba_6m"], "전체 평균(6개월)"), unsafe_allow_html=True)

# -------------------- Risk Map --------------------
with t_map:
    st.markdown("### 🗺️ Risk Map — 지역별 위험도")
    geo_path = APP_DIR / "assets" / "seoul_districts.geojson"
    if pred.empty or "district" not in pred:
        st.info("district 컬럼이 필요합니다.")
    else:
        g3 = pred.groupby("district", as_index=False, observed=False)["risk_proba_3m"].mean()
        g6 = pred.groupby("district", as_index=False, observed=False)["risk_proba_6m"].mean() if "risk_proba_6m" in pred else None
        if geo_path.exists():
            import json
            with open(geo_path, encoding="utf-8") as f: geojson = json.load(f)
            which = st.radio("표시 지표", ["3개월 위험도(평균)"] + (["6개월 위험도(평균)"] if g6 is not None else []), horizontal=True)
            val_col = "risk_proba_3m" if which.startswith("3개월") else "risk_proba_6m"
            g = g3 if val_col=="risk_proba_3m" else g6
            vals = pd.to_numeric(g[val_col], errors="coerce"); vmin, vmax = float(vals.min()), float(vals.max())
            if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin==vmax: vmin, vmax = 0.0, 1.0
            fig = px.choropleth_mapbox(
                g, geojson=geojson, locations="district", color=val_col, featureidkey="properties.name",
                color_continuous_scale="Reds", range_color=(vmin, vmax),
                mapbox_style="carto-positron", zoom=11.7, center={"lat":37.547,"lon":127.035},
                opacity=0.68, height=680,
                title=("행정동별 평균 3개월 위험도" if val_col=="risk_proba_3m" else "행정동별 평균 6개월 위험도"),
            )
            fig.update_traces(hovertemplate="<b>행정동:</b> %{location}<br>" +
                              ("<b>3개월 위험도:</b> %{z:.1%}" if val_col=="risk_proba_3m" else "<b>6개월 위험도:</b> %{z:.1%}") + "<extra></extra>")
            fig.update_layout(
                hoverlabel=dict(bgcolor="rgba(255,255,255,0.96)", font_size=16, font_color="black", font_family="Arial"),
                margin=dict(l=0,r=0,t=60,b=0),
                coloraxis_colorbar=dict(title="위험 확률(%)", tickformat=".0%"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("지도 파일이 없어 표로 대체합니다. `dashboard/assets/seoul_districts.geojson` 를 준비하세요.")
            show = g3.merge(g6, on="district", how="outer") if g6 is not None else g3
            if "risk_proba_3m" in show: show["3M(%)"] = (show["risk_proba_3m"]*100).round(1)
            if "risk_proba_6m" in show: show["6M(%)"] = (show["risk_proba_6m"]*100).round(1)
            st.dataframe(show.rename(columns={"district":"행정동"})[["행정동"]+[c for c in ["3M(%)","6M(%)"] if c in show]],
                         use_container_width=True, height=580)

# -------------------- Store Explorer --------------------
with t_store:
    st.markdown("### 🏪 상점명 기반 상세 분석")
    st.caption("필터로 후보를 좁힌 후 점포를 선택해 주세요. **선택 점포의 최신 기준**으로 출력됩니다.")
    c_dist, c_cat, c_shop = st.columns([1.0, 1.0, 1.4])
    with c_dist:
        dist_sel = st.multiselect("행정동", sorted(pred["district"].dropna().unique()) if "district" in pred else [], placeholder="행정동 선택")
    with c_cat:
        cat_sel = st.multiselect("업종", sorted(pred["category"].dropna().unique()) if "category" in pred else [], placeholder="업종 선택")
    cand = pred.copy()
    if dist_sel and "district" in cand: cand = cand[cand["district"].isin(dist_sel)]
    if cat_sel and "category" in cand: cand = cand[cand["category"].isin(cat_sel)]
    cand_latest = latest_per_store(cand)

    name_col = "MCT_NM_mask" if "MCT_NM_mask" in cand_latest.columns else ("MCT_NM" if "MCT_NM" in cand_latest.columns else None)
    if not name_col: cand_latest["__tmp_name__"] = cand_latest["store_id"].astype(str); name_col = "__tmp_name__"
    def _fmt_label(r):
        nm = str(r.get(name_col,"")).strip(); dong = str(r.get("district","")).strip()
        base = nm if nm else str(r.get("store_id","")); return f"{base} · {dong}" if dong else base
    opts = cand_latest[["store_id","district",name_col]].copy(); opts["__label"] = opts.apply(_fmt_label, axis=1)
    dup = opts["__label"].duplicated(keep=False)
    if dup.any(): opts.loc[dup,"__label"] = opts.loc[dup].apply(lambda r: f"{r['__label']} ({str(r['store_id'])[-5:]})", axis=1)
    opts = opts.sort_values("__label"); label_to_id = dict(zip(opts["__label"], opts["store_id"].astype(str)))
    with c_shop:
        sel_label = st.selectbox("점포 선택 (가게명)", options=list(label_to_id.keys()), index=0 if len(label_to_id) else None, placeholder="가게명 선택")
    if not label_to_id:
        st.warning("선택한 행정동/업종에 해당하는 점포가 없습니다."); st.stop()

    sel_id = label_to_id[sel_label]; sel_id_str = str(sel_id)
    sel_row_latest = cand_latest[cand_latest["store_id"].astype(str)==sel_id_str].iloc[0]
    st.session_state["sel_store_id"] = sel_id_str
    st.session_state["sel_district"] = str(sel_row_latest.get("district",""))
    st.session_state["sel_category"] = str(sel_row_latest.get("category",""))

    sdf = pred[pred["store_id"].astype(str)==sel_id_str].sort_values("month")
    if sdf.empty: st.info("선택한 점포의 시계열 데이터가 아직 없어요."); st.stop()

    store_disp = str(sel_row_latest.get("MCT_NM", sel_id_str))
    st.markdown(f"#### 📍 선택 점포: **{store_disp}**")

    # === 점수(%) 중심 상단 3칸 ===
    cA, cB, cC = st.columns(3)
    last = sdf.iloc[-1]
    proba3 = float(last.get("risk_proba_3m", np.nan))
    proba6 = float(last.get("risk_proba_6m", np.nan)) if "risk_proba_6m" in sdf.columns else np.nan

    # 보조 등급 텍스트 (규칙: 3M=1 이면 고위험, 아니면 6M=1 이면 '위험', 둘다 0이면 '안정')
    t3 = int(last.get("risk_label_3m", 0))
    t6 = int(last.get("risk_label_6m", 0))
    tier3 = "고위험" if t3==1 else "안정"
    tier6 = "고위험" if t3==1 else ("위험" if t6==1 else "안정")

    with cA:
        st.metric("3개월 폐업 위험 점수", "N/A" if not np.isfinite(proba3) else f"{proba3*100:.1f}%")
        st.markdown(f"<span class='small'>등급: {tier3}</span>", unsafe_allow_html=True)
    with cB:
        st.metric("6개월 폐업 위험 점수", "N/A" if not np.isfinite(proba6) else f"{proba6*100:.1f}%")
        st.markdown(f"<span class='small'>등급: {tier6}</span>", unsafe_allow_html=True)
    with cC:
        sdf_local = sdf[["month","risk_proba_3m"]].dropna()
        if len(sdf_local) >= 2:
            w = 3 if len(sdf_local) >= 4 else (len(sdf_local)-1)
            recent_delta = float(sdf_local["risk_proba_3m"].iloc[-1] - sdf_local["risk_proba_3m"].iloc[-(w+1)])
            st.metric("최근 변화(3개월)", f"{recent_delta*100:+.1f}%p")
        else:
            st.metric("최근 변화(3개월)", "N/A")

    g1,g2 = st.columns(2)
    if {"month","risk_proba_3m"}.issubset(sdf.columns):
        g1.markdown(_intro_text("shop3"), unsafe_allow_html=True)
        fig = px.line(sdf, x="month", y="risk_proba_3m", markers=True, height=360, title="점포 위험 확률 추세 (3개월)")
        fig.update_layout(yaxis_title="위험 확률(점포)", xaxis_title="월"); g1.plotly_chart(fig, use_container_width=True)
        g1.markdown(_describe_ts(sdf["month"], sdf["risk_proba_3m"], "이 점포(3개월)"), unsafe_allow_html=True)
    if {"month","risk_proba_6m"}.issubset(sdf.columns):
        g2.markdown(_intro_text("shop6"), unsafe_allow_html=True)
        fig2 = px.line(sdf, x="month", y="risk_proba_6m", markers=True, height=360, title="점포 위험 확률 추세 (6개월)")
        fig2.update_layout(yaxis_title="위험 확률(점포)", xaxis_title="월"); g2.plotly_chart(fig2, use_container_width=True)
        g2.markdown(_describe_ts(sdf["month"], sdf["risk_proba_6m"], "이 점포(6개월)"), unsafe_allow_html=True)

    st.markdown("#### ⚠️ 폐업 조기위험 원인 분석")
    st.caption("아래 항목은 최근 변화가 큰 지표를 모아 보여드립니다!")
    glabel = _kdict()
    contrib_cols = [c for c in sdf.columns if c.startswith("contrib_") and c.endswith("_3m")]
    top3_groups: list[str] = []
    if contrib_cols:
        lastc = sdf.iloc[-1:][contrib_cols].T; lastc.columns=["val"]
        lastc["group"] = lastc.index.str.replace("contrib_","",regex=False).str.replace("_3m","",regex=False)
        top3_groups = lastc.sort_values("val", ascending=False).head(3)["group"].map(lambda g: glabel.get(g, g)).tolist()
        st.markdown("**영향 그룹(최신월 Top3)**: " + " · ".join(top3_groups) if top3_groups else "_-_")
    else:
        st.markdown("<span class='small'>그룹 기여 정보가 없어 생략합니다.</span>", unsafe_allow_html=True)

    # ---- 경고 사유 bullets 생성
    bullets: list[str] = []
    if not alerts.empty and {"store_id","month"}.issubset(alerts.columns):
        a = alerts.copy(); a["store_id"] = a["store_id"].astype(str).str.strip(); a = a[a["store_id"]==sel_id_str]
        if not a.empty:
            row = a.sort_values("month").iloc[-1]
            for k in ["reason_1","reason_2","reason_3"]:
                txt = str(row.get(k,"")).strip()
                if txt: bullets.append(f"- {txt}")

    if not bullets and (not sigrec.empty) and {"store_id","month"}.issubset(sigrec.columns):
        s = sigrec.copy(); s["store_id"]=s["store_id"].astype(str).str.strip(); s = s[s["store_id"]==sel_id_str].sort_values("month")
        if not s.empty:
            last_s = s.iloc[-1]; candidates = sorted({c[:-8] for c in s.columns if c.endswith("_delta3m")})
            scores = []
            for k in candidates:
                d = pd.to_numeric(last_s.get(f"{k}_delta3m", np.nan), errors="coerce")
                g = pd.to_numeric(last_s.get(f"{k}_peer_gap", np.nan), errors="coerce")
                if not (np.isfinite(d) or np.isfinite(g)): continue
                label_map = {"youth_share":"20대 이하 고객 비중","revisit_share":"재방문 고객 비중","new_share":"신규 고객 비중",
                             "delivery_ratio":"배달 매출 비중","peer_sales_ratio":"동일 업종 매출 비율(=100)",
                             "peer_trx_ratio":"동일 업종 매출건수 비율(=100)","industry_rank_pct":"업종 내 매출 순위%",
                             "zone_rank_pct":"상권 내 매출 순위%","industry_close_ratio":"업종 내 해지 가맹점 비중",
                             "zone_close_ratio":"상권 내 해지 가맹점 비중","resident_share":"거주 고객 비율",
                             "worker_share":"직장 고객 비율","floating_share":"유동 고객 비율"}
                negative_is_bad = {"industry_rank_pct","zone_rank_pct","industry_close_ratio","zone_close_ratio"}
                higher_is_better = k not in negative_is_bad
                sign = -1.0 if higher_is_better else +1.0
                score = np.nanmean([sign*(d if np.isfinite(d) else 0.0), sign*(g if np.isfinite(g) else 0.0)])
                scores.append((k, score, d, g, higher_is_better, label_map.get(k,k)))
            scores.sort(key=lambda x: (x[1] if np.isfinite(x[1]) else -1e18), reverse=True)
            for k,_, d,g,good,lab in scores[:3]:
                txt = (f"{lab} 하락(최근 3개월 Δ {d:+.2f}), 행정동·업종 동월 평균 대비 {g:+.2f}%p"
                       if good else
                       f"{lab} 상승(최근 3개월 Δ {d:+.2f}), 행정동·업종 동월 평균 대비 {g:+.2f}%p")
                bullets.append(f"- {txt}")

    # ✅ 폴백
    used_fallback = False
    if not bullets:
        if contrib_cols:
            gl = _kdict()
            lastc = sdf.iloc[-1:][contrib_cols].T; lastc.columns = ["val"]
            lastc["group"] = lastc.index.str.replace("contrib_","",regex=False).str.replace("_3m","",regex=False)
            top = (lastc.sort_values("val", ascending=False).head(3)["group"].map(lambda g: gl.get(g, g)).tolist())
            bullets = [f"- {g} 관련 지표가 최근월에 높은 영향" for g in top]
            used_fallback = True
        else:
            sdf_local = sdf[["month","risk_proba_3m"]].dropna()
            if len(sdf_local) >= 2:
                ch = float(sdf_local["risk_proba_3m"].iloc[-1] - sdf_local["risk_proba_3m"].iloc[-2])
                bullets = [f"- 최근 1개월 위험확률 변화: {ch*100:+.1f}%p"]
                used_fallback = True

    if bullets:
        st.markdown("**📌 체크 포인트**")
        if used_fallback:
            st.caption("파일이 없거나 비어 있어 자동 생성된 요약입니다.")
        st.markdown("\n".join(bullets))
    else:
        st.info("설명에 활용 가능한 신호가 부족합니다. (alerts/sigrec 파일 또는 관련 컬럼 생성 필요)")

    col_ai1, _ = st.columns([1,3])
    try: ai_toggle = col_ai1.toggle("AI 설명 켜기 🔎", value=False)
    except Exception: ai_toggle = col_ai1.checkbox("AI 설명 켜기 🔎", value=False)

    def _friendly_tone(text: str) -> str:
        for a,b in {"악화":"아쉬운 흐름","급감":"큰 조정","감소":"조정","하락":"조정","문제":"과제"}.items():
            text = text.replace(a,b)
        return text

    if ai_toggle:
        if _HAS_GEMINI: _ = _get_gemini_key_from_user()
        store_name = str(sel_row_latest.get("MCT_NM", sel_id_str))
        district = st.session_state.get("sel_district",""); category = st.session_state.get("sel_category","")
        sdf_local = sdf[["month","risk_proba_3m"]].dropna()
        recent_delta = np.nan
        if len(sdf_local) >= 2:
            w = 3 if len(sdf_local) >= 4 else (len(sdf_local)-1)
            recent_delta = float(sdf_local["risk_proba_3m"].iloc[-1] - sdf_local["risk_proba_3m"].iloc[-(w+1)])
        extra_metrics = {}
        if np.isfinite(proba3): extra_metrics["위험확률_현재(3M)"] = f"{proba3*100:.1f}%"
        if np.isfinite(recent_delta): extra_metrics["최근변화(3M)"] = f"{recent_delta*100:+.1f}%p"

        prompt = build_ai_prompt(
            store_name=store_name, store_id=sel_id_str,
            district=district, category=category,
            top_groups=top3_groups, reasons=bullets,
            score_now=proba3 if np.isfinite(proba3) else np.nan,
            extra_metrics=extra_metrics,
        )

        with st.spinner("AI가 점포 상황을 정리하는 중입니다..."):
            ai_text = _friendly_tone(_gemini_generate(prompt))

        st.markdown("**✨ AI 요약**")
        parts = [s.strip(" -•·\t.") for s in ai_text.replace("•","\n").replace("·","\n").split("\n") if s.strip(" -•·\t.")]
        items = parts[:3]
        if items:
            st.markdown("<ul style='margin-top:0.4rem; line-height:1.6;'>"+ "".join(f"<li>{it}</li>" for it in items)+"</ul>", unsafe_allow_html=True)
        else:
            st.caption("생성된 문장이 없습니다.")
        _save_ai_log(sel_id_str, prompt, ai_text, extra_metrics)

# -------------------- AI Policy Lab --------------------
with t_policy:
    st.markdown("### 💡 AI Policy Lab — 위험유형별 맞춤 액션 & 정책 추천")
    if pred.empty: st.info("예측 파일이 필요합니다."); st.stop()
    st.markdown("##### 점포 선택")
    c_dist2, c_cat2, c_shop2 = st.columns([1.0,1.0,1.4])
    with c_dist2:
        dist_sel2 = st.multiselect("행정동", sorted(pred["district"].dropna().unique()) if "district" in pred else [],
                                   default=[st.session_state.get("sel_district","")] if st.session_state.get("sel_district") else None)
    with c_cat2:
        cat_sel2 = st.multiselect("업종", sorted(pred["category"].dropna().unique()) if "category" in pred else [],
                                  default=[st.session_state.get("sel_category","")] if st.session_state.get("sel_category") else None)
    cand2 = pred.copy()
    if dist_sel2 and "district" in cand2: cand2 = cand2[cand2["district"].isin(dist_sel2)]
    if cat_sel2 and "category" in cand2: cand2 = cand2[cand2["category"].isin(cat_sel2)]
    cand2_latest = latest_per_store(cand2)
    name_col2 = "MCT_NM_mask" if "MCT_NM_mask" in cand2_latest.columns else ("MCT_NM" if "MCT_NM" in cand2_latest.columns else None)
    if not name_col2: cand2_latest["__tmp_name__"] = cand2_latest["store_id"].astype(str); name_col2 = "__tmp_name__"
    def _fmt_label2(r):
        nm = str(r.get(name_col2,"")).strip(); dong = str(r.get("district","")).strip()
        base = nm if nm else str(r.get("store_id","")); return f"{base} · {dong}" if dong else base
    opts2 = cand2_latest[["store_id","district",name_col2]].copy(); opts2["__label"] = opts2.apply(_fmt_label2, axis=1)
    dup2 = opts2["__label"].duplicated(keep=False)
    if dup2.any(): opts2.loc[dup2,"__label"] = opts2.loc[dup2].apply(lambda r: f"{r['__label']} ({str(r['store_id'])[-5:]})", axis=1)
    opts2 = opts2.sort_values("__label"); label_to_id2 = dict(zip(opts2["__label"], opts2["store_id"].astype(str)))
    default_index = 0
    if st.session_state.get("sel_store_id") and label_to_id2:
        try: default_index = list(label_to_id2.values()).index(st.session_state["sel_store_id"])
        except ValueError: default_index = 0
    with c_shop2:
        sel_label2 = st.selectbox("점포 선택 (가게명)", options=list(label_to_id2.keys()),
                                  index=default_index if len(label_to_id2) else None)
    if not label_to_id2: st.warning("선택한 행정동/업종에 해당하는 점포가 없습니다."); st.stop()

    sel_id2 = label_to_id2[sel_label2]; sel_id_str2 = str(sel_id2)
    row_latest2 = cand2_latest[cand2_latest["store_id"].astype(str)==sel_id_str2].iloc[0]
    district = str(row_latest2.get("district","")); category = str(row_latest2.get("category",""))
    _last_line2 = pred[pred["store_id"].astype(str)==sel_id_str2].sort_values("month").iloc[-1]
    # risk_tier 없으면 레이블로 계산
    if "risk_tier" in _last_line2.index:
        risk_tier = str(_last_line2.get("risk_tier","안정"))
    else:
        _t3 = int(_last_line2.get("risk_label_3m",0)); _t6 = int(_last_line2.get("risk_label_6m",0))
        risk_tier = "고위험" if _t3==1 else ("위험" if _t6==1 else "안정")
    st.session_state.update({"sel_store_id":sel_id_str2,"sel_district":district,"sel_category":category,"sel_risk_tier":risk_tier})

    sdf2 = pred[pred["store_id"].astype(str)==sel_id_str2].sort_values("month")
    if sdf2.empty: st.info("선택한 점포의 데이터가 없습니다."); st.stop()

    c1,c2,c3 = st.columns(3)
    c1.metric("현재 등급", risk_tier); c2.metric("행정동", district or "-"); c3.metric("업종", category or "-")

    tabs = st.tabs(["정책 추천","금융/보험 제안","마케팅/고객확장","공동구매/원가절감"])
    if policy_map.empty:
        for t in tabs:
            with t: st.info("정책 매핑 파일이 없어 데모 카드로 대체됩니다. outputs 폴더의 정책 매핑 xlsx 파일명을 확인하세요.")
    else:
        def show_cards(df):
            if df.empty: st.info("추천 항목이 없습니다."); return
            top = df.head(8)
            for _, r in top.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{r.get('title','(무제)')}**  \n"
                                f"<span class='small'>{r.get('owner','')}</span>", unsafe_allow_html=True)
                    if str(r.get("summary","")).strip(): st.caption(str(r.get("summary","")).strip())
                    cols = st.columns(4)
                    cols[0].write(f"유형: **{r.get('support_type','-')}**")
                    cols[1].write(f"지역: **{r.get('region','-')}**")
                    cols[2].write(f"업종: **{r.get('industry','-')}**")
                    ddl = r.get("deadline"); cols[3].write(f"마감: **{ddl.date() if pd.notna(ddl) else '상시'}**")
                    url = str(r.get("url","")).strip()
                    if url: st.link_button("신청/안내 바로가기", url)
        with tabs[0]: st.subheader("정부/지자체 정책 추천"); show_cards(policy_map[policy_map["support_type"].isin(["policy","grant","advisory","subsidy","gov","public"])])
        with tabs[1]: st.subheader("금융/보험 제안"); show_cards(policy_map[policy_map["support_type"].isin(["loan","credit","bnpl","insurance","fintech"])])
        with tabs[2]: st.subheader("마케팅/고객확장"); show_cards(policy_map[policy_map["support_type"].isin(["marketing","coupon","ad","growth"])])
        with tabs[3]:
            st.subheader("공동구매/원가절감")
            show_cards(policy_map[policy_map["support_type"].isin(["sourcing","procurement","costdown","rent"])])

            # === ⬇️ 이미지 참고한 '공동구매 & 리워드' 추가 카드 ===
            # === 공동구매 & 리워드 (텍스트 설명형) ===
            st.markdown("#### 공동구매 & 리워드")
            
            with st.container(border=True):
                st.markdown(
                    """
            ##### 프로그램 구조
            - **데이터 기반 매칭**: 위험상권·소상공인 데이터를 활용해 적합한 공급사를 자동 매칭
            - **지자체·카드사 협업**: 성동구 × 신한카드와 함께 **식자재 공동구매** 운영  
              · **고위험도 점포 우선 알림/선착순 기회 부여**
            - **AI 공동구매 그룹**: 동일 업종·지역 점포를 묶어 **AI 기반 공동구매 그룹** 자동 구성
            - **결제·포인트 연계**: 카드결제 및 제휴 포인트와 연동
            
            ##### 리워드(포인트) 정책
            - **고객 유지·재방문 시 포인트 적립** (점포/카드사 공동 운영)
            - **공동구매 결제 시 포인트 사용** 가능 (현금성 할인 효과)
            - **제휴 포인트 통합**: 예) 신한포인트 등과 상호 전환/사용
            
            ##### 수익모델
            - **거래 수수료**: B2B 판매액의 **1–2%**
            - **프리미엄 공급사 등록비**: 상단 노출·입찰권 등 부가 혜택 포함
            - **성과공유(성공보수)**: **비용절감액의 일정 비율**을 성공보수로 수취
            
            ##### 비즈니스 라인
            - **공동구매·비용절감 (Shared Sourcing)**  
              원재료·포장재·서비스를 공동구매하는 **B2B 상생 마켓플레이스** 구축  
              · **판매수수료/플랫폼 거래 수익** · **협력업체 납품계약 수익**
                    """,
                    unsafe_allow_html=True,
                )

