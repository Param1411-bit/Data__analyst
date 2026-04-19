import io
import json
import traceback
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataGPT — AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:       #0e1117;  --surf:     #161b27;  --surf2:    #1c2333;
  --border:   #262d3d;  --border2:  #38445c;
  --text:     #cdd5ef;  --dim:      #68778f;  --dimmer:   #3a4355;
  --blue:     #4d8ef5;  --blue-bg:  #162040;
  --green:    #3ec98a;  --green-bg: #0c3020;
  --amber:    #f0a020;  --amber-bg: #3a2800;
  --red:      #ef5e5e;  --red-bg:   #3a1010;
  --mono: 'IBM Plex Mono', monospace;
  --sans: 'IBM Plex Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: var(--sans); color: var(--text);
}
[data-testid="stSidebar"] {
  background: var(--surf) !important;
  border-right: 1px solid var(--border);
}

/* ── wordmark ── */
.wm { padding:1.2rem 0 1rem; border-bottom:1px solid var(--border); margin-bottom:1rem; }
.wm-title { font-family:var(--mono); font-size:1.3rem; font-weight:600; color:var(--text); }
.wm-title span { color:var(--blue); }
.wm-sub { font-size:0.62rem; color:var(--dimmer); text-transform:uppercase; letter-spacing:2.5px; margin-top:3px; }

/* ── page header ── */
.ph { border-bottom:1px solid var(--border); padding-bottom:0.9rem; margin-bottom:1.4rem; }
.ph-pill { font-family:var(--mono); font-size:0.62rem; font-weight:600; color:var(--blue);
           background:var(--blue-bg); border:1px solid var(--blue); border-radius:3px;
           padding:2px 9px; letter-spacing:1.5px; text-transform:uppercase;
           display:inline-block; margin-bottom:6px; }
.ph h1  { font-family:var(--mono); font-size:1.4rem; font-weight:600; color:var(--text); margin:0; }
.ph-sub { font-size:0.8rem; color:var(--dim); margin-top:3px; }

/* ── insight block ── */
.ins { background:var(--surf); border:1px solid var(--border); border-left:3px solid var(--blue);
       border-radius:0 5px 5px 0; padding:0.9rem 1.1rem; margin:0.7rem 0;
       font-size:0.84rem; line-height:1.75; color:var(--text); }
.ins .lbl { font-family:var(--mono); font-size:0.6rem; color:var(--blue);
            letter-spacing:2px; text-transform:uppercase; margin-bottom:0.4rem; }

/* ── chart rationale ── */
.rat { background:var(--surf2); border:1px solid var(--border); border-radius:5px;
       padding:0.75rem 1rem; margin-top:0.25rem; font-size:0.78rem;
       line-height:1.65; color:var(--dim); }
.rat .lbl { font-family:var(--mono); font-size:0.58rem; color:var(--amber);
            letter-spacing:2px; text-transform:uppercase; margin-bottom:0.3rem; }
.rat b { color:var(--text); font-weight:500; }

/* ── dataset info card (new — matches Session 28 "write a summary") ── */
.dinfo { background:var(--surf2); border:1px solid var(--border); border-radius:6px;
         padding:1rem 1.2rem; margin:0.6rem 0; font-size:0.82rem; line-height:1.7; }
.dinfo .lbl { font-family:var(--mono); font-size:0.58rem; color:var(--blue);
              letter-spacing:2px; text-transform:uppercase; margin-bottom:0.35rem; }

/* ── quality dimension badges (Session 28) ── */
.qdim { display:inline-block; font-family:var(--mono); font-size:0.68rem;
        padding:3px 9px; border-radius:3px; margin:3px 3px 3px 0; font-weight:600; }
.qd-completeness { background:#1a2a40; color:#5ba4f5; border:1px solid #2d5fa0; }
.qd-validity      { background:#2a1a10; color:#f0a020; border:1px solid #a06010; }
.qd-accuracy      { background:#2a1010; color:#ef5e5e; border:1px solid #a03030; }
.qd-consistency   { background:#0f2a1a; color:#3ec98a; border:1px solid #1a7040; }
.qd-tidiness      { background:#1e1a30; color:#a07af5; border:1px solid #604fa0; }

/* ── stat card ── */
.sc { background:var(--surf); border:1px solid var(--border); border-radius:6px; padding:0.9rem 1rem; }
.sc .v { font-family:var(--mono); font-size:1.55rem; font-weight:600; color:var(--text); line-height:1; margin-bottom:3px; }
.sc .v.ok   { color:var(--green); }  .sc .v.warn { color:var(--amber); }  .sc .v.bad { color:var(--red); }
.sc .l { font-size:0.65rem; color:var(--dimmer); text-transform:uppercase; letter-spacing:1.5px; }

/* ── inline tags ── */
.tag { display:inline-block; font-family:var(--mono); font-size:0.68rem;
       padding:2px 7px; border-radius:3px; margin:2px 3px 2px 0; }
.tg { background:var(--green-bg); color:var(--green); border:1px solid var(--green); }
.ta { background:var(--amber-bg); color:var(--amber); border:1px solid var(--amber); }
.tr { background:var(--red-bg);   color:var(--red);   border:1px solid var(--red); }
.tb { background:var(--blue-bg);  color:var(--blue);  border:1px solid var(--blue); }

/* ── cleaning step row ── */
.sr { display:flex; align-items:flex-start; gap:0.6rem; padding:0.5rem 0;
      border-bottom:1px solid var(--border); font-size:0.8rem; }
.sn { font-family:var(--mono); font-size:0.65rem; color:var(--dimmer); min-width:26px; margin-top:1px; }

/* ── scope question card ── */
.qc { background:var(--surf2); border:1px solid var(--border); border-radius:4px;
      padding:0.55rem 0.85rem; margin:0.25rem 0; font-size:0.82rem; line-height:1.5; }
.qn { font-family:var(--mono); font-size:0.65rem; color:var(--blue); margin-right:0.45rem; }

/* ── chat ── */
.cu { background:var(--surf2); border:1px solid var(--border); border-radius:5px 5px 0 5px;
      padding:0.65rem 0.9rem; margin:0.45rem 0 0.45rem 2rem; font-size:0.83rem; }
.ca { background:var(--surf); border:1px solid var(--border); border-left:3px solid var(--blue);
      border-radius:0 5px 5px 5px; padding:0.65rem 0.9rem;
      margin:0.45rem 2rem 0.45rem 0; font-size:0.83rem; line-height:1.7; }

/* ── report ── */
.ra { background:var(--surf); border:1px solid var(--border); border-radius:5px;
      padding:1.4rem 1.8rem; font-size:0.83rem; line-height:1.8;
      color:var(--text); white-space:pre-wrap; word-break:break-word; }

/* ── buttons ── */
.stButton > button {
  background:var(--surf2) !important; color:var(--text) !important;
  font-family:var(--mono) !important; font-size:0.76rem !important;
  font-weight:600 !important; border:1px solid var(--border2) !important;
  border-radius:4px !important; padding:0.45rem 1.3rem !important;
  transition:border-color 0.2s,color 0.2s !important;
}
.stButton > button:hover { border-color:var(--blue) !important; color:var(--blue) !important; }

/* ── inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background:var(--surf2) !important; color:var(--text) !important;
  border-color:var(--border) !important; font-family:var(--sans) !important;
  font-size:0.84rem !important;
}
.stSelectbox > div > div { background:var(--surf2) !important; color:var(--text) !important; }

/* ── tabs ── */
[data-testid="stTabs"] [role="tab"] {
  font-family:var(--mono) !important; font-size:0.73rem !important; color:var(--dim) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color:var(--blue) !important; border-bottom-color:var(--blue) !important;
}

/* ── expander ── */
details > summary {
  font-family:var(--mono) !important; font-size:0.76rem !important;
  color:var(--dim) !important; background:var(--surf) !important;
  border:1px solid var(--border) !important; border-radius:4px !important;
  padding:0.45rem 0.75rem !important;
}

code { font-family:var(--mono) !important; background:var(--surf2) !important;
       color:var(--blue) !important; padding:1px 5px !important;
       border-radius:3px !important; font-size:0.82em !important; }
hr   { border-color:var(--border) !important; margin:1.1rem 0 !important; }
.stDataFrame { border-radius:5px; overflow:hidden; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track  { background:var(--bg); }
::-webkit-scrollbar-thumb  { background:var(--border2); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    """Initialise all session-state keys once with safe defaults."""
    defaults = {
        "df_original":   None,   # raw frame — never modified
        "df_clean":      None,   # cleaned frame — used everywhere downstream
        "filename":      "",     # uploaded filename for display
        "questions":     [],     # scope questions (Framework 1)
        "cleaning_log":  [],     # list[str] of applied steps
        "suggestions":   [],     # list[dict] from assess_data()
        "assessment":    {},     # full assessment dict
        "conclusions":   "",     # AI-generated conclusions text
        "stress_test":   "",     # Framework 3 stress-test output
        "case_report":   "",     # final markdown report
        "chat_history":  [],     # EDA chat messages
        "stage":         "01 · Define Scope",
        "groq_ok":       False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────────
# GROQ — current non-deprecated models only
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "llama-3.3-70b-versatile": "LLaMA 3.3 · 70B  (best quality)",
    "llama-3.1-8b-instant":    "LLaMA 3.1 · 8B   (fastest)",
    "mixtral-8x7b-32768":      "Mixtral · 8×7B   (long context)",
    "gemma2-9b-it":            "Gemma 2 · 9B     (balanced)",
}

SYSTEM_PROMPT = """You are DataGPT — a senior data analyst with 10+ years of experience.
Rules:
- Direct and precise. No filler phrases, no flattery.
- Every claim is backed by a specific number from the data.
- Prefix uncertain statements with "Assumption:".
- If data is insufficient, say so — do not fill the gap.
- Use numbered lists for findings.
- Always state what the data CANNOT tell us."""


def make_client(api_key: str) -> Optional[Groq]:
    """Return a Groq client if key is non-empty, else None."""
    k = (api_key or "").strip()
    return Groq(api_key=k) if k else None


def call_llm(
    client: Groq,
    model: str,
    user_msg: str,
    system: str = SYSTEM_PROMPT,
    temperature: float = 0.35,
    max_tokens: int = 1800,
) -> str:
    """
    Call Groq chat completion with granular error handling.

    Raises ValueError with a user-readable message on auth/rate/model errors.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "invalid_api_key" in msg.lower():
            raise ValueError("Invalid API key — verify at console.groq.com.") from exc
        if "429" in msg or "rate_limit" in msg.lower():
            raise ValueError("Rate limit hit — wait a moment and retry.") from exc
        if "404" in msg or "model_not_found" in msg.lower():
            raise ValueError(f"Model '{model}' not found — pick another.") from exc
        raise ValueError(f"Groq API error: {msg[:220]}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY BASE LAYOUT  ← BUG FIX: xaxis/yaxis removed from PBASE
# ─────────────────────────────────────────────────────────────────────────────
# Root cause of the TypeError: when px/go already set xaxis/yaxis on the figure
# and PBASE also contained xaxis/yaxis, update_layout() got duplicate kwargs.
# Fix: PBASE contains only paper/plot/font/margin.
# GRID is a separate dict merged per-chart where needed.

PBASE = dict(
    paper_bgcolor="#161b27",
    plot_bgcolor="#161b27",
    font=dict(family="IBM Plex Sans", color="#cdd5ef", size=11),
    margin=dict(l=44, r=18, t=42, b=38),
)
GRID = dict(gridcolor="#262d3d", linecolor="#262d3d", zerolinecolor="#262d3d")


def _layout(fig: go.Figure, title: str, xt: str = "", yt: str = "",
            extra_x: dict = None, extra_y: dict = None):
    """
    Apply consistent dark theme to any Plotly figure without duplicate-kwarg errors.

    Args:
        fig:     The figure to update.
        title:   Chart title text.
        xt:      x-axis label.
        yt:      y-axis label.
        extra_x: Additional x-axis kwargs (merged with GRID).
        extra_y: Additional y-axis kwargs (merged with GRID).
    """
    xd = {**GRID, **(extra_x or {})}
    yd = {**GRID, **(extra_y or {})}
    if xt:
        xd["title"] = xt
    if yt:
        yd["title"] = yt
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=xd,
        yaxis=yd,
        **PBASE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='wm'>
      <div class='wm-title'>Data<span>GPT</span></div>
      <div class='wm-sub'>AI Data Analyst · Data -gpt</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-family:var(--mono);font-size:0.65rem;color:var(--dimmer);"
        "letter-spacing:2px;text-transform:uppercase;margin-bottom:0.35rem;'>GROQ</p>",
        unsafe_allow_html=True,
    )
    # FIX: label must be non-empty; use label_visibility to hide it visually
    api_key = st.text_input(
        "Groq API Key", type="password", placeholder="gsk_...",
        help="Free at console.groq.com", label_visibility="collapsed",
    )
    model = st.selectbox("Model", list(MODELS.keys()),
                         format_func=lambda m: MODELS[m])

    groq_client = make_client(api_key)
    if api_key:
        st.session_state["groq_ok"] = bool(groq_client)
        cls = "tg" if groq_client else "tr"
        lbl = "● connected" if groq_client else "● key format invalid"
        st.markdown(f"<span class='tag {cls}'>{lbl}</span>", unsafe_allow_html=True)
    else:
        st.session_state["groq_ok"] = False
        st.markdown("<span class='tag ta'>● no key entered</span>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-family:var(--mono);font-size:0.65rem;color:var(--dimmer);"
        "letter-spacing:2px;text-transform:uppercase;margin-bottom:0.35rem;'>WORKFLOW</p>",
        unsafe_allow_html=True,
    )
    STAGES = [
        "01 · Define Scope",
        "02 · Load & Assess",
        "03 · Clean Data",
        "04 · Explore (EDA)",
        "05 · Conclusions",
        "06 · Report",
    ]
    # FIX: label must be non-empty string (Streamlit ≥1.35 will raise on "")
    stage = st.radio(
        "Workflow Stage", STAGES,
        index=STAGES.index(st.session_state["stage"]),
        label_visibility="collapsed",
    )
    st.session_state["stage"] = stage

    st.markdown("<hr>", unsafe_allow_html=True)
    df_ref = st.session_state.get("df_clean")
    if df_ref is not None:
        st.markdown(
            "<p style='font-family:var(--mono);font-size:0.65rem;color:var(--dimmer);"
            "letter-spacing:2px;text-transform:uppercase;margin-bottom:0.35rem;'>ACTIVE DATASET</p>",
            unsafe_allow_html=True,
        )
        fname = st.session_state.get("filename", "")
        if fname:
            st.markdown(
                f"<span class='tag tb' style='font-size:0.6rem;'>{fname}</span>",
                unsafe_allow_html=True,
            )
        ca, cb = st.columns(2)
        ca.metric("Rows", f"{df_ref.shape[0]:,}")
        cb.metric("Cols", df_ref.shape[1])
        mp = round(df_ref.isnull().mean().mean() * 100, 1)
        st.markdown(
            f"<span class='tag {'tr' if mp > 5 else 'tg'}'>{mp}% missing</span>",
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div style='position:absolute;bottom:1rem;left:0;right:0;text-align:center;
                font-family:var(--mono);font-size:0.58rem;color:var(--dimmer);letter-spacing:1px;'>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def page_header(pill: str, title: str, sub: str = ""):
    """Render consistent stage header."""
    sub_html = f"<div class='ph-sub'>{sub}</div>" if sub else ""
    st.markdown(f"""
    <div class='ph'>
      <div class='ph-pill'>{pill}</div>
      <h1>{title}</h1>
      {sub_html}
    </div>""", unsafe_allow_html=True)


def insight_block(text: str, label: str = "DataGPT Analysis", color: str = "blue"):
    """Render a structured analyst insight box."""
    cmap = {"blue": "var(--blue)", "amber": "var(--amber)", "green": "var(--green)"}
    c = cmap.get(color, "var(--blue)")
    st.markdown(f"""
    <div class='ins' style='border-left-color:{c};'>
      <div class='lbl' style='color:{c};'>{label}</div>
      {text}
    </div>""", unsafe_allow_html=True)


def chart_rationale(why_this: str, alternatives: str, question: str):
    """
    Render analytical justification beneath every chart.
    This is the core educational differentiator — maps to Framework 2 (Execution Chain).
    """
    st.markdown(f"""
    <div class='rat'>
      <div class='lbl'>Chart Rationale — Why This Chart?</div>
      <b>Chosen because:</b> {why_this}<br>
      <b>Alternatives considered & rejected:</b> {alternatives}<br>
      <b>Analytical question answered:</b> {question}
    </div>""", unsafe_allow_html=True)


def quality_badge(dim: str, text: str):
    """Render a Session-28 data-quality-dimension badge."""
    cls_map = {
        "Completeness": "qd-completeness",
        "Validity":     "qd-validity",
        "Accuracy":     "qd-accuracy",
        "Consistency":  "qd-consistency",
        "Tidiness":     "qd-tidiness",
    }
    cls = cls_map.get(dim, "tb")
    st.markdown(
        f"<span class='qdim {cls}'>{dim}</span> {text}",
        unsafe_allow_html=True,
    )


def stat_cards(items: list):
    """
    Render a row of metric stat cards.
    Each item: dict with keys val(str), lbl(str), cls(""/"ok"/"warn"/"bad").
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(f"""
        <div class='sc'>
          <div class='v {item.get("cls","")}'>{item["val"]}</div>
          <div class='l'>{item["lbl"]}</div>
        </div>""", unsafe_allow_html=True)


def pf(fig: go.Figure):
    """
    Render a Plotly figure full-width.

    Uses width='stretch' (Streamlit 1.35+) which is the API-stable replacement
    for the deprecated use_container_width=True parameter.
    Passing use_container_width triggers a console warning every render cycle.
    """
    st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# DATA FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse uploaded file bytes into a DataFrame.

    Mirrors Session 27 content:
    - CSV with encoding fallback (utf-8 → latin-1 → cp1252)
    - Excel via openpyxl

    Raises ValueError with a user-readable message on failure.
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError("Cannot decode CSV — try re-saving with UTF-8 encoding.")
    if ext in ("xlsx", "xls"):
        try:
            return pd.read_excel(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Excel parse error: {e}") from e
    raise ValueError(f"Unsupported format '.{ext}'. Upload CSV or Excel.")


def assess_data(df: pd.DataFrame) -> dict:
    """
    Full structural audit — maps directly to Session 28 assessment methodology.

    Checks (in Session 28 order):
      - Shape, dtypes, sample
      - Missing values (Completeness)
      - Duplicates (Validity / Accuracy)
      - Cardinality of object columns
      - IQR outliers (Accuracy)
      - Skewness (Validity)
      - Date-like strings (Validity)
      - Semantic range violations (Accuracy)

    Returns dict with keys:
        shape, dtypes, missing, missing_pct, duplicate_count,
        cardinality, outliers, skewness, suggestions
    """
    a = {}
    a["shape"]   = df.shape
    a["dtypes"]  = df.dtypes.astype(str).to_dict()

    # ── Missing (Completeness) ──
    miss = df.isnull().sum()
    a["missing"]     = miss[miss > 0].to_dict()
    a["missing_pct"] = {k: round(v / len(df) * 100, 2)
                        for k, v in a["missing"].items()}

    # ── Duplicates ──
    a["duplicate_count"] = int(df.duplicated().sum())

    # ── Cardinality ──
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    a["cardinality"] = {c: int(df[c].nunique()) for c in obj_cols}

    # ── Numeric analysis ──
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # IQR outliers (Accuracy)
    outliers = {}
    for c in num_cols:
        s = df[c].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        n = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        if n:
            outliers[c] = n
    a["outliers"] = outliers

    # Skewness (Validity)
    skew = {}
    for c in num_cols:
        s = df[c].dropna()
        if len(s) >= 10:
            sk = float(s.skew())
            if abs(sk) > 1.0:
                skew[c] = round(sk, 3)
    a["skewness"] = skew

    # ── Build suggestions (Session 28: Define step) ──
    suggestions = []
    already = set()

    # 1. Date-like strings → datetime  [Validity]
    for c in obj_cols:
        sample = df[c].dropna().head(60).astype(str)
        rate = sample.str.match(
            r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})|(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
        ).mean()
        if rate > 0.6:
            suggestions.append({
                "type": "to_datetime", "col": c, "dim": "Validity",
                "reason": (
                    f"'{c}' is object dtype but {int(rate*100)}% of values "
                    "match date patterns. Text dtype prevents sorting, groupby, "
                    "and date-part extraction."
                ),
            })
            already.add(c)

    # 2. Numeric stored as string  [Validity]
    for c in obj_cols:
        if c in already:
            continue
        sample = df[c].dropna().head(120).astype(str).str.strip()
        cleaned = sample.str.replace(r"[₹$€£,%\s]", "", regex=True)
        rate = cleaned.str.match(r"^-?\d+(\.\d+)?$").mean()
        if rate > 0.8:
            suggestions.append({
                "type": "to_numeric", "col": c, "dim": "Validity",
                "reason": (
                    f"'{c}' is object dtype but {int(rate*100)}% of values are "
                    "numeric strings (after stripping ₹/$/%/£ symbols). "
                    "Float/int dtype unlocks descriptive stats and correlation."
                ),
            })
            already.add(c)

    # 3. Low-cardinality object → category  [Validity / memory]
    for c, uniq in a["cardinality"].items():
        if c in already:
            continue
        if uniq <= 25 and len(df) >= 50:
            suggestions.append({
                "type": "to_category", "col": c, "dim": "Validity",
                "reason": (
                    f"'{c}' has {uniq} unique values in {len(df):,} rows "
                    f"({round(uniq/len(df)*100,1)}% cardinality). "
                    "category dtype cuts memory by up to 5× and speeds up groupby."
                ),
            })

    # 4. Duplicates  [Accuracy]
    if a["duplicate_count"]:
        suggestions.append({
            "type": "drop_duplicates", "col": "_all_", "dim": "Accuracy",
            "reason": (
                f"{a['duplicate_count']:,} fully duplicate rows inflate row counts, "
                "skew frequency distributions, and bias any model trained on this data."
            ),
        })

    # 5. High-null columns (>50%) → drop  [Completeness]
    for c, pct in a["missing_pct"].items():
        if pct > 50:
            suggestions.append({
                "type": "drop_col", "col": c, "dim": "Completeness",
                "reason": (
                    f"'{c}' is {pct}% null. Imputing >50% introduces more synthetic "
                    "bias than the column removes. Dropping is the correct decision."
                ),
            })
        elif pct > 0:
            dtype = str(df[c].dtype)
            fill = "median" if ("int" in dtype or "float" in dtype) else "mode"
            extra = (
                f"Median preferred over mean because '{c}' also has "
                f"{outliers.get(c,0)} IQR outliers — median is outlier-robust."
                if fill == "median" and c in outliers else
                "Median preferred over mean — robust to skew." if fill == "median"
                else "Mode used for text/category — preserves modal class."
            )
            suggestions.append({
                "type": f"fill_{fill}", "col": c, "dim": "Completeness",
                "reason": f"'{c}' is {pct}% null. {extra}",
            })

    # 6. Semantic range violations  [Accuracy] — mirrors Session 28 example
    NON_NEG = ["age", "price", "cost", "count", "qty", "quantity",
                "revenue", "sales", "salary", "amount", "weight",
                "height", "duration", "distance", "volume", "units"]
    for c in num_cols:
        if any(h in c.lower() for h in NON_NEG):
            n_neg = int((df[c] < 0).sum())
            if n_neg:
                suggestions.append({
                    "type": "range_flag", "col": c, "dim": "Accuracy",
                    "reason": (
                        f"'{c}' has {n_neg} negative values. "
                        "Domain definition says this column must be non-negative — "
                        "likely data entry errors or sign-convention bugs."
                    ),
                })

    a["suggestions"] = suggestions
    return a


def _dataset_info_block(df: pd.DataFrame, filename: str):
    """
    Render the Session 28 'Write a dataset summary + column descriptions' block.

    This is shown automatically on load — mirrors the first two steps
    of the Session 28 assessment methodology.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols  = df.select_dtypes(include=["datetime64"]).columns.tolist()
    mem_mb   = round(df.memory_usage(deep=True).sum() / 1024**2, 2)

    st.markdown("""
    <div class='dinfo'>
      <div class='lbl'>Step 1 — Dataset Summary (Session 28)</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**File:** `{filename}`")
    c1.markdown(f"**Shape:** `{df.shape[0]:,} rows × {df.shape[1]} cols`")
    c1.markdown(f"**Memory:** `{mem_mb} MB`")
    c2.markdown(f"**Numeric cols ({len(num_cols)}):**")
    for c in num_cols[:8]:
        c2.markdown(f"  - `{c}` ({df[c].dtype})")
    c3.markdown(f"**Categorical cols ({len(cat_cols)}):**")
    for c in cat_cols[:8]:
        c3.markdown(f"  - `{c}` ({df[c].nunique()} unique)")
    if dt_cols:
        st.markdown(f"**Datetime cols:** `{', '.join(dt_cols)}`")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Step 2: Auto-generated column descriptions ──
    st.markdown("""
    <div class='dinfo' style='margin-top:0.6rem;'>
      <div class='lbl'>Step 2 — Column Descriptions (auto-generated)</div>
    """, unsafe_allow_html=True)

    col_rows = []
    for c in df.columns:
        nn   = int(df[c].notna().sum())
        uniq = int(df[c].nunique())
        dtype_str = str(df[c].dtype)
        sample = str(df[c].dropna().iloc[0]) if nn else "—"
        col_rows.append({
            "Column":       c,
            "Dtype":        dtype_str,
            "Non-Null":     nn,
            "Null %":       round((1 - nn / len(df)) * 100, 1),
            "Unique":       uniq,
            "Cardinality%": round(uniq / len(df) * 100, 1),
            "Sample Value": sample[:40],
        })
    st.dataframe(pd.DataFrame(col_rows), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def apply_cleaning(df: pd.DataFrame, suggestions: list) -> tuple:
    """
    Apply all queued suggestions — maps to Session 28 'Code' step.

    Each step is independent (try/except) so one failure does not block the rest.

    Args:
        df:          Raw DataFrame.
        suggestions: List of suggestion dicts from assess_data().

    Returns:
        (cleaned_df, log): tuple of cleaned DataFrame and list[str] of steps.
    """
    df  = df.copy()
    log = []

    for sug in suggestions:
        t, col = sug["type"], sug["col"]
        try:
            if t == "drop_duplicates":
                before = len(df)
                df     = df.drop_duplicates()
                log.append(f"Removed {before - len(df):,} duplicate rows. [Accuracy]")

            elif t == "drop_col" and col in df.columns:
                df.drop(columns=[col], inplace=True)
                log.append(f"Dropped '{col}' (>50% null). [Completeness]")

            elif t == "to_datetime" and col in df.columns:

                import warnings as _warnings

                def _smart_date_parse(series: pd.Series) -> pd.Series:
                    FORMATS = [
                        "%Y-%m-%d",    # 2024-12-29  ISO — highest priority, unambiguous
                        "%Y/%m/%d",    # 2024/12/29  ISO slash
                        "%d/%m/%Y",    # 17/04/2022  EU day-first (day>12 disambiguates)
                        "%m/%d/%Y",    # 10/22/2022  US month-first (day slot >12)
                        "%d-%m-%Y",    # 17-04-2022  EU dash
                        "%m-%d-%Y",    # 10-22-2022  US dash
                        "%d/%m/%y",    # 17/04/22    EU short year
                        "%m/%d/%y",    # 10/22/22    US short year
                        "%d-%m-%y",    # 17-04-22    EU short year dash
                        "%d %b %Y",    # 17 Apr 2022
                        "%d %B %Y",    # 17 April 2022
                        "%b %d, %Y",   # Apr 17, 2022
                        "%B %d, %Y",   # April 17, 2022
                        "%Y%m%d",      # 20220417 compact
                    ]

                    # Convert to strings; keep original null positions as None
                    str_series = series.astype(str).where(series.notna(), other=None)
                    result = pd.Series(
                        pd.NaT, index=series.index, dtype="datetime64[ns]"
                    )

                    for fmt in FORMATS:
                        still_nat = result.isna() & series.notna()
                        if not still_nat.any():
                            break
                        with _warnings.catch_warnings():
                            _warnings.simplefilter("ignore")
                            parsed = pd.to_datetime(
                                str_series[still_nat], format=fmt, errors="coerce"
                            )
                        # update() only fills NaT positions — never overwrites
                        # a successfully parsed value with NaT
                        result.update(parsed[parsed.notna()])

                    # Final fallback: pandas free-form with dayfirst=True
                    still_nat = result.isna() & series.notna()
                    if still_nat.any():
                        with _warnings.catch_warnings():
                            _warnings.simplefilter("ignore")
                            fallback = pd.to_datetime(
                                str_series[still_nat], dayfirst=True, errors="coerce"
                            )
                        result.update(fallback[fallback.notna()])

                    return result

                before_nulls = int(df[col].isna().sum())
                df[col]      = _smart_date_parse(df[col])
                after_nulls  = int(df[col].isna().sum())
                new_nats     = max(0, after_nulls - before_nulls)
                nat_note     = (
                    f" ⚠ {new_nats} value(s) could not be parsed in any known "
                    "format → NaT (inspect those rows manually)"
                    if new_nats
                    else " — 0 new NaTs created"
                )
                log.append(f"'{col}' → datetime64{nat_note}. [Validity]")

            elif t == "to_numeric" and col in df.columns:
                cleaned = df[col].astype(str).str.replace(r"[₹$€£,%\s]", "", regex=True)
                df[col] = pd.to_numeric(cleaned, errors="coerce")
                log.append(f"'{col}' → numeric (symbols stripped). [Validity]")

            elif t == "to_category" and col in df.columns:
                df[col] = df[col].astype("category")
                log.append(f"'{col}' → category dtype. [Validity]")

            elif t == "fill_median" and col in df.columns:
                if df[col].isnull().any():
                    med = df[col].median()
                    df[col] = df[col].fillna(med)
                    log.append(f"'{col}' nulls → median ({med:.4g}). [Completeness]")

            elif t == "fill_mode" and col in df.columns:
                if df[col].isnull().any():
                    mode_s = df[col].mode()
                    if not mode_s.empty:
                        df[col] = df[col].fillna(mode_s.iloc[0])
                        log.append(
                            f"'{col}' nulls → mode ('{mode_s.iloc[0]}'). [Completeness]"
                        )

            elif t == "range_flag":
                log.append(
                    f"⚠ '{col}' has negative values — flagged, manual review needed. [Accuracy]"
                )

        except Exception as exc:
            log.append(f"⚠ Skipped '{col}' ({t}): {exc}")

    return df, log


def build_eda_charts(df: pd.DataFrame) -> list:
    charts = []

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols  = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # ── 1. Histogram + mean/median overlays ───────────────────────────────
    # First thing a real analyst does: look at each numeric column's shape.
    # Determines imputation strategy and whether transformations are needed.
    for col in num_cols[:5]:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        skew_val = round(float(s.skew()), 2)
        n_bins   = min(40, max(10, int(np.sqrt(len(s)))))

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=s, nbinsx=n_bins,
            marker_color="#4d8ef5", opacity=0.8, showlegend=False,
        ))
        fig.add_vline(x=float(s.mean()),   line_dash="dash", line_color="#f0a020",
                      annotation_text=f"mean={s.mean():.2f}",
                      annotation_position="top right")
        fig.add_vline(x=float(s.median()), line_dash="dot",  line_color="#3ec98a",
                      annotation_text=f"median={s.median():.2f}",
                      annotation_position="top left")
        _layout(fig, f"Distribution — {col}", xt=col, yt="Frequency")

        skew_note = (
            f"Skew = {skew_val:+.2f} — right-skewed. Consider log-transform before modelling."
            if skew_val > 1 else
            f"Skew = {skew_val:+.2f} — left-skewed. Check for floor/ceiling effects."
            if skew_val < -1 else
            f"Skew = {skew_val:+.2f} — approximately symmetric."
        )
        charts.append({
            "title": f"Distribution — {col}", "fig": fig,
            "why_this": (
                f"Histogram with mean (amber dashed) and median (green dotted) overlaid. "
                f"{skew_note} Gap between mean and median = skew exists. "
                "Skew determines imputation choice (median, not mean) and model assumptions."
            ),
            "alternatives": (
                "Box plot shows median/IQR/outliers but hides shape (bimodality, gaps). "
                "Line chart requires ordered time data. "
                "Bar chart needs manual binning. "
                "Histogram is the mandatory first chart for any numeric column."
            ),
            "question": (
                f"What is the shape of '{col}'? Normal, skewed, or multimodal? "
                "Any data entry anomalies visible?"
            ),
        })

    # ── 2. Z-score box plot grid — outlier audit ──────────────────────────
    # After individual distributions, compare spread and outliers across ALL columns.
    if len(num_cols) >= 2:
        df_z = df[num_cols].copy()
        for c in num_cols:
            s = df_z[c].dropna()
            if s.std() > 0:
                df_z[c] = (df_z[c] - s.mean()) / s.std()
        df_melt = df_z.melt(var_name="Column", value_name="Z-Score")

        fig = px.box(
            df_melt.dropna(), x="Column", y="Z-Score",
            color="Column",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            points="outliers",
        )
        # FIX: px.box already sets xaxis/yaxis — pass extra args separately
        _layout(fig,
                "Outlier Audit — Z-Score Box Plots (all numeric columns)",
                xt="Column", yt="Z-Score (normalised)")
        fig.update_layout(showlegend=False)

        charts.append({
            "title": "Outlier Audit — Box Plots", "fig": fig,
            "why_this": (
                "Z-score normalisation puts all columns on the same axis regardless of units. "
                "Points beyond whiskers (±1.5×IQR) are individual outliers — "
                "this is the fastest cross-column outlier QA view."
            ),
            "alternatives": (
                "Individual histograms (above) give shape but not side-by-side outlier comparison. "
                "Violin plot adds KDE — useful but heavy with many columns. "
                "Box plot is the standard tool for comparing spread and spotting outliers."
            ),
            "question": (
                "Which columns have the most outliers? "
                "How does spread compare? Are outliers symmetric or one-sided?"
            ),
        })

    # ── 3. Pearson correlation heatmap ───────────────────────────────────
    # Before feature selection or modelling, survey all pairwise linear relationships.
    if len(num_cols) >= 3:
        corr = df[num_cols].corr(numeric_only=True).round(2)
        fig  = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=9),
            hovertemplate="%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>",
        ))
        _layout(fig, "Pearson Correlation Matrix")

        corr_abs = corr.abs()
        np.fill_diagonal(corr_abs.values, 0)
        top_pair = corr_abs.stack().idxmax()
        top_r    = round(corr.loc[top_pair[0], top_pair[1]], 3)

        charts.append({
            "title": "Correlation Matrix", "fig": fig,
            "why_this": (
                f"Covers all {len(num_cols)*(len(num_cols)-1)//2} column pairs in one view. "
                f"Strongest pair: {top_pair[0]} vs {top_pair[1]} (r={top_r}). "
                "Blue = positive, Red = negative. Essential for spotting multicollinearity."
            ),
            "alternatives": (
                "Scatter matrix (pair plot) shows raw points but unreadable beyond ~5 columns. "
                "Individual scatters require checking n×(n-1)/2 combinations manually. "
                "Heatmap is the only scalable correlation summary."
            ),
            "question": (
                "Which numeric columns are linearly related? "
                "Are there multicollinearity concerns for regression?"
            ),
        })

    # ── 4. Scatter for most-correlated pair ──────────────────────────────
    # The heatmap gives r — scatter tells you if the relationship is truly linear.
    if len(num_cols) >= 2:
        corr2 = df[num_cols].corr(numeric_only=True).abs()
        np.fill_diagonal(corr2.values, 0)
        pair  = corr2.stack().idxmax()
        c1, c2 = pair[0], pair[1]
        r_val  = round(df[[c1, c2]].corr().iloc[0, 1], 3)
        color_col = cat_cols[0] if cat_cols else None

        fig = px.scatter(
            df.dropna(subset=[c1, c2]),
            x=c1, y=c2,
            color=color_col,
            opacity=0.55,
            trendline="ols",
            trendline_color_override="#f0a020",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        # FIX: px.scatter already sets xaxis/yaxis — _layout merges cleanly
        _layout(fig, f"Scatter — {c1} vs {c2}  (r = {r_val})", xt=c1, yt=c2)

        charts.append({
            "title": f"Scatter — {c1} vs {c2}", "fig": fig,
            "why_this": (
                f"Scatter for the strongest correlated pair (r={r_val}). "
                "OLS trendline (amber) confirms linearity or reveals that r is "
                "inflated by a cluster of outliers. "
                + (f"Coloured by '{color_col}' to check subgroup effects." if color_col else "")
            ),
            "alternatives": (
                "Heatmap shows r but not non-linearity, heteroskedasticity, or clusters. "
                "Line chart assumes time-ordering — wrong here. "
                "Scatter is the only chart showing every individual observation relationship."
            ),
            "question": (
                f"Is {c1}–{c2} truly linear? Are outliers driving the correlation? "
                "Do subgroups behave differently?"
            ),
        })

    # ── 5. Horizontal bar — categorical frequency ─────────────────────────
    # First question for any categorical column: how is frequency distributed?
    for col in cat_cols[:3]:
        vc = df[col].value_counts().head(20).reset_index()
        vc.columns = [col, "count"]
        vc["pct"]  = (vc["count"] / len(df) * 100).round(1)

        fig = px.bar(
            vc, x="count", y=col,
            orientation="h",
            color="count",
            color_continuous_scale=["#1c2333", "#4d8ef5"],
            text=[f"{p}%" for p in vc["pct"]],
        )
        fig.update_traces(textposition="outside")
        # FIX: px.bar already set xaxis/yaxis; pass categoryorder as extra_y
        _layout(
            fig,
            f"Category Frequency — {col}",
            xt="Count",
            extra_y={"categoryorder": "total ascending"},
        )
        fig.update_layout(coloraxis_showscale=False)

        dom_pct = vc["pct"].iloc[-1]
        charts.append({
            "title": f"Frequency — {col}", "fig": fig,
            "why_this": (
                "Horizontal bar because category labels are text — they read left-to-right "
                "on the y-axis without truncation. Sorted ascending so the dominant "
                f"category is at the top (represents {dom_pct}% of records). "
                + ("⚠ Class imbalance this extreme will bias classifiers."
                   if dom_pct > 70 else "")
            ),
            "alternatives": (
                "Pie chart requires estimating angles — humans do this poorly for >3 slices. "
                "Pie hides absolute counts. Vertical bar is hard to read with long labels. "
                "Treemap is for hierarchical data — not needed for a flat category."
            ),
            "question": (
                f"What is the frequency distribution of '{col}'? "
                "Is there class imbalance? Are there rare values that are data errors?"
            ),
        })

    # ── 6. Time series + rolling mean ─────────────────────────────────────
    # If datetime detected: is there a trend? Line chart is the only correct choice.
    if dt_cols and num_cols:
        dt_col  = dt_cols[0]
        val_col = num_cols[0]
        ts = df[[dt_col, val_col]].dropna().sort_values(dt_col)
        if len(ts) >= 5:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts[dt_col], y=ts[val_col],
                mode="lines", line=dict(color="#4d8ef5", width=1.2),
                name="Raw", opacity=0.6,
            ))
            if len(ts) >= 20:
                window = max(5, len(ts) // 20)
                roll   = ts[val_col].rolling(window, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=ts[dt_col], y=roll,
                    mode="lines", line=dict(color="#f0a020", width=2, dash="dot"),
                    name=f"{window}-period rolling mean",
                ))
            _layout(fig, f"Time Series — {val_col} over {dt_col}",
                    xt=dt_col, yt=val_col)

            charts.append({
                "title": f"Time Series — {val_col}", "fig": fig,
                "why_this": (
                    "Line chart because time is continuous and ordered — connecting points "
                    "encodes the direction and rate of change. "
                    "Raw data (blue) shows volatility; rolling mean (amber dashed) reveals trend."
                ),
                "alternatives": (
                    "Bar chart suits discrete period counts (monthly totals) "
                    "but hides intra-period variation on dense series. "
                    "Scatter loses temporal ordering. "
                    "Area chart suits cumulative metrics — line is better for rate-of-change."
                ),
                "question": (
                    f"Is there trend, seasonality, or a structural break in '{val_col}' over time?"
                ),
            })

    # ── 7. Null positional heatmap ────────────────────────────────────────
    # Count per column tells you HOW MUCH is missing. Position tells you WHERE.
    miss_cols = [c for c in df.columns if df[c].isnull().any()]
    if miss_cols:
        sample = df[miss_cols].head(250).isnull().astype(int)
        fig = go.Figure(go.Heatmap(
            z=sample.values,
            x=sample.columns.tolist(),
            y=list(range(len(sample))),
            colorscale=[[0, "#161b27"], [1, "#ef5e5e"]],
            showscale=False,
            hovertemplate="Col: %{x}<br>Row: %{y}<br>Missing: %{z}<extra></extra>",
        ))
        _layout(fig, "Missing Value Map (first 250 rows) — red = missing",
                xt="Column", yt="Row index")

        charts.append({
            "title": "Missing Value Map", "fig": fig,
            "why_this": (
                "Positional heatmap: rows = records, columns = features, red = null. "
                "Vertical streaks → whole column is sparse (consider dropping). "
                "Horizontal streaks → certain rows have many nulls simultaneously "
                "(different source / system failure). "
                "Random scatter → Missing Completely At Random (MCAR) → safe to impute."
            ),
            "alternatives": (
                "Bar chart of null counts per column tells you HOW MUCH is missing — not WHERE. "
                "Pattern reveals MCAR vs systematic missingness, "
                "which determines the correct imputation strategy."
            ),
            "question": (
                "Are missing values random (MCAR) or do they cluster, "
                "suggesting a systematic data collection failure?"
            ),
        })

    return charts


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 01 — DEFINE SCOPE  (Framework 1: 5-Question Framework)
# ─────────────────────────────────────────────────────────────────────────────
if stage == "01 · Define Scope":

    page_header(
        "STAGE 01", "Define Scope",
        "Framework 1: Write the 5 scope questions before touching the data.",
    )

    insight_block(
        "A real analyst defines what they are trying to find out <b>before</b> loading the data. "
        "This prevents post-hoc fishing (p-hacking). "
        "Every chart and conclusion produced later is evaluated against these questions.",
        label="Why This Step Matters (Session 28)", color="amber",
    )

    st.markdown("""
    <div class='dinfo'>
      <div class='lbl'>Framework 1 — The 5-Question Framework (@askdatadawn)</div>
      Answer these before you begin:<br>
      <b>1.</b> What decision does this analysis need to inform?<br>
      <b>2.</b> Who is the audience, and what do they already believe?<br>
      <b>3.</b> What data is available?<br>
      <b>4.</b> What are the likely follow-up questions or objections?<br>
      <b>5.</b> What similar past analyses or data points do we have?
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("**Add a Scope Question**")
        # FIX: label must be non-empty string
        new_q = st.text_area(
            "Scope question input",
            placeholder=(
                "Examples:\n"
                "• What drives customer churn in this dataset?\n"
                "• Is there seasonality in monthly sales?\n"
                "• Which product category has the highest return rate?"
            ),
            height=105,
            label_visibility="collapsed",
        )
        if st.button("Add Question"):
            q = new_q.strip()
            if q and q not in st.session_state["questions"]:
                st.session_state["questions"].append(q)
                st.rerun()
            elif not q:
                st.warning("Enter a question first.")
            else:
                st.info("Already in scope list.")

    with c2:
        st.markdown("**Scope List**")
        if st.session_state["questions"]:
            for i, q in enumerate(st.session_state["questions"], 1):
                st.markdown(
                    f"<div class='qc'><span class='qn'>Q{i}.</span>{q}</div>",
                    unsafe_allow_html=True,
                )
            if st.button("Clear All"):
                st.session_state["questions"] = []
                st.rerun()
        else:
            st.markdown(
                "<p style='color:var(--dimmer);font-size:0.82rem;'>No questions yet.</p>",
                unsafe_allow_html=True,
            )

    if st.session_state["questions"] and st.session_state["groq_ok"]:
        st.markdown("---")
        if st.button("AI Scope Review"):
            qs_text = "\n".join(
                f"{i+1}. {q}" for i, q in enumerate(st.session_state["questions"])
            )
            prompt = (
                f"Analyst scope questions:\n{qs_text}\n\n"
                "For each question reply with:\n"
                "1. What columns/data types are needed to answer it?\n"
                "2. Most appropriate statistical method or chart type.\n"
                "3. Is it too vague? If yes, rewrite precisely.\n"
                "3 sentences per question max. Be direct."
            )
            with st.spinner("Reviewing scope…"):
                try:
                    reply = call_llm(groq_client, model, prompt, max_tokens=800)
                    insight_block(reply, label="DataGPT · Scope Review")
                except ValueError as e:
                    st.error(str(e))
    elif st.session_state["questions"]:
        st.info("Add your Groq API key in the sidebar to get an AI scope review.")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 02 — LOAD & ASSESS  (Session 28: Data Accessing + Assessment)
# ─────────────────────────────────────────────────────────────────────────────
elif stage == "02 · Load & Assess":

    page_header(
        "STAGE 02", "Load & Assess",
        "Session 28: Gather → Summary → Column Descriptions → Assess → Document Issues",
    )

    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            with st.spinner("Parsing file…"):
                raw = load_file(uploaded.read(), uploaded.name)
            st.session_state.update({
                "df_original":  raw,
                "df_clean":     raw.copy(),
                "filename":     uploaded.name,
                "cleaning_log": [],
                "suggestions":  [],
                "assessment":   {},
            })
            st.success(
                f"✓ Loaded **{uploaded.name}** — "
                f"{raw.shape[0]:,} rows × {raw.shape[1]} columns"
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

    df = st.session_state.get("df_original")
    if df is None:
        insight_block("Upload a file above to begin.", label="Waiting", color="amber")
        st.stop()

    # ── Session 28: Step 1+2 — Summary + Column Descriptions ──
    _dataset_info_block(df, st.session_state.get("filename", ""))

    # ── Session 28: Step 3 — Programmatic Assessment ──
    st.markdown("---")
    st.markdown(
        "<div class='dinfo'><div class='lbl'>Step 3 — Programmatic Assessment (Session 28)</div></div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Running audit…"):
        a = assess_data(df)
        st.session_state["assessment"]  = a
        st.session_state["suggestions"] = a["suggestions"]

    n_null_cols  = len(a["missing"])
    n_null_cells = sum(a["missing"].values())
    null_pct     = round(n_null_cells / (df.shape[0] * df.shape[1]) * 100, 1)
    dup          = a["duplicate_count"]

    stat_cards([
        {"val": f"{df.shape[0]:,}",          "lbl": "Rows",          "cls": ""},
        {"val": str(df.shape[1]),             "lbl": "Columns",       "cls": ""},
        {"val": str(n_null_cols),             "lbl": "Cols w/ Nulls", "cls": "bad" if n_null_cols else "ok"},
        {"val": f"{null_pct}%",              "lbl": "Cells Missing", "cls": "bad" if null_pct > 5 else "ok"},
        {"val": str(dup),                    "lbl": "Duplicates",    "cls": "bad" if dup else "ok"},
        {"val": str(len(a["outliers"])),     "lbl": "Outlier Cols",  "cls": "warn" if a["outliers"] else "ok"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    t_head, t_null, t_stats, t_issues = st.tabs([
        "Head / Sample", "Nulls & Outliers", "Numeric Stats", "Issues Found",
    ])

    with t_head:
        st.markdown("*First 10 rows — raw, unmodified:*")
        st.dataframe(df.head(10), width="stretch")
        st.markdown("*Random sample (10 rows):*")
        st.dataframe(df.sample(min(10, len(df)), random_state=42), width="stretch")

    with t_null:
        if a["missing"]:
            md = pd.DataFrame({
                "Column": list(a["missing"].keys()),
                "Count":  list(a["missing"].values()),
                "Pct":    [a["missing_pct"][k] for k in a["missing"]],
            }).sort_values("Pct", ascending=False)
            fig_m = go.Figure(go.Bar(
                x=md["Column"], y=md["Pct"],
                marker=dict(
                    color=md["Pct"],
                    colorscale=[[0, "#3ec98a"], [0.3, "#f0a020"], [1, "#ef5e5e"]],
                    cmin=0, cmax=100,
                ),
                text=[f"{v:.1f}%" for v in md["Pct"]], textposition="auto",
            ))
            _layout(fig_m, "Null % per Column", xt="Column", yt="% Missing")
            pf(fig_m)
            st.dataframe(md, width="stretch")
        else:
            st.markdown(
                "<span class='tag tg'>✓ No missing values detected</span>",
                unsafe_allow_html=True,
            )

        if a["outliers"]:
            st.markdown("**IQR Outlier flags:**")
            for c, n in sorted(a["outliers"].items(), key=lambda x: -x[1]):
                pct = round(n / len(df) * 100, 1)
                st.markdown(
                    f"<span class='tag {'tr' if pct>5 else 'ta'}'>"
                    f"{c}: {n} rows ({pct}%)</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<span class='tag tg'>✓ No IQR outliers detected</span>",
                unsafe_allow_html=True,
            )

        if a["skewness"]:
            st.markdown("**Highly skewed columns (|skew| > 1):**")
            for c, sk in sorted(a["skewness"].items(), key=lambda x: -abs(x[1])):
                direction = "right" if sk > 0 else "left"
                st.markdown(
                    f"<span class='tag {'tr' if abs(sk)>2 else 'ta'}'>"
                    f"{c}: {sk:+.2f} ({direction}-skewed)</span>",
                    unsafe_allow_html=True,
                )

    with t_stats:
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        if nc:
            desc = df[nc].describe().round(3).T
            desc["skew"]     = df[nc].skew().round(3)
            desc["kurtosis"] = df[nc].kurtosis().round(3)
            st.dataframe(desc, width="stretch")
        else:
            st.info("No numeric columns in this dataset.")

    with t_issues:
        # ── Session 28: Document Issues with Quality Dimension labels ──
        sugs = a["suggestions"]
        ICONS = {
            "to_datetime":     ("🔄", "Convert → datetime"),
            "to_numeric":      ("🔄", "Convert → numeric"),
            "to_category":     ("🔄", "Convert → category"),
            "drop_duplicates": ("🗑",  "Remove duplicates"),
            "drop_col":        ("❌", "Drop column"),
            "fill_median":     ("🩹", "Impute with median"),
            "fill_mode":       ("🩹", "Impute with mode"),
            "range_flag":      ("⚠",  "Range violation"),
        }
        if sugs:
            st.markdown(f"**{len(sugs)} issue(s) found — classified by quality dimension:**")
            for i, s in enumerate(sugs, 1):
                icon, lbl = ICONS.get(s["type"], ("?", s["type"]))
                dim = s.get("dim", "")
                with st.expander(f"{icon}  [{i}]  {s['col']}  ·  {lbl}"):
                    if dim:
                        quality_badge(dim, "")
                    st.markdown(
                        f"<div class='ins'><div class='lbl'>Why this needs fixing</div>"
                        f"{s['reason']}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            insight_block(
                "No structural issues detected. Verify the correct file was uploaded "
                "before skipping Stage 03.",
                label="Audit Result", color="green",
            )

    # ── AI audit summary ──
    if st.session_state["groq_ok"]:
        if st.button("Get AI Audit Summary"):
            ns = {}
            for c in df.select_dtypes(include=[np.number]).columns[:6]:
                ns[c] = {
                    "mean": round(df[c].mean(), 3),
                    "std":  round(df[c].std(), 3),
                    "min":  round(df[c].min(), 3),
                    "max":  round(df[c].max(), 3),
                }
            prompt = (
                f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns\n"
                f"Columns: {list(df.columns)}\n"
                f"Dtypes: {a['dtypes']}\n"
                f"Missing %: {a['missing_pct']}\n"
                f"Duplicates: {a['duplicate_count']}\n"
                f"Outlier cols: {a['outliers']}\n"
                f"Skewed cols: {a['skewness']}\n"
                f"Numeric stats: {json.dumps(ns)}\n"
                f"Scope questions: {st.session_state['questions']}\n\n"
                "Reply:\n"
                "1. Dataset quality summary (2 sentences).\n"
                "2. Top 3 issues — cite specific column names.\n"
                "3. Can each scope question be answered with this data? "
                "If not, what exactly is missing?\n"
                "No preamble. Be direct."
            )
            with st.spinner("Analysing…"):
                try:
                    reply = call_llm(groq_client, model, prompt, max_tokens=650)
                    insight_block(reply)
                except ValueError as e:
                    st.error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 03 — CLEAN DATA  (Session 28: Define → Code → Test)
# ─────────────────────────────────────────────────────────────────────────────
elif stage == "03 · Clean Data":

    page_header(
        "STAGE 03", "Clean Data",
        "Session 28: Define → Code → Test. Every step is documented.",
    )

    df_orig = st.session_state.get("df_original")
    if df_orig is None:
        st.warning("Load a dataset in Stage 02 first.")
        st.stop()

    sugs = st.session_state.get("suggestions", [])
    ICONS = {
        "to_datetime":     ("🔄", "Convert → datetime"),
        "to_numeric":      ("🔄", "Convert → numeric"),
        "to_category":     ("🔄", "Convert → category"),
        "drop_duplicates": ("🗑",  "Remove duplicates"),
        "drop_col":        ("❌", "Drop column"),
        "fill_median":     ("🩹", "Impute with median"),
        "fill_mode":       ("🩹", "Impute with mode"),
        "range_flag":      ("⚠",  "Flag — manual review"),
    }

    if not sugs:
        insight_block(
            "Audit found no structural issues. No cleaning needed — proceed to Stage 04.",
            label="Nothing to Clean", color="green",
        )
    else:
        st.markdown(
            f"**{len(sugs)} transformation(s) queued.** "
            "Review each below, then apply all at once."
        )
        for i, s in enumerate(sugs, 1):
            icon, lbl = ICONS.get(s["type"], ("?", s["type"]))
            dim = s.get("dim", "")
            with st.expander(f"{icon}  [{i}]  {s['col']}  ·  {lbl}"):
                if dim:
                    quality_badge(dim, "issue")
                st.markdown(
                    f"<div class='rat'><div class='lbl'>Define — Analyst Reasoning</div>"
                    f"{s['reason']}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    bc1, bc2 = st.columns([1, 5])
    with bc1:
        run_btn = st.button("▶ Apply All", type="primary")
    with bc2:
        rst_btn = st.button("↩ Reset to Original")

    if run_btn:
        if not sugs:
            st.info("Nothing to apply.")
        else:
            with st.spinner("Applying transformations…"):
                try:
                    df_c, log = apply_cleaning(df_orig, sugs)
                    st.session_state["df_clean"]     = df_c
                    st.session_state["cleaning_log"] = log
                    st.success(f"Done — {len(log)} step(s) applied.")
                except Exception as e:
                    st.error(f"Failed: {e}")
                    st.error(traceback.format_exc())

    if rst_btn:
        st.session_state["df_clean"]     = df_orig.copy()
        st.session_state["cleaning_log"] = []
        st.info("Reset to original.")

    # ── Test step (Session 28) ──
    log = st.session_state.get("cleaning_log", [])
    if log:
        st.markdown("#### Cleaning Log — Test (Session 28: verify each step worked)")
        for i, entry in enumerate(log, 1):
            st.markdown(
                f"<div class='sr'>"
                f"<span class='sn'>#{i:02d}</span>"
                f"<span>{entry}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    df_c = st.session_state.get("df_clean")
    if df_c is not None and log:
        st.markdown("---")
        st.markdown("#### Before vs After")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Original**")
            st.markdown(
                f"<span class='tag ta'>"
                f"{df_orig.shape[0]:,} rows · {df_orig.shape[1]} cols · "
                f"{df_orig.isnull().sum().sum():,} nulls</span>",
                unsafe_allow_html=True,
            )
            st.dataframe(df_orig.head(6), width="stretch")
        with cb:
            st.markdown("**After Cleaning**")
            remaining = df_c.isnull().sum().sum()
            st.markdown(
                f"<span class='tag tg'>"
                f"{df_c.shape[0]:,} rows · {df_c.shape[1]} cols · "
                f"{remaining:,} nulls</span>",
                unsafe_allow_html=True,
            )
            st.dataframe(df_c.head(6), width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 04 — EXPLORE (EDA)  (Framework 2: Execution Chain)
# ─────────────────────────────────────────────────────────────────────────────
elif stage == "04 · Explore (EDA)":

    page_header(
        "STAGE 04", "Exploratory Data Analysis",
        "Framework 2: Every chart is chosen for a specific analytical reason.",
    )

    df_eda = st.session_state.get("df_clean")
    if df_eda is None:
        st.warning("Load a dataset in Stage 02 first.")
        st.stop()

    with st.spinner("Building charts…"):
        charts = build_eda_charts(df_eda)

    if not charts:
        st.error(
            "No charts could be generated — dataset may have no numeric or "
            "categorical columns. Check Stage 02 for type conversion suggestions."
        )
        st.stop()

    # Render each chart immediately followed by its rationale
    for ch in charts:
        pf(ch["fig"])
        chart_rationale(
            why_this=ch["why_this"],
            alternatives=ch["alternatives"],
            question=ch["question"],
        )
        st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Descriptive Statistics Table (df.describe())"):
        nd = df_eda.select_dtypes(include=[np.number])
        if not nd.empty:
            d = nd.describe().round(3).T
            d["skew"]     = nd.skew().round(3)
            d["kurtosis"] = nd.kurtosis().round(3)
            st.dataframe(d, width="stretch")

    st.markdown("---")
    st.markdown("#### Ask DataGPT About the Data")

    if not st.session_state["groq_ok"]:
        st.info("Add your Groq API key in the sidebar to enable the analyst chat.")
    else:
        for msg in st.session_state["chat_history"]:
            cls = "cu" if msg["role"] == "user" else "ca"
            st.markdown(
                f"<div class='{cls}'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )

        # FIX: label must be non-empty
        user_q = st.text_input(
            "EDA question",
            placeholder="e.g. Should I log-transform revenue before regression?",
            label_visibility="collapsed",
        )
        qa_col, clr_col = st.columns([1, 5])
        with qa_col:
            ask_btn = st.button("Ask")
        with clr_col:
            if st.button("Clear Chat"):
                st.session_state["chat_history"] = []
                st.rerun()

        if ask_btn:
            if not user_q.strip():
                st.warning("Enter a question first.")
            else:
                num_stats = df_eda.describe(include=[np.number]).round(3).to_string()[:2000]
                cat_info  = {
                    c: df_eda[c].value_counts().head(5).to_dict()
                    for c in df_eda.select_dtypes(
                        include=["object", "category"]
                    ).columns[:4]
                }
                ctx = (
                    f"Dataset: {df_eda.shape[0]} rows × {df_eda.shape[1]} cols\n"
                    f"Columns & dtypes: {df_eda.dtypes.to_dict()}\n"
                    f"Numeric stats:\n{num_stats}\n"
                    f"Categorical top values: {json.dumps(cat_info, default=str)}\n"
                    f"Cleaning done: {st.session_state['cleaning_log']}\n\n"
                    f"Question: {user_q}"
                )
                with st.spinner("Thinking…"):
                    try:
                        reply = call_llm(groq_client, model, ctx, max_tokens=700)
                        st.session_state["chat_history"] += [
                            {"role": "user",      "content": user_q},
                            {"role": "assistant", "content": reply},
                        ]
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 05 — CONCLUSIONS  (Framework 3: Stress-Test Approach)
# ─────────────────────────────────────────────────────────────────────────────
elif stage == "05 · Conclusions":

    page_header(
        "STAGE 05", "Conclusions",
        "Framework 3: Answer scope questions with evidence, then stress-test the findings.",
    )

    df_c = st.session_state.get("df_clean")
    if df_c is None:
        st.warning("Load a dataset in Stage 02 first.")
        st.stop()
    if not st.session_state["groq_ok"]:
        st.info("Add your Groq API key to generate conclusions.")
        st.stop()

    qs  = st.session_state["questions"]
    log = st.session_state["cleaning_log"]
    a   = st.session_state.get("assessment", {})

    if qs:
        with st.expander("Scope Questions (Stage 01)"):
            for i, q in enumerate(qs, 1):
                st.markdown(f"**Q{i}.** {q}")
    if log:
        with st.expander("Cleaning Log (Stage 03)"):
            for s in log:
                st.markdown(f"- {s}")

    st.markdown("---")

    # ── Conclusions ──
    if st.button("Generate Conclusions", type="primary"):
        nc = df_c.select_dtypes(include=[np.number]).columns[:8].tolist()
        cc = df_c.select_dtypes(include=["object", "category"]).columns[:5].tolist()

        ns = {}
        for c in nc:
            s = df_c[c].dropna()
            ns[c] = {
                "mean": round(float(s.mean()), 3),
                "median": round(float(s.median()), 3),
                "std":  round(float(s.std()), 3),
                "min":  round(float(s.min()), 3),
                "max":  round(float(s.max()), 3),
                "skew": round(float(s.skew()), 3),
                "nulls": int(df_c[c].isnull().sum()),
            }
        cs = {c: df_c[c].value_counts().head(5).to_dict() for c in cc}

        corr_sum = ""
        if len(nc) >= 2:
            corr = df_c[nc].corr()
            top  = (
                corr.abs()
                .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack().nlargest(5)
            )
            corr_sum = str(top.round(3).to_dict())

        prompt = (
            f"Dataset: {df_c.shape[0]:,} rows × {df_c.shape[1]} columns\n"
            f"Columns: {list(df_c.columns)}\n"
            f"Numeric stats: {json.dumps(ns)}\n"
            f"Categorical top values: {json.dumps(cs, default=str)}\n"
            f"Top correlations: {corr_sum}\n"
            f"Outlier cols: {a.get('outliers', {})}\n"
            f"Skewed cols: {a.get('skewness', {})}\n"
            f"Cleaning done: {log}\n\n"
            "Scope questions:\n"
            + (
                "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs))
                if qs else "(none — general investigation)"
            ) +
            "\n\nStructured reply:\n"
            "1. Data quality verdict (2 sentences).\n"
            "2. Answer each scope question with specific numbers. "
            "If unanswerable, say exactly why.\n"
            "3. Top 3 unexpected findings — cite numbers.\n"
            "4. What this data CANNOT tell us.\n"
            "5. Recommended next steps.\n"
            "Be specific. No hedging."
        )
        with st.spinner("Generating conclusions…"):
            try:
                conc = call_llm(groq_client, model, prompt, max_tokens=1600)
                st.session_state["conclusions"] = conc
            except ValueError as e:
                st.error(str(e))

    if st.session_state["conclusions"]:
        insight_block(st.session_state["conclusions"], label="DataGPT · Conclusions")

        st.markdown("---")
        # ── Framework 3: Stress-Test ──
        st.markdown(
            "<div class='dinfo'><div class='lbl'>Framework 3 — Stress-Test (@askdatadawn)</div>"
            "Ask a skeptical senior data scientist to critique the methodology before presenting.</div>",
            unsafe_allow_html=True,
        )
        if st.button("Run Stress-Test on Conclusions"):
            stress_prompt = (
                "Act as a skeptical senior data scientist reviewing this analysis.\n\n"
                f"Methodology: Standard EDA on a tabular dataset. "
                f"Cleaning steps: {log}\n"
                f"Conclusions:\n{st.session_state['conclusions'][:1000]}\n\n"
                "Your job:\n"
                "1. Find the 3 weakest points in this methodology.\n"
                "2. Identify alternative explanations not ruled out.\n"
                "3. What question will the sharpest person in the room ask — "
                "and do we have a good answer?\n"
                "4. Rate confidence 1–5 and what would move it higher.\n"
                "Be direct. Be critical."
            )
            with st.spinner("Stress-testing…"):
                try:
                    st_result = call_llm(
                        groq_client, model, stress_prompt, max_tokens=900
                    )
                    st.session_state["stress_test"] = st_result
                except ValueError as e:
                    st.error(str(e))

        if st.session_state["stress_test"]:
            insight_block(
                st.session_state["stress_test"],
                label="DataGPT · Stress-Test (Framework 3)",
                color="amber",
            )

        # Supporting stats
        nd = df_c.select_dtypes(include=[np.number])
        if not nd.empty:
            st.markdown("---")
            st.markdown("#### Supporting Numeric Summary")
            cols_s = nd.columns[:4].tolist()
            stat_cards([
                {
                    "val": f"{nd[c].mean():.3g}",
                    "lbl": f"{c[:18]} mean",
                    "cls": "warn" if c in a.get("outliers", {}) else "",
                }
                for c in cols_s
            ])


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 06 — REPORT
# ─────────────────────────────────────────────────────────────────────────────
elif stage == "06 · Report":

    page_header("STAGE 06", "Report", "Complete, downloadable analysis record.")

    df_f = st.session_state.get("df_clean")
    if df_f is None:
        st.warning("Load a dataset in Stage 02 first.")
        st.stop()

    log = st.session_state.get("cleaning_log", [])
    qs  = st.session_state.get("questions", [])
    a   = st.session_state.get("assessment", {})

    stat_cards([
        {"val": f"{df_f.shape[0]:,}",            "lbl": "Final Rows",      "cls": ""},
        {"val": str(df_f.shape[1]),               "lbl": "Final Columns",   "cls": ""},
        {"val": str(len(log)),                    "lbl": "Cleaning Steps",  "cls": ""},
        {"val": str(len(qs)),                     "lbl": "Scope Questions", "cls": ""},
        {
            "val": str(df_f.isnull().sum().sum()),
            "lbl": "Remaining Nulls",
            "cls": "bad" if df_f.isnull().sum().sum() else "ok",
        },
    ])

    st.markdown("---")

    if st.session_state["groq_ok"]:
        if st.button("Generate Report", type="primary"):
            nc = df_f.select_dtypes(include=[np.number]).columns.tolist()
            cc = df_f.select_dtypes(include=["object", "category"]).columns.tolist()
            dc = df_f.select_dtypes(include=["datetime64"]).columns.tolist()

            prompt = (
                "Write a professional data analysis report in Markdown.\n\n"
                f"Dataset: {df_f.shape[0]:,} rows × {df_f.shape[1]} cols\n"
                f"Numeric cols: {nc}\nCategorical cols: {cc}\nDatetime cols: {dc}\n"
                f"Cleaning done: {log}\nScope questions: {qs}\n"
                f"Key conclusions: {st.session_state['conclusions'][:1200]}\n"
                f"Stress-test findings: {st.session_state['stress_test'][:600]}\n"
                f"Outliers: {a.get('outliers',{})}\n"
                f"Missing resolved: {a.get('missing_pct',{})}\n\n"
                "Use exactly these section headings:\n"
                "# Data Analysis Report\n"
                "## 1. Executive Summary\n"
                "## 2. Dataset Overview\n"
                "## 3. Data Quality Findings\n"
                "## 4. Cleaning Actions Taken\n"
                "## 5. Key Analytical Findings\n"
                "## 6. Limitations\n"
                "## 7. Recommended Next Steps\n"
                "## Appendix: Full Cleaning Log\n\n"
                "Write for a business stakeholder — "
                "plain English, specific numbers, honest limitations."
            )
            with st.spinner("Drafting report…"):
                try:
                    ts     = datetime.now().strftime("%Y-%m-%d %H:%M")
                    report = call_llm(
                        groq_client, model, prompt, max_tokens=2000, temperature=0.3
                    )
                    report += f"\n\n---\n*Generated by DataGPT· {ts}*"
                    st.session_state["case_report"] = report
                except ValueError as e:
                    st.error(str(e))
    else:
        st.info("Add your Groq API key to generate the AI report.")

    if st.session_state["case_report"]:
        st.markdown("---")
        st.markdown("#### Analysis Report")
        st.markdown(
            f"<div class='ra'>{st.session_state['case_report']}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Downloads")
    d1, d2, d3 = st.columns(3)

    with d1:
        st.download_button(
            "⬇ Cleaned Dataset (.csv)",
            data=df_f.to_csv(index=False).encode("utf-8"),
            file_name="datagpt_cleaned.csv",
            mime="text/csv",
        )
    with d2:
        if st.session_state["case_report"]:
            st.download_button(
                "⬇ Analysis Report (.md)",
                data=st.session_state["case_report"].encode("utf-8"),
                file_name="datagpt_report.md",
                mime="text/markdown",
            )
        else:
            st.markdown(
                "<p style='color:var(--dimmer);font-size:0.82rem;'>"
                "Generate the report first.</p>",
                unsafe_allow_html=True,
            )
    with d3:
        if log:
            log_txt = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(log))
            st.download_button(
                "⬇ Cleaning Log (.txt)",
                data=log_txt.encode("utf-8"),
                file_name="datagpt_cleaning_log.txt",
                mime="text/plain",
            )