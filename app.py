import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gensim import corpora
from gensim.models import LdaModel
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import plotly.colors as pc

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* -- Reset & base -- */
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
    section[data-testid="stSidebar"] { background: #0f1117; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] input {color: #1e293b !important;}
    section[data-testid="stSidebar"] .stDateInput * {color: #1e293b !important;}
    section[data-testid="stSidebar"] .stDateInput input {color: #1e293b !important; background: #ffffff !important;}
    section[data-testid="stSidebar"] .stDateInput div[data-baseweb="input"] {background: #ffffff !important;}
    section[data-testid="stSidebar"] .stRadio label { padding: 0.35rem 0.75rem; border-radius: 6px; transition: background 0.15s; }
    section[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.07); }
    section[data-testid="stSidebar"] .stButton button * {
        background: #ffffff;
        color: #1e293b !important;
        font-weight: 600;
        transition: background 0.05s, color 0.05s !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: #6366f1 !important;
        border-color: #6366f1 !important;
    }

    section[data-testid="stSidebar"] .stButton button:hover * {
        color: #ffffff !important;
        background-color: transparent !important;
    }

    /* -- Page header -- */
    .page-header {
        display: flex; align-items: flex-start; gap: 1rem;
        padding: 1.4rem 1.8rem; margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%);
        border-radius: 14px; border-left: 4px solid #6366f1;
    }
    .page-title { font-size: 1.65rem; font-weight: 800; color: #f1f5f9; margin: 0; letter-spacing: -0.02em; }
    .page-subtitle { color: #94a3b8; font-size: 0.88rem; margin-top: 0.3rem; }

    /* -- Section headers -- */
    .section-header {
        font-size: 0.72rem; font-weight: 700; color: #6366f1;
        text-transform: uppercase; letter-spacing: 0.12em;
        padding: 0.5rem 0 0.5rem 0; margin: 1.5rem 0 0.8rem 0;
        border-bottom: 1px solid #e2e8f0;
    }

    /* -- KPI cards -- */
    .kpi-grid { display: flex; gap: 0.8rem; margin-bottom: 1.2rem; }
    .kpi-card {
        flex: 1; background: #ffffff; border-radius: 12px;
        padding: 1.1rem 1.3rem; text-align: left;
        border: 1px solid #e8eaf0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
        position: relative; overflow: hidden;
    }
    .kpi-card::before {
        content: ''; position: absolute; top: 0; left: 0;
        width: 4px; height: 100%; background: var(--accent, #6366f1);
    }
    .kpi-card.pos::before { background: #10b981; }
    .kpi-card.neg::before { background: #ef4444; }
    .kpi-card.neu::before { background: #f59e0b; }
    .kpi-card.blue::before { background: #3b82f6; }
    .kpi-card.purple::before { background: #6366f1; }
    .kpi-value { font-size: 1.9rem; font-weight: 800; color: #1e293b; line-height: 1.1; letter-spacing: -0.03em; }
    .kpi-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.25rem; font-weight: 600; }
    .kpi-delta { font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem; }

    /* -- Insight boxes -- */
    .insight-pos { background: #f0fdf4; border-left: 4px solid #10b981; border-radius: 8px; padding: 0.85rem 1.2rem; margin: 0.5rem 0; }
    .insight-neg { background: #fef2f2; border-left: 4px solid #ef4444; border-radius: 8px; padding: 0.85rem 1.2rem; margin: 0.5rem 0; }
    .insight-neu { background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 0.85rem 1.2rem; margin: 0.5rem 0; }
    .insight-pos p, .insight-neg p, .insight-neu p { margin: 0; font-size: 0.88rem; color: #1e293b; }

    /* -- Recommendation cards -- */
    .rec-card {
        background: white; border-radius: 12px; padding: 1.1rem 1.4rem;
        border: 1px solid #e8eaf0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        margin-bottom: 0.75rem; display: flex; gap: 1rem; align-items: flex-start;
    }
    .rec-priority { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; padding: 0.2rem 0.55rem; border-radius: 4px; white-space: nowrap; margin-top: 0.15rem; }
    .rec-priority.high { background: #fee2e2; color: #dc2626; }
    .rec-priority.medium { background: #fef9c3; color: #a16207; }
    .rec-priority.low { background: #dcfce7; color: #16a34a; }
    .rec-content.rec-title { font-weight: 700; color: #1e293b; font-size: 0.95rem; margin-bottom: 0.3rem; }
    .rec-content.rec-body { color: #475569; font-size: 0.86rem; line-height: 1.5; }
    .rec-content.rec-evidence { color: #94a3b8; font-size: 0.8rem; margin-top: 0.4rem; font-style: italic; }

    /* -- Badges -- */
    .badge { display: inline-block; padding: 0.2rem 0.65rem; border-radius: 20px; font-size: 0.73rem; font-weight: 600; margin-right: 0.35rem; }
    .badge-bert { background: #ede9fe; color: #5b21b6; }
    .badge-lda { background: #d1fae5; color: #065f46; }
    .badge-absa { background: #dbeafe; color: #1e40af; }

    /* -- Review cards -- */
    .review-card {
        background: #f8fafc; border-radius: 10px; padding: 0.9rem 1.1rem;
        border: 1px solid #e8eaf0; margin: 0.45rem 0; font-size: 0.875rem;
        line-height: 1.6; color: #334155;
    }
    .review-card .stars { color: #f59e0b; font-weight: 700; margin-bottom: 0.3rem; font-size: 0.8rem; letter-spacing: 0.04em; }

    /* -- Metric pill -- */
    .metric-pill {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: #f1f5f9; border-radius: 20px; padding: 0.25rem 0.75rem;
        font-size: 0.8rem; font-weight: 600; color: #475569;
    }
</style>
"""

st.set_page_config(
    page_title="NLP Business Intelligence Dashboard",
    page_icon="📊", layout="wide", initial_sidebar_state="expanded"
)
st.markdown(CSS, unsafe_allow_html=True)


# -- Data loading --
@st.cache_data
def load_data():
    df = pd.read_csv("yelp_reviews_cleaned.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_merged():
    try:
        df = pd.read_csv("yelp_reviews_merged.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except:
        return None

@st.cache_data
def load_absa():
    try:
        return pd.read_pickle("absa_results.pkl")
    except:
        return None

@st.cache_resource
def load_lda():
    try:
        lda = LdaModel.load("lda_model_final")
        dic = corpora.Dictionary.load("lda_dictionary_final.gensim")
        return lda, dic
    except:
        return None, None

# -- 3-file results loader (results_summary.csv, absa_results.csv, topic_results.csv) --
@st.cache_data
def load_results_summary():
    try:
        return pd.read_csv("results_summary.csv")
    except:
        return None

@st.cache_data
def load_absa_summary():
    try:
        return pd.read_csv("absa_results.csv")
    except:
        return None

@st.cache_data
def load_topic_results():
    try:
        return pd.read_csv("topic_results.csv")
    except:
        return None

@st.cache_data
def load_absa_eval_ci():
    try:
        return pd.read_csv("absa_eval_ci.csv")
    except:
        return None

@st.cache_data
def load_bert_error_analysis():
    """bert_error_analysis.csv - columns: text, true, bert_pred, vader_pred, lr_pred
    Export from the notebook error-analysis cell by adding:
        error_df.to_csv('bert_error_analysis.csv', index=False)
    """
    try:
        df = pd.read_csv("bert_error_analysis.csv")
        if "text_length" not in df.columns and "text" in df.columns:
            df["text_length"] = df["text"].str.len()
        return df
    except:
        return None

@st.cache_data
def load_absa_flagged():
    """absa_flagged_cases_for_manual_coding.csv - already exported by the notebook."""
    try:
        df = pd.read_csv("absa_flagged_cases_for_manual_coding.csv")
        df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
        return df
    except:
        return None

def get_kpi(results_df, model, metric, default=None):
    """Helper to look up a single value from results_summary.csv"""
    if results_df is None:
        return default
    row = results_df[(results_df["Model"] == model) & (results_df["Metric"] == metric)]
    if row.empty:
        return default
    return row["Value"].iloc[0]

# -- N/A-safe formatting helpers (no hardcoded fallback numbers) --
def fmt_pct(val, decimals=1):
    """Format a 0-1 ratio as a percentage string, or 'N/A' if missing."""
    if val is None:
        return "N/A"
    return f"{float(val)*100:.{decimals}f}%"

def fmt_pct_raw(val, decimals=1):
    """Format a value already in 0-100 scale as a percentage string, or 'N/A'."""
    if val is None:
        return "N/A"
    return f"{float(val):.{decimals}f}%"

def fmt_score(val, decimals=4):
    """Format a numeric score (e.g. F1, coherence), or 'N/A' if missing."""
    if val is None:
        return "N/A"
    return f"{float(val):.{decimals}f}"

def fmt_int(val):
    """Format an integer-like value, or 'N/A' if missing."""
    if val is None:
        return "N/A"
    return f"{int(val)}"

df_full = load_data()
df_merged_st = load_merged()
absa_df = load_absa()
lda_model, dictionary = load_lda()
results_summary_df = load_results_summary()
absa_summary_df = load_absa_summary()
topic_results_df = load_topic_results()
bert_error_df = load_bert_error_analysis()
absa_flagged_df = load_absa_flagged()
absa_eval_ci_df = load_absa_eval_ci()

ASPECTS = ["food","service","ambience","price","location"]
TOPIC_LABELS = {
    0:"General Experience",
    1:"Fast Food & American",
    2:"Hotels",
    3:"Hair & Auto Services",
    4:"Cafes & Desserts",
    5:"Fine Dining",
    6:"Complaints",
    7:"Bars & Nightlife",
    8:"Pet & Retail",
    9:"Asian Cuisine",
    10:"Breakfast & Brunch"
}
C = {
    "pos": "#10b981",
    "neg": "#ef4444",
    "neu": "#f59e0b",
    "blue": "#3b82f6",
    "purple": "#6366f1",
    "red": "#e94560",
    "slate": "#64748b",
    "indigo": "#4f46e5",
}

THEME = dict(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", size=12, color="#334155"),
    margin=dict(t=48, b=36, l=36, r=24),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter"),
)

def apply_clean_axes(fig, x_grid=False, y_grid=True):
    """Apply clean, minimal axis styling to any figure."""
    fig.update_xaxes(showgrid=x_grid, gridcolor="#f1f5f9", zeroline=False,
                     linecolor="#e2e8f0", tickfont=dict(size=11))
    fig.update_yaxes(showgrid=y_grid, gridcolor="#f1f5f9", zeroline=False,
                     linecolor="#e2e8f0", tickfont=dict(size=11))
    return fig

def page_header(title, subtitle="", icon=""):
    st.markdown(f'''<div class="page-header">
        <div>
            <div class="page-title">{icon} {title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
    </div>''', unsafe_allow_html=True)

def section(label):
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)

def kpi_card(val, label, delta="", accent="purple"):
    return f'''<div class="kpi-card {accent}">
        <div class="kpi-value">{val}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-delta">{delta}</div>
    </div>'''

def ibox(text, kind="neu"):
    cls = {"pos":"insight-pos","neg":"insight-neg","neu":"insight-neu"}.get(kind,"insight-neu")
    icon = {"pos":"✅","neg":"⚠️","neu":"💡"}.get(kind,"💡")
    st.markdown(f'<div class="{cls}"><p><b>{icon}</b> {text}</p></div>', unsafe_allow_html=True)

def fmt_p(p):
    return "< 0.0001" if p < 0.0001 else f"= {p:.4f}"

def fmt_peq(p):
    return "< 0.0001" if p < 0.0001 else f"= {p:.4f}"

def fmt_pval(p):
    return "N/A" if p is None else f"{float(p):.3e}"

# -- Sidebar --
with st.sidebar:
    st.markdown("### 📊 NLP Dashboard")
    st.markdown('<span style="font-size:0.8rem;color:#94a3b8;">Yelp Review Intelligence - MSc DS</span>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("**Navigate**", [
        "🏠 Executive Dashboard",
        "🏢 Category Analysis",
        "📈 Q1 Overall Sentiment",
        "🔍 Q2 Aspect Analysis",
        "🗂️ Q3 Topic Modelling",
        "📅 Q4 Sentiment Trends",
        "🔑 Q5 Review Drivers",
        "💡 Recommendations",
        "🤖 Technical Appendix",
    ], label_visibility="visible")
    st.divider()
    st.markdown("**⚙️ Filters**")
    min_d = df_full["date"].min().date()
    max_d = df_full["date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        key="date_range")
    star_filter = st.multiselect(
        "Star Rating",
        [1.0,2.0,3.0,4.0,5.0],
        default=[1.0,2.0,3.0,4.0,5.0],
        format_func=lambda x: f"{'⭐'*int(x)} {int(x)}★",
        key="star_filter")
    sent_filter = st.multiselect(
        "Sentiment",
        ["positive","negative","neutral"],
        default=["positive","negative","neutral"],
        format_func=lambda x: {"positive":"✅ Positive","negative":"⚠️ Negative","neutral":"➖ Neutral"}[x],
        key="sent_filter"
        )
    st.divider()
    st.caption("50,000 Yelp reviews - BERT - DeBERTa - LDA")

    def clear_filters():
      st.session_state.date_range = (min_d, max_d)
      st.session_state.star_filter = [1.0, 2.0, 3.0, 4.0, 5.0]
      st.session_state.sent_filter = ["positive", "negative", "neutral"]

    st.button(
      "🔄 Clear Filters",
      use_container_width=True,
      on_click=clear_filters
      )

df = df_full.copy()
if len(date_range) == 2:
    df = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]
if star_filter:
    df = df[df["stars"].isin(star_filter)]
if sent_filter:
    df = df[df["sentiment"].isin(sent_filter)]

# ==========================================================
# HOME
# ==========================================================
if page == "🏠 Executive Dashboard":
    page_header("NLP Business Intelligence Dashboard",
                "Question-Driven Customer Review Analysis - Yelp Open Dataset - MSc Data Science", "📊")

    sc = df_full["sentiment"].value_counts()
    total = len(df)
    pos_pct = (df["sentiment"]=="positive").mean()*100
    neg_pct = (df["sentiment"]=="negative").mean()*100
    avg_star = df["stars"].mean()
    n_topics = get_kpi(results_summary_df, "LDA", "KPI_N_Topics", None)
    bert_acc_home = get_kpi(results_summary_df, "BERT", "Accuracy", None)

    section("Key Performance Indicators")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col,val,lbl,delta,acc in [
        (c1,f"{total:,}","Total Reviews","50k sample","blue"),
        (c2,f"{avg_star:.2f} ★","Avg Rating","Yelp dataset","neu"),
        (c3,f"{pos_pct:.1f}%","Positive","Sentiment","pos"),
        (c4,f"{neg_pct:.1f}%","Negative","Sentiment","neg"),
        (c5,fmt_pct(bert_acc_home),"BERT Accuracy","Best model","purple"),
        (c6,fmt_int(n_topics),"LDA Topics","Discovered","blue"),
    ]:
        with col:
            st.markdown(kpi_card(val, lbl, delta, acc), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Sentiment Overview")
    col1,col2,col3 = st.columns(3)
    with col1:
        labels_ordered = ["Positive","Negative","Neutral"]
        vals_ordered = [sc.get(l.lower(),0) for l in labels_ordered]
        fig = go.Figure(go.Pie(
            labels=labels_ordered, values=vals_ordered, hole=0.6,
            marker_colors=[C["pos"],C["neg"],C["neu"]],
            marker_line=dict(color="white", width=3),
            textinfo="percent", textfont_size=12,
        ))
        fig.update_layout(**THEME, title="Sentiment Split",
                          legend=dict(orientation="h", yanchor="bottom", y=-0.2))
        fig.add_annotation(text=f"<b>{total:,}</b><br>reviews",
                           x=0.5, y=0.5, showarrow=False, font_size=13, font_color="#334155")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=labels_ordered, y=vals_ordered,
            marker=dict(
                color=[C["pos"],C["neg"],C["neu"]],
                line=dict(color="white", width=1.5)
            ),
            text=[f"{v:,}<br><span style='font-size:10px'>{v/total*100:.1f}%</span>" for v in vals_ordered],
            textposition="outside",
        ))
        fig.update_layout(**THEME, title="Review Counts", yaxis_title="Reviews",
                          showlegend=False, yaxis=dict(range=[0, max(vals_ordered)*1.25]))
        apply_clean_axes(fig, x_grid=False, y_grid=True)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        star_c = df["stars"].value_counts().sort_index()
        star_colors = [C["neg"],"#f87171",C["neu"],"#6ee7b7",C["pos"]]
        fig = go.Figure(go.Bar(
            x=[f"{int(s)}★" for s in star_c.index], y=star_c.values,
            marker=dict(color=star_colors, line=dict(color="white",width=1.5)),
            text=[f"{v:,}" for v in star_c.values], textposition="outside",
        ))
        fig.update_layout(**THEME, title="Star Distribution", yaxis_title="Reviews", showlegend=False)
        apply_clean_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

    section("Review Volume Over Time")
    if len(df) > 0:
        df["month"] = df["date"].dt.to_period("M").astype(str)
        monthly_trend = df.groupby("month").agg(Reviews=("stars","count"), Avg_Stars=("stars","mean")).reset_index()
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Bar(
                x=monthly_trend["month"], y=monthly_trend["Reviews"],
                marker=dict(color=C["blue"], opacity=0.85, line=dict(color="white",width=0.5)),
                name="Reviews",
            ))
            fig.update_layout(**THEME, title="Review Volume Trend", showlegend=False,
                              xaxis=dict(tickangle=45, showgrid=False), yaxis=dict(title="Reviews"))
            apply_clean_axes(fig)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure(go.Scatter(
                x=monthly_trend["month"], y=monthly_trend["Avg_Stars"],
                mode="lines+markers",
                line=dict(width=2.5, color=C["pos"]),
                marker=dict(size=5, color=C["pos"]),
                fill="tozeroy", fillcolor="rgba(16,185,129,0.08)",
            ))
            fig.update_layout(**THEME, title="Average Rating Trend", showlegend=False,
                              xaxis=dict(tickangle=45, showgrid=False),
                              yaxis=dict(title="Avg Stars", range=[1,5]))
            apply_clean_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

    section("Executive Summary")
    bert_neu_home = get_kpi(results_summary_df, "BERT", "F1_Neutral", None)
    ibox(f"Positive reviews dominate at <b>{sc.get('positive',0)/total*100:.1f}%</b>, consistent with positivity bias in online review platforms.", "pos")
    ibox(f"Neutral reviews account for only <b>{sc.get('neutral',0)/total*100:.1f}%</b> - the hardest class to classify (BERT F1={fmt_score(bert_neu_home,2)}).", "neu")


# ==========================================================
# CATEGORY ANALYSIS
# ==========================================================
elif page == "🏢 Category Analysis":
    page_header("How does sentiment differ across business categories?", "Business category comparison using Yelp metadata")
    if df_merged_st is None:
        st.warning("yelp_reviews_merged.csv not found. Run Business Category Analysis cells in notebook first.")
    else:
        cat_list = ["Restaurants","Hotels","Hair Salons","Auto Repair","Coffee","Bars","Shopping"]
        cat_res = []
        for cat in cat_list:
            mask = df_merged_st["categories"].str.contains(cat, case=False, na=False)
            sub = df_merged_st[mask]
            if len(sub) < 50: continue
            cat_res.append({"Category":cat,"Reviews":len(sub),
                "Positive %":round(float((sub["sentiment"]=="positive").mean())*100,1),
                "Negative %":round(float((sub["sentiment"]=="negative").mean())*100,1),
                "Avg Stars":round(float(sub["stars"].mean()),2)})
        cat_df = pd.DataFrame(cat_res).sort_values("Positive %", ascending=False)

        c1,c2,c3 = st.columns(3)
        for col,val,lbl,delta in [
            (c1,str(len(cat_df)),"Categories Analysed",""),
            (c2,cat_df.iloc[0]["Category"],"Top Category",f"{cat_df.iloc[0]['Positive %']:.1f}% positive"),
            (c3,cat_df.loc[cat_df["Reviews"].idxmax(),"Category"],"Most Reviewed",f"{cat_df['Reviews'].max():,} reviews"),

        ]:
            with col:
                st.markdown(kpi_card(val, lbl, delta), unsafe_allow_html=True)

        section("Category Sentiment Analysis")

        # Chart 1: Positive vs Negative grouped
        fig = go.Figure()
        fig.add_trace(
            go.Bar(name="Positive %",
                   y=cat_df["Category"],
                   x=cat_df["Positive %"],
                   orientation="h",
                   marker_color=C["pos"],
                   text=[f"{v:.1f}%" for v in cat_df["Positive %"]],
                   textposition="outside")
            )
        fig.add_trace(
            go.Bar(name="Negative %",
                   y=cat_df["Category"],
                   x=cat_df["Negative %"],
                   orientation="h",
                   marker_color=C["neg"],
                   text=[f"{v:.1f}%" for v in cat_df["Negative %"]],
                   textposition="outside")
            )
        fig.update_layout(
            **THEME,
            title="Positive vs Negative Sentiment by Category",
            barmode="group",
            xaxis_title="%",
            xaxis=dict(range=[0, cat_df["Positive %"].max()*1.3]),
            yaxis=dict(autorange="reversed"), height=380)
        st.plotly_chart(
            fig,
            use_container_width=True
            )
        # Chart 2 & 3: Avg Stars + Volume
        col1, col2 = st.columns(2)
        with col1:
            srt = cat_df.sort_values("Avg Stars", ascending=False)
            fig = go.Figure(
                go.Bar(
                    y=srt["Category"],
                    x=srt["Avg Stars"],
                    orientation="h",
                    marker_color=C["blue"],
                    text=[f"{v:.2f}" for v in srt["Avg Stars"]],
                    textposition="outside")
                )
            fig.update_layout(
                **THEME,
                title="Average Star Rating by Category",
                xaxis_title="Avg Stars",
                xaxis=dict(range=[0, 5.8]),
                yaxis=dict(autorange="reversed"), height=320
                )
            st.plotly_chart(
                fig,
                use_container_width=True
                )
        with col2:
            srt2 = cat_df.sort_values("Reviews", ascending=False)
            fig = go.Figure(
                go.Bar(
                    y=srt2["Category"],
                    x=srt2["Reviews"],
                    orientation="h",
                    marker_color=C["purple"],
                    text=[f"{v:,}" for v in srt2["Reviews"]],
                    textposition="outside")
                )
            fig.update_layout(
                **THEME,
                title="Review Volume by Category",
                xaxis_title="Reviews",
                yaxis=dict(autorange="reversed"),
                height=320
                )
            st.plotly_chart(
                fig,
                use_container_width=True
                )

        section("Category Summary Table")
        st.dataframe(cat_df.reset_index(drop=True), use_container_width=True, hide_index=True)
        top_c = cat_df.iloc[0]; bot_c = cat_df.iloc[-1]

        section("Executive Summary")
        ibox(f"<b>{top_c['Category']}</b> leads with {top_c['Positive %']:.1f}% positive sentiment. Avg rating: {top_c['Avg Stars']:.2f} stars.","pos")
        ibox(f"<b>{bot_c['Category']}</b> has the lowest positive rate ({bot_c['Positive %']:.1f}%).","neg")
        ibox(f"Positive sentiment ranges from {cat_df['Positive %'].min():.1f}% to {cat_df['Positive %'].max():.1f}% across categories. This is descriptive: no significance test was run and categories overlap.","neu")

# ==========================================================
# Q1
# ==========================================================
elif page == "📈 Q1 Overall Sentiment":
    page_header('Q1: What is the overall sentiment of Yelp reviews?')
    sc = df["sentiment"].value_counts()
    total = len(df)

    c1,c2,c3 = st.columns(3)
    for col,val,lbl,delta in [
        (c1,f"{sc.get('positive',0)/total*100:.1f}%","Positive",f"{sc.get('positive',0):,}"),
        (c2,f"{sc.get('negative',0)/total*100:.1f}%","Negative",f"{sc.get('negative',0):,}"),
        (c3,f"{sc.get('neutral',0)/total*100:.1f}%","Neutral",f"{sc.get('neutral',0):,}"),
    ]:
        with col:
            st.markdown(kpi_card(val, lbl, delta), unsafe_allow_html=True)


    section("Sentiment Distribution")
    col1,col2 = st.columns(2)
    with col1:
        fig = go.Figure(
            go.Pie(
                labels=[s.capitalize() for s in sc.index],
                values=sc.values,
                hole=0.5,
                marker_colors=[C["pos"],C["neg"],C["neu"]],
                marker_line=dict(color="white",width=2))
            )
        fig.update_layout(
            **THEME,
            title="Sentiment Distribution"
            )
        st.plotly_chart(
            fig,
            use_container_width=True
            )
    with col2:
        fig = go.Figure(
            go.Bar(
                x=[s.capitalize() for s in sc.index],
                y=sc.values,
                marker_color=[C["pos"],C["neg"],C["neu"]],
                text=[f"{v:,} ({v/total*100:.1f}%)" for v in sc.values],
                textposition="outside"))
        fig.update_layout(
            **THEME,
            title="Review Count by Sentiment",
            yaxis_title="Reviews",
            showlegend=False,
            yaxis=dict(range=[0,sc.max()*1.2])
            )
        st.plotly_chart(
            fig,
            use_container_width=True
            )

    pivot = df.groupby(["stars","sentiment"]).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1),axis=0)*100
    fig = px.imshow(
        pivot_pct.round(1),
        color_continuous_scale=[
            [0.0, "#fff5f5"],
            [0.5, "#fde68a"],
            [1.0, "#6ee7b7"],
            ],
        zmin=0,
        zmax=100,
        title="Sentiment % by Star Rating",
        text_auto=True,
        labels=dict(x="Sentiment",
                    y="Stars",
                    color="%")
        )
    fig.update_layout(**THEME)
    st.plotly_chart(
        fig,
        use_container_width=True
        )

    t1,t2,t3 = st.tabs(["✅ Positive","⚠️ Negative","➖ Neutral"])
    for tab,sent,color in [(t1,"positive",C["pos"]),(t2,"negative",C["neg"]),(t3,"neutral",C["neu"])]:
        with tab:
            samples = df[df["sentiment"]==sent][["text","stars"]].sample(min(3,len(df[df["sentiment"]==sent])),random_state=42)
            for _,row in samples.iterrows():
                st.markdown(f'''<div class="review-card">
                    <div class="stars">{"⭐" * int(row["stars"])} {int(row["stars"])} stars</div>
                    {row["text"][:250]}…
                </div>''', unsafe_allow_html=True)

    section("Executive Summary")
    bert_neu_q1 = get_kpi(results_summary_df, "BERT", "F1_Neutral", None)
    ibox(f"Positive reviews dominate at {sc.get('positive',0)/total*100:.1f}%.","pos")
    ibox(f"Neutral reviews account for {sc.get('neutral',0)/total*100:.1f}% - hardest class (BERT F1={fmt_score(bert_neu_q1,2)}).","neu")

# ==========================================================
# Q2 - ASPECT ANALYSIS
# ==========================================================
elif page == "🔍 Q2 Aspect Analysis":
    page_header('Q2: Which aspects receive the most praise and criticism?','DeBERTa ABSA 2,000 review sample')

    if absa_df is None:
        st.warning("ABSA results not loaded. Run absa_results.pkl in notebook.")
    else:
        section("Aspect Sentiment Analysis")
        pos_pcts = [round(float((absa_df[f"absa_{a}"]=="Positive").mean())*100, 1) for a in ASPECTS]
        neg_pcts = [round(float((absa_df[f"absa_{a}"]=="Negative").mean())*100, 1) for a in ASPECTS]
        neu_pcts = [round(float((absa_df[f"absa_{a}"]=="Neutral").mean())*100, 1)  for a in ASPECTS]
        col1,col2 = st.columns(2)
        with col1:
            sp = sorted(zip(ASPECTS,pos_pcts),key=lambda x:x[1],reverse=True)
            fig = go.Figure(
                go.Bar(
                    y=[a[0].capitalize() for a in sp],
                    x=[a[1] for a in sp],
                    orientation="h",
                    marker_color=C["pos"],
                    text=[f"{a[1]:.1f}%" for a in sp],
                    textposition="outside"))
            fig.update_layout(
                **THEME,
                title="Positive Aspect Scores",
                xaxis_title="%",
                yaxis=dict(autorange="reversed"
                ))
            st.plotly_chart(
                fig,
                use_container_width=True
                )

        with col2:
            sn = sorted(
                zip(ASPECTS,neg_pcts),
                key=lambda x:x[1],
                reverse=True)
            bc = [C["neg"] if i==0 else "#4E79A7" for i in range(len(sn))]
            fig = go.Figure(
                go.Bar(
                    y=[a[0].capitalize() for a in sn],
                    x=[a[1] for a in sn],
                    orientation="h",
                    marker_color=bc,
                    text=[f"{a[1]:.1f}%" for a in sn],
                    textposition="outside"))
            fig.update_layout(
                **THEME,
                title="Negative Aspect Scores",
                xaxis_title="%",
                yaxis=dict(autorange="reversed")
                )
            st.plotly_chart(
                fig,
                use_container_width=True)

        col3,col4 = st.columns(2)
        with col3:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Positive",x=ASPECTS,y=pos_pcts,marker_color=C["pos"]))
            fig.add_trace(go.Bar(name="Negative",x=ASPECTS,y=neg_pcts,marker_color=C["neg"]))
            fig.add_trace(go.Bar(name="Neutral", x=ASPECTS,y=neu_pcts,marker_color=C["neu"]))
            fig.update_layout(
                **THEME,
                title="Aspect Sentiment Ranking",
                barmode="group",
                yaxis_title="%"
                )
            st.plotly_chart(
                fig,
                use_container_width=True
                )

        with col4:
            hm = pd.DataFrame({"Positive":pos_pcts,"Negative":neg_pcts,"Neutral":neu_pcts}, index=[a.capitalize() for a in ASPECTS])
            hm = hm.sort_values("Positive",ascending=False)
            fig = px.imshow(
                hm.T.round(1),
                color_continuous_scale="RdYlGn",
                title="Aspect Sentiment Heatmap",
                text_auto=True,
                labels=dict(x="Aspect", y="Sentiment", color="%")
                )
            fig.update_layout(**THEME)
            st.plotly_chart(
                fig,
                use_container_width=True)

        # Aspect x Star Rating
        section("Aspect Sentiment by Star Rating")
        if "stars" in absa_df.columns:
            sel_asp = st.selectbox("Select aspect:", [a.capitalize() for a in ASPECTS])
            asp_col = f"absa_{sel_asp.lower()}"
            if asp_col in absa_df.columns:
                rows = []
                for star_val, grp in absa_df.groupby("stars"):
                    vc = grp[asp_col].value_counts(normalize=True) * 100
                    for sent_label, pct in vc.items():
                        rows.append({"Star Rating": f"{int(star_val)}★", "Sentiment": sent_label, "%": round(float(pct), 1)})
                star_long = pd.DataFrame(rows)
                fig = px.bar(
                    star_long,
                    x="Star Rating",
                    y="%",
                    color="Sentiment",
                    title=f"{sel_asp} Sentiment by Star Rating",
                    color_discrete_map={"Positive":C["pos"],"Negative":C["neg"],"Neutral":C["neu"]},
                    barmode="stack", text="%"
                    )
                fig.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="inside"
                    )
                fig.update_layout(
                    **THEME,
                    yaxis_title="%",
                    xaxis_title="Star Rating"
                    )
                st.plotly_chart(
                    fig,
                    use_container_width=True
                    )
                ibox(f"Chart shows how <b>{sel_asp}</b> sentiment shifts across star tiers - higher stars correlate with more positive aspect mentions.","neu")
        else:
            st.info("To enable this chart: merge absa_df with df_reviews on review_id to add the 'stars' column.")

        # Customer Satisfaction Matrix
        section("Importance-Satisfaction Matrix")
        asp_colors_hex = {"food":"#3498db","service":"#e74c3c","ambience":"#2ecc71","price":"#e67e22","location":"#9b59b6"}
        total_ab = len(absa_df)
        mat_data = []
        for i, a in enumerate(ASPECTS):
            col_a = f"absa_{a}"
            men = int((absa_df[col_a]!="Neutral").sum())
            mat_data.append({
                "Aspect": a.capitalize(),
                "Positive": pos_pcts[i],
                "Negative": neg_pcts[i],
                "Mentions": men,
                "Color": asp_colors_hex[a]
            })
        mat_df = pd.DataFrame(mat_data)
        avg_p = mat_df["Positive"].mean()
        avg_m = mat_df["Mentions"].mean()
        fig = go.Figure()
        for _,row in mat_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["Mentions"]],
                    y=[row["Positive"]],
                    mode="markers+text",
                    marker=dict(size=row["Mentions"]/25,
                                color=row["Color"],
                                opacity=0.85),
                    text=[row["Aspect"]],
                    textposition="top center",
                    name=row["Aspect"])
                )
        fig.add_hline(
            y=avg_p,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="Avg Satisfaction")
        fig.add_vline(
            x=avg_m,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="Avg Signal Frequency")
        fig.update_layout(
            **THEME,
            title="Customer Satisfaction Matrix",
            xaxis_title="Non-Neutral ABSA Predictions (Signal Count)",
            yaxis_title="Positive Sentiment %",
            showlegend=False, height=460)
        fig.add_annotation(
            x=mat_df["Mentions"].min(),
            y=mat_df["Positive"].max(),
            text="HIGH SATISFACTION<br>LOW SIGNAL FREQUENCY",
            showarrow=False,
            font=dict(size=16, color="#2ecc71"),
            xanchor="left",
            yanchor="top"
        )

        fig.add_annotation(
            x=avg_m + (mat_df["Mentions"].max() - avg_m) * 0.05,
            y=mat_df["Positive"].max(),
            text="HIGH SATISFACTION<br>HIGH SIGNAL FREQUENCY",
            showarrow=False,
            font=dict(size=16, color="#2ecc71"),
            xanchor="left",
            yanchor="top"
        )

        fig.add_annotation(
            x=mat_df["Mentions"].min(),
            y=mat_df["Positive"].min(),
            text="LOW SATISFACTION<br>LOW SIGNAL FREQUENCY",
            showarrow=False,
            font=dict(size=16, color="#e74c3c"),
            xanchor="left",
            yanchor="bottom"
        )

        fig.add_annotation(
            x=avg_m + (mat_df["Mentions"].max() - avg_m) * 0.05,
            y=mat_df["Positive"].min(),
            text="LOW SATISFACTION<br>HIGH SIGNAL FREQUENCY",
            showarrow=False,
            font=dict(size=16, color="#e74c3c"),
            xanchor="left",
            yanchor="bottom"
        )
        st.plotly_chart(fig, use_container_width=True)

        section("Executive Summary")
        most_pos = ASPECTS[pos_pcts.index(max(pos_pcts))]
        most_neg = ASPECTS[neg_pcts.index(max(neg_pcts))]
        ca,cb = st.columns(2)
        with ca:
            top_pos = [a for a,p in zip(ASPECTS,pos_pcts) if round(p,1)==round(max(pos_pcts),1)]
            if len(top_pos) > 1:
              ibox(f"{' and '.join(a.capitalize() for a in top_pos)} are tied for the strongest positive sentiment ({max(pos_pcts):.1f}% each). Key competitive strengths.","pos")
            else:
              ibox(f"{most_pos.capitalize()} receives strongest positive sentiment ({max(pos_pcts):.1f}%). Key competitive strengths.","pos")
        with cb:
            ibox(f"{most_neg.capitalize()} generates highest negative sentiment ({max(neg_pcts):.1f}%). Directional only: differences between aspects may fall within sampling uncertainty.","neg")

# ==========================================================
# Q3 - TOPIC MODELLING
# ==========================================================
elif page == "🗂️ Q3 Topic Modelling":
    lda_coherence_val = get_kpi(results_summary_df, "LDA", "Coherence", None)
    n_topics_q3 = get_kpi(results_summary_df, "LDA", "KPI_N_Topics", None)
    page_header('Q3: What topics are most frequently discussed?',f'LDA {fmt_int(n_topics_q3)} Topics C_V Coherence: {fmt_score(lda_coherence_val)}')
    if "dominant_topic" not in df_full.columns:
        st.warning("Topic assignments not found. Run topic assignment in notebook first.")
    else:
        df_full["topic_label"] = df_full["dominant_topic"].map(TOPIC_LABELS)
        topic_counts = df_full["dominant_topic"].value_counts().sort_index()
        topic_df = pd.DataFrame({"topic_id":list(topic_counts.index),
            "label":[TOPIC_LABELS.get(i,"Unknown") for i in topic_counts.index],
            "count":topic_counts.values, "pct":topic_counts.values/len(df_full)*100})
        section("Topic Summary")
        c1,c2,c3 = st.columns(3)
        for col,val,lbl,delta in [
            (c1,fmt_int(n_topics_q3),"Total Topics","LDA optimised"),
            (c2,fmt_score(lda_coherence_val),"Coherence Score","Final Result"),
            (c3,TOPIC_LABELS.get(topic_counts.idxmax(),"Unknown"),"Largest Topic",f"{topic_counts.max():,} reviews")
        ]:
            with col:
                st.markdown(kpi_card(val, lbl, delta), unsafe_allow_html=True)

        section("Topic Distribution")
        # Reviews per Topic - full width horizontal bar (cleaner than treemap)
        topic_df = topic_df.sort_values("count", ascending=False)
        colors = pc.sample_colorscale("Blues", [i / (len(topic_df) - 1) for i in range(len(topic_df))])[::-1]
        fig = go.Figure(
            go.Bar(
                y=topic_df["label"],
                x=topic_df["count"],
                orientation="h",
                marker=dict(
                    color=colors,
                    line=dict(width=0)
                ),
                text=topic_df["count"].apply(lambda x: f"{x:,}"),
                textposition="outside"
            )
        )
        fig.update_layout(
            **THEME,
            title="Reviews per Topic",
            xaxis_title="Number of Reviews",
            yaxis=dict(autorange="reversed"),
            xaxis=dict(
                range=[0, topic_df["count"].max()*1.25]
            ),
            height=450,
            showlegend=False
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

        # Topic x Sentiment Matrix
        section("Topic x Sentiment Matrix")
        topic_sent = (df_full.groupby(["dominant_topic", "sentiment"]).size().unstack(fill_value=0))
        topic_sent_pct = (topic_sent.div(topic_sent.sum(axis=1), axis=0)* 100)
        topic_sent_pct = topic_sent_pct.sort_values("positive",ascending=False)
        topic_sent_pct.index = [TOPIC_LABELS.get(i, f"Topic {i}") for i in topic_sent_pct.index]
        col_order = [
            c for c in ["negative", "neutral", "positive"]
            if c in topic_sent_pct.columns
        ]
        tsm_display = topic_sent_pct[col_order].copy()
        tsm_display.columns = [c.capitalize() for c in tsm_display.columns]
        fig = go.Figure()
        colors = {"Negative": "#f87171", "Neutral": "#fbbf24", "Positive": "#34d399"}
        for col in tsm_display.columns:
            fig.add_trace(go.Bar(
                name=col,
                x=tsm_display.index,
                y=tsm_display[col],
                marker_color=colors[col],
                text=tsm_display[col].round(1).astype(str) + "%",
                textposition="inside"
            ))
        fig.update_layout(
            **THEME,
            barmode="stack",
            height=450,
            xaxis_title="Topic",
            yaxis_title="%",
            xaxis_tickangle=-30
        )

        fig.update_traces(
            textfont_size=12
        )
        st.plotly_chart(fig, use_container_width=True)

        # Topic Deep Dive
        section("Topic Deep Dive")
        selected = st.selectbox("Select a topic:", [f"Topic {i+1}: {TOPIC_LABELS[i]}" for i in range(11)])
        t_idx = int(selected.split(":")[0].replace("Topic","").strip()) - 1

        tr = df_full[df_full["dominant_topic"] == t_idx]
        sc2 = tr["sentiment"].value_counts()
        avg_star = tr["stars"].mean()
        positive_pct = (sc2.get("positive", 0) / len(tr)) * 100
        negative_pct = (sc2.get("negative", 0) / len(tr)) * 100
        st.info(
            f"""
            **📌 Topic Summary**
            • Reviews: **{len(tr):,}**
            • Average Rating: **{avg_star:.2f} ⭐**
            • Positive Reviews: **{positive_pct:.1f}%**
            • Negative Reviews: **{negative_pct:.1f}%**
            • Dataset Share: **{len(tr)/len(df_full)*100:.1f}%**
            """
            )

        col_a,col_b = st.columns(2)
        with col_a:
            if lda_model:
                tw = lda_model.show_topic(t_idx, topn=10)
                words   = [w for w,_ in tw]
                weights = [round(w*100,3) for _,w in tw]
                fig = go.Figure(
                    go.Bar(
                        y=list(reversed(words)),
                        x=list(reversed(weights)),
                        orientation="h",
                        marker=dict(
                            color=list(reversed(weights)),
                            colorscale="Blues",
                            showscale=False),
                        text=[f"{w:.2f}%" for w in reversed(weights)], textposition="outside")
                    )
                fig.update_layout(
                    **THEME,
                    title=f"Keywords: {TOPIC_LABELS[t_idx]}",
                    xaxis_title="Weight (%)",
                    height=380
                    )
                st.plotly_chart(
                    fig,
                    use_container_width=True
                    )


        with col_b:
            tr = df_full[df_full["dominant_topic"]==t_idx]
            if len(tr) > 0:
                sc2 = tr["sentiment"].value_counts()
                fig = go.Figure(
                    go.Pie(
                        labels=[s.capitalize() for s in sc2.index],
                        values=sc2.values,
                        hole=0.5,
                        marker_colors=[{"positive":C["pos"],"negative":C["neg"],"neutral":C["neu"]}[x] for x in sc2.index],
                        marker_line=dict(color="white",width=2))
                    )
                fig.update_layout(
                    **THEME,
                    title=f"Sentiment: {TOPIC_LABELS[t_idx]}",
                    height=320
                    )
                st.plotly_chart(
                    fig,
                    use_container_width=True
                    )
                # st.metric("Reviews in Topic", f"{len(tr):,}", f"{len(tr)/len(df_full)*100:.1f}% of total")

        st.markdown("**Sample Reviews**")

        if len(tr) > 0:
            ts = df_full[
                df_full["dominant_topic"]==t_idx][["text","sentiment","stars"]].sample(
                    min(3,len(df_full[df_full["dominant_topic"]==t_idx])),
                    random_state=42
                    )
            for _,row in ts.iterrows():
                color = {"positive":C["pos"],"negative":C["neg"],"neutral":C["neu"]}.get(row["sentiment"],C["neu"])
                st.markdown(f'''<div class="review-card">
                    <div class="stars">{"⭐" * int(row["stars"])} {int(row["stars"])} stars</div>
                    {row["text"][:200]}…
                </div>''', unsafe_allow_html=True)

        top_t = topic_counts.idxmax()
        ibox(f"11 topics identified. {TOPIC_LABELS.get(top_t,'Unknown')} is the most discussed ({topic_counts[top_t]:,} reviews, {topic_counts[top_t]/len(df_full)*100:.1f}%). Topic 7 Complaints has the highest share of negative reviews, consistent with the sentiment findings.","neu")

# ==========================================================
# Q4 - SENTIMENT TRENDS
# ==========================================================
elif page == "📅 Q4 Sentiment Trends":
    page_header("Q4: How has customer sentiment changed over time?", "Monthly sentiment trend analysis")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby(["month","sentiment"]).size().reset_index(name="count")
    monthly_total = df.groupby("month").size().reset_index(name="total")
    monthly = monthly.merge(monthly_total, on="month")
    monthly["pct"] = monthly["count"] / monthly["total"] * 100

    col1,_ = st.columns([3,1])
    with col1:
        view = st.radio("View:", ["Percentage (%)", "Raw Count"], horizontal=True)
    y_col = "pct" if "%" in view else "count"
    y_title = "Percentage (%)" if "%" in view else "Number of Reviews"
    fig = px.line(
        monthly,
        x="month",
        y=y_col,
        color="sentiment",
        color_discrete_map={"positive":C["pos"],"negative":C["neg"],"neutral":C["neu"]},
        title="Monthly Sentiment Trend",
        markers=True
        )
    fig.update_layout(
        **THEME,
        xaxis_title="Month",
        yaxis_title=y_title,
        xaxis=dict(tickangle=45),
        height=420
        )
    st.plotly_chart(
        fig,
        use_container_width=True
        )

    col3,col4 = st.columns(2)
    with col3:
        vol = df.groupby("month").size().reset_index(name="count")
        fig = px.bar(
            vol,
            x="month",
            y="count",
            title="Review Volume by Month",
            color_discrete_sequence=[C["blue"]]
            )
        fig.update_layout(
            **THEME,
            xaxis=dict(tickangle=45),
            yaxis_title="Reviews"
            )
        st.plotly_chart(
            fig,
            use_container_width=True
            )
    with col4:
        df["year"] = df["date"].dt.year
        df["month_num"] = df["date"].dt.month
        hm_df = df[df["sentiment"]=="positive"].groupby(["year","month_num"]).size().unstack(fill_value=0)
        if not hm_df.empty and len(hm_df.columns) > 0:
            month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            cols_av = [month_names[i-1] for i in hm_df.columns if 1 <= i <= 12]
            hm_display = hm_df.copy()
            hm_display.index = [str(y) for y in hm_display.index]
            fig = px.imshow(
                hm_display,
                title="Positive Reviews - Year x Month",
                color_continuous_scale="Greens",
                labels=dict(x="Month", y="Year", color="Reviews"),
                x=cols_av, aspect="auto"
                )
            fig.update_layout(
                **THEME
                )
            st.plotly_chart(
                fig,
                use_container_width=True
                )
        else:
            st.info("Not enough data to display Year x Month heatmap.")

    # -- Mann-Kendall trend test (overall, full dataset) --
    st.markdown('<div class="section-header">Statistical Trend Test</div>', unsafe_allow_html=True)
    try:
        import pymannkendall as mk
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pymannkendall", "-q"])
        import pymannkendall as mk

    monthly_pos_series = (
        df.assign(month=df["date"].dt.to_period("M"))
          .groupby("month")
          .apply(lambda g: (g["sentiment"] == "positive").mean() * 100)
          .dropna()
    )
    if len(monthly_pos_series) < 2:
      st.info("Not enough data to display Year x Month heatmap.")
    else:
      mk_result = mk.original_test(monthly_pos_series.values)
      mk1, mk2, mk3, mk4 = st.columns(4)
      for col, val, lbl, sub in [
          (mk1, mk_result.trend.capitalize(), "Trend Direction", "Mann-Kendall"),
          (mk2, f"{mk_result.p:.4f}", "p-value",
          "significant" if mk_result.p < 0.05 else "not significant"),
          (mk3, f"{mk_result.Tau:+.3f}", "Kendall's Tau", "strength of trend"),
          (mk4, f"{mk_result.slope:+.4f} pts/mo", "Sen's Slope", f"n={len(monthly_pos_series)} months"),
      ]:
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-label">{lbl}</div>
                    <div style="font-size:11px;color:#888">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    pos_monthly = monthly[monthly["sentiment"]=="positive"]
    if len(pos_monthly) >= 2:
        latest   = pos_monthly["pct"].iloc[-1]
        earliest = pos_monthly["pct"].iloc[0]
        trend = "increased" if latest > earliest else "decreased"
        ibox(f"Positive sentiment shows a statistically significant {mk_result.trend} trend over the observed period (Mann-Kendall p={mk_result.p:.4f}).","neu")


# ==========================================================
# Q5 - REVIEW DRIVERS
# ==========================================================
elif page == "🔑 Q5 Review Drivers":
    page_header("Q5: What words are most frequently associated with positive and negative reviews?", "Word frequency and TF-IDF analysis - 1-star vs 5-star reviews")

    df_full["clean_text"] = df_full["clean_text"].fillna("")
    cc1,cc2 = st.columns(2)
    with cc1:
        top_n = st.slider("Number of words:", 10, 25, 15)
    with cc2:
        method = st.radio("Analysis method:", ["Raw Frequency","TF-IDF Weighted"], horizontal=True)
    five_star = df_full[df_full["stars"]==5.0]["clean_text"]
    one_star = df_full[df_full["stars"]==1.0]["clean_text"]

    if method == "TF-IDF Weighted":
        from sklearn.feature_extraction.text import TfidfVectorizer
        def get_tfidf(texts, n):
            tfidf = TfidfVectorizer(max_features=5000)
            mat = tfidf.fit_transform(texts)
            scores = np.asarray(mat.mean(axis=0)).flatten()
            top_idx = scores.argsort()[-n:][::-1]
            vocab = tfidf.get_feature_names_out()
            return [(vocab[i], float(scores[i])) for i in top_idx]
        five_words = get_tfidf(five_star.tolist(), top_n)
        one_words  = get_tfidf(one_star.tolist(), top_n)
        x_label = "TF-IDF Score"
    else:
        five_words = Counter(" ".join(five_star).split()).most_common(top_n)
        one_words  = Counter(" ".join(one_star).split()).most_common(top_n)
        x_label = "Frequency"

    section("Positive Review Drivers")
    col1,col2 = st.columns(2)
    with col1:
        if five_words:
            w5,c5 = zip(*five_words)
            fig = go.Figure(
                go.Bar(
                    y=list(reversed(w5)),
                    x=list(reversed(c5)),
                    orientation="h",
                    marker_color=C["pos"],
                    text=[f"{v:.4f}" if method=="TF-IDF Weighted" else f"{int(v):,}" for v in reversed(c5)],
                    textposition="outside")
                )
            fig.update_layout(
                **THEME,
                title="5-star Top Keywords",
                xaxis_title=x_label,
                height=480)
            st.plotly_chart(
                fig,
                use_container_width=True
                )

    with col2:
        st.markdown("**Sample 5-Star Reviews**")
        for _,row in df_full[df_full["stars"]==5.0][["text"]].sample(3,random_state=42).iterrows():
            st.markdown(f'''<div class="review-card">
                <div class="stars">⭐⭐⭐⭐⭐ 5 stars</div>
                {row["text"][:250]}…
                </div>''', unsafe_allow_html=True)
        if five_words:
            ibox(f"5-star reviews feature: <b>{five_words[0][0]}</b>, <b>{five_words[1][0]}</b>, <b>{five_words[2][0]}</b>.","pos")

    section("Negative Review Drivers")
    col3,col4 = st.columns(2)
    with col3:
        if one_words:
            w1,c1 = zip(*one_words)
            fig = go.Figure(
                go.Bar(
                    y=list(reversed(w1)),
                    x=list(reversed(c1)),
                    orientation="h",
                    marker_color=C["neg"],
                    text=[f"{v:.4f}" if method=="TF-IDF Weighted" else f"{int(v):,}" for v in reversed(c1)],
                    textposition="outside")
                )
            fig.update_layout(
                **THEME,
                title="1-star Top Keywords",
                xaxis_title=x_label,
                height=480
                )
            st.plotly_chart(
                fig,
                use_container_width=True
                )
    with col4:
        st.markdown("**Sample 1-Star Reviews**")
        for _,row in df_full[df_full["stars"]==1.0][["text"]].sample(3,random_state=42).iterrows():
            st.markdown(f'''<div class="review-card">
                <div class="stars">⭐ 1 star</div>
                {row["text"][:250]}…
                </div>''', unsafe_allow_html=True)
        if one_words:
            ibox(f"1-star reviews feature: <b>{one_words[0][0]}</b>, <b>{one_words[1][0]}</b>, <b>{one_words[2][0]}</b>.","neg")

    section("Driver Comparison")
    if five_words and one_words:
        comp = pd.DataFrame({"Rank":range(1,6),
            "Positive Drivers (5 star)": [w for w,_ in five_words[:5]],
            "Score": [f"{c:.4f}" if method=="TF-IDF Weighted" else f"{int(c):,}" for _,c in five_words[:5]],
            "Negative Drivers (1 star)": [w for w,_ in one_words[:5]],
            "Score ": [f"{c:.4f}" if method=="TF-IDF Weighted" else f"{int(c):,}" for _,c in one_words[:5]],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)
    ibox(f"Method: <b>{method}</b>. Many of the most frequent terms appear in both lists, so frequency alone does not show sentiment-specific language.","neg")


# ==========================================================
# RECOMMENDATIONS
# ==========================================================
elif page == "💡 Recommendations":
    page_header("Business Recommendations", "Indicative suggestions from NLP pipeline outputs - directional only, not precision-ranked findings")
    ibox("Priorities are researcher-assigned and indicative only. ABSA aspect differences are small relative to sampling uncertainty (see Technical Appendix), so treat these as hypotheses to investigate, not validated findings.", "neu")
    recs = [
        {"priority":"HIGH","title":"Address Pricing Strategy",
         "body":"Price had the highest negative rate among the five aspects, although the differences between aspects are small. Worth investigating value proposition and price-quality communication.",
         "evidence":"ABSA: Price highest negative aspect."},
        {"priority":"HIGH","title":"Improve Service Consistency",
         "body":"Service shows both high praise and substantial criticism. Review service-related complaints for recurring patterns before acting.",
         "evidence":"ABSA: Service has both high positive and substantial negative rates."},
        {"priority":"MEDIUM","title":"Leverage Service and Ambience as Marketing Asset",
         "body":"Service and Ambience are the most praised aspects. Highlight friendly staff and atmosphere together in marketing communications and visual content.",
         "evidence":"ABSA: Service and Ambience tied for highest positive score."},
        {"priority":"MEDIUM","title":"Monitor Complaint Topic Cluster",
         "body":"Topic 7 Complaints emerged as distinct cluster. Monitor for early signals of service deterioration.",
         "evidence":"LDA Topic 7 keywords: time, would, back, didnt."},
        {"priority":"LOW","title":"Maintain Food Quality Standards",
         "body":"Food is among the most frequent terms in both 5-star and 1-star reviews, so food quality is likely central to customer feedback. No driver analysis was performed.",
         "evidence":"Word frequency: food = top word in both positive and negative reviews."},
    ]
    for rec in recs:
        p_color = {"HIGH":"#e53e3e","MEDIUM":"#d69e2e","LOW":"#38a169"}.get(rec["priority"],"#3182ce")
        icon = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(rec["priority"],"🔵")
        st.markdown(f'''<div class="rec-card">
            <div class="rec-title">{icon} {rec["title"]}
                <span style="color:{p_color};font-size:0.75rem;font-weight:700;"> {rec["priority"]} PRIORITY</span>
            </div>
            <div class="rec-body">{rec["body"]}</div>
            <div class="rec-body" style="color:#718096;margin-top:0.3rem;font-style:italic;">Evidence: {rec["evidence"]}</div>
        </div>''', unsafe_allow_html=True)
    sc_full = df_full["sentiment"].value_counts()

    vader_acc = get_kpi(results_summary_df, "VADER", "Accuracy", None)
    lr_acc    = get_kpi(results_summary_df, "Logistic Regression", "Accuracy", None)
    bert_acc  = get_kpi(results_summary_df, "BERT", "Accuracy", None)
    lda_coh   = get_kpi(results_summary_df, "LDA", "Coherence", None)
    bert_coh  = get_kpi(results_summary_df, "BERTopic", "Coherence", None)

    lines = ["NLP BUSINESS INSIGHT REPORT","="*50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Dataset: Yelp Open Dataset 50,000 reviews","",
        "SENTIMENT SUMMARY","-"*30,
        f"Positive: {sc_full.get('positive',0):,} ({sc_full.get('positive',0)/len(df_full)*100:.1f}%)",
        f"Negative: {sc_full.get('negative',0):,} ({sc_full.get('negative',0)/len(df_full)*100:.1f}%)",
        f"Neutral: {sc_full.get('neutral',0):,}  ({sc_full.get('neutral',0)/len(df_full)*100:.1f}%)",
        "","MODEL PERFORMANCE","-"*30,
        f"VADER: {fmt_pct(vader_acc)} | Logistic Regression: {fmt_pct(lr_acc)} | BERT: {fmt_pct(bert_acc)}",
        f"LDA Coherence: {fmt_score(lda_coh)} | BERTopic Coherence: {fmt_score(bert_coh)}",
        "","RECOMMENDATIONS","-"*30] + [f"[{r['priority']}] {r['title']}: {r['body'][:80]}..." for r in recs]
    st.download_button("Download Full Report (.txt)", "\n".join(lines),
        f"NLP_Report_{datetime.now().strftime('%Y%m%d')}.txt", "text/plain")

# ==========================================================
# TECHNICAL APPENDIX
# ==========================================================
elif page == "🤖 Technical Appendix":
    page_header("Technical Appendix - Model Evaluation", "Performance metrics for all pipeline components - for examiner review")

    tab1,tab2,tab3 = st.tabs(["BERT Sentiment","ABSA","LDA Topic Model"])
    #BERT Sentiment
    with tab1:
        st.markdown('<div class="section-header">BERT Sentiment Classification</div>', unsafe_allow_html=True)

        bert_acc  = get_kpi(results_summary_df, "BERT", "Accuracy", None)
        bert_pos  = get_kpi(results_summary_df, "BERT", "F1_Positive", None)
        bert_neg  = get_kpi(results_summary_df, "BERT", "F1_Negative", None)
        bert_neu  = get_kpi(results_summary_df, "BERT", "F1_Neutral", None)
        vader_acc = get_kpi(results_summary_df, "VADER", "Accuracy", None)
        delta_pp  = (float(fmt_score(bert_acc)) - float(fmt_score(vader_acc))) * 100 if (bert_acc is not None and vader_acc is not None) else None

        c1,c2,c3,c4 = st.columns(4)
        for col,val,lbl,delta in [
            (c1,fmt_pct(bert_acc),"Accuracy",f"+{delta_pp:.0f}pp vs VADER" if delta_pp is not None else "N/A"),
            (c2,fmt_score(bert_pos,2),"Positive F1","Strong"),
            (c3,fmt_score(bert_neg,2),"Negative F1","Good"),
            (c4,fmt_score(bert_neu,2),"Neutral F1","Challenging class"),
        ]:
            with col:
                st.markdown(f'''<div class="kpi-card">
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-label">{lbl}</div>
                    <div class="kpi-delta">{delta}</div>
                </div>''',unsafe_allow_html=True)

        eval_df = None
        if results_summary_df is not None:
            model_order = ["VADER", "Logistic Regression", "BERT"]
            eval_rows = []
            for m in model_order:
                for cls in ["Positive", "Negative", "Neutral"]:
                    f1 = get_kpi(results_summary_df, m, f"F1_{cls}", None)
                    prec = get_kpi(results_summary_df, m, f"Precision_{cls}", None)
                    rec = get_kpi(results_summary_df, m, f"Recall_{cls}", None)
                    if f1 is not None:
                        eval_rows.append({
                            "Model": m, "Class": cls,
                            "F1": round(float(f1), 2),
                            "Precision": round(float(prec), 2) if prec is not None else None,
                            "Recall": round(float(rec), 2) if rec is not None else None,
                        })
            if eval_rows:
                eval_df = pd.DataFrame(eval_rows)

        if eval_df is None or eval_df.empty:
            st.warning("results_summary.csv not found or missing F1 metrics. Run the export cells in the notebook first.")
        else:
            test_n = get_kpi(results_summary_df, "BERT", "KPI_Test_N", None)
            model_order = ["VADER", "Logistic Regression", "BERT"]
            acc_rows = []
            for m in model_order:
                acc = get_kpi(results_summary_df, m, "Accuracy", None)
                lo = get_kpi(results_summary_df, m, "Accuracy_CI_Lower", None)
                hi = get_kpi(results_summary_df, m, "Accuracy_CI_Upper", None)
                acc_rows.append({
                    "Model": m,
                    "Accuracy_pct": float(acc) * 100 if acc is not None else None,
                    "CI_lower_pct": float(lo) * 100 if lo is not None else None,
                    "CI_upper_pct": float(hi) * 100 if hi is not None else None,
                    "err_minus": (float(acc) - float(lo)) * 100 if (acc is not None and lo is not None) else 0,
                    "err_plus": (float(hi) - float(acc)) * 100 if (acc is not None and hi is not None) else 0,
                })
            acc_df = pd.DataFrame(acc_rows)

            col1, col2 = st.columns(2)
            with col1:
                model_colors = {"VADER": C["neg"], "Logistic Regression": C["neu"], "BERT": C["pos"]}
                fig = go.Figure()
                for _, row in acc_df.iterrows():
                    fig.add_trace(go.Bar(
                        name=row["Model"],
                        x=[row["Model"]],
                        y=[row["Accuracy_pct"]],
                        marker_color=model_colors.get(row["Model"], C["blue"]),
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=[row["err_plus"]],
                            arrayminus=[row["err_minus"]],
                            color="#555",
                            thickness=2,
                            width=6,
                        ),
                        text=[f"{row['Accuracy_pct']:.1f}%<br>[{row['CI_lower_pct']:.1f}–{row['CI_upper_pct']:.1f}]"
                              if row["CI_lower_pct"] is not None else fmt_pct_raw(row["Accuracy_pct"])],
                        textposition="outside",
                        showlegend=False,
                    ))
                n_label = f"n={int(test_n):,}" if test_n is not None else "n=unknown"
                fig.update_layout(
                    **THEME,
                    title=f"Model Accuracy with 95% Wilson CI ({n_label} test samples)",
                    yaxis=dict(title="Accuracy (%)", range=[0, 105]),
                    xaxis_title="Model",
                    barmode="group",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                f1_pivot = eval_df.pivot(index="Model", columns="Class", values="F1")
                fig = px.imshow(
                    f1_pivot,
                    color_continuous_scale="RdYlGn",
                    title="F1 Score Heatmap by Class",
                    text_auto=True,
                    zmin=0, zmax=1
                )
                fig.update_layout(**THEME)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Classification Report**")
            st.dataframe(eval_df.set_index(["Model","Class"])[["F1","Precision","Recall"]].round(2), use_container_width=True)
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**Strengths**")
            st.markdown(f"- BERT {fmt_pct(bert_acc)} accuracy\n- Positive F1={fmt_score(bert_pos,2)}\n- Clear progressive improvement")
        with c2:
            st.markdown("**Limitations**")
            st.markdown(f"- Neutral F1={fmt_score(bert_neu,2)}\n- Star-rating proxy labels\n- Computationally expensive")
        with c3:
            st.markdown("**Future Work**")
            st.markdown("- RoBERTa or DeBERTa\n- Manual neutral annotation\n- Domain-adaptive pre-training")

        # -- STATISTICAL VALIDATION: McNemar's test --------------------------
        st.markdown("---")
        st.markdown('<div class="section-header">Statistical Validation - McNemar Test</div>', unsafe_allow_html=True)
        st.caption(
            "McNemar's test (χ² with Yates correction, p < 0.05) checks whether BERT's error reduction over baseline models is statistically significant, not random. "
        )

        mcn_bv_stat = get_kpi(results_summary_df, "McNemar", "BERT_vs_VADER_Statistic", None)
        mcn_bv_p = get_kpi(results_summary_df, "McNemar", "BERT_vs_VADER_PValue", None)
        mcn_bl_stat = get_kpi(results_summary_df, "McNemar", "BERT_vs_LR_Statistic", None)
        mcn_bl_p = get_kpi(results_summary_df, "McNemar", "BERT_vs_LR_PValue", None)

        def _sig_label(p):
            if p is None: return "N/A - run export snippet above"
            return "✅ Significant (p < 0.05)" if float(p) < 0.05 else "⬜ Not significant (p ≥ 0.05)"

        mc1, mc2 = st.columns(2)
        with mc1:
            p_str = fmt_pval(mcn_bv_p) if mcn_bv_p is not None else "N/A"
            χ_str = f"{float(mcn_bv_stat):.4f}" if mcn_bv_stat is not None else "N/A"
            winner = "BERT performs better" if (mcn_bv_p is not None and float(mcn_bv_p) < 0.05) else ""
            st.markdown(
                f'''
                <div class="kpi-card" style="text-align:left;padding:18px;">
                    <div style="font-size:13px;font-weight:700;margin-bottom:8px;color:#1e293b;">
                        🧪 BERT vs VADER
                    </div>
                    <div style="font-size:13px;line-height:1.8;">
                        <span style="color:black;">χ² statistic:</span>
                        <b style="color:black;">{χ_str}</b><br>
                        <span style="color:black;">p-value:</span>
                        <b style="color:black;">{p_str}</b><br>
                        <span style="color:black;">Result:</span>
                        <b style="color:black;">{_sig_label(float(mcn_bv_p))}</b>
                        {"<br><span style='color:#90EE90;font-weight:700'>" + winner + "</span>" if winner else ""}
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
                )
        with mc2:
            p_str2 = fmt_pval(mcn_bl_p) if mcn_bl_p is not None else "N/A"
            χ_str2 = f"{float(mcn_bl_stat):.4f}" if mcn_bl_stat is not None else "N/A"
            winner2 = "BERT performs better" if (mcn_bl_p is not None and float(mcn_bl_p) < 0.05) else ""
            st.markdown(
                f'''
                <div class="kpi-card" style="text-align:left;padding:18px">
                    <div style="font-size:13px;font-weight:700;margin-bottom:8px;color:#1e293b;">
                    🧪 BERT vs Logistic Regression
                    </div>
                    <div style="font-size:13px;line-height:1.8;">
                        <span style="color:black;">χ² statistic:</span>
                        <b style="color:black;">{χ_str2}</b><br>
                        <span style="color:black;">p-value:</span>
                        <b style="color:black;">{p_str2}</b><br>
                        <span style="color:black;">Result:</span>
                        <b style="color:black;">{_sig_label(float(mcn_bl_p))}</b>
                        {"<br><span style='color:#90EE90;font-weight:700'>" + winner + "</span>" if winner else ""}
                    </div>
                </div>''', unsafe_allow_html=True)

        if mcn_bv_p is not None and mcn_bl_p is not None:
            if float(mcn_bv_p) < 0.05 and float(mcn_bl_p) < 0.05:
                ibox("Both McNemar tests significant (p < 0.05) - BERT's gains over VADER and Logistic Regression are statistically significant.", "pos")
            elif float(mcn_bv_p) < 0.05 or float(mcn_bl_p) < 0.05:
                ibox("One McNemar test is significant. BERT shows a confirmed advantage over one baseline but not both at p < 0.05.", "neu")
            else:
                ibox("Neither McNemar test reaches p < 0.05 on this test set. Accuracy differences may be within random variation - consider increasing test-set size.", "neg")

        # -- ERROR ANALYSIS ---------------------------------------------------
        st.markdown("---")
        st.markdown('<div class="section-header">Error Analysis</div>', unsafe_allow_html=True)

        if bert_error_df is None:
            st.info(
                "**bert_error_analysis.csv not found.** "
                "Add the line below to the notebook error-analysis cell, then re-run it:"
            )
            st.code("error_df.to_csv('bert_error_analysis.csv', index=False)", language="python")
            st.caption(
                "`error_df` already exists in the notebook - it contains the columns "
                "`text`, `true`, `bert_pred`, `vader_pred`, `lr_pred`."
            )
        else:
            edf = bert_error_df.copy()
            edf["bert_correct"] = edf["true"] == edf["bert_pred"]
            edf["vader_correct"] = edf["true"] == edf["vader_pred"]
            edf["lr_correct"] = edf["true"] == edf["lr_pred"]
            errors_only = edf[~edf["bert_correct"]]
            all3_wrong = edf[~edf["bert_correct"] & ~edf["vader_correct"] & ~edf["lr_correct"]]
            only_bert_wrong = edf[~edf["bert_correct"] & edf["vader_correct"] & edf["lr_correct"]]
            n_total  = len(edf)
            n_errors = len(errors_only)

            # KPI row
            ea1,ea2,ea3,ea4 = st.columns(4)
            for col,val,lbl,sub in [
                (ea1, f"{n_total:,}", "Test Samples", "held-out set"),
                (ea2, f"{n_errors:,}", "BERT Errors", f"{n_errors/n_total*100:.1f}% error rate"),
                (ea3, f"{len(all3_wrong):,}", "All 3 Models Wrong", f"{len(all3_wrong)/n_total*100:.2f}% of test set"),
                (ea4, f"{len(only_bert_wrong):,}", "Only BERT Wrong",f"{len(only_bert_wrong)/n_total*100:.2f}% of test set"),
            ]:
                with col:
                    st.markdown(
                        f'''<div class="kpi-card">
                            <div class="kpi-value">{val}</div>
                            <div class="kpi-label">{lbl}</div>
                            <div class="kpi-delta">{sub}</div>
                        </div>''', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            ec1, ec2 = st.columns(2)

            # Confusion heatmap of errors only
            with ec1:
                CLS = ["negative","neutral","positive"]
                ep = errors_only.groupby(["true","bert_pred"]).size().unstack(fill_value=0)
                ep = ep.reindex(index=[c for c in CLS if c in ep.index],
                                columns=[c for c in CLS if c in ep.columns], fill_value=0)
                ep.index   = [c.capitalize() for c in ep.index]
                ep.columns = [c.capitalize() for c in ep.columns]
                fig = px.imshow(ep, text_auto=True, color_continuous_scale="Reds",
                                title="Error Confusion Matrix (BERT misclassifications only)",
                                labels=dict(x="BERT Predicted", y="True Label", color="Count"))
                fig.update_layout(**THEME, height=320)
                st.plotly_chart(fig, use_container_width=True)

            # Dominant error patterns bar chart
            with ec2:
                pat_rows = []
                for tc in ["negative","neutral","positive"]:
                    for pc in ["negative","neutral","positive"]:
                        if tc == pc: continue
                        n = len(errors_only[(errors_only["true"]==tc)&(errors_only["bert_pred"]==pc)])
                        if n:
                            pat_rows.append({
                                "Pattern": f"True {tc.capitalize()} → Pred {pc.capitalize()}",
                                "Count": n,
                                "% of Errors": round(n/n_errors*100,1)
                            })
                if pat_rows:
                    pf = pd.DataFrame(pat_rows).sort_values("Count", ascending=True)
                    fig = go.Figure(go.Bar(
                        y=pf["Pattern"], x=pf["% of Errors"], orientation="h",
                        marker_color=C["neg"],
                        text=[f"{v:.1f}%" for v in pf["% of Errors"]],
                        textposition="outside"
                    ))
                    fig.update_layout(
                        **THEME, title="Dominant Error Patterns (% of all BERT errors)",
                        xaxis_title="% of All Errors",
                        xaxis=dict(range=[0, pf["% of Errors"].max()*1.35]),
                        height=320
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Review-length vs error (Welch t-test)
            if "text_length" in edf.columns:
                from scipy import stats as _sp
                len_ok  = edf.loc[ edf["bert_correct"], "text_length"].dropna()
                len_err = edf.loc[~edf["bert_correct"], "text_length"].dropna()
                t_stat, p_val = _sp.ttest_ind(len_ok, len_err, equal_var=False)
                direction = "longer" if len_err.mean() > len_ok.mean() else "shorter"

                lc1, lc2 = st.columns(2)
                with lc1:
                    fig = go.Figure()
                    fig.add_trace(go.Box(y=len_ok.clip(upper=3000), name="Correct", marker_color=C["pos"], boxmean=True))
                    fig.add_trace(go.Box(y=len_err.clip(upper=3000), name="Incorrect", marker_color=C["neg"], boxmean=True))
                    fig.update_layout(**THEME, title="Review Length: Correct vs Incorrect",
                                      yaxis_title="Characters (capped 3,000)", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                with lc2:
                    st.markdown("**Welch's t-test - does review length predict errors?**")
                    st.markdown(f"""
                    | Metric | Value |
                    |---|---|
                    | Mean length - correct | {len_ok.mean():.0f} chars |
                    | Mean length - incorrect | {len_err.mean():.0f} chars |
                    | t-statistic | {t_stat:.3f} |
                    | p-value | {fmt_p(p_val)} |
                    | Significant? | {"✅ Yes (p < 0.05)" if p_val < 0.05 else "❌ No (p ≥ 0.05)"} |
                    """)
                    if p_val < 0.05:
                        ibox(f"Misclassified reviews are statistically significantly <b>{direction}</b> (Welch t-test p {fmt_peq(p_val)}). Length may confound the neutral class.", "neg")
                    else:
                        ibox(f"No significant length difference between correct and incorrect predictions (p {fmt_peq(p_val)}). Review length does not reliably predict BERT errors.", "neu")

            # Qualitative: sample misclassified neutrals
            neu2pos = edf[(edf["true"]=="neutral") & (edf["bert_pred"]=="positive")]
            if len(neu2pos) > 0 and "text" in edf.columns:
                n_neu = len(edf[edf["true"]=="neutral"])
                st.markdown(
                    f"**Sample Misclassified Neutral Reviews** "
                    f"(true = neutral → predicted positive | {len(neu2pos)} cases, "
                    f"{len(neu2pos)/n_neu*100:.1f}% of all true-neutral test reviews)"
                )
                for _, row in neu2pos.sample(min(3,len(neu2pos)), random_state=42).iterrows():
                    txt = str(row["text"])[:300]
                    st.markdown(
                        f'''<div class="review-card" style="border-left:4px solid {C["neu"]};padding:10px 14px;margin:6px 0;background:#fffbea;border-radius:6px">
                            <span style="font-size:11px;color:#888">True: Neutral → Predicted: Positive</span><br>
                            <span style="font-size:13px">{txt}…</span>
                        </div>''', unsafe_allow_html=True)

    # -- ABSA EVALUATION tab  --------
    #ABSA Evaluation
    with tab2:
        st.markdown('<div class="section-header">ABSA - Error Analysis</div>', unsafe_allow_html=True)

        # -- PART 1: Signal-based flagged cases (absa_flagged_cases_for_manual_coding.csv) --
        if absa_flagged_df is None:
            st.info(
                "**absa_flagged_cases_for_manual_coding.csv not found.** "
                "This file is already exported by the notebook error-analysis cell. "
                "Make sure it is in the same directory as app.py."
            )
        else:
            flagged = absa_flagged_df.copy()
            all_eval_absa = None
            try:
                all_eval_absa = pd.read_csv("absa_evaluation_100_samples.csv")
                all_eval_absa["stars"] = pd.to_numeric(all_eval_absa["stars"], errors="coerce")
            except:
                pass

            n_flagged = len(flagged)
            n_total_absa = len(all_eval_absa) if all_eval_absa is not None else None

            # KPI row
            fa1,fa2,fa3,fa4 = st.columns(4)
            for col,val,lbl,sub in [
                (fa1, f"{n_flagged:,}", "Flagged Cases", "signal-label mismatch"),
                (fa2, f"{flagged['aspect'].nunique() if 'aspect' in flagged.columns else 'N/A'}", "Aspects Affected", "across all 5 aspects"),
                (fa3, f"{n_total_absa:,}" if n_total_absa else "N/A", "Eval Sample Size", "total ABSA evaluated"),
                (fa4, f"{n_flagged/n_total_absa*100:.1f}%" if n_total_absa else "N/A", "Flagged Rate", "overall"),
            ]:
                with col:
                    st.markdown(
                        f'''<div class="kpi-card">
                            <div class="kpi-value">{val}</div>
                            <div class="kpi-label">{lbl}</div>
                            <div class="kpi-delta">{sub}</div>
                        </div>''', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            ab1, ab2 = st.columns(2)

            # Flagged cases by aspect + predicted label heatmap
            with ab1:
                if "aspect" in flagged.columns and "predicted" in flagged.columns:
                    flag_piv = flagged.groupby(["aspect","predicted"]).size().unstack(fill_value=0)
                    flag_piv.index = [a.capitalize() for a in flag_piv.index]
                    fig = px.imshow(
                        flag_piv, text_auto=True, color_continuous_scale="OrRd",
                        title="Flagged Cases by Aspect × Predicted Label",
                        labels=dict(x="Predicted Label", y="Aspect", color="Count")
                    )
                    fig.update_layout(**THEME, height=320)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Expected columns 'aspect' and 'predicted' not found in flagged CSV.")

            # Flagged rate per aspect bar
            with ab2:
                if all_eval_absa is not None and "aspect" in flagged.columns and "aspect" in all_eval_absa.columns:
                    rate_rows = []
                    for asp in ASPECTS:
                        tot = len(all_eval_absa[all_eval_absa["aspect"]==asp])
                        flg = len(flagged[flagged["aspect"]==asp])
                        if tot > 0:
                            rate_rows.append({"Aspect": asp.capitalize(),
                                              "Flagged": flg,
                                              "Total": tot,
                                              "Rate (%)": round(flg/tot*100,1)})
                    if rate_rows:
                        rdf = pd.DataFrame(rate_rows).sort_values("Rate (%)", ascending=True)
                        fig = go.Figure(go.Bar(
                            y=rdf["Aspect"], x=rdf["Rate (%)"], orientation="h",
                            marker_color=C["neg"],
                            text=[f"{v:.1f}%" for v in rdf["Rate (%)"]],
                            textposition="outside"
                        ))
                        fig.update_layout(**THEME,
                                          title="Flagged Rate per Aspect",
                                          xaxis_title="% Flagged",
                                          xaxis=dict(range=[0, rdf["Rate (%)"].max()*1.4]),
                                          height=320)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback: just show count per aspect
                    if "aspect" in flagged.columns:
                        fc = flagged["aspect"].value_counts().reset_index()
                        fc.columns = ["Aspect","Count"]
                        fc["Aspect"] = fc["Aspect"].str.capitalize()
                        fig = go.Figure(go.Bar(
                            x=fc["Aspect"], y=fc["Count"], marker_color=C["neg"],
                            text=fc["Count"], textposition="outside"
                        ))
                        fig.update_layout(**THEME, title="Flagged Cases per Aspect", height=320)
                        st.plotly_chart(fig, use_container_width=True)

            # Star-rating distribution in flagged vs correct
            if "stars" in flagged.columns:
                star_col1, star_col2 = st.columns(2)
                with star_col1:
                    star_cnt = flagged["stars"].value_counts().sort_index().reset_index()
                    star_cnt.columns = ["Stars","Count"]
                    mid_pct = len(flagged[flagged["stars"].isin([3,4])])/len(flagged)*100 if len(flagged) else 0
                    fig = go.Figure(go.Bar(
                        x=star_cnt["Stars"].astype(str), y=star_cnt["Count"],
                        marker_color=[C["neg"],C["neg"],C["neu"],C["neu"],C["pos"]],
                        text=star_cnt["Count"], textposition="outside"
                    ))
                    fig.update_layout(**THEME, title="Star-Rating Distribution - Flagged Cases",
                                      xaxis_title="Stars", yaxis_title="Count", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                with star_col2:
                    st.markdown("**Flagged cases at 3–4 stars (plausibly mixed-sentiment):**")
                    mid_n = len(flagged[flagged["stars"].isin([3,4])])
                    st.markdown(f"""
                    | Metric | Value |
                    |---|---|
                    | Mean stars - flagged | {flagged["stars"].mean():.2f} |
                    | Flagged at 3–4 stars | {mid_n} ({mid_pct:.1f}%) |
                    | Total flagged | {n_flagged} |
                    """
                    )
                    ibox(
                        f"{mid_pct:.0f}% of flagged cases occur at 3–4 stars - mixed-sentiment reviews where the star proxy is ambiguous, so many may not be clear model errors. "
                        "This is indicative only: no manually annotated ABSA ground truth was available.", "neu"
                    )

            # Sample flagged records per aspect
            if "aspect" in flagged.columns and "predicted" in flagged.columns and "text" in flagged.columns:
                st.markdown("**Sample Flagged Predictions by Aspect**")
                asp_sel = st.selectbox("Select aspect", [a.capitalize() for a in ASPECTS], key="absa_err_asp")
                sub_f = flagged[flagged["aspect"]==asp_sel.lower()]
                if len(sub_f) == 0:
                    st.info(f"No flagged cases for {asp_sel}.")
                else:
                    for _, row in sub_f.head(5).iterrows():
                        stars_val = int(row["stars"]) if pd.notna(row.get("stars")) else "?"
                        pred_col = C["neg"] if str(row["predicted"])=="Negative" else C["pos"] if str(row["predicted"])=="Positive" else C["neu"]
                        st.markdown(
                            f'''<div class="review-card" style="border-left:4px solid {pred_col};padding:10px 14px;margin:6px 0;background:#fafafa;border-radius:6px">
                                <span style="font-size:11px;color:#888">
                                    Predicted: <b>{row["predicted"]}</b> | Stars: {"⭐"*stars_val} ({stars_val})
                                </span><br>
                                <span style="font-size:13px">{str(row["text"])[:280]}…</span>
                            </div>''', unsafe_allow_html=True)

        # -- PART Sample Validation -----------------------
        st.markdown('<div class="section-header">Signal-correctness by Aspect with 95% Wilson CI (proxy)</div>', unsafe_allow_html=True)

        if absa_eval_ci_df is None:
            st.warning("absa_eval_ci.csv not found. Run the ABSA CI cell in the notebook first.")
        else:
            # -- Overall KPI cards --
            overall = absa_eval_ci_df[absa_eval_ci_df["aspect"] == "overall"].iloc[0]
            aspect_rows = absa_eval_ci_df[absa_eval_ci_df["aspect"] != "overall"]

            c1,c2,c3,c4 = st.columns(4)
            for col,val,lbl,delta in [
                (c1, f"{int(overall['n_total'])}", "Total Samples", "n varies by aspect"),
                (c2, f"{overall['accuracy_pct']:.1f}%", "Overall signal-correctness", f"{int(overall['k_correct'])}/{int(overall['n_total'])} correct"),
                (c3, f"{overall['ci_lower_pct']:.1f}%", "95% CI Lower", "Wilson score"),
                (c4, f"{overall['ci_upper_pct']:.1f}%", "95% CI Upper", "Wilson score"),
            ]:
                with col:
                    st.markdown(f'''<div class="kpi-card">
                        <div class="kpi-value">{val}</div>
                        <div class="kpi-label">{lbl}</div>
                        <div class="kpi-delta">{delta}</div>
                    </div>''', unsafe_allow_html=True)

            st.markdown('<div class="page-subtitle">100 randomly sampled review-aspect predictions - star-rating alignment used as a proxy signal - Wilson score CI</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns([3, 2])
            with col_a:
                asp_colors = {"food":"#3498db","service":"#e74c3c","ambience":"#2ecc71","price":"#e67e22","location":"#9b59b6"}
                fig = go.Figure()
                for _, row in aspect_rows.iterrows():
                    asp = row["aspect"]
                    lo_str = f"{row['ci_lower_pct']:.1f}"
                    hi_str = f"{row['ci_upper_pct']:.1f}"
                    fig.add_trace(go.Bar(
                        name=asp.capitalize(),
                        x=[asp.capitalize()],
                        y=[row["accuracy_pct"]],
                        marker_color=asp_colors.get(asp, C["blue"]),
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=[row["ci_upper_pct"] - row["accuracy_pct"]],
                            arrayminus=[row["accuracy_pct"] - row["ci_lower_pct"]],
                            color="#444", thickness=2, width=8,
                        ),
                        text=[f"{row['accuracy_pct']:.1f}%<br>[{lo_str}–{hi_str}]"],
                        textposition="outside",
                        showlegend=False,
                    ))
                # Overall reference line
                fig.add_hline(
                    y=overall["accuracy_pct"],
                    line_dash="dash", line_color="gray", opacity=0.6,
                    annotation_text=f"Overall {overall['accuracy_pct']:.1f}%",
                    annotation_position="top right",
                )
                fig.update_layout(
                    **THEME,
                    title=f"ABSA Signal-Correctness per Aspect (n differs by aspect, 95% Wilson CI)",
                    yaxis=dict(range=[0, 115], title="Signal-correctness (%)"),
                    xaxis_title="Aspect",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                # Summary table
                tbl = aspect_rows[["aspect","k_correct","n_total","accuracy_pct","ci_lower_pct","ci_upper_pct"]].copy()
                tbl.columns = ["Aspect","Correct","N","Signal-correct %","CI Lower %","CI Upper %"]
                tbl["Aspect"] = tbl["Aspect"].str.capitalize()
                tbl["95% CI"] = tbl.apply(lambda r: f"[{r['CI Lower %']:.1f}–{r['CI Upper %']:.1f}]", axis=1)
                st.dataframe(tbl[["Aspect","Correct","N","Signal-correct %","95% CI"]],
                             use_container_width=True,
                             hide_index=True,
                             column_config={"Signal-correct %": st.column_config.NumberColumn(format="%.1f%%")}
                             )


        # -- PART 2: Mann-Kendall Trend Test per Aspect -----------------------
        st.markdown("---")
        st.markdown('<div class="section-header">Mann-Kendall Trend Test per Aspect</div>', unsafe_allow_html=True)
        st.caption(
            "Kendall's τ-based non-parametric test on monthly negative-sentiment rates per aspect. "
            "Computed live from absa_results.pkl. Threshold: p < 0.05."
        )

        if absa_df is None:
            st.warning("absa_results.pkl not found. Run the ABSA notebook cells and save the pickle.")
        else:
            try:
                import pymannkendall as mk
            except ImportError:
                import subprocess, sys
                subprocess.check_call([sys.executable,"-m","pip","install","pymannkendall","-q"])
                import pymannkendall as mk

            _adf = absa_df.copy()
            _adf["date"]  = pd.to_datetime(_adf["date"], errors="coerce")
            _adf["month"] = _adf["date"].dt.to_period("M")

            # Monthly negative-sentiment rate per aspect
            monthly_neg = {}
            for asp in ASPECTS:
                col = f"absa_{asp}"
                if col not in _adf.columns:
                    continue
                m = (_adf.groupby("month")
                         .apply(lambda g: (g[col]=="Negative").mean()*100))
                monthly_neg[asp] = m

            trend_df_mk = pd.DataFrame(monthly_neg)

            mk_rows = []
            for asp in ASPECTS:
                if asp not in trend_df_mk.columns:
                    continue
                series = trend_df_mk[asp].dropna().values
                if len(series) < 4:
                    mk_rows.append({"Aspect": asp.capitalize(), "Trend": "Skipped (n<4)",
                                    "p-value": None, "Tau (τ)": None,
                                    "Slope (pts/month)": None, "Months": len(series), "Significant": False})
                    continue
                res = mk.original_test(series)
                mk_rows.append({
                    "Aspect": asp.capitalize(),
                    "Trend": res.trend.capitalize(),
                    "p-value": round(float(res.p), 4),
                    "Tau (τ)": round(float(res.Tau), 3),
                    "Slope (pts/month)": round(float(res.slope), 4),
                    "Months": len(series),
                    "Significant": res.p < 0.05
                })
            mk_df = pd.DataFrame(mk_rows)

            # -- Benjamini-Hochberg FDR correction across the 5 aspect tests --
            # Five simultaneous tests were run above; correct for multiple comparisons
            # rather than treating each raw p < 0.05 as independently significant.
            _testable = mk_df["p-value"].notna()
            if _testable.sum() > 0:
                try:
                    from statsmodels.stats.multitest import multipletests
                except ImportError:
                    import subprocess, sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels", "-q"])
                    from statsmodels.stats.multitest import multipletests
                _reject, _p_adj, _, _ = multipletests(
                    mk_df.loc[_testable, "p-value"].values, alpha=0.05, method="fdr_bh"
                )
                mk_df.loc[_testable, "p-value (FDR)"] = _p_adj.round(4)
                mk_df.loc[_testable, "Significant (FDR)"] = _reject
                mk_df["p-value (FDR)"] = mk_df["p-value (FDR)"].astype(object)
                mk_df["Significant (FDR)"] = mk_df["Significant (FDR)"].fillna(False)
            else:
                mk_df["p-value (FDR)"] = None
                mk_df["Significant (FDR)"] = False

            # KPI: how many significant trends (FDR-corrected is the primary figure;
            # raw is shown as context since it is what the uncorrected table displays)
            n_sig_mk = mk_df["Significant (FDR)"].sum() if len(mk_df) else 0
            n_sig_raw = mk_df["Significant"].sum() if len(mk_df) else 0
            mka,mkb,mkc = st.columns(3)
            for col,val,lbl,sub in [
                (mka, f"{len(mk_df)}", "Aspects Tested", "food, service, ambience, price, location"),
                (mkb, f"{n_sig_mk}", "Significant Trends (FDR-corrected)", f"{n_sig_raw} of {len(mk_df)} before correction, p < 0.05"),
                (mkc, f"{len(trend_df_mk)}", "Months Covered", "in ABSA sample"),
            ]:
                with col:
                    st.markdown(
                        f'''<div class="kpi-card">
                            <div class="kpi-value">{val}</div>
                            <div class="kpi-label">{lbl}</div>
                            <div class="kpi-delta">{sub}</div>
                        </div>''', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            mk1, mk2 = st.columns(2)

            # Summary table with colour coding (FDR-corrected significance drives the colour,
            # since the raw per-aspect p-value alone overstates confidence across 5 simultaneous tests)
            with mk1:
                def _sig_colour(row):
                    if row["Significant (FDR)"]:
                        return ["background-color:#d4edda"]*len(row)
                    return [""]*len(row)
                disp_mk = mk_df[["Aspect","Trend","p-value","p-value (FDR)","Tau (τ)","Slope (pts/month)","Months","Significant (FDR)"]].copy()
                disp_mk["Significant (FDR)"] = disp_mk["Significant (FDR)"].map({True:"✅ Yes", False:"❌ No"})
                st.markdown("**Mann-Kendall Results Table**")
                st.caption("p-value (FDR) is the Benjamini-Hochberg-adjusted p-value across the 5 aspect tests; significance is assessed on the corrected value.")
                st.dataframe(
                    disp_mk.style.apply(_sig_colour, axis=1),
                    use_container_width=True, hide_index=True
                )

            # Monthly trend line chart
            with mk2:
                if not trend_df_mk.empty:
                    fig = go.Figure()
                    palette = [C["pos"],C["neg"],C["neu"],C["blue"],C["purple"]]
                    for i,asp in enumerate(ASPECTS):
                        if asp not in trend_df_mk.columns: continue
                        series_plot = trend_df_mk[asp].dropna()
                        row_mk = mk_df[mk_df["Aspect"]==asp.capitalize()]
                        sig_mk = bool(row_mk["Significant (FDR)"].values[0]) if len(row_mk) else False
                        fig.add_trace(go.Scatter(
                            x=[str(m) for m in series_plot.index],
                            y=series_plot.values,
                            mode="lines+markers",
                            name=asp.capitalize() + (" *" if sig_mk else ""),
                            line=dict(color=palette[i%5], width=2 if sig_mk else 1,
                                      dash="solid" if sig_mk else "dot"),
                            marker=dict(size=5)
                        ))
                    fig.update_layout(
                        **THEME,
                        title="Monthly Negative-Sentiment Rate per Aspect (★ = significant trend)",
                        xaxis_title="Month", yaxis_title="Negative Rate (%)",
                        legend=dict(title="Aspect (* = sig.)"),
                        xaxis=dict(tickangle=30),
                        height=380
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Tau bar chart
            if not mk_df.empty and mk_df["Tau (τ)"].notna().any():
                tau_df = mk_df.dropna(subset=["Tau (τ)"]).sort_values("Tau (τ)")
                fig = go.Figure(go.Bar(
                    y=tau_df["Aspect"],
                    x=tau_df["Tau (τ)"],
                    orientation="h",
                    marker_color=[C["neg"] if v > 0 else C["pos"] for v in tau_df["Tau (τ)"]],
                    text=[f"{v:+.3f}" for v in tau_df["Tau (τ)"]],
                    textposition="outside"
                ))
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    **THEME,
                    title="Kendall's τ per Aspect (positive = worsening trend in negative sentiment)",
                    xaxis_title="Kendall's τ",
                    height=280
                )
                st.plotly_chart(fig, use_container_width=True)

            # Insight box
            ibox(f"Overall signal-based correctness: <b>{overall['accuracy_pct']:.1f}%</b> "
                 f"({int(overall['k_correct'])}/{int(overall['n_total'])} predictions aligned with the star-rating-derived signal). "
                 f"95% Wilson CI: [{overall['ci_lower_pct']:.1f}%–{overall['ci_upper_pct']:.1f}%]. "
                 f"Signal-correctness differs by aspect, but per-aspect samples are small and confidence intervals overlap.", "pos")
            if n_sig_mk == 0:
                ibox(
                    "No aspect shows a statistically significant trend in negative-sentiment rate (all p ≥ 0.05). "
                    "Month-to-month variation is consistent with random noise given the small per-month ABSA sample sizes - this does not mean sentiment is stable, but that the 2,000-review ABSA sample lacks the statistical power to detect gradual trends.",
                    "neu"
                )
            else:
                sig_aspects = mk_df[mk_df["Significant (FDR)"]]["Aspect"].tolist()
                ibox(
                    f"{n_sig_mk} aspect(s) remain statistically significant after Benjamini-Hochberg FDR correction for the 5 simultaneous tests: <b>{', '.join(sig_aspects)}</b>. "
                    "These trends are exploratory: monthly counts in the 2,000-review sample are sparse, so they should not by themselves drive business recommendations.", "neu"
                )

    #LDA Topic Model
    with tab3:
        st.markdown('<div class="section-header">LDA Topic Model Evaluation</div>', unsafe_allow_html=True)

        n_topics_kpi = get_kpi(results_summary_df, "LDA", "KPI_N_Topics", None)
        lda_coh_kpi = get_kpi(results_summary_df, "LDA", "Coherence", None)
        bertopic_coh = get_kpi(results_summary_df, "BERTopic", "Coherence", None)
        bertopic_cov = get_kpi(results_summary_df, "BERTopic", "KPI_Coverage_Pct", None)
        proxy_align = get_kpi(results_summary_df, "LDA", "KPI_Proxy_GT_Alignment", None)
        proxy_n_cats = get_kpi(results_summary_df, "LDA", "KPI_Proxy_GT_NumCategories", None)

        c1,c2,c3,c4 = st.columns(4)
        for col,val,lbl,delta in [
            (c1,fmt_int(n_topics_kpi),"Optimal Topics","Grid search"),
            (c2,fmt_score(lda_coh_kpi),"C_V Coherence","Final result"),
            (c3,"15","Training Passes","Final model"),
            (c4,fmt_pct_raw(proxy_align),"Category-Topic Alignment",f"{fmt_int(proxy_n_cats)} categories"),
        ]:
            with col:
                st.markdown(f'''<div class="kpi-card">
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-label">{lbl}</div>
                    <div class="kpi-delta">{delta}</div>
                </div>''',unsafe_allow_html=True)

        col1,col2 = st.columns(2)
        with col1:
            tuning_json = get_kpi(results_summary_df, "LDA", "TuningHistory_JSON", None)
            if tuning_json is not None:
                tuning_df = pd.DataFrame(json.loads(tuning_json))
            else:
                tuning_df = None

            if tuning_df is None or tuning_df.empty:
                st.warning("LDA tuning history not found in results_summary.csv. Run the notebook's grid-search and export cells first.")
            else:
                fig = go.Figure(
                    go.Bar(
                        x=tuning_df["Configuration"],
                        y=tuning_df["C_V Score"],
                        marker_color=[C["neg"] if "Rejected" in d else C["pos"] if d in ["Improved","Final"] else C["neu"] for d in tuning_df["Decision"]],
                        text=tuning_df["C_V Score"].round(4),
                        textposition="outside")
                    )
                fig.update_layout(
                    **THEME,
                    title="LDA Hyperparameter Tuning",
                    yaxis=dict(range=[0.4, max(0.65, tuning_df["C_V Score"].max()*1.1)],title="C_V Score"),
                    xaxis=dict(tickangle=30))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if lda_coh_kpi is None or bertopic_coh is None:
                st.warning("Coherence scores not found in results_summary.csv.")
            else:
                cdf = pd.DataFrame({"Model":["LDA","BERTopic"],
                    "C_V Coherence":[lda_coh_kpi, bertopic_coh],
                    "Coverage (%)":[100.0, bertopic_cov if bertopic_cov is not None else None],
                    "Selected":["Yes","No"]})
                fig = go.Figure(
                    go.Bar(
                        x=cdf["Model"],
                        y=cdf["C_V Coherence"],
                        marker_color=[C["pos"],C["neg"]],
                        text=cdf["C_V Coherence"].round(4),
                        textposition="outside")
                    )

                fig.update_layout(
                    **THEME,
                    title="LDA vs BERTopic Comparison",
                    yaxis=dict(range=[0, max(0.7, float(lda_coh_kpi)*1.1)],title="C_V Score")
                    )
                st.plotly_chart(fig, use_container_width=True)

        # Topic table from topic_results.csv (topic_id, keywords, size)
        if topic_results_df is not None:
            idf = topic_results_df.copy()
            idf["Topic"] = idf["topic_id"].apply(lambda i: f"Topic {i+1}")
            idf = idf.rename(columns={"label":"Label","keywords":"Top Keywords","size":"Size"})
            idf = idf[["Topic","Label","Top Keywords","Size"]]
            st.dataframe(idf, use_container_width=True, hide_index=True)
        else:
            st.warning("topic_results.csv not found. Run the notebook's export cells first.")

        if lda_coh_kpi is not None and bertopic_coh is not None:
            ibox(f"LDA {'scored higher than' if lda_coh_kpi > bertopic_coh else 'did not score higher than'} BERTopic on coherence ({fmt_score(lda_coh_kpi)} vs {fmt_score(bertopic_coh)}) and coverage (100% vs {fmt_pct_raw(bertopic_cov)}). Category-Topic Alignment Analysis achieved {fmt_pct_raw(proxy_align)} alignment across {fmt_int(proxy_n_cats)} business categories.","neu")
