"""
app.py — Moteur de Recherche Sémantique
Dashboard pro · Design inspiré Linear/Vercel
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from search_engine import load_engine_from_file

st.set_page_config(page_title="Recherche Sémantique · NLP", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #0A0B0E !important;
    font-family: 'Inter', sans-serif !important; color: #E2E8F0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1400px; }
[data-testid="stSidebar"] { background-color: #111318 !important; border-right: 1px solid #1C1F28 !important; }
[data-testid="stSidebar"] * { color: #94A3B8 !important; font-family: 'Inter', sans-serif !important; }
.app-header { padding: 0 0 2rem 0; border-bottom: 1px solid #1C1F28; margin-bottom: 2rem; }
.app-eyebrow { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #F59E0B; margin-bottom: 0.5rem; }
.app-title { font-size: 2rem; font-weight: 700; color: #F1F5F9; line-height: 1.15; letter-spacing: -0.02em; }
.app-subtitle { font-size: 0.88rem; color: #64748B; margin-top: 0.35rem; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.9rem; margin-bottom: 2rem; }
.kpi-card { background: #111318; border: 1px solid #1C1F28; border-top: 2px solid #F59E0B;
    border-radius: 10px; padding: 1.1rem 1.4rem; }
.kpi-label { font-size: 0.68rem; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #F1F5F9; font-variant-numeric: tabular-nums; line-height: 1; }
.kpi-sub { font-size: 0.72rem; color: #475569; margin-top: 0.3rem; }
.section-title { font-size: 0.72rem; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin: 1.5rem 0 0.9rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #1C1F28; }
.result-card { background: #111318; border: 1px solid #1C1F28;
    border-left: 3px solid #F59E0B; border-radius: 0 10px 10px 0;
    padding: 1.1rem 1.4rem; margin-bottom: 0.7rem; transition: border-color 0.15s; }
.result-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem; }
.result-rank { font-size: 0.65rem; font-weight: 700; color: #475569;
    font-variant-numeric: tabular-nums; min-width: 1.4rem; }
.result-title { font-size: 0.92rem; font-weight: 600; color: #F1F5F9; flex: 1; }
.result-score { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    font-weight: 500; color: #F59E0B; background: #1C1610;
    border: 1px solid #3D2E0A; border-radius: 5px; padding: 2px 8px; }
.result-cat { font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; padding: 2px 7px; border-radius: 4px; }
.result-preview { font-size: 0.82rem; color: #64748B; line-height: 1.65; }
.quick-query { display: inline-block; background: #111318; border: 1px solid #1C1F28;
    border-radius: 6px; padding: 4px 10px; margin: 2px; font-size: 0.75rem;
    color: #94A3B8; cursor: pointer; transition: all 0.15s; }
.search-input-wrap { position: relative; }
[data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #1C1F28 !important; }
[data-baseweb="tab"] { background: transparent !important; color: #475569 !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 0.55rem 1.1rem !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important; }
[aria-selected="true"][data-baseweb="tab"] {
    color: #E2E8F0 !important; border-bottom: 2px solid #F59E0B !important; background: transparent !important; }
[data-testid="stTextInput"] input {
    background: #111318 !important; border: 1px solid #2D3142 !important;
    color: #F1F5F9 !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.95rem !important;
    padding: 0.7rem 1rem !important; }
[data-testid="stTextInput"] input:focus { border-color: #F59E0B !important; outline: none !important; }
.compare-card { background: #111318; border: 1px solid #1C1F28;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.5rem; }
.compare-method { font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 0.6rem; }
.compare-row { display: flex; justify-content: space-between; align-items: center;
    padding: 0.4rem 0; border-bottom: 1px solid #1C1F28; }
.compare-row:last-child { border-bottom: none; }
.compare-doc { font-size: 0.78rem; color: #94A3B8; flex: 1; }
.compare-sc { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
    font-weight: 500; padding: 1px 6px; border-radius: 4px; margin-left: 0.5rem; }
</style>
""", unsafe_allow_html=True)

CAT_META = {
    "data_emploi":    {"label":"Data & Emploi",   "bg":"#1e3a8a","fg":"#93c5fd","accent":"#3b82f6"},
    "sante_publique": {"label":"Santé publique",  "bg":"#064e3b","fg":"#6ee7b7","accent":"#10b981"},
    "logement":       {"label":"Logement",        "bg":"#78350f","fg":"#fcd34d","accent":"#f59e0b"},
    "education":      {"label":"Éducation",       "bg":"#4c1d95","fg":"#c4b5fd","accent":"#8b5cf6"},
    "economie":       {"label":"Économie",        "bg":"#881337","fg":"#fda4af","accent":"#f43f5e"},
}
PALETTE = ["#6366F1","#34D399","#F59E0B","#F87171","#A78BFA"]
LAYOUT  = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111318",
               font=dict(family="Inter", color="#94A3B8", size=11),
               margin=dict(t=32,b=12,l=12,r=12),
               xaxis=dict(gridcolor="#1C1F28",linecolor="#1C1F28"))

QUICK_QUERIES = [
    "data scientist machine learning Python",
    "pipeline ETL Spark données massives",
    "déserts médicaux accès aux soins",
    "logement social HLM rénovation",
    "décrochage scolaire jeunes",
    "évaluation politiques sociales économétrie",
]

@st.cache_resource
def load_engine(w):
    return load_engine_from_file(str(Path(__file__).parent/"data"/"corpus.json"), tfidf_weight=w)

@st.cache_data
def get_corpus_df():
    with open(Path(__file__).parent/"data"/"corpus.json") as f:
        docs = json.load(f)
    return pd.DataFrame(docs)

with st.sidebar:
    st.markdown("### Paramètres")
    tfidf_w  = st.slider("Poids TF-IDF (α)", 0.0, 1.0, 0.5, 0.1)
    method   = st.selectbox("Méthode", ["hybrid","tfidf","bm25"])
    top_k    = st.slider("Résultats", 3, 15, 8)
    st.markdown("---")
    st.markdown("### Catégorie")
    cats_options = ["Toutes"] + [v["label"] for v in CAT_META.values()]
    cat_sel = st.radio("", cats_options, label_visibility="collapsed")
    cat_filter = None
    if cat_sel != "Toutes":
        cat_filter = next(k for k,v in CAT_META.items() if v["label"]==cat_sel)

engine   = load_engine(tfidf_w)
stats    = engine.get_stats()
df_corp  = get_corpus_df()

st.markdown(f"""
<div class="app-header">
  <div class="app-eyebrow">NLP · Information Retrieval</div>
  <div class="app-title">Moteur de Recherche<br>Sémantique</div>
  <div class="app-subtitle">TF-IDF cosinus · BM25 Okapi · Score hybride configurable · {stats["n_documents"]} documents indexés</div>
</div>
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-label">Documents</div><div class="kpi-value">{stats["n_documents"]}</div><div class="kpi-sub">5 catégories</div></div>
  <div class="kpi-card"><div class="kpi-label">Vocabulaire</div><div class="kpi-value">{stats["vocabulary_size"]:,}</div><div class="kpi-sub">Tokens indexés</div></div>
  <div class="kpi-card"><div class="kpi-label">Méthode</div><div class="kpi-value" style="font-size:1.3rem;padding-top:0.2rem">{method.upper()}</div><div class="kpi-sub">α = {tfidf_w:.1f}</div></div>
  <div class="kpi-card"><div class="kpi-label">Top-K</div><div class="kpi-value">{top_k}</div><div class="kpi-sub">Résultats max</div></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Recherche","Corpus","Comparaison méthodes"])

with tab1:
    query = st.text_input("", placeholder="Rechercher dans le corpus…",
                          label_visibility="collapsed")

    query_html = '<div style="margin: 0.6rem 0 1.2rem 0">'
    for qex in QUICK_QUERIES:
        query_html += f'<span class="quick-query">{qex}</span>'
    query_html += '</div>'
    st.markdown(query_html, unsafe_allow_html=True)

    if not query:
        st.markdown('<p style="color:#475569;font-size:0.85rem">Entrez une requête pour explorer le corpus.</p>', unsafe_allow_html=True)
    else:
        results = engine.search(query, top_k=top_k, category_filter=cat_filter, method=method)
        if not results:
            st.markdown('<p style="color:#475569;font-size:0.85rem">Aucun résultat. Essayez d\'autres termes.</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="font-size:0.8rem;color:#475569;margin-bottom:1rem"><strong style="color:#94A3B8">{len(results)}</strong> résultat(s) pour <em style="color:#F1F5F9">"{query}"</em></p>', unsafe_allow_html=True)

            col_res, col_chart = st.columns([3,2], gap="large")
            with col_res:
                for i, r in enumerate(results):
                    meta = CAT_META.get(r["category"], {"label":r["category"],"bg":"#1C1F28","fg":"#94A3B8","accent":"#F59E0B"})
                    st.markdown(f"""
<div class="result-card" style="border-left-color:{meta['accent']}">
  <div class="result-header">
    <span class="result-rank">#{i+1}</span>
    <span class="result-title">{r["title"]}</span>
    <span class="result-cat" style="background:{meta['bg']};color:{meta['fg']}">{meta['label']}</span>
    <span class="result-score">{r["score"]:.3f}</span>
  </div>
  <div class="result-preview">{r["preview"]}</div>
</div>""", unsafe_allow_html=True)

            with col_chart:
                df_r = pd.DataFrame(results[:8])
                short_titles = [t[:30]+"…" if len(t)>30 else t for t in df_r["title"]]
                fig = go.Figure(go.Bar(
                    x=df_r["score"], y=short_titles, orientation="h",
                    marker=dict(color="#F59E0B", opacity=0.85, cornerradius=4),
                    text=[f'{s:.3f}' for s in df_r["score"]],
                    textposition="outside", textfont=dict(color="#94A3B8", size=9),
                ))
                fig.update_layout(**LAYOUT, height=max(200, len(results)*38),
                                   showlegend=False, xaxis_showgrid=False,
                                   title="Scores de pertinence", title_font_size=12,
                                   title_font_color="#94A3B8")
                fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    ca, cb = st.columns(2, gap="large")
    with ca:
        st.markdown('<div class="section-title">Répartition par catégorie</div>', unsafe_allow_html=True)
        cat_counts = df_corp["category"].value_counts().reset_index()
        cat_counts.columns = ["category","count"]
        cat_counts["label"] = cat_counts["category"].map(lambda x: CAT_META.get(x,{}).get("label",x))
        fig = go.Figure(go.Pie(
            labels=cat_counts["label"], values=cat_counts["count"],
            hole=0.55, marker=dict(colors=PALETTE,
                                   line=dict(color="#0A0B0E", width=2)),
            textfont=dict(size=11, color="#94A3B8"),
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280,
                          font=dict(family="Inter", color="#94A3B8"),
                          margin=dict(t=20,b=20,l=20,r=20),
                          legend=dict(font=dict(color="#94A3B8"),bgcolor="rgba(0,0,0,0)"),
                          showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        st.markdown('<div class="section-title">Longueur des documents</div>', unsafe_allow_html=True)
        df_corp["words"] = df_corp["content"].str.split().str.len()
        fig = px.box(df_corp, x="category", y="words",
                     color="category", color_discrete_sequence=PALETTE)
        fig.update_layout(**LAYOUT, height=280, showlegend=False,
                          xaxis_title="", yaxis_title="Nb mots",
                          title="Distribution longueur", title_font_size=12,
                          title_font_color="#94A3B8")
        fig.update_yaxes(gridcolor="#1C1F28", linecolor="#1C1F28")
        fig.update_traces(marker=dict(color="#F59E0B", opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<div class="section-title">Comparer TF-IDF · BM25 · Hybride</div>', unsafe_allow_html=True)
    ref = st.text_input("Requête de référence :", value="data scientist machine learning Python",
                        key="cmp")
    if ref:
        r_tf  = engine.search(ref, top_k=5, method="tfidf")
        r_bm  = engine.search(ref, top_k=5, method="bm25")
        r_hy  = engine.search(ref, top_k=5, method="hybrid")

        c1, c2, c3 = st.columns(3, gap="medium")
        for col, label, results, accent in [
            (c1,"TF-IDF cosinus",r_tf,"#6366F1"),
            (c2,"BM25 Okapi",r_bm,"#34D399"),
            (c3,f"Hybride α={tfidf_w}",r_hy,"#F59E0B"),
        ]:
            html_content = f'<div class="compare-card"><div class="compare-method" style="color:{accent}">{label}</div>'
            for r in results:
                short = r["title"][:38]+"…" if len(r["title"])>38 else r["title"]
                html_content += f'<div class="compare-row"><span class="compare-doc">{short}</span><span class="compare-sc" style="background:{accent}1A;color:{accent};border:1px solid {accent}33">{r["score"]:.3f}</span></div>'
            html_content += '</div>'
            col.markdown(html_content, unsafe_allow_html=True)

        ids_tf = {r["id"] for r in r_tf}
        ids_bm = {r["id"] for r in r_bm}
        ids_hy = {r["id"] for r in r_hy}
        st.markdown("---")
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("TF-IDF ∩ BM25", len(ids_tf & ids_bm))
        cm2.metric("Communs 3 méthodes", len(ids_tf & ids_bm & ids_hy))
        cm3.metric("Uniquement hybride", len(ids_hy - ids_tf - ids_bm))
