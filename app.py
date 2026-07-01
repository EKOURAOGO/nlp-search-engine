"""
app.py — Moteur de Recherche Sémantique
TF-IDF + BM25 + Hybride · Politiques publiques & Emploi Data
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from search_engine import SearchEngine, load_engine_from_file

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Moteur de Recherche Sémantique",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.stApp { background-color: #0f1117; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e2130, #252a3d);
    border: 1px solid #2d3550; border-radius: 12px; padding: 16px 20px;
}
[data-testid="stMetricValue"] { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
h1 { color: #e2e8f0 !important; font-weight: 800 !important; }
h2, h3 { color: #cbd5e1 !important; font-weight: 600 !important; }
[data-testid="stSidebar"] { background-color: #141624 !important; border-right: 1px solid #2d3550; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
button[data-baseweb="tab"] { color: #94a3b8 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #60a5fa !important; border-bottom-color: #60a5fa !important; }
.result-card {
    background: linear-gradient(135deg, #1a1f2e, #1e2540);
    border: 1px solid #2d3550; border-left: 3px solid #60a5fa;
    border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;
}
.result-card:hover { border-left-color: #34d399; }
.score-badge {
    display: inline-block; background: #1e3a5f; color: #60a5fa;
    border-radius: 999px; padding: 2px 10px; font-size: 0.78rem; font-weight: 700;
}
.cat-badge {
    display: inline-block; border-radius: 6px;
    padding: 2px 8px; font-size: 0.72rem; font-weight: 600; margin-left: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────

CAT_COLORS = {
    "data_emploi":    ("#1d4ed8", "#dbeafe"),
    "sante_publique": ("#065f46", "#d1fae5"),
    "logement":       ("#92400e", "#fef3c7"),
    "education":      ("#7c3aed", "#ede9fe"),
    "economie":       ("#9f1239", "#ffe4e6"),
}

CAT_LABELS = {
    "data_emploi":    "Data & Emploi",
    "sante_publique": "Santé publique",
    "logement":       "Logement",
    "education":      "Éducation",
    "economie":       "Économie",
}

EXAMPLE_QUERIES = [
    "data scientist machine learning Python",
    "pipeline ETL Apache Spark traitement données",
    "déserts médicaux accès aux soins santé",
    "logement social HLM rénovation urbaine",
    "décrochage scolaire jeunes insertion",
    "évaluation impact politiques sociales économétrie",
    "NLP traitement texte classification",
    "minima sociaux RSA indicateurs DREES",
]

# ── Chargement ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_engine(tfidf_weight: float):
    corpus_path = Path(__file__).parent / "data" / "corpus.json"
    return load_engine_from_file(str(corpus_path), tfidf_weight=tfidf_weight)


@st.cache_data
def load_corpus_df():
    path = Path(__file__).parent / "data" / "corpus.json"
    with open(path) as f:
        docs = json.load(f)
    return pd.DataFrame(docs)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding:24px 0 8px 0">
  <h1 style="margin:0;font-size:2rem">🔎 Moteur de Recherche Sémantique</h1>
  <p style="color:#64748b;margin:4px 0 0 0;font-size:0.95rem">
    TF-IDF cosinus · BM25 Okapi · Score hybride configurable · 115 documents
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    tfidf_weight = st.slider("Poids TF-IDF (α)", 0.0, 1.0, 0.5, step=0.1,
                              help="α=1 → TF-IDF pur · α=0 → BM25 pur · α=0.5 → hybride")
    method = st.selectbox("Méthode de scoring", ["hybrid", "tfidf", "bm25"])
    top_k = st.slider("Nombre de résultats (top-k)", 3, 20, 8)

    st.divider()
    st.markdown("### 🗂️ Catégories")
    all_cats = ["Toutes"] + list(CAT_LABELS.values())
    selected_cat_label = st.radio("Filtrer par catégorie", all_cats)
    selected_cat = None
    if selected_cat_label != "Toutes":
        selected_cat = [k for k, v in CAT_LABELS.items() if v == selected_cat_label][0]

    st.divider()
    st.markdown("### 📖 À propos")
    st.caption("Score hybride = α × TF-IDF cosinus + (1-α) × BM25 Okapi normalisé")

# ── Chargement moteur ─────────────────────────────────────────────────────────

engine = load_engine(tfidf_weight)
stats = engine.get_stats()
df_corpus = load_corpus_df()

# ── KPIs ──────────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("📄 Documents indexés", stats["n_documents"])
c2.metric("📚 Vocabulaire", f"{stats['vocabulary_size']:,}")
c3.metric("🏷️ Catégories", len(stats["categories"]))
c4.metric("⚖️ Poids TF-IDF", f"{tfidf_weight:.1f}")

st.divider()

# ── Onglets ───────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍 Recherche", "📊 Corpus & Analyse", "⚖️ Comparaison des méthodes"])

# ── Tab 1 : Recherche ─────────────────────────────────────────────────────────

with tab1:
    col_query, col_btn = st.columns([4, 1])

    with col_query:
        query = st.text_input(
            "Requête :", placeholder="Ex: data scientist machine learning Python...",
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("🔍 Rechercher", use_container_width=True)

    # Exemples rapides
    st.markdown("**Exemples rapides :**")
    ex_cols = st.columns(4)
    for i, ex in enumerate(EXAMPLE_QUERIES[:4]):
        if ex_cols[i].button(ex[:30] + "...", key=f"ex_{i}"):
            query = ex

    if query:
        with st.spinner("Recherche en cours..."):
            results = engine.search(
                query, top_k=top_k,
                category_filter=selected_cat,
                method=method,
            )

        if results:
            st.markdown(f"**{len(results)} résultat(s)** pour *\"{query}\"*")
            st.divider()

            for i, r in enumerate(results):
                bg, fg = CAT_COLORS.get(r["category"], ("#1e2130", "#e2e8f0"))
                cat_label = CAT_LABELS.get(r["category"], r["category"])

                st.markdown(f"""
<div class="result-card">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
    <span style="color:#94a3b8;font-size:0.8rem;font-weight:600">#{i+1}</span>
    <span style="color:#e2e8f0;font-weight:700;font-size:1rem">{r['title']}</span>
    <span class="cat-badge" style="background:{bg};color:{fg}">{cat_label}</span>
    <span class="score-badge" style="margin-left:auto">{r['score']:.3f}</span>
  </div>
  <p style="color:#94a3b8;font-size:0.85rem;margin:0;line-height:1.6">{r['preview']}</p>
</div>
""", unsafe_allow_html=True)

            # Graphique des scores
            df_results = pd.DataFrame(results)
            fig = px.bar(
                df_results.head(10), x="score", y="title",
                orientation="h", color="score",
                color_continuous_scale="Blues",
                title="Scores de pertinence",
                labels={"score": "Score", "title": "Document"},
            )
            fig.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#141624",
                font_color="#cbd5e1", height=350,
                yaxis=dict(autorange="reversed"),
                margin=dict(t=40, b=10, l=10, r=10),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun résultat pour cette requête. Essayez d'autres termes.")

# ── Tab 2 : Corpus ───────────────────────────────────────────────────────────

with tab2:
    st.subheader("Répartition du corpus")

    col_a, col_b = st.columns(2)

    with col_a:
        cat_counts = df_corpus["category"].map(CAT_LABELS).value_counts().reset_index()
        cat_counts.columns = ["Catégorie", "Documents"]
        fig_pie = px.pie(
            cat_counts, values="Documents", names="Catégorie",
            color_discrete_sequence=["#60a5fa", "#34d399", "#f59e0b", "#f87171", "#a78bfa"],
            title="Répartition par catégorie",
        )
        fig_pie.update_layout(
            paper_bgcolor="#0f1117", font_color="#cbd5e1",
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        df_corpus["word_count"] = df_corpus["content"].str.split().str.len()
        fig_hist = px.histogram(
            df_corpus, x="word_count",
            color="category",
            color_discrete_sequence=["#60a5fa", "#34d399", "#f59e0b", "#f87171", "#a78bfa"],
            nbins=30, title="Distribution de la longueur des documents (mots)",
        )
        fig_hist.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#141624",
            font_color="#cbd5e1", barmode="stack",
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Parcourir le corpus")
    cat_filter_table = st.selectbox("Catégorie :", ["Toutes"] + list(CAT_LABELS.values()), key="table_cat")
    df_show = df_corpus.copy()
    if cat_filter_table != "Toutes":
        cat_key = [k for k, v in CAT_LABELS.items() if v == cat_filter_table][0]
        df_show = df_show[df_show["category"] == cat_key]

    df_show["Catégorie"] = df_show["category"].map(CAT_LABELS)
    df_show["Aperçu"] = df_show["content"].str[:100] + "..."
    st.dataframe(
        df_show[["id", "Catégorie", "title", "Aperçu"]].rename(columns={"title": "Titre", "id": "ID"}),
        use_container_width=True, hide_index=True, height=300,
    )

# ── Tab 3 : Comparaison des méthodes ─────────────────────────────────────────

with tab3:
    st.subheader("Comparaison TF-IDF vs BM25 vs Hybride sur une requête")

    ref_query = st.text_input(
        "Requête de référence :",
        value="data scientist machine learning Python",
        key="compare_query",
    )

    if ref_query:
        with st.spinner("Calcul des 3 méthodes..."):
            r_tfidf  = engine.search(ref_query, top_k=5, method="tfidf")
            r_bm25   = engine.search(ref_query, top_k=5, method="bm25")
            r_hybrid = engine.search(ref_query, top_k=5, method="hybrid")

        cols = st.columns(3)
        for col, label, results, color in [
            (cols[0], "TF-IDF cosinus", r_tfidf, "#60a5fa"),
            (cols[1], "BM25 Okapi", r_bm25, "#34d399"),
            (cols[2], f"Hybride (α={tfidf_weight})", r_hybrid, "#f59e0b"),
        ]:
            col.markdown(f"**{label}**")
            for r in results:
                col.markdown(f"""
<div style="background:#1a1f2e;border-left:3px solid {color};border-radius:8px;
            padding:10px 12px;margin-bottom:8px">
  <p style="color:#e2e8f0;font-size:0.82rem;font-weight:600;margin:0 0 2px 0">{r['title'][:45]}...</p>
  <p style="color:{color};font-size:0.78rem;font-weight:700;margin:0">Score: {r['score']:.3f}</p>
</div>
""", unsafe_allow_html=True)

        # Venn-style : documents en commun
        ids_tfidf  = {r["id"] for r in r_tfidf}
        ids_bm25   = {r["id"] for r in r_bm25}
        ids_hybrid = {r["id"] for r in r_hybrid}
        common_all = ids_tfidf & ids_bm25 & ids_hybrid
        common_tf_bm = ids_tfidf & ids_bm25

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Communs TF-IDF ∩ BM25", len(common_tf_bm))
        c2.metric("Communs les 3 méthodes", len(common_all))
        c3.metric("Uniquement dans Hybride", len(ids_hybrid - ids_tfidf - ids_bm25))
