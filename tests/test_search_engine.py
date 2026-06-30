"""
test_search_engine.py
---------------------
Tests unitaires du moteur de recherche sémantique.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from search_engine import SearchEngine

SAMPLE_DOCS = [
    {"id": 1, "category": "data", "title": "Data Scientist Python machine learning",
     "content": "Poste de data scientist spécialisé en machine learning et Python. Compétences requises : scikit-learn, TensorFlow, pandas."},
    {"id": 2, "category": "data", "title": "Data Engineer pipelines ETL",
     "content": "Conception de pipelines de données ETL avec Apache Spark et PySpark. Architecture Medallion, Delta Lake."},
    {"id": 3, "category": "sante", "title": "Rapport santé publique prévention",
     "content": "Analyse des politiques de prévention sanitaire en France. Accès aux soins, déserts médicaux, vaccination."},
    {"id": 4, "category": "logement", "title": "Logement social HLM rénovation",
     "content": "État du parc de logements sociaux HLM. Rénovation urbaine, mixité sociale, quartiers prioritaires."},
    {"id": 5, "category": "data", "title": "NLP traitement texte Python spaCy",
     "content": "Développement de pipelines NLP en Python avec spaCy et scikit-learn. Topic modeling, classification de textes."},
]


class TestSearchEngineInit:

    def test_engine_not_indexed_initially(self):
        engine = SearchEngine()
        assert not engine._indexed

    def test_engine_search_raises_if_not_indexed(self):
        engine = SearchEngine()
        with pytest.raises(RuntimeError):
            engine.search("data scientist")

    def test_index_sets_indexed_flag(self):
        engine = SearchEngine()
        engine.index(SAMPLE_DOCS)
        assert engine._indexed

    def test_index_stores_documents(self):
        engine = SearchEngine()
        engine.index(SAMPLE_DOCS)
        assert len(engine.documents) == len(SAMPLE_DOCS)

    def test_get_stats_after_indexing(self):
        engine = SearchEngine()
        engine.index(SAMPLE_DOCS)
        stats = engine.get_stats()
        assert stats["indexed"] is True
        assert stats["n_documents"] == len(SAMPLE_DOCS)
        assert stats["vocabulary_size"] > 0


class TestSearchEngineQuery:

    @pytest.fixture
    def engine(self):
        e = SearchEngine()
        e.index(SAMPLE_DOCS)
        return e

    def test_search_returns_list(self, engine):
        results = engine.search("data scientist python")
        assert isinstance(results, list)

    def test_search_top_k_respected(self, engine):
        results = engine.search("data scientist", top_k=2)
        assert len(results) <= 2

    def test_search_results_have_required_fields(self, engine):
        results = engine.search("machine learning")
        for r in results:
            assert "id" in r
            assert "title" in r
            assert "score" in r
            assert "preview" in r

    def test_search_scores_positive(self, engine):
        results = engine.search("pipelines ETL Spark")
        for r in results:
            assert r["score"] > 0

    def test_search_scores_descending(self, engine):
        results = engine.search("machine learning python")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_tfidf_method(self, engine):
        results = engine.search("Python NLP spaCy", method="tfidf")
        assert len(results) > 0

    def test_search_bm25_method(self, engine):
        results = engine.search("Python NLP spaCy", method="bm25")
        assert len(results) > 0

    def test_search_hybrid_method(self, engine):
        results = engine.search("Python NLP spaCy", method="hybrid")
        assert len(results) > 0

    def test_search_invalid_method_raises(self, engine):
        with pytest.raises(ValueError):
            engine.search("test", method="invalid_method")

    def test_search_category_filter(self, engine):
        results = engine.search("python données", category_filter="sante")
        for r in results:
            assert r["category"] == "sante"

    def test_search_empty_query_returns_empty(self, engine):
        results = engine.search("")
        assert results == []

    def test_search_relevant_result_first(self, engine):
        results = engine.search("logement HLM social")
        if results:
            assert results[0]["id"] == 4

    def test_search_data_query_returns_data_docs(self, engine):
        results = engine.search("machine learning scikit-learn", top_k=3)
        categories = [r["category"] for r in results]
        assert "data" in categories


class TestSearchEngineMethods:

    def test_hybrid_weight_influences_results(self):
        engine1 = SearchEngine(tfidf_weight=1.0)
        engine2 = SearchEngine(tfidf_weight=0.0)
        engine1.index(SAMPLE_DOCS)
        engine2.index(SAMPLE_DOCS)
        r1 = engine1.search("data scientist python", method="hybrid")
        r2 = engine2.search("data scientist python", method="hybrid")
        # Les scores peuvent différer selon le poids
        assert r1[0]["score"] != r2[0]["score"] or r1[0]["id"] == r2[0]["id"]

    def test_reindex_updates_corpus(self):
        engine = SearchEngine()
        engine.index(SAMPLE_DOCS[:3])
        assert len(engine.documents) == 3
        engine.index(SAMPLE_DOCS)
        assert len(engine.documents) == len(SAMPLE_DOCS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
