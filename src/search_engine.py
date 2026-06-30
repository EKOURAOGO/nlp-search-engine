"""
search_engine.py
----------------
Moteur de recherche sémantique hybride :
  - TF-IDF (cosine similarity)
  - BM25 Okapi (Robertson & Spärck Jones, 1994)
  - Score hybride combinant les deux méthodes
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import preprocess


class SearchEngine:
    """
    Moteur de recherche hybride TF-IDF + BM25.

    Attributs:
        documents : Liste des documents indexés.
        tfidf_weight : Poids de TF-IDF dans le score hybride (0-1).
    """

    def __init__(self, tfidf_weight: float = 0.5):
        """
        Initialise le moteur de recherche.

        Args:
            tfidf_weight: Poids du score TF-IDF dans la combinaison hybride.
                          BM25 reçoit (1 - tfidf_weight).
        """
        self.tfidf_weight = tfidf_weight
        self.documents: List[Dict] = []
        self._processed_docs: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._bm25: Optional[BM25Okapi] = None
        self._indexed = False

    def index(self, documents: List[Dict]) -> None:
        """
        Indexe une liste de documents.

        Args:
            documents: Liste de {'id', 'title', 'content', 'category'}.
        """
        self.documents = documents
        texts = [doc["title"] + " " + doc["content"] for doc in documents]
        self._processed_docs = [preprocess(t) for t in texts]

        # Index TF-IDF
        self._vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(self._processed_docs)

        # Index BM25
        tokenized = [doc.split() for doc in self._processed_docs]
        self._bm25 = BM25Okapi(tokenized)

        self._indexed = True

    def _tfidf_scores(self, query_processed: str) -> np.ndarray:
        """
        Calcule les scores TF-IDF cosinus pour une requête.

        Args:
            query_processed: Requête prétraitée.

        Returns:
            Tableau de scores (un par document).
        """
        query_vec = self._vectorizer.transform([query_processed])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        return scores

    def _bm25_scores(self, query_processed: str) -> np.ndarray:
        """
        Calcule les scores BM25 pour une requête.

        Args:
            query_processed: Requête prétraitée.

        Returns:
            Tableau de scores normalisé (0-1).
        """
        tokens = query_processed.split()
        scores = np.array(self._bm25.get_scores(tokens), dtype=float)
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
        return scores

    def search(
        self,
        query: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
        method: str = "hybrid",
    ) -> List[Dict]:
        """
        Recherche les documents les plus pertinents pour une requête.

        Args:
            query: Requête utilisateur (texte libre).
            top_k: Nombre de résultats à retourner.
            category_filter: Si fourni, filtre les résultats par catégorie.
            method: "tfidf" | "bm25" | "hybrid".

        Returns:
            Liste triée de résultats avec scores.

        Raises:
            RuntimeError: Si le moteur n'a pas été indexé.
        """
        if not self._indexed:
            raise RuntimeError("Le moteur n'est pas indexé. Appelez index() d'abord.")

        query_processed = preprocess(query)
        if not query_processed.strip():
            return []

        if method == "tfidf":
            scores = self._tfidf_scores(query_processed)
        elif method == "bm25":
            scores = self._bm25_scores(query_processed)
        elif method == "hybrid":
            tfidf = self._tfidf_scores(query_processed)
            bm25 = self._bm25_scores(query_processed)
            scores = self.tfidf_weight * tfidf + (1 - self.tfidf_weight) * bm25
        else:
            raise ValueError(f"Méthode inconnue : {method}. Utilisez 'tfidf', 'bm25' ou 'hybrid'.")

        # Tri décroissant
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                break
            doc = self.documents[idx]
            if category_filter and doc.get("category") != category_filter:
                continue
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "category": doc.get("category", ""),
                "score": round(float(scores[idx]), 4),
                "preview": doc["content"][:200] + "...",
            })
            if len(results) >= top_k:
                break

        return results

    def get_stats(self) -> Dict:
        """Retourne les statistiques du moteur indexé."""
        if not self._indexed:
            return {"indexed": False}
        return {
            "indexed": True,
            "n_documents": len(self.documents),
            "vocabulary_size": len(self._vectorizer.vocabulary_),
            "categories": list({d.get("category", "") for d in self.documents}),
        }


def load_engine_from_file(corpus_path: str, tfidf_weight: float = 0.5) -> "SearchEngine":
    """
    Charge un moteur de recherche depuis un corpus JSON.

    Args:
        corpus_path: Chemin vers le fichier corpus.json.
        tfidf_weight: Poids TF-IDF dans le score hybride.

    Returns:
        Moteur indexé et prêt à l'emploi.
    """
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    engine = SearchEngine(tfidf_weight=tfidf_weight)
    engine.index(corpus)
    return engine
