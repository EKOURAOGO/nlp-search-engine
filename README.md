# NLP - Moteur de recherche sémantique (TF-IDF + BM25)

Moteur de recherche sur corpus textuel combinant **TF-IDF cosinus** et **BM25 Okapi** dans un score hybride configurable. Appliqué à un corpus de documents sur les politiques publiques et l'emploi data (115 documents). Architecture modulaire avec 20 tests unitaires.

---

## Fonctionnement

```
Requête utilisateur
      │
      ▼
Prétraitement (nettoyage, stopwords, tokenisation)
      │
      ├──► Score TF-IDF cosinus   ──┐
      │                              ├──► Score hybride = α·TF-IDF + (1-α)·BM25
      └──► Score BM25 Okapi      ──┘
                │
                ▼
         Ranking décroissant
                │
                ▼
         Top-K résultats (avec filtre catégorie optionnel)
```

**Pourquoi combiner TF-IDF et BM25 ?**

| Méthode | Force | Limite |
|---------|-------|--------|
| TF-IDF cosinus | Bonne sur longs documents, robuste à la longueur | Pas de saturation des termes répétés |
| BM25 Okapi | Saturation des termes, normalisation de longueur | Sensible aux paramètres k1, b |
| **Hybride** | **Combine les deux avantages** | Nécessite de calibrer α |

---

## Structure du projet

```
nlp-search-engine/
├── src/
│   ├── build_corpus.py    # Construction du corpus (115 documents)
│   ├── preprocessing.py   # Nettoyage et tokenisation
│   └── search_engine.py   # Moteur TF-IDF + BM25 + hybride
├── data/
│   └── corpus.json        # Corpus indexé
├── tests/
│   └── test_search_engine.py  # 20 tests unitaires
├── requirements.txt
└── README.md
```

---

## Utilisation

```python
from src.search_engine import load_engine_from_file

# Charger et indexer
engine = load_engine_from_file("data/corpus.json", tfidf_weight=0.6)

# Recherche hybride
results = engine.search("data scientist machine learning Python", top_k=5)

# Avec filtre par catégorie
results = engine.search("pipelines ETL", category_filter="data_emploi", method="bm25")

for r in results:
    print(f"[{r['score']:.3f}] {r['title']}")
    print(f"  {r['preview']}")
```

---

## Catégories du corpus

| Catégorie | Documents | Contenu |
|-----------|-----------|---------|
| `data_emploi` | 50 | Offres d'emploi data, fiches de poste |
| `sante_publique` | 25 | Rapports DREES, politiques sanitaires |
| `logement` | 15 | Logement social, rénovation urbaine |
| `education` | 10 | Décrochage scolaire, service civique |
| `economie` | 15 | Conjoncture INSEE, évaluation d'impact |

---

## Tests

```bash
python3 -m pytest tests/ -v
```

Sortie attendue : `20 passed`

Les tests couvrent : initialisation, indexation, recherche (3 méthodes), filtre catégorie, requête vide, tri décroissant, gestion des erreurs.

---

## Stack technique

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![BM25](https://img.shields.io/badge/rank--bm25-BM25%20Okapi-blue?style=flat-square)
![pytest](https://img.shields.io/badge/pytest-20%20tests-red?style=flat-square)

---

## Auteur

**Emmanuel KOURAOGO** 
[GitHub](https://github.com/EKOURAOGO) · [Email](mailto:ekouraogo73@gmail.com)
