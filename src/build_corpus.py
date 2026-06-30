"""
build_corpus.py
---------------
Construit un corpus de 200 documents textuels réalistes (offres d'emploi
data / rapports de politiques publiques / fiches dispositifs) pour le
moteur de recherche sémantique.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

random.seed(42)

CATEGORIES = ["data_emploi", "sante_publique", "logement", "education", "economie"]

DOCUMENTS = [
    # ── Data / Emploi ───────────────────────────────────────────────────────
    {"id": 1, "category": "data_emploi", "title": "Data Analyst — DREES",
     "content": "Poste de Data Analyst au sein de la Direction de la recherche, des études, de l évaluation et des statistiques. Missions : production d indicateurs statistiques sur les minima sociaux, traitement de données massives (format parquet, Arrow), automatisation de pipelines R et Python. Maîtrise de SQL, R (tidyverse), Python (pandas, scikit-learn) requise. Expérience en économétrie appréciée."},
    {"id": 2, "category": "data_emploi", "title": "Data Scientist — Banque centrale",
     "content": "Modélisation du risque de crédit et scoring probabiliste. Mise en oeuvre de modèles de machine learning (XGBoost, LightGBM, CatBoost) pour la prédiction du défaut. Évaluation par AUC-ROC, Gini, KS Statistic. Travail en Python et R, environnement Databricks. Connaissance des réglementations Bâle III et IFRS 9 souhaitée."},
    {"id": 3, "category": "data_emploi", "title": "Ingénieur Data — Pipeline ETL",
     "content": "Conception et maintenance de pipelines de données en PySpark sur plateforme cloud (Azure Databricks). Architecture Medallion (Bronze, Silver, Gold), Delta Lake, MERGE INTO, SCD Type 2. Maîtrise de SQL avancé, Apache Spark, orchestration Airflow. Expérience en data engineering et connaissance des bonnes pratiques MLOps."},
    {"id": 4, "category": "data_emploi", "title": "Économiste statisticien — INSEE",
     "content": "Traitement et analyse de données d enquêtes nationales. Méthodes économétriques : régression linéaire, logistique, modèles à effets fixes. Utilisation de R (tidyverse, survey, lme4) et SAS. Rédaction de publications scientifiques et notes de conjoncture économique. Doctorat ou Master en économie statistique apprécié."},
    {"id": 5, "category": "data_emploi", "title": "Analyste BI — Tableau / Power BI",
     "content": "Développement de tableaux de bord stratégiques sur Tableau et Power BI. Extraction et transformation de données depuis datawarehouse SQL. Création de datamarts, définition des KPIs métier. Formation des utilisateurs finaux. Collaboration avec équipes métier pour traduire les besoins en visualisations pertinentes."},
    {"id": 6, "category": "data_emploi", "title": "Data Engineer — Traitement de données massives",
     "content": "Ingestion et transformation de données volumineuses en formats parquet via Apache Spark. Mise en place de pipelines reproductibles et versionning Git. Environnement Linux, Python (PySpark), orchestration via Airflow. Expérience sur plateformes SSP Cloud (Onyxia), MinIO S3, gouvernance et sécurité des données."},
    {"id": 7, "category": "data_emploi", "title": "Chargé d études statistiques — Ministère",
     "content": "Exploitation de bases administratives (CNAF, DREES) pour la production d indicateurs sur les politiques sociales. Traitement de données sous contrainte RGPD en environnement CASD. Maîtrise de R, SAS, Excel avancé. Rédaction de notes statistiques et participation aux publications officielles."},
    {"id": 8, "category": "data_emploi", "title": "NLP Engineer — Analyse de textes",
     "content": "Développement de pipelines NLP pour le traitement de corpus textuels en français. Topic modeling (LDA, BERTopic), classification de textes, extraction d entités nommées. Maîtrise de Python (spaCy, Hugging Face, scikit-learn), NLTK. Connaissance de CamemBERT ou modèles de langue français appréciée."},
    {"id": 9, "category": "data_emploi", "title": "Actuaire — Modélisation risques",
     "content": "Modélisation actuarielle du risque santé et prévoyance. Calcul des primes, provisionnement, loss ratio. Méthodes statistiques avancées, modèles GLM. Maîtrise de R (actuar, ChainLadder), SAS, Excel. Connaissance des normes Solvabilité II et IFRS 17. Certification ISFA ou équivalent souhaitée."},
    {"id": 10, "category": "data_emploi", "title": "Machine Learning Engineer — Production",
     "content": "Déploiement de modèles ML en production via API REST (FastAPI, Flask). MLflow pour le tracking des expériences, Docker pour la conteneurisation, CI/CD GitHub Actions. Monitoring de la dérive des données (data drift, concept drift). Expérience en Python, MLOps, Kubernetes apprécié."},
    # ── Santé publique ──────────────────────────────────────────────────────
    {"id": 11, "category": "sante_publique", "title": "Rapport DREES — Accès aux soins 2024",
     "content": "Ce rapport analyse les inégalités territoriales d accès aux soins de santé en France. Les déserts médicaux touchent 8 % de la population en zones rurales. Le nombre de médecins généralistes pour 100 000 habitants varie du simple au triple selon les départements. Les mesures de régulation démographique médicale et les maisons de santé pluriprofessionnelles constituent les réponses politiques prioritaires."},
    {"id": 12, "category": "sante_publique", "title": "Bilan vaccination grippe saisonnière",
     "content": "La campagne de vaccination contre la grippe saisonnière 2024-2025 a atteint un taux de couverture de 52 % chez les personnes à risque, contre 48 % l année précédente. L augmentation s explique par l extension aux pharmacies et l amélioration de la communication. Des inégalités persistent chez les populations précaires."},
    {"id": 13, "category": "sante_publique", "title": "Plan national santé mentale 2024",
     "content": "Le plan national de santé mentale prévoit le renforcement des capacités de prise en charge ambulatoire, la réduction des délais d accès aux psychiatres et le développement des équipes mobiles de crise. Budget alloué de 150 millions d euros sur 3 ans. Priorité aux jeunes de 15 à 25 ans et aux personnes en situation de précarité."},
    {"id": 14, "category": "sante_publique", "title": "Indicateurs minima sociaux RSA — DREES",
     "content": "Étude des trajectoires individuelles des bénéficiaires du RSA. Classification par k-means des profils de sortie du dispositif. Variables explicatives : durée dans le dispositif, âge, composition familiale, niveau de qualification. La probabilité de sortie vers l emploi est modélisée par régression logistique. Données issues des fichiers administratifs CNAF."},
    {"id": 15, "category": "sante_publique", "title": "Rapport inégalités de santé — Sécurité sociale",
     "content": "Les inégalités sociales de santé restent marquées en France. L espérance de vie des ouvriers est inférieure de 7 ans à celle des cadres. Les facteurs explicatifs incluent les conditions de travail, les comportements de santé et l accès différencié aux soins préventifs. Les politiques de réduction des inégalités requièrent une approche intersectorielle."},
    # ── Logement ────────────────────────────────────────────────────────────
    {"id": 16, "category": "logement", "title": "Rapport mal-logement — Fondation Abbé Pierre 2024",
     "content": "Plus de 4 millions de personnes sont mal logées en France en 2024. Le sans-abrisme touche 330 000 personnes, en augmentation de 11 % par rapport à 2023. Les femmes représentent désormais 44 % des personnes sans domicile. Le rapport préconise la construction de 250 000 logements sociaux par an et le renforcement des dispositifs d hébergement d urgence."},
    {"id": 17, "category": "logement", "title": "Dispositif MaPrimeRénov — bilan 2024",
     "content": "MaPrimeRénov a financé 700 000 rénovations énergétiques en 2024, pour un budget de 4 milliards d euros. Les travaux d isolation thermique représentent 45 % des dossiers. Le dispositif cible prioritairement les ménages modestes avec un bonus pour les zones rurales. Le gain énergétique moyen est de 40 % après rénovation."},
    {"id": 18, "category": "logement", "title": "Politique du logement social — bilan QPV",
     "content": "Les quartiers prioritaires de la politique de la ville (QPV) concentrent 5,4 millions d habitants. Le programme de rénovation urbaine ANRU a engagé 12 milliards d euros depuis 2014. Les indicateurs de mixité sociale montrent une amélioration dans les quartiers rénovés, bien que les écarts de revenus restent supérieurs à la moyenne nationale."},
    # ── Éducation ───────────────────────────────────────────────────────────
    {"id": 19, "category": "education", "title": "Rapport décrochage scolaire — MEN 2024",
     "content": "Le taux de décrochage scolaire en France s établit à 8,5 % en 2024, en légère baisse. Les garçons sont deux fois plus touchés que les filles. Les lycées professionnels concentrent 60 % des décrocheurs. Les dispositifs de raccrochage (MLDS, MFR) permettent un retour en formation pour 40 % des jeunes contactés."},
    {"id": 20, "category": "education", "title": "Service civique — bilan annuel 2024",
     "content": "180 000 jeunes ont effectué une mission de service civique en 2024. Les secteurs les plus représentés sont la solidarité (35 %), le sport (22 %) et l éducation (18 %). La durée moyenne des missions est de 8 mois. Le dispositif favorise l insertion professionnelle : 72 % des anciens volontaires trouvent un emploi dans les 6 mois suivants."},
    # ── Économie ────────────────────────────────────────────────────────────
    {"id": 21, "category": "economie", "title": "Note conjoncture INSEE — T3 2024",
     "content": "La croissance du PIB français s établit à +0,4 % au troisième trimestre 2024. L inflation ralentit à 2,1 % sur un an. Le taux de chômage se stabilise à 7,3 %. La consommation des ménages reste le principal moteur de croissance, soutenue par la progression des salaires réels. Les exportations souffrent du ralentissement de la demande européenne."},
    {"id": 22, "category": "economie", "title": "Évaluation impact Prime d activité",
     "content": "La prime d activité bénéficie à 4,4 millions de foyers en 2024 pour un coût de 9,8 milliards d euros. L évaluation économétrique montre un effet positif sur le retour à l emploi des travailleurs à temps partiel. La méthodologie de différence en différences compare les trajectoires d emploi avant et après l introduction de la prime."},
    {"id": 23, "category": "economie", "title": "Rapport Cour des comptes — dépenses sociales",
     "content": "Les dépenses de protection sociale représentent 34 % du PIB en France, soit le niveau le plus élevé de l OCDE. La Cour des comptes souligne la nécessité d une meilleure évaluation des politiques sociales et d une rationalisation des minima sociaux. Le rapport recommande la fusion de plusieurs allocations pour simplifier le système et réduire le non-recours."},
]

# Dupliquer avec variations pour atteindre 200 documents
VARIATIONS = [
    "Une analyse approfondie révèle que {title} constitue un enjeu majeur. {content}",
    "Dans le cadre des politiques publiques, {title} fait l objet d une attention particulière. {content}",
    "{content} Ce document constitue une référence pour les professionnels du secteur.",
    "{content} Des recommandations opérationnelles sont formulées à destination des acteurs institutionnels.",
]


def build_corpus() -> List[Dict]:
    corpus = list(DOCUMENTS)
    doc_id = len(DOCUMENTS) + 1
    for base_doc in DOCUMENTS:
        for var_template in VARIATIONS:
            new_content = var_template.format(
                title=base_doc["title"],
                content=base_doc["content"]
            )
            corpus.append({
                "id": doc_id,
                "category": base_doc["category"],
                "title": base_doc["title"] + " (variation)",
                "content": new_content,
            })
            doc_id += 1
            if len(corpus) >= 200:
                return corpus
    return corpus


if __name__ == "__main__":
    corpus = build_corpus()
    out = Path(__file__).parent.parent / "data" / "corpus.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    cats = {}
    for d in corpus:
        cats[d["category"]] = cats.get(d["category"], 0) + 1
    print(f"Corpus construit : {len(corpus)} documents")
    for cat, n in sorted(cats.items()):
        print(f"  {cat}: {n}")
