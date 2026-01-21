# 🚀 Baseline Projet Veille Automatique - NLP

**Système end-to-end de collecte, analyse et synthèse d'articles techniques**

## 📋 Contenu du Projet

```
baseline_project/
├── config.json                 # Configuration (sources, modèles, etc.)
├── main.py                     # Script orchestrateur principal
├── requirements.txt            # Dépendances Python
│
├── src/                        # Code source
│   ├── news_collector.py       # Collecte articles
│   ├── text_preprocessor.py    # Prétraitement NLP
│   ├── news_classifier.py      # Classification
│   └── report_generator.py     # Génération rapport
│
├── data/                       # Données
│   ├── articles_raw.jsonl      # Articles bruts (output)
│   ├── articles_processed.jsonl# Articles traités (output)
│   ├── articles_classified.jsonl# Articles classifiés (output)
│   └── collection_errors.json  # Log d'erreurs (output)
│
├── output/                     # Résultats
│   └── veille_report.txt       # Rapport final (output)
│
└── notebooks/                  # Jupyter notebooks (optionnel)
    └── 01_exploratory_data_analysis.ipynb
```

## 🎯 Qu'est-ce qu'un Baseline?

Ce projet est un **point de départ fonctionnel** que tu **dois étendre et améliorer**:

### ✅ Ce qui est DÉJÀ IMPLÉMENTÉ (Baseline)
- ✓ Collecte HackerNews + RSS (sources partielles)
- ✓ Normalisation basique (HTML, URLs, case)
- ✓ Tokenization spaCy français
- ✓ Classification zero-shot (modèle pré-entraîné)
- ✓ Sentiment analysis basique
- ✓ Détection doublons (cosinus similarity)
- ✓ Génération rapport structurée

### 🎓 Ce que VOUS devez AMÉLIORER (Travail Étudiant)

**Niveau 0 (Fondamental)** : Utiliser asis
```python
# "C'est bon, j'ai un système qui fonctionne"
```

**Niveau 1 ** : Ajouter sophistication
```python
# "Je vais améliorer la confiance du NER"
# "Je vais fine-tuner le classifier sur nos données"
# "Je vais implémenter semantic similarity pour duplicates"
#....
```

**Niveau 2 ** : Production-ready + innovations
```python
# "Je vais ajouter caching + async pour scalabilité"
# "Je vais implémenter custom NER avec spaCy"
# "Je vais créer metrics d'évaluation rigoureuses"
```

---

## 🚀 Démarrage Rapide

###  Installation

```bash
# Cloner/télécharger le baseline
cd baseline_project

# Créer virtualenv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer dépendances
pip install -r requirements.txt

# Télécharger modèles spaCy français
python -m spacy download fr_core_news_sm
```

###  Configuration

Éditer `config.json` selon tes besoins:
```json
{
  "collection": {
    "hacker_news": {
      "enabled": true,
      "num_pages": 2  // Changer nombre pages
    },
    "pycoders_rss": {
      "enabled": true
    }
  },
  "preprocessing": {
    "language": "fr",
    "remove_stopwords": true
  }
  // ...
}
```

### Exécuter le Pipeline

```bash
# Mode simple
python main.py

# Mode avec logging détaillé
python main.py 2>&1 | tee run.log

# Mode avec profiling
python -m cProfile -s cumtime main.py > profile.txt
```

### 4️⃣ Afficher Résultats

```bash
# Lire rapport généré
cat output/veille_report.txt

# Analyser données
head -10 data/articles_raw.jsonl
head -10 data/articles_processed.jsonl
head -10 data/articles_classified.jsonl
```

---

## 📚 Structure Code

### Chaque Module est Indépendant

```python
# ✅ Peux utiliser séparément:

# 1. Juste collecte
from src.news_collector import NewsCollector
collector = NewsCollector(config)
articles = collector.collect_from_hacker_news()

# 2. Juste prétraitement
from src.text_preprocessor import TextPreprocessor
preprocessor = TextPreprocessor(config)
processed = preprocessor.process_batch(articles)

# 3. Juste classification
from src.news_classifier import NewsClassifier
classifier = NewsClassifier(config)
classified = classifier.classify_batch(articles)

# 4. Juste rapport
from src.report_generator import ReportGenerator
generator = ReportGenerator(config)
report = generator.generate(articles)
```

---

## 🔍 Explications Code

### Chaque module a des commentaires détaillés

```python
# src/news_collector.py
"""
📥 Module Collecte d'Articles

DESIGN DECISIONS documentés :
1. Pourquoi BeautifulSoup? Car structure HTML stable
2. Pourquoi 2 pages HackerNews? Assez pour démo
3. Gestion timeouts : Retry logic avec delays

À AMÉLIORER :
- Ajouter YouTube API (actuellement TODO)
- Ajouter GitHub API (trending repos)
- Implémenter caching (ne pas re-scraper même URLs)
"""
```

---

## 📊 Format Données

### Tout est JSON (facile à analyser)

```jsonl
# data/articles_raw.jsonl
{"title": "BERT Tutorial", "url": "...", "content": "...", "source": "HackerNews", ...}
{"title": "Fine-tuning...", "url": "...", "content": "...", "source": "RSS", ...}

# data/articles_processed.jsonl
{...plus: "tokens": [...], "normalized_content": "...", "token_loss_pct": 45.2}

# data/articles_classified.jsonl
{...plus: "topic_prediction": "Advanced", "sentiment_label": "Positif", "is_duplicate": false}
```

---

## ✅ Checklist pour Étudiants

### À compléter/améliorer:

- [ ] **Collecte**
  - [ ] Implémenter YouTube scraper (au lieu de stub)
  - [ ] Ajouter GitHub API
  - [ ] Ajouter Medium API
  - [ ] Implémenter retry + exponential backoff

- [ ] **Prétraitement**
  - [ ] Tester impact remove_accents: True vs False
  - [ ] Comparer spaCy vs NLTK sur timing
  - [ ] Analyser token_loss_pct (est-ce normal 45%?)
  - [ ] Ajouter stemming optionnel

- [ ] **Classification**
  - [ ] Fine-tuner sur 100+ articles annotés
  - [ ] Évaluer P/R/F1 sur test set
  - [ ] Comparer modèles: distilbert vs roberta
  - [ ] Implémenter custom NER (technos spécifiques)

- [ ] **Rapport**
  - [ ] Ajouter visualisations (wordcloud, charts)
  - [ ] Calculer trend analysis (topics semaine précédente vs actuelle)
  - [ ] Ajouter insights qualitatifs
  - [ ] Exporter aussi en JSON/Markdown

- [ ] **Optionnel (E4)**
  - [ ] Implémenter caching (SQLite/Redis)
  - [ ] Async processing pour speed
  - [ ] API FastAPI pour servir rapport
  - [ ] Dashboard web (Streamlit)
  - [ ] Notifications email/Slack

---

## 🐛 Troubleshooting

### Erreur: "module 'spacy' has no attribute 'load'"
```bash
# Solution: Télécharger le modèle
python -m spacy download fr_core_news_sm
```

### Erreur: "CUDA out of memory"
```python
# Dans config.json, changer:
"device": -1  # Forcer CPU au lieu de GPU
```

### Articles collectés = 0
```bash
# Vérifier: HackerNews accessible?
curl https://news.ycombinator.com/

# Sinon: Activer juste RSS dans config.json
```

### Classification lente
```python
# Solution 1: Utiliser modèle plus petit
"model_name": "distilbert-base-multilingual-uncased"  # Plus rapide

# Solution 2: Batch processing avec huggingface
# (déjà implémenté dans le code)
```

---

## 📖 Ressources Documentation

### Modules Utilisés (official docs)

- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [spaCy](https://spacy.io/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [scikit-learn](https://scikit-learn.org/)

### Papers Académiques

- Vaswani et al. (2017) - "Attention is All You Need"
- Devlin et al. (2018) - "BERT: Pre-training..."

---

## 🎯 Conseils Étudiant

### 1. Comprendre CHAQUE module
```python
# Ne pas copier-coller aveuglément!
# Lire code de NewsCollector:
# - Pourquoi BeautifulSoup et pas Selenium?
# - Pourquoi ce regex pour URLs?
# - Où est la gestion d'erreurs?
```

### 2. Ajouter vos propres améliorations
```python
# Ne pas étendre, REMPLACER:
# "Je vais faire un meilleur NER"
# "Je vais implémenter semantic similarity"
# NOT: "J'ai juste enlevé un print() du code"
```

### 3. Justifier chaque décision
```python
# Dans votre présentation:
# "Nous avons choisi spaCy car X, Y, Z"
# "Trade-off: A vs B, nous choisissons A car..."
# "Limitation connue: C, amélioration future: D"
```

### 4. Évaluer rigoureusement
```python
# Toujours calculer metrics:
# - Accuracy globale
# - Precision/Recall/F1 par classe
# - Confusion matrix
# - Analyse erreurs qualitative
```

---

## 💡 Idées d'Extensions

**Easy (+5 pts)**
- Ajouter visualisations (wordcloud, bar charts)
- Implémenter caching de URLs scrapées
- Ajouter plus sources RSS

**Moyen (+10 pts)**
- Fine-tuning classifier sur 100 articles annotés
- Implémentation custom NER (frameworks spécifiques)
- Dashboard web (Streamlit)

**Difficile (+15 pts)**
- Production deployment (FastAPI API + Docker)
- Async processing (asyncio, concurrent.futures)
- Machine Learning pipeline (MLflow)
- Benchmark multiples modèles

---

## 📝 Fonctionnement du Rapport

Le rapport généré contient:

```
📰 VEILLE AUTOMATIQUE : NLP & Python
================================================

📊 RÉSUMÉ EXÉCUTIF
  Articles collectés: 50
  Articles uniques: 45 (dédupli rate: 10%)

🔥 TRENDING TOPICS (Sujets du moment)
  1. Fine-tuning LLMs (12 articles)
  2. RAG Systems (8 articles)
  3. French Models (5 articles)

✨ ARTICLES À NE PAS MANQUER
  1. "Complete Guide to LoRA"
     Niveau: Advanced | Confiance: 0.95
  
  2. "Evaluating RAG Systems"
     Niveau: Intermediate | Confiance: 0.88

📊 ANALYSE THÉMATIQUE
  Top Keywords:
    transformer (45 mentions)
    llm (38 mentions)
    fine-tuning (32 mentions)

😊 ANALYSE SENTIMENTS
  Positif: 55% (enthousiasme, innovations)
  Critique: 25% (limitations, coûts)
  Neutre: 20% (annonces)
```

---

## 🎓 Bon Travail!

Ce baseline est un **point de départ**, pas un produit fini.

**L'objectif n'est PAS** : "Faire tourner le code"

**L'objectif EST** : "Comprendre chaque ligne + améliorer + justifier"

Good luck! 🚀
