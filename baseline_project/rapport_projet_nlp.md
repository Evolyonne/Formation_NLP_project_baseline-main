# Rapport – Projet Veille NLP

## 1. Objectif du projet
Mettre en place une **pipeline de veille automatique en NLP** permettant de :
- Collecter des articles (HackerNews, RSS)
- Prétraiter les textes
- Classifier les articles (niveau / sentiment)
- Générer automatiquement un **rapport de veille**

Le projet est implémenté en **Python**, structuré de façon modulaire et versionné avec **Git**.

---

## 2. Structure finale du projet

```
baseline_project/
│
├── data/                   # Données générées (ignorées par Git)
│   └── .gitkeep
│
├── output/                 # Rapports générés (ignorés par Git)
│   └── .gitkeep
│
├── src/                    # Code source
│   ├── news_collector.py
│   ├── text_preprocessor.py
│   ├── news_classifier.py
│   └── report_generator.py
│
├── config.json             # Configuration du pipeline
├── main.py                 # Orchestrateur du pipeline
├── requirements.txt
├── README.md
└── .gitignore
```

👉 **Principe clé** : seuls le code et la configuration sont versionnés. Les fichiers générés (`data/`, `output/`, `__pycache__`) sont exclus de Git.

---

## 3. Pipeline NLP implémentée

### Étape 1 – Collecte des articles
- Sources : HackerNews + flux RSS
- Module : `news_collector.py`
- Résultat : liste d’articles structurés (titre, contenu, source, URL)

### Étape 2 – Prétraitement NLP
- Module : `text_preprocessor.py`
- Nettoyage du texte
- Tokenisation avec **spaCy**
- Rapport de qualité (perte de tokens, statistiques)

### Étape 3 – Classification
- Module : `news_classifier.py`
- Classification du **niveau** (Intermediate / Advanced)
- Analyse de **sentiment** (Positif / Neutre / Critique)
- Détection de doublons

### Étape 4 – Génération du rapport
- Module : `report_generator.py`
- Agrégation des résultats
- Trending topics
- Articles “must‑read”
- Sortie : `output/veille_report.txt`

---

## 4. Problèmes rencontrés et solutions

### 4.1 Dossiers `data/` et `output/` dupliqués
**Problème** : chemins absolus et incohérents → génération de plusieurs dossiers.

**Solution** :
- Centralisation des chemins avec `Path(__file__).parent`
- Utilisation systématique de chemins relatifs
- Un seul `data/` et un seul `output/`

---

### 4.2 Conflits Git lors des merges
**Problème** : fichiers générés suivis par Git (`.jsonl`, `.txt`, `__pycache__`, `.pyc`).

**Solution définitive** :
- Nettoyage de l’index Git
- Mise à jour du `.gitignore`

```gitignore
# Python cache
__pycache__/
*.pyc

# Generated data
data/*.jsonl
data/*.json
output/*.txt
*.log
```

Résultat : merges propres et reproductibles.

---

### 4.3 Problèmes d’environnement Python
- Conflits spaCy / Typer / Click
- Modèles spaCy manquants

Décision : **stabilisation du pipeline existant**, sans ajout d’améliorations expérimentales.

---

## 5. Tentative d’amélioration : Custom NER (abandonnée)

Objectif initial :
- Annotation manuelle avec **Doccano**
- Création d’un modèle NER personnalisé (technologies : PyTorch, TensorFlow, FastAPI…)

Ce qui a été fait :
- Déploiement de Doccano via Docker
- Création d’un projet d’annotation (sequence labeling)

Raison de l’abandon :
- Conflits d’environnement
- Temps limité
- Priorité donnée à la stabilité du pipeline principal

---

## 6. État final du projet

✅ Pipeline fonctionnelle
✅ Architecture propre et modulaire
✅ Git propre (aucun fichier généré versionné)
✅ Rapport automatique reproductible

❌ Améliorations avancées (Custom NER) reportées

---

## 7. Commande principale

Pour exécuter la veille :

```bash
python main.py
```

---

## 8. Conclusion

Le projet atteint son objectif principal : **une veille NLP automatisée, stable et maintenable**.

Les bases sont solides pour de futures extensions (NER custom, dashboards, orchestration), mais le socle actuel est fonctionnel et propre.

---

📌 *Rapport généré à des fins académiques – Projet NLP*

