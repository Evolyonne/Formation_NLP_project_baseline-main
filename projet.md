# 📌 INSTRUCTIONS POUR ÉTUDIANTS - Projet Veille Automatique NLP

## ⚠️ IMPORTANT À LIRE EN PREMIER

**Le baseline n'est PAS votre projet final.**

C'est un **point de départ fonctionnel** que vous **devez améliorer**.

---

## 🎯 Règle Fondamentale

### Baseline Seul = MAX 10/20

Si vous utilisez le baseline **sans aucune amélioration réelle** :
- Code fonctionne ✓
- Rapport généré ✓
- **MAIS: Score maximum = 10/20 (Fondamental)**

**Pourquoi ?** Parce qu'utiliser du code qui marche n'est pas un apprentissage.

### Votre Objectif: Améliorer le Baseline

Pour obtenir **12/20 ou plus**, vous DEVEZ améliorer le système.

---

## 📋 3 Niveaux de Compétence

Choisissez le VÔTRE et suivez les exigences.

### NIVEAU E2 : "Je veux juste réussir" (12/20)

**Effort minimal mais reconnu:**
```
Baseline + 1 amélioration superficielle
  ✓ Code baseline fonctionne
  ✓ Ajouter 1 source supplémentaire OU
  ✓ Tweaks mineurs (seuils, paramètres)
  ✗ Aucune amélioration algorithmique
  
  Temps estimé: 3-4h additional
  Score: 16/20 (Fondamental)
  
  Exemples:
  - Ajouter un regex pour mieux nettoyer URLs
  - Changer le seuil de similarité (0.85 → 0.80)
  - Améliorer les rapports visuellement
```

**Bon à faire:**
```
"J'ajoute GitHub API pour collecter trending repos"
 → C'est une vraie amélioration + collecte
 → Utile pour le système
 → Effort réél
```

**Pas bon à faire:**
```
"J'ajoute des commentaires au code"
 → C'est cosmétique, pas une amélioration
```

---

### NIVEAU E3 : "Je veux vraiment apprendre" (15/20)

**Apprentissage réel avec mesures:**
```
Baseline + amélioration SIGNIFICATIVE

OPTION A : Fine-tuning
  ✓ Annoter 50+ articles pour dataset
  ✓ Fine-tuner classifier (distilbert → custom)
  ✓ Montrer: Accuracy baseline 0.65 → 0.82 (17 points gain!)
  ✓ Metrics: Precision/Recall/F1 par classe
  ✓ Confusion matrix
  Temps: 10-15h additional
  
  Exemple résultat attendu:
  ```
  Baseline (zero-shot): Accuracy 65%
  After fine-tuning: Accuracy 82%
  
  Class        Precision  Recall  F1
  Débutant     0.85       0.88    0.86
  Intermediate 0.80       0.78    0.79
  Advanced     0.81       0.80    0.80
  ```

OPTION B : Custom NER
  ✓ Créer NER pour technos spécifiques
    (PyTorch, TensorFlow, FastAPI, etc.)
  ✓ Annoter 50+ articles
  ✓ Train spaCy custom model
  ✓ Metrics: Precision/Recall/F1 > 0.80

  
  Exemple résultat attendu:
  ```
  Custom NER extracts:
  "PyTorch implementation of BERT"
  → Technology: [PyTorch, BERT]
  
  Metrics:
  Precision: 0.87
  Recall: 0.85
  F1: 0.86
  ```

OPTION C : Ajouter sources
  ✓ GitHub API (trending repos)
  ✓ Medium API (technical posts)
  ✓ Intégrer dans collecteur
  ✓ Collecter 50+ articles par source
  ✓ Analysis: "GitHub: 40% advanced vs HN: 20%"

Raison: "Code amélioré + évaluation rigoureuse"
```

**Bon à faire:**
```
"Je fine-tuner le classifier.
 Baseline accuracy: 65%
 Mon version: 82%
 Voici les metrics:"
 → Apprentissage réel
 → Mesurable
 → Justifiable
```

**Pas bon à faire:**
```
"Je change la config.json (num_pages: 2 → 3)"
 → C'est pas une amélioration, c'est un paramètre
```

---

### NIVEAU E4 : "Je veux être excellent" (18/20)

**Système production-ready avec multiples améliorations:**
```
Baseline + MULTIPLES améliorations SUBSTANTIELLES

Minimum 2 de ces options:
  ✓ Fine-tuning classifier (50+ exemples)
  ✓ Custom NER implementation
  ✓ Semantic similarity (embeddings)
  ✓ Production features:
    - Caching layer (SQLite/Redis)
    - Async processing
    - FastAPI deployment
    - Docker containerization
  ✓ 3+ sources supplémentaires
  ✓ Visualisations avancées
    - Wordcloud
    - Timeline trends
    - Interactive dashboard

PLUS: Benchmarking rigoureux
  ✓ Model comparison table
  ✓ Latency analysis
  ✓ Memory footprint
  ✓ Scalability considerations

Temps: 20-25h additional
Score: 20/20 (Expert)
Raison: "Baseline transformed into production system"

Exemple résultat attendu:
```
Model Comparison:
┌─────────────────┬──────────┬────────┬──────────┐
│ Model           │ Accuracy │ Latency│ Memory   │
├─────────────────┼──────────┼────────┼──────────┤
│ Baseline        │ 65%      │ 120ms  │ 450MB    │
│ Fine-tuned      │ 82%      │ 150ms  │ 500MB    │
│ Custom NER      │ 85%      │ 200ms  │ 600MB    │
│ Deployment-opt  │ 80%      │  80ms  │ 250MB    │
└─────────────────┴──────────┴────────┴──────────┘

Conclusion: Use fine-tuned for accuracy,
            deployment-opt for speed
```
```

**Bon à faire:**
```
"Je vais faire du fine-tuning ET custom NER.
 Puis je compare avec baseline.
 Je vais aussi ajouter caching pour speed.
 Enfin je déploie avec FastAPI."
 → Multiple substantial improvements
 → Production-ready
 → Rigorous evaluation
```

**Pas bon à faire:**
```
"J'ai changé les couleurs du rapport"
 → Cosmétique, pas technique
```

---

## 📊 Barème

### Checkpoint 1 : Collecte & Prétraitement (10 pts)
Attendu: Montrez que vous avez compris data collection + preprocessing

```
À minimiser:
  - Code baseline compile? ✓ (2 pts)
  - Données nettoyées? ✓ (2 pts)

À maximiser:
  - Amélioration collecte?
  - Justification choix design?
  - Metrics qualité données?
  - E2: Code fonctionne → 8/10
  - E3: + amélioration source → 9/10
  - E4: + multiple sources + analysis → 10/10
```

### Checkpoint 2 : Classification (20 pts)
Attendu: Fine-tuning OU custom NER, pas juste baseline

```
À minimizer:
  - Code compile? ✓ (2 pts)
  - Baseline zero-shot works? ✓ (2 pts)
  
À maximizer:
  - FINE-TUNING implémenté? (4 pts)
    - Accuracy improvement mesurable
    - Metrics complets (P/R/F1)
  - ERROR ANALYSIS (3 pts)
    - 5+ erreurs analysées
    - Patterns identifiés
  
  - E2: Baseline seulement → 10/20 (MAX)
  - E3: Baseline + fine-tuning → 15-18/20
  - E4: Baseline + fine-tuning + custom NER → 20/20
```

### Checkpoint 3 : Rapport (15 pts)
Attendu: Visualisations + insights, pas juste baseline

```
À minimizer:
  - Rapport texte ✓ (7 pts)
  
À maximizer:
  - Visualisations personnalisées? (4 pts)
    - Wordcloud (technos)
    - Bar charts (sentiments)
    - Timeline (trends)
  - Insights nouveaux? (3 pts)
    - Pas juste ce que baseline dit
    - Analyse qualitative
  
  - E2: Baseline rapport → 10/15
  - E3: + visualisations → 13/15
  - E4: + insights originaux → 15/15
```

### Présentation (20 pts)
Attendu: Pouvoir JUSTIFIER vos choix et améliorations

```
Questions que vous aurez:

"Montrez-moi UNE amélioration clé"
 → Prêt? Oui: "J'ai fine-tuné et..."
 → Pas prêt? Non: "Euh... j'ai utilisé le baseline..."
 
"Justifiez avec chiffres"
 → Prêt: "Accuracy 65% → 82%, +17 points"
 → Pas prêt: "Euh... ça marche"

"Qu'aurait fait le baseline là?"
 → Prêt: "Baseline predirait 'Intermediate', mais c'est Advanced"
 → Pas prêt: "Je sais pas..."

La présentation teste si vous COMPRENEZ votre code.
Si vous juste copiez le baseline, vous échouez ces questions.
```

---

## 🚀 WORKFLOW RECOMMANDÉ

### 1 Collecte + Preprocessing

```
  - Décompresser baseline_project.zip
  - Installer dependances: pip install -r requirements.txt
  - Run: python main.py
  - Vérifier que ça fonctionne ✓
  - Lire src/news_collector.py
  - Lire src/text_preprocessor.py
  - Comprendre le flux données
 - Décider amélioration CP1:
    Option A (E2): Rien (baseline ok) → 8/10
    Option B (E3): +1 source (GitHub API) → 9/10
    Option C (E4): +2 sources → 10/10
  - Montrer improvements
  - Documenter choix
```

### 2 Classification

```
  - Annoter 50+ articles pour fine-tuning
    (Créer CSV: text, label, confidence)
  - Lire src/news_classifier.py
  - Implémenter fine-tuning:
    ```python
    from transformers import Trainer, TrainingArguments
    trainer = Trainer(...)
    trainer.train()
    ```
  - Calculer metrics: Precision, Recall, F1
  - Confusion matrix
  - Montrer: Baseline 65% → Fine-tuned 82%
  - Montrer metrics table
  - Montrer error analysis (5+ erreurs)
```

### 3 : Rapport + Présentations

```
  - Ajouter visualisations (wordcloud, bar charts, ...)
  - Écrire insights qualitatifs
  - Rapport final
  - Visualisations
  - Documentation complète
 (PRESENTATIONS):
  - 5 min: démo système live
  - 10 min: justifier améliorations
  - 5 min: Q&A technique
  
  Questions possibles:
  "Why fine-tuning?"
  "Show me your accuracy improvement"
  "What error patterns did you find?"
  "How would you improve further?"
```

---

## ❓ FAQ

**Q: Est-ce que je peux juste utiliser le baseline?**
```
R: Oui, mais score MAX 16/20 (E2 seulement).
   Pour 18+ vous DEVEZ améliorer.
```

**Q: Qu'est-ce qui compte comme "amélioration"?**
```
R: AMÉLIORATION (compte) :
  ✓ Fine-tuning classifier
  ✓ Custom NER
  ✓ Ajouter sources
  ✓ Semantic similarity
  ✓ Production features (async, caching)
  
  PAS une amélioration (cosmétique) :
  ✗ Renommer variables
  ✗ Ajouter commentaires
  ✗ Changer couleurs rapport
  ✗ Reformater texte
```

**Q: Fine-tuning, c'est dur?**
```
R: Non, HuggingFace le rend facile:
   ```python
   from transformers import Trainer
   trainer = Trainer(model, args, train, eval)
   trainer.train()
   ```
   Cherchez "HuggingFace fine-tuning tutorial"





## ✅ FINAL CHECKLIST

Avant de commencer:

- [ ] J'ai compris que baseline ≠ projet final
- [ ] J'ai choisi mon niveau (E2, E3, E4)
- [ ] J'ai compris l'exigence d'amélioration
- [ ] Je peux justifier mes choix
- [ ] J'ai un plan (quoi améliorer)
- [ ] Je peux montrer metrics si E3/E4



## 📞 Questions?

Si quelque chose n'est pas clair:
- Relire cette page
- Regarder les exemples dans les grilles d'évaluation
- Demander à Nastasia (votre instructrice)

Mais soyez sûr d'une chose:
**Utiliser baseline sans amélioration = MAX 10/20**
**Pour 15+ : FAUT améliorer et justifier**

Bon courage! 🎓