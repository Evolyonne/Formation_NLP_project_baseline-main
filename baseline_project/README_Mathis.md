## Ce qui j'ai implemente

### Configuration (config.json)
- Ajout d'une source Wikipedia avec termes de recherche NLP/IA, langue en, limite d'articles. l24
- Passage du preprocessing en anglais (language=en, spacy_model=en_core_web_sm). l40
- Classification zero-shot avec le modele `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`. l53
- Deduplication par TF-IDF + cosine similarity avec seuil 0.80. l62

### Collecte (src/news_collector.py)
- Integration de Wikipedia via `wikipedia-api`.
- Pour chaque terme de recherche: ouverture de la page, recuperation du texte et limite a 5000 caracteres.
- Gestion des erreurs par terme avec log dans `data/collection_errors.json`. l75

### Pretraitement (src/text_preprocessor.py)
- Normalisation plus robuste: suppression des blocs `script/style` l102
- Texte passe en anglais (stopwords anglais et modele spaCy en).l67 78 327

### Classification (src/news_classifier.py)
- Zero-shot force en PyTorch et modele multilingue mDeBERTa. l80
- Troncature du texte pour la classification (limite 512 tokens approx).
- Deduplication TF-IDF avec `max_features=5000` + stopwords anglais.
- Seuil de similarite ajuste a 0.80. l327
