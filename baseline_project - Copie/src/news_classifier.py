# src/news_classifier.py
"""
🤖 Module Classification & Extraction NLP

Tasks :
1. Topic Classification : Débutant / Intermédiaire / Avancé
2. Sentiment Analysis : Positif / Neutre / Critique
3. Duplicate Detection : Similarité cosinus entre articles

Usage:
    classifier = NewsClassifier(config)
    classified_articles = classifier.classify_batch(articles)
"""

import logging
import torch
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import json

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# NEWS CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

class NewsClassifier:
    """
    Classification articles avec HuggingFace + similarité cosinus
    
    DESIGN DECISIONS :
    
    1. Topic Classification : Zero-shot vs Fine-tuned?
        → Choix : Zero-shot (pas besoin données entraînement)
        → Avantage : Rapide, flexible sur labels
        → Limitation : Accuracy inférieure à fine-tuned
    
    2. Modèle multilingue : distilbert-base-multilingual-uncased
        → Support français + anglais + vitesse
        → Alternative : roberta (meilleur accuracy, plus lent)
    
    3. Duplicate detection : Cosine similarity vs Semantic?
        → Choix : TF-IDF cosine (fast, transparent)
        → Alternative : Embeddings (better but slower)
    
    4. Seuil similarité : 0.85
        → Validation : Tester manuellement sur 20 paires
        → Justification : Équilibre false positives/negatives
    """
    
    def __init__(self, config: Dict):
        self.config = config

        model_name = config.get("classification", {}).get(
            "model_name",
            "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        )

        # --- Zero-shot topic (FORCER PyTorch) ---
        try:
            logger.info(f"📥 Chargement modèle: {model_name}...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=-1,
                framework="pt"
            )
            logger.info(f"✅ Modèle chargé: {model_name}")
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {str(e)}")
            self.classifier = None

        # --- Sentiment (FORCER PyTorch) ---
        try:
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                device=-1,
                framework="pt"
            )
            logger.info("✅ Sentiment classifier chargé")
        except Exception as e:
            logger.warning(f"⚠️ Sentiment classifier: {str(e)}")
            self.sentiment_classifier = None

        # --- Vectorizer pour duplicate detection ---
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            lowercase=True,
            stop_words="english"
        )
        self.tfidf_matrix = None

    
    # ═══════════════════════════════════════════════════════════════════════
    # TASK 1 : TOPIC CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def classify_topic(self, text: str) -> Dict:
        """
        Classifier article par niveau : Débutant/Intermédiaire/Avancé
        
        Args:
            text: Contenu article à classifier
        
        Returns:
            {
                'predicted_label': str,
                'confidence': float,
                'all_scores': dict de tous labels et scores
            }
        """
        if not text or not self.classifier:
            return {
                'predicted_label': 'Unknown',
                'confidence': 0.0,
                'all_scores': {}
            }
        
        try:
            labels = self.config.get('classification', {}).get('labels', 
                                                               ['Beginner', 'Intermediate', 'Advanced'])
            
            # Limiter texte à 512 tokens (limite BERT)
            text_truncated = ' '.join(text.split()[:400])
            
            result = self.classifier(text_truncated, labels)
            
            # result format:
            # {
            #   'sequence': texte original,
            #   'labels': ['Intermediate', 'Beginner', 'Advanced'],
            #   'scores': [0.95, 0.04, 0.01]
            # }
            
            return {
                'predicted_label': result['labels'][0],
                'confidence': round(result['scores'][0], 4),
                'all_scores': {label: round(score, 4) 
                              for label, score in zip(result['labels'], result['scores'])}
            }
        
        except Exception as e:
            logger.warning(f"Erreur classification: {str(e)}")
            return {
                'predicted_label': 'Unknown',
                'confidence': 0.0,
                'all_scores': {}
            }
    
    # ═══════════════════════════════════════════════════════════════════════
    # TASK 2 : SENTIMENT ANALYSIS
        # ═══════════════════════════════════════════════════════════════════════
        
    def analyze_sentiment(self, text: str) -> Dict:
        if not text or not self.sentiment_classifier:
            return {'sentiment': 'NEUTRAL', 'score': 0.5, 'label': 'Neutre'}

        try:
            # Tronquer proprement avec le tokenizer (évite >512 tokens)
            tokenizer = self.sentiment_classifier.tokenizer
            inputs = tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            truncated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

            result = self.sentiment_classifier(truncated_text)[0]

            hf_label = str(result.get('label', '')).lower()
            score = float(result.get('score', 0.0))

            # Certains modèles renvoient LABEL_0/1/2 -> mapping
            if hf_label in ["label_2"]:
                hf_label = "positive"
            elif hf_label in ["label_0"]:
                hf_label = "negative"
            elif hf_label in ["label_1"]:
                hf_label = "neutral"

            if hf_label == "positive":
                return {'sentiment': 'POSITIVE', 'score': round(score, 4), 'label': 'Positif'}
            elif hf_label == "negative":
                return {'sentiment': 'NEGATIVE', 'score': round(score, 4), 'label': 'Critique'}
            else:
                return {'sentiment': 'NEUTRAL', 'score': round(score, 4), 'label': 'Neutre'}

        except Exception as e:
            logger.warning(f"Erreur sentiment: {str(e)}")
            return {'sentiment': 'NEUTRAL', 'score': 0.5, 'label': 'Neutre'}


    
    # ═══════════════════════════════════════════════════════════════════════
    # TASK 3 : DUPLICATE DETECTION
    # ═══════════════════════════════════════════════════════════════════════
    
    def detect_duplicates(self, articles: List[Dict], threshold: float = 0.85) -> List[Dict]:
        """
        Détecter articles dupliqués via similarité cosinus
        
        JUSTIFICATION APPROCHE :
        - TF-IDF cosine : Fast, transparent, interpretable
        - Alternative : Embeddings (plus lent, meilleur accuracy)
        
        Args:
            articles: Liste articles à dédupliquer
            threshold: Seuil similarité (0-1)
        
        Returns:
            Articles avec flag 'is_duplicate' ajouté
        """
        logger.info(f"🔍 Duplicate detection (threshold={threshold})...")
        
        if len(articles) < 2:
            for article in articles:
                article['is_duplicate'] = False
            return articles
        
        try:
            # Extraire contenus
            contents = [article.get('normalized_content', article.get('title', '')) 
                       for article in articles]
            
            # Construire matrice TF-IDF
            self.tfidf_matrix = self.vectorizer.fit_transform(contents)
            
            # Calculer similarités pairwise
            similarity_matrix = cosine_similarity(self.tfidf_matrix)
            
            # Marquer doublons
            duplicates = set()
            for i in range(len(articles)):
                for j in range(i + 1, len(articles)):
                    if similarity_matrix[i][j] >= threshold:
                        # Marquer le plus jeune comme doublon
                        duplicates.add(j)
                        
                        logger.debug(f"  Doublon détecté: {i} ↔ {j} (sim={similarity_matrix[i][j]:.3f})")
            
            # Ajouter flag
            for idx, article in enumerate(articles):
                article['is_duplicate'] = idx in duplicates
                article['duplicate_score'] = max(similarity_matrix[idx]) if len(articles) > 1 else 0.0
            
            logger.info(f"✅ Duplicate detection: {len(duplicates)} doublons trouvés")
            return articles
        
        except Exception as e:
            logger.error(f"❌ Erreur duplicate detection: {str(e)}")
            for article in articles:
                article['is_duplicate'] = False
            return articles
    
    # ═══════════════════════════════════════════════════════════════════════
    # BATCH CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def classify_batch(self, articles: List[Dict]) -> List[Dict]:
        """
        Classifier batch d'articles
        
        Ajoute à chaque article :
        - topic_prediction
        - topic_confidence
        - topic_scores
        - sentiment
        - sentiment_score
        - sentiment_label
        """
        logger.info(f"🤖 Classification {len(articles)} articles...")
        
        classified = []
        
        for i, article in enumerate(articles):
            try:
                # Topic -> texte normalisé (OK)
                text_topic = article.get("normalized_content", article.get("content", article.get("title", "")))

                # Sentiment -> texte BRUT (title + content) (mieux pour éviter "Neutre")
                text_sentiment = f"{article.get('title','')} {article.get('content','')}".strip()
                if not text_sentiment:
                    text_sentiment = article.get("title", "")

                # Task 1: Topic
                topic_result = self.classify_topic(text_topic)
                article["topic_prediction"] = topic_result["predicted_label"]
                article["topic_confidence"] = topic_result["confidence"]
                article["topic_scores"] = topic_result["all_scores"]
 
                # Task 2: Sentiment
                if article.get("source") == "Wikipedia":
                    # Wikipedia = encyclopédique → sentiment neutre par défaut
                    sentiment_result = {
                        "sentiment": "NEUTRAL",
                        "score": 0.5,
                        "label": "Neutre"
                    }
                else:
                    sentiment_result = self.analyze_sentiment(text_sentiment)

                article["sentiment"] = sentiment_result["sentiment"]
                article["sentiment_score"] = sentiment_result["score"]
                article["sentiment_label"] = sentiment_result["label"]



                classified.append(article)

                if (i + 1) % 10 == 0:
                    logger.info(f"  ✓ {i + 1}/{len(articles)} articles")

            except Exception as e:
                logger.warning(f"  ✗ Article {i}: {str(e)}")
                continue

        
        # Task 3: Duplicate detection
        classified = self.detect_duplicates(classified, 
                                           threshold=self.config.get('deduplication', {}).get('threshold', 0.85))
        
        logger.info(f"✅ Classification terminée: {len(classified)} articles")
        return classified
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def print_classification_summary(self, articles: List[Dict]):
        """Afficher résumé classification"""
        if not articles:
            return
        
        # Compter par topic
        topics = {}
        for article in articles:
            topic = article.get('topic_prediction', 'Unknown')
            topics[topic] = topics.get(topic, 0) + 1
        
        # Compter par sentiment
        sentiments = {}
        for article in articles:
            sentiment = article.get('sentiment_label', 'Unknown')
            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
        
        # Compter doublons
        duplicates = sum(1 for a in articles if a.get('is_duplicate', False))
        
        print("\n" + "="*70)
        print("📊 RÉSUMÉ CLASSIFICATION")
        print("="*70)
        print("\n📌 DISTRIBUTION TOPICS:")
        for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(articles)) * 100
            print(f"  {topic:15s}: {count:3d} ({pct:5.1f}%)")
        
        print("\n😊 DISTRIBUTION SENTIMENTS:")
        for sentiment, count in sorted(sentiments.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(articles)) * 100
            print(f"  {sentiment:15s}: {count:3d} ({pct:5.1f}%)")
        
        print(f"\n🔄 DUPLICATES: {duplicates} ({(duplicates/len(articles)*100):.1f}%)")
        print("="*70)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN - TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    # Charger config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Créer classifier
    classifier = NewsClassifier(config)
    
    # Test articles
    test_articles = [
        {
            'title': 'Beginner Guide to Python',
            'content': 'This is a simple tutorial for beginners learning Python programming...',
            'normalized_content': 'simple tutorial beginners learning python programming'
        },
        {
            'title': 'Advanced Fine-tuning Techniques',
            'content': 'Advanced techniques for fine-tuning large language models...',
            'normalized_content': 'advanced techniques fine-tuning large language models'
        }
    ]
    
    # Classifier
    classified = classifier.classify_batch(test_articles)
    
    # Afficher résultats
    for article in classified:
        print(f"\nArticle: {article['title']}")
        print(f"  Topic: {article['topic_prediction']} ({article['topic_confidence']})")
        print(f"  Sentiment: {article['sentiment_label']}")
