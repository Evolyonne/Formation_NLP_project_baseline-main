# src/report_generator.py
"""
📄 Module Génération Rapport

Synthétise articles classifiés en rapport professionnel

Sections :
1. Trending Topics
2. Must-Read Articles
3. Thematic Analysis
4. Sentiment Distribution
5. Resources by Level

Usage:
    generator = ReportGenerator(config)
    report = generator.generate(classified_articles)
    generator.save_report(report, 'output/report.txt')
"""

import logging
from typing import List, Dict
from datetime import datetime
from collections import Counter
import json

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Génère rapport de veille professionnel"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1 : TRENDING TOPICS
    # ═══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def extract_trending_topics(articles: List[Dict], num_topics: int = 3) -> List[Dict]:
        """
        Extraire trending topics
        
        Critères :
        - Nombre articles sur le sujet
        - Mentions dans les titres
        - Sentiment dominant
        """
        logger.info("📈 Extraction trending topics...")
        
        # Compter par topic
        topic_counts = Counter()
        topic_articles = {}
        
        for article in articles:
            if not article.get('is_duplicate', False):  # Ignorer doublons
                topic = article.get('topic_prediction', 'Other')
                topic_counts[topic] += 1
                
                if topic not in topic_articles:
                    topic_articles[topic] = []
                topic_articles[topic].append(article)
        
        # Top N topics
        trending = []
        for topic, count in topic_counts.most_common(num_topics):
            articles_topic = topic_articles.get(topic, [])
            
            # Sentiment moyen
            sentiments = [a.get('sentiment_label', 'Neutre') for a in articles_topic]
            sentiment_counts = Counter(sentiments)
            dominant_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else 'Neutre'
            
            trending.append({
                'topic': topic,
                'count': count,
                'articles_sample': [a.get('title', '')[:50] for a in articles_topic[:3]],
                'dominant_sentiment': dominant_sentiment
            })
        
        logger.info(f"✅ {len(trending)} trending topics trouvés")
        return trending
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2 : MUST-READ ARTICLES
    # ═══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def extract_must_read_articles(articles: List[Dict], num_articles: int = 5) -> List[Dict]:
        """
        Extraire articles must-read
        
        Critères de ranking :
        - Confiance classification élevée
        - Score sentiment distinctif (pas neutre)
        - Pas doublon
        """
        logger.info("✨ Extraction must-read articles...")
        
        # Filtrer
        candidates = [a for a in articles if not a.get('is_duplicate', False)]
        
        # Score ranking
        scored = []
        for article in candidates:
            # Score confiance
            confidence = article.get('topic_confidence', 0.0)
            
            # Score sentiment (intéressant si pas neutre)
            sentiment = article.get('sentiment_label', 'Neutre')
            sentiment_score = 0.5 if sentiment == 'Neutre' else 1.0
            
            # Score global
            score = (confidence * 0.7) + (sentiment_score * 0.3)
            
            scored.append({
                'article': article,
                'score': score
            })
        
        # Top N
        must_read = [s['article'] for s in sorted(scored, key=lambda x: x['score'], reverse=True)[:num_articles]]
        
        logger.info(f"✅ {len(must_read)} must-read articles sélectionnés")
        return must_read
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 3 : THEMATIC ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def thematic_analysis(articles: List[Dict]) -> Dict:
        """
        Analyser thèmes principaux
        
        Extrait keywords des articles via tokens
        """
        logger.info("📊 Analyse thématique...")
        
        # Compter tokens
        all_tokens = []
        for article in articles:
            tokens = article.get('tokens', [])
            all_tokens.extend(tokens)
        
        # Top keywords
        token_counts = Counter(all_tokens)
        top_keywords = token_counts.most_common(15)
        
        # Distribution par topic
        topics_dist = Counter(a.get('topic_prediction', 'Other') for a in articles 
                              if not a.get('is_duplicate', False))
        
        return {
            'top_keywords': [{'keyword': k, 'count': c} for k, c in top_keywords],
            'topic_distribution': dict(topics_dist),
            'num_unique_tokens': len(token_counts)
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 4 : SENTIMENT ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def sentiment_analysis(articles: List[Dict]) -> Dict:
        """Analyser distribution sentiments"""
        logger.info("😊 Analyse sentiments...")
        
        sentiments = Counter(a.get('sentiment_label', 'Neutre') for a in articles 
                            if not a.get('is_duplicate', False))
        
        total = sum(sentiments.values()) or 1
        
        return {
            'distribution': {
                sentiment: {
                    'count': count,
                    'percentage': round((count / total) * 100, 1)
                }
                for sentiment, count in sentiments.items()
            },
            'dominant_sentiment': sentiments.most_common(1)[0][0] if sentiments else 'Neutre'
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # GÉNÉRATION RAPPORT COMPLET
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate(self, articles: List[Dict]) -> str:
        """
        Générer rapport complet
        
        Format :
        - Header avec métadonnées
        - Trending topics
        - Must-read articles
        - Thematic analysis
        - Sentiment analysis
        - Resources by level
        """
        logger.info("📝 Génération rapport...")
        
        # Extraire sections
        trending = self.extract_trending_topics(articles)
        must_read = self.extract_must_read_articles(articles)
        thematic = self.thematic_analysis(articles)
        sentiments = self.sentiment_analysis(articles)
        
        # Compter articles
        num_total = len(articles)
        num_unique = sum(1 for a in articles if not a.get('is_duplicate', False))
        
        # Construire rapport
        report = []
        
        # HEADER
        report.append("=" * 75)
        report.append("📰 VEILLE AUTOMATIQUE : NLP & Python")
        report.append(f"Généré le {datetime.now().strftime('%d %B %Y à %H:%M')}")
        report.append("=" * 75)
        report.append("")
        
        # RÉSUMÉ EXÉCUTIF
        report.append("📊 RÉSUMÉ EXÉCUTIF")
        report.append("-" * 75)
        report.append(f"Articles collectés : {num_total}")
        report.append(f"Articles uniques : {num_unique} (dédupli rate: {((num_total-num_unique)/max(num_total,1)*100):.1f}%)")
        report.append("")
        
        # TRENDING TOPICS
        report.append("🔥 TRENDING TOPICS (Sujets du moment)")
        report.append("-" * 75)
        for i, trend in enumerate(trending, 1):
            report.append(f"\n{i}. {trend['topic'].upper()}")
            report.append(f"   Articles: {trend['count']}")
            report.append(f"   Sentiment dominant: {trend['dominant_sentiment']}")
            report.append(f"   Exemples:")
            for article_title in trend['articles_sample']:
                report.append(f"     • {article_title}")
        report.append("")
        
        # MUST-READ
        report.append("\n✨ ARTICLES À NE PAS MANQUER")
        report.append("-" * 75)
        for i, article in enumerate(must_read, 1):
            report.append(f"\n📌 {i}. {article.get('title', 'No Title')}")
            report.append(f"   Source: {article.get('source', 'Unknown')}")
            report.append(f"   URL: {article.get('url', 'N/A')}")
            report.append(f"   Niveau: {article.get('topic_prediction', 'Unknown')}")
            report.append(f"   Confiance: {article.get('topic_confidence', 0):.1%}")
            report.append(f"   Sentiment: {article.get('sentiment_label', 'Neutre')}")
        report.append("")
        
        # ANALYSE THÉMATIQUE
        report.append("\n📊 ANALYSE THÉMATIQUE")
        report.append("-" * 75)
        report.append(f"\nTop Keywords ({thematic['num_unique_tokens']} tokens uniques):")
        for keyword_data in thematic['top_keywords'][:10]:
            keyword = keyword_data['keyword']
            count = keyword_data['count']
            report.append(f"  • {keyword:20s} ({count} mentions)")
        
        report.append(f"\nDistribution par Niveau:")
        for topic, count in sorted(thematic['topic_distribution'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / max(num_unique, 1)) * 100
            report.append(f"  • {topic:20s}: {count:3d} ({pct:5.1f}%)")
        report.append("")
        
        # SENTIMENT ANALYSIS
        report.append("\n😊 ANALYSE SENTIMENTS")
        report.append("-" * 75)
        for sentiment, data in sentiments['distribution'].items():
            report.append(f"{sentiment:15s}: {data['count']:3d} articles ({data['percentage']:5.1f}%)")
        report.append("")
        
        # FOOTER
        report.append("\n" + "=" * 75)
        report.append("Fin du rapport")
        report.append("=" * 75)
        
        return "\n".join(report)
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAUVEGARDE
    # ═══════════════════════════════════════════════════════════════════════
    
    def save_report(self, report: str, filepath: str):
        """Sauvegarder rapport en fichier"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"✅ Rapport sauvegardé: {filepath}")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {str(e)}")

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
    
    # Créer générateur
    generator = ReportGenerator(config)
    
    # Test articles
    test_articles = [
        {
            'title': 'BERT Tutorial',
            'content': 'Learn BERT...',
            'topic_prediction': 'Beginner',
            'topic_confidence': 0.95,
            'sentiment_label': 'Positif',
            'tokens': ['bert', 'tutorial', 'nlp'],
            'source': 'Medium',
            'url': 'https://example.com/bert',
            'is_duplicate': False
        },
        {
            'title': 'Advanced Fine-tuning',
            'content': 'Advanced techniques...',
            'topic_prediction': 'Advanced',
            'topic_confidence': 0.88,
            'sentiment_label': 'Neutre',
            'tokens': ['fine', 'tuning', 'llm'],
            'source': 'ArXiv',
            'url': 'https://example.com/advanced',
            'is_duplicate': False
        }
    ]
    
    # Générer rapport
    report = generator.generate(test_articles)
    print(report)
    
    # Sauvegarder
    generator.save_report(report, 'test_report.txt')