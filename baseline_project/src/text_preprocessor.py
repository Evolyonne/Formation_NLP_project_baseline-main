# src/text_preprocessor.py
"""
🔧 Module Prétraitement NLP

Pipeline complet de nettoyage texte :
- Normalisation (HTML, URLs, whitespace, casse, accents)
- Tokenization (spaCy français)
- Cleaning (stopwords, lemmatization, filtrage tokens)
- Évaluation qualité

Usage:
    preprocessor = TextPreprocessor(config)
    processed_articles = preprocessor.process_batch(articles)
    metrics = preprocessor.get_quality_metrics()
"""

import re
import spacy
import logging
from typing import List, Dict, Optional
import json
from collections import Counter
import unicodedata

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

class TextPreprocessor:
    """
    Pipeline NLP français avec justifications
    
    DESIGN DECISIONS (à documenter) :
    
    1. Normalisation casse : minuscules OUI/NON?
       → Choix : Minuscules (réduit vocabulaire, aide generalisation)
       → MAIS : Conserver acronymes techniques (détecté avec regex)
    
    2. Tokenization : spaCy vs NLTK vs regex?
       → Choix : spaCy (support français, POS tagging, speed)
    
    3. Lemmatization : appliquer?
       → Choix : OUI (réduit bruit, aides clustering)
       → Tradeoff : Perte info fine-grained (ex: "running" → "run")
    
    4. Stopwords : supprimer?
       → Choix : OUI (bruit pour classification)
       → Mais : Documenter impact
    
    5. Accents français : normer?
       → Choix : NON (unicodedata lossless)
    """
    
    def __init__(self, config: Dict):
        """
        Initialiser preprocessor
        
        Args:
            config: Dict de configuration (voir config.json)
        """
        self.config = config
        
        # Charger modèle spaCy
        lang = config.get('preprocessing', {}).get('language', 'fr')
        model_name = config.get('preprocessing', {}).get('spacy_model', 'fr_core_news_sm')
        
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"✅ Modèle spaCy chargé: {model_name}")
        except OSError:
            logger.error(f"❌ Modèle {model_name} non trouvé. Installez avec :")
            logger.error(f"   python -m spacy download {model_name}")
            raise
        
        # Stopwords français
        self.french_stopwords = self._load_french_stopwords()
        
        # Métriques qualité
        self.quality_metrics = {}
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 1 : NORMALISATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def normalize(self, text: str) -> str:
        """
        Normaliser texte brut
        
        JUSTIFICATIONS :
        - Supprimer tags HTML (contenu non-pertinent)
        - Remplacer URLs (ne contribuent pas au sens)
        - Normaliser whitespace (aide tokenizer)
        - Lowercase (réduit vocabulaire, mais voir note casse)
        - PRESERVER accents français
        """
        if not text:
            return ""
        
        # 1. Supprimer tags HTML
        text = re.sub(r'<[^>]+>', '', text)
        
        # 2. Remplacer URLs par token
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', text)
        
        # 3. Remplacer mentions Twitter
        text = re.sub(r'@\w+', '<MENTION>', text)
        
        # 4. Remplacer hashtags
        text = re.sub(r'#\w+', '<HASHTAG>', text)
        
        # 5. Normaliser whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # 6. Conversion minuscules (MAIS: conserver acronymes?)
        # Pour ce baseline, on garde minuscules simples
        # Les acronymes seront détectés dans NER
        text = text.lower()
        
        # 7. Supprimer caractères spéciaux extrêmes (mais pas accents)
        # Garder : lettres, chiffres, accents, espaces, ponctuation basique
        text = re.sub(r'[^\w\s\-\.\'àâäæçéèêëìîïòôöœùûüœÿñ]', '', text)
        
        # 8. Supprimer espaces inutiles
        text = text.strip()
        
        return text
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 2 : TOKENIZATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizer avec spaCy
        
        JUSTIFICATIONS :
        - spaCy: Support français, POS tagging, pipeline modulaire
        - Alternatives:
          * NLTK: Plus flexible mais lent, moins support français
          * Regex: Rapide mais fragile sur edge cases
        """
        if not text:
            return []
        
        try:
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
            return tokens
        except Exception as e:
            logger.warning(f"Erreur tokenization: {str(e)}")
            # Fallback: split simple
            return text.split()
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 3 : CLEANING
    # ═══════════════════════════════════════════════════════════════════════
    
    def clean(self, text: str) -> List[str]:
        """
        Tokenize + clean
        
        Étapes :
        1. Tokenization
        2. Suppression stopwords français
        3. Suppression tokens très courts
        4. Lemmatization via spaCy
        
        JUSTIFICATIONS :
        - Stopwords: BERT/transformers gèrent bien, mais réduit bruit
        - Min length: Tokens 1-char peu informatifs
        - Lemmatization: Réduit sparsité (running→run)
        """
        if not text:
            return []
        
        try:
            doc = self.nlp(text)
            
            tokens = []
            for token in doc:
                # 1. Skip stopwords
                if token.is_stop:
                    continue
                
                # 2. Skip tokens trop courts
                if len(token.text) < self.config.get('preprocessing', {}).get('min_token_length', 2):
                    continue
                
                # 3. Skip punctuation/numbers uniquement
                if token.is_punct:
                    continue
                
                # 4. Lemmatization
                lemma = token.lemma_
                tokens.append(lemma)
            
            return tokens
        
        except Exception as e:
            logger.warning(f"Erreur cleaning: {str(e)}")
            return []
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 4 : PROCESS COMPLET
    # ═══════════════════════════════════════════════════════════════════════
    
    def process(self, text: str, include_metrics: bool = False) -> Dict:
        """
        Pipeline complet
        
        Returns:
            {
                'original': texte original,
                'normalized': texte normalisé,
                'tokens': liste tokens nettoyés,
                'num_tokens_original': compte tokens avant,
                'num_tokens_final': compte tokens après,
                'token_loss_pct': % tokens perdus
            }
        """
        # Étape 1: Normalisation
        normalized = self.normalize(text)
        
        # Étape 2: Tokenization brut
        tokens_raw = self.tokenize(normalized)
        
        # Étape 3: Cleaning
        tokens_clean = self.clean(normalized)
        
        # Calculer metrics
        num_original = len(tokens_raw) if tokens_raw else 0
        num_final = len(tokens_clean) if tokens_clean else 0
        token_loss_pct = (1 - num_final / max(num_original, 1)) * 100
        
        return {
            'original': text,
            'normalized': normalized,
            'tokens': tokens_clean,
            'num_tokens_original': num_original,
            'num_tokens_final': num_final,
            'token_loss_pct': round(token_loss_pct, 2)
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # PROCESS BATCH
    # ═══════════════════════════════════════════════════════════════════════
    
    def process_batch(self, articles: List) -> List[Dict]:
        """
        Traiter batch d'articles
        
        Optimisé avec nlp.pipe() pour speed
        """
        logger.info(f"🔧 Preprocessing {len(articles)} articles...")
        
        processed = []
        token_losses = []
        
        for i, article in enumerate(articles):
            try:
                result = self.process(article.content or article.title)
                
                # Enrichir article
                article_dict = article.to_dict()
                article_dict['tokens'] = result['tokens']
                article_dict['num_tokens'] = result['num_tokens_final']
                article_dict['token_loss_pct'] = result['token_loss_pct']
                article_dict['normalized_content'] = result['normalized']
                
                processed.append(article_dict)
                token_losses.append(result['token_loss_pct'])
                
                if (i + 1) % 20 == 0:
                    logger.info(f"  ✓ {i + 1}/{len(articles)} articles")
            
            except Exception as e:
                logger.warning(f"  ✗ Article {i}: {str(e)}")
                continue
        
        # Sauvegarder metrics
        self.quality_metrics = {
            'num_articles_processed': len(processed),
            'avg_token_loss_pct': round(sum(token_losses) / len(token_losses), 2) if token_losses else 0,
            'token_loss_distribution': {
                'min': round(min(token_losses), 2) if token_losses else 0,
                'max': round(max(token_losses), 2) if token_losses else 0,
                'median': round(sorted(token_losses)[len(token_losses)//2], 2) if token_losses else 0,
            }
        }
        
        logger.info(f"✅ Preprocessing terminé: {len(processed)} articles")
        return processed
    
    # ═══════════════════════════════════════════════════════════════════════
    # MÉTRIQUES QUALITÉ
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_quality_metrics(self) -> Dict:
        """Retourner métriques qualité"""
        return self.quality_metrics
    
    def print_quality_report(self):
        """Afficher rapport qualité"""
        if not self.quality_metrics:
            logger.warning("Aucune métrique disponible")
            return
        
        print("\n" + "="*70)
        print("📊 RAPPORT QUALITÉ PRÉTRAITEMENT")
        print("="*70)
        print(f"Articles traités: {self.quality_metrics['num_articles_processed']}")
        print(f"Perte tokens moyenne: {self.quality_metrics['avg_token_loss_pct']}%")
        print(f"  Min: {self.quality_metrics['token_loss_distribution']['min']}%")
        print(f"  Max: {self.quality_metrics['token_loss_distribution']['max']}%")
        print(f"  Médiane: {self.quality_metrics['token_loss_distribution']['median']}%")
        print("="*70)
    
    # ═══════════════════════════════════════════════════════════════════════
    # UTILITAIRES
    # ═══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _load_french_stopwords() -> set:
        """Charger stopwords français"""
        # Stopwords français courants
        stopwords = {
            'le', 'la', 'les', 'de', 'des', 'du', 'un', 'une', 'des',
            'et', 'ou', 'mais', 'donc', 'car', 'ni', 'soit',
            'à', 'au', 'aux', 'par', 'pour', 'avec', 'sans', 'sous',
            'dans', 'sur', 'entre', 'vers', 'chez', 'depuis', 'jusqu',
            'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
            'moi', 'toi', 'lui', 'elle', 'nous', 'vous', 'eux',
            'suis', 'es', 'est', 'sommes', 'êtes', 'sont',
            'ce', 'cela', 'celui', 'celle', 'ceux', 'celles',
            'que', 'qui', 'quoi', 'quel', 'quelle', 'quels', 'quelles',
            'où', 'quand', 'comment', 'pourquoi', 'combien',
            'très', 'trop', 'plus', 'moins', 'aussi', 'bien', 'mal',
            'ne', 'pas', 'rien', 'jamais', 'toujours', 'encore',
            'peu', 'beaucoup', 'assez', 'tout', 'autre'
        }
        return stopwords

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
    
    # Créer preprocessor
    preprocessor = TextPreprocessor(config)
    
    # Test sur texte simple
    test_text = """
    <p>🚀 Découvrez BERT 2.0 : L'révolution du NLP en 2025 !
    URL: https://example.com/article
    Auteur: @john_doe | 25 commentaires
    
    C'est incroyable, vraiment ! ✨ La nouvelle API HuggingFace...</p>
    """
    
    result = preprocessor.process(test_text)
    
    print("\n📝 TEST PREPROCESSING")
    print(f"Original ({len(result['original'])} chars): {result['original']}")
    print(f"Normalized ({len(result['normalized'])} chars): {result['normalized']}")
    print(f"Tokens ({result['num_tokens_final']} tokens): {result['tokens']}")
    print(f"Token loss: {result['token_loss_pct']}%")
