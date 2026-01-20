#!/usr/bin/env python
# -*- coding: utf-8 -*-
# main.py
"""
🚀 MAIN SCRIPT - Pipeline Complet Veille NLP

Orchestre :
1. Collecte articles (news_collector.py)
2. Prétraitement NLP (text_preprocessor.py)
3. Classification & extraction (news_classifier.py)
4. Génération rapport (report_generator.py)

Usage:
    python main.py
    
Output:
    - data/articles_raw.jsonl : Articles bruts collectés
    - data/articles_processed.jsonl : Articles après prétraitement
    - data/articles_classified.jsonl : Articles après classification
    - output/veille_report.txt : Rapport final
"""

import sys
import io
import json
import logging
from pathlib import Path
from datetime import datetime

# Forcer UTF-8 pour la console Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from news_collector import NewsCollector
from text_preprocessor import TextPreprocessor
from news_classifier import NewsClassifier
from report_generator import ReportGenerator

# ═══════════════════════════════════════════════════════════════════════════
# SETUP LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(config: dict):
    """Configurer logging"""
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_file = config.get('logging', {}).get('log_file', 'veille_system.log')
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Pipeline Veille NLP - Démarrage")
    logger.info(f"Configuration chargée depuis: config.json")
    
    return logger

# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 : COLLECTE
# ═══════════════════════════════════════════════════════════════════════════

def step_collect(config: dict, logger) -> list:
    """Collecter articles de sources multiples"""
    logger.info("\n" + "="*75)
    logger.info("ÉTAPE 1 : COLLECTE ARTICLES")
    logger.info("="*75)
    
    try:
        collector = NewsCollector(config)
        articles = collector.collect_all()
        
        # Sauvegarder
        collector.save_to_jsonl('data/articles_raw.jsonl')
        collector.save_errors_log('data/collection_errors.json')
        
        logger.info(f"✅ ÉTAPE 1 COMPLÉTÉE : {len(articles)} articles collectés")
        return articles
    
    except Exception as e:
        logger.error(f"❌ ERREUR ÉTAPE 1 : {str(e)}")
        raise

# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 : PRÉTRAITEMENT
# ═══════════════════════════════════════════════════════════════════════════

def step_preprocess(articles: list, config: dict, logger) -> list:
    """Prétraiter articles (normalisation + tokenization)"""
    logger.info("\n" + "="*75)
    logger.info("ÉTAPE 2 : PRÉTRAITEMENT NLP")
    logger.info("="*75)
    
    try:
        preprocessor = TextPreprocessor(config)
        processed_articles = preprocessor.process_batch(articles)
        
        # Afficher rapport qualité
        preprocessor.print_quality_report()
        
        # Sauvegarder
        with open('data/articles_processed.jsonl', 'w', encoding='utf-8') as f:
            for article in processed_articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ ÉTAPE 2 COMPLÉTÉE : {len(processed_articles)} articles traités")
        return processed_articles
    
    except Exception as e:
        logger.error(f"❌ ERREUR ÉTAPE 2 : {str(e)}")
        raise

# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 : CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def step_classify(articles: list, config: dict, logger) -> list:
    """Classifier articles (topics, sentiments, duplicates)"""
    logger.info("\n" + "="*75)
    logger.info("ÉTAPE 3 : CLASSIFICATION & EXTRACTION")
    logger.info("="*75)
    
    try:
        classifier = NewsClassifier(config)
        classified_articles = classifier.classify_batch(articles)
        
        # Afficher résumé
        classifier.print_classification_summary(classified_articles)
        
        # Sauvegarder
        with open('data/articles_classified.jsonl', 'w', encoding='utf-8') as f:
            for article in classified_articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ ÉTAPE 3 COMPLÉTÉE : {len(classified_articles)} articles classifiés")
        return classified_articles
    
    except Exception as e:
        logger.error(f"❌ ERREUR ÉTAPE 3 : {str(e)}")
        raise

# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 : GÉNÉRATION RAPPORT
# ═══════════════════════════════════════════════════════════════════════════

def step_generate_report(articles: list, config: dict, logger) -> str:
    """Générer rapport final"""
    logger.info("\n" + "="*75)
    logger.info("ÉTAPE 4 : GÉNÉRATION RAPPORT")
    logger.info("="*75)
    
    try:
        generator = ReportGenerator(config)
        report = generator.generate(articles)
        
        # Sauvegarder
        output_file = config.get('output', {}).get('report_name', 'veille_report.txt')
        output_dir = config.get('output', {}).get('output_dir', './output')
        
        # Créer répertoire s'il n'existe pas
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_path = Path(output_dir) / output_file
        generator.save_report(report, str(output_path))
        
        logger.info(f"✅ ÉTAPE 4 COMPLÉTÉE : Rapport sauvegardé")
        
        return report
    
    except Exception as e:
        logger.error(f"❌ ERREUR ÉTAPE 4 : {str(e)}")
        raise

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Pipeline complet"""
    
    # Créer répertoires
    Path('data').mkdir(exist_ok=True)
    Path('output').mkdir(exist_ok=True)
    
    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = BASE_DIR / "config.json"

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ ERREUR : config.json non trouvé")
        print(f"Chemin attendu : {CONFIG_PATH}")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(config)
    
    try:
        # ═══════════════════════════════════════════════════════════════════
        # ÉTAPE 1 : Collecter
        # ═══════════════════════════════════════════════════════════════════
        articles = step_collect(config, logger)
        
        if not articles:
            logger.warning("⚠️  Aucun article collecté")
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # ÉTAPE 2 : Prétraiter
        # ═══════════════════════════════════════════════════════════════════
        processed_articles = step_preprocess(articles, config, logger)
        
        if not processed_articles:
            logger.error("❌ Aucun article après prétraitement")
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # ÉTAPE 3 : Classifier
        # ═══════════════════════════════════════════════════════════════════
        classified_articles = step_classify(processed_articles, config, logger)
        
        if not classified_articles:
            logger.error("❌ Aucun article après classification")
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # ÉTAPE 4 : Générer rapport
        # ═══════════════════════════════════════════════════════════════════
        report = step_generate_report(classified_articles, config, logger)
        
        # ═══════════════════════════════════════════════════════════════════
        # RÉSUMÉ FINAL
        # ═══════════════════════════════════════════════════════════════════
        logger.info("\n" + "="*75)
        logger.info("✅ PIPELINE COMPLET - SUCCÈS")
        logger.info("="*75)
        logger.info(f"Rapport généré: output/{config.get('output', {}).get('report_name', 'veille_report.txt')}")
        logger.info(f"Articles traités: {len(classified_articles)}")
        logger.info(f"Temps total: {datetime.now().strftime('%H:%M:%S')}")
        logger.info("="*75)
        
        # Afficher début du rapport
        print("\n" + "="*75)
        print("📄 APERÇU RAPPORT")
        print("="*75)
        lines = report.split('\n')[:30]
        print('\n'.join(lines))
        print("...")
        print(f"\n✅ Rapport complet sauvegardé en output/{config.get('output', {}).get('report_name', 'veille_report.txt')}")
    
    except Exception as e:
        logger.error(f"\n❌ ERREUR FATALE : {str(e)}")
        logger.exception("Traceback:")
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
