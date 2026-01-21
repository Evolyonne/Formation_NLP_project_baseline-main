#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import io
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# --- Paths robustes ---
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)  # évite les soucis de lancement depuis un autre dossier

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Forcer UTF-8 pour la console Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ajouter src au path
sys.path.insert(0, str(BASE_DIR / 'src'))

from news_collector import NewsCollector
from text_preprocessor import TextPreprocessor
from news_classifier import NewsClassifier
from report_generator import ReportGenerator


def setup_logging(config: dict):
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
    logger.info("Configuration chargée depuis: config.json")
    return logger


def step_collect(config: dict, logger) -> list:
    logger.info("\n" + "="*75)
    logger.info("ÉTAPE 1 : COLLECTE ARTICLES")
    logger.info("="*75)

    collector = NewsCollector(config)
    articles = collector.collect_all()

    collector.save_to_jsonl(str(DATA_DIR / "articles_raw.jsonl"))
    collector.save_errors_log(str(DATA_DIR / "collection_errors.json"))

    logger.info(f"✅ ÉTAPE 1 COMPLÉTÉE : {len(articles)} articles collectés")
    return articles


def step_preprocess(articles: list, config: dict, logger) -> list:
    logger.info("\n" + "="*75)
    logger.info("ÉTAPE 2 : PRÉTRAITEMENT NLP")
    logger.info("="*75)

    preprocessor = TextPreprocessor(config)
    processed_articles = preprocessor.process_batch(articles)

    preprocessor.print_quality_report()

    with open(DATA_DIR / "articles_processed.jsonl", "w", encoding="utf-8") as f:
        for a in processed_articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    logger.info(f"✅ ÉTAPE 2 COMPLÉTÉE : {len(processed_articles)} articles traités")
    return processed_articles


def step_classify(articles: list, config: dict, logger) -> list:
    logger.info("\n" + "="*75)
    logger.info("ÉTAPE 3 : CLASSIFICATION & EXTRACTION")
    logger.info("="*75)

    classifier = NewsClassifier(config)
    classified_articles = classifier.classify_batch(articles)

    classifier.print_classification_summary(classified_articles)

    with open(DATA_DIR / "articles_classified.jsonl", "w", encoding="utf-8") as f:
        for a in classified_articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    logger.info(f"✅ ÉTAPE 3 COMPLÉTÉE : {len(classified_articles)} articles classifiés")
    return classified_articles


def step_generate_report(articles: list, config: dict, logger) -> str:
    logger.info("\n" + "="*75)
    logger.info("ÉTAPE 4 : GÉNÉRATION RAPPORT")
    logger.info("="*75)

    generator = ReportGenerator(config)
    report = generator.generate(articles)

    output_file = config.get("output", {}).get("report_name", "veille_report.txt")
    output_path = OUTPUT_DIR / output_file

    generator.save_report(report, str(output_path))
    logger.info("✅ ÉTAPE 4 COMPLÉTÉE : Rapport sauvegardé")

    return report


def main():
<<<<<<< HEAD
    """Pipeline complet"""
    
    # Créer répertoires
    Path('data').mkdir(exist_ok=True)
    Path('output').mkdir(exist_ok=True)
    
    # Charger configuration
=======
    CONFIG_PATH = BASE_DIR / "config.json"

>>>>>>> origin/main
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ ERREUR : config.json non trouvé")
        print("   Place config.json dans le répertoire courant")
        sys.exit(1)

    logger = setup_logging(config)

    try:
        articles = step_collect(config, logger)
        if not articles:
            logger.warning("⚠️ Aucun article collecté")
            return

        processed = step_preprocess(articles, config, logger)
        if not processed:
            logger.error("❌ Aucun article après prétraitement")
            return

        classified = step_classify(processed, config, logger)
        if not classified:
            logger.error("❌ Aucun article après classification")
            return

        report = step_generate_report(classified, config, logger)

        logger.info("\n" + "="*75)
        logger.info("✅ PIPELINE COMPLET - SUCCÈS")
        logger.info("="*75)
        logger.info(f"Rapport généré: {OUTPUT_DIR / config.get('output', {}).get('report_name', 'veille_report.txt')}")
        logger.info(f"Articles traités: {len(classified)}")
        logger.info(f"Temps total: {datetime.now().strftime('%H:%M:%S')}")
        logger.info("="*75)

        print("\n" + "="*75)
        print("📄 APERÇU RAPPORT")
        print("="*75)
        print("\n".join(report.split("\n")[:30]))
        print("...")
        print(f"\n✅ Rapport complet sauvegardé en {OUTPUT_DIR / config.get('output', {}).get('report_name', 'veille_report.txt')}")

    except Exception as e:
        logger.error(f"\n❌ ERREUR FATALE : {str(e)}")
        logger.exception("Traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
