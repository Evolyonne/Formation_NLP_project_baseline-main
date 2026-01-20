# src/news_collector.py
"""
📥 Module Collecte d'Articles

Collecte articles de sources multiples :
- HackerNews (BeautifulSoup)
- RSS feeds (feedparser)
- YouTube (optionnel, API)

Usage:
    collector = NewsCollector(config)
    articles = collector.collect_all()
"""

import requests
from bs4 import BeautifulSoup
import feedparser
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import wikipediaapi


# ═══════════════════════════════════════════════════════════════════════════
# LOGGER
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Article:
    """Représentation standardisée d'un article"""
    title: str
    url: str
    content: str
    source: str
    date: Optional[str] = None
    author: Optional[str] = None
    score: Optional[int] = None  # Pour HN: points/upvotes
    
    def to_dict(self):
        return asdict(self)

# ═══════════════════════════════════════════════════════════════════════════
# NEWS COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════

class NewsCollector:
    """Collecte articles de multiples sources"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Dict de configuration (voir config.json)
        """
        self.config = config
        self.articles: List[Article] = []
        self.errors: List[Dict] = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; NewsCollector/1.0)'
        })

            # ═══════════════════════════════════════════════════════════════════════
    # WIKIPEDIA
    # ═══════════════════════════════════════════════════════════════════════

    def collect_from_wikipedia(self) -> List[Article]:
        """
        Récupère des pages Wikipédia à partir de titres (search_terms).
        NOTE: wikipedia-api ne fait pas une recherche "full search", il ouvre des pages.
        """
        cfg = self.config.get('collection', {}).get('wikipedia', {})
        if not cfg.get('enabled', False):
            return []

        logger.info("🔄 Collecte Wikipedia...")
        articles: List[Article] = []

        lang = cfg.get('language', 'fr')
        terms = cfg.get('search_terms', [])
        max_articles = int(cfg.get('max_articles', 10))

        try:
            wiki = wikipediaapi.Wikipedia(
                language=lang,
                user_agent='Mozilla/5.0 (compatible; NewsCollector/1.0; wikipedia-api)'
            )

            seen_urls = set()

            for term in terms:
                if len(articles) >= max_articles:
                    break

                try:
                    page = wiki.page(term)

                    if not page.exists():
                        logger.warning(f"  ✗ Wikipedia: page introuvable pour '{term}' (lang={lang})")
                        continue

                    url = page.fullurl
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    text = (page.text or "").strip()
                    if not text:
                        continue

                    # (optionnel) limiter taille pour éviter des fichiers énormes
                    text = text[:5000]

                    article = Article(
                        title=page.title,
                        url=url,
                        content=text,
                        source='Wikipedia',
                        date=None,
                        author=None,
                        score=None
                    )
                    articles.append(article)
                    logger.debug(f"    ✓ {page.title[:50]}...")

                except Exception as e:
                    self.errors.append({
                        'source': 'Wikipedia',
                        'term': term,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    continue

            logger.info(f"✅ Wikipedia: {len(articles)} articles collectés")
            return articles

        except Exception as e:
            logger.error(f"❌ Wikipedia collection failed: {str(e)}")
            self.errors.append({
                'source': 'Wikipedia',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return []

    
    # ═══════════════════════════════════════════════════════════════════════
    # HACKER NEWS
    # ═══════════════════════════════════════════════════════════════════════
    
    def collect_from_hacker_news(self) -> List[Article]:
        """
        Scrape Hacker News
        
        Structure HTML :
        <table class="itemlist">
          <tr class="athing">
            <td class="title">
              <span class="titleline">
                <a href="...">Title</a>
              </span>
            </td>
          </tr>
          <tr class="subtext">
            <span class="score">42 points</span>
            <a class="hnuser">author</a>
            <span class="age">2 hours ago</span>
            <a class="togg">↓</a>
            <a href="item?id=...">123 comments</a>
          </tr>
        </table>
        """
        logger.info("🔄 Collecte HackerNews...")
        articles = []
        
        try:
            config = self.config.get('collection', {}).get('hacker_news', {})
            url = config.get('url', 'https://news.ycombinator.com/')
            num_pages = config.get('num_pages', 2)
            timeout = config.get('timeout', 10)
            delay = config.get('delay_between_requests', 2)
            
            for page_num in range(num_pages):
                page_url = f"{url}?p={page_num + 1}" if page_num > 0 else url
                
                logger.info(f"  Scraping page {page_num + 1}/{num_pages}")
                
                try:
                    response = self.session.get(page_url, timeout=timeout)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    rows = soup.find_all('tr', class_='athing')
                    
                    for row in rows:
                        try:
                            # Extraire titre et URL
                            title_link = row.find('span', class_='titleline').find('a')
                            if not title_link:
                                continue
                            
                            title = title_link.get_text(strip=True)
                            article_url = title_link.get('href', '')
                            
                            # Extraire metadata (score, auteur, temps)
                            subtext_row = row.find_next('tr', class_='subtext')
                            if subtext_row:
                                score_span = subtext_row.find('span', class_='score')
                                score = int(score_span.get_text().split()[0]) if score_span else None
                                
                                author_link = subtext_row.find('a', class_='hnuser')
                                author = author_link.get_text() if author_link else None
                                
                                age_span = subtext_row.find('span', class_='age')
                                date = age_span.get('title') if age_span else None
                            else:
                                score = author = date = None
                            
                            article = Article(
                                title=title,
                                url=article_url,
                                content=title,  # Placeholder: titre = contenu initial
                                source='HackerNews',
                                date=date,
                                author=author,
                                score=score
                            )
                            
                            articles.append(article)
                            logger.debug(f"    ✓ {title[:50]}...")
                        
                        except Exception as e:
                            self.errors.append({
                                'source': 'HackerNews',
                                'error': str(e),
                                'timestamp': datetime.now().isoformat()
                            })
                            continue
                    
                    # Délai entre requêtes (respectueux)
                    time.sleep(delay)
                
                except requests.RequestException as e:
                    self.errors.append({
                        'source': 'HackerNews',
                        'page': page_num + 1,
                        'error': f"Request failed: {str(e)}",
                        'timestamp': datetime.now().isoformat()
                    })
                    logger.warning(f"  ✗ Erreur page {page_num + 1}: {str(e)}")
            
            logger.info(f"✅ HackerNews: {len(articles)} articles collectés")
            return articles
        
        except Exception as e:
            logger.error(f"❌ HackerNews collection failed: {str(e)}")
            self.errors.append({
                'source': 'HackerNews',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return []
    
    # ═══════════════════════════════════════════════════════════════════════
    # RSS FEEDS
    # ═══════════════════════════════════════════════════════════════════════
    
    def collect_from_rss_feed(self, feed_url: str) -> List[Article]:
        """
        Parse RSS feed
        
        Structure RSS standard :
        <rss>
          <channel>
            <item>
              <title>Article Title</title>
              <link>https://example.com/article</link>
              <description>Article summary</description>
              <pubDate>Sun, 15 Jan 2025 10:30:00 GMT</pubDate>
              <author>john@example.com</author>
            </item>
          </channel>
        </rss>
        """
        logger.info(f"🔄 Collecte RSS: {feed_url}")
        articles = []
        
        try:
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                logger.warning(f"  ⚠️ RSS parsing warning: {feed.bozo_exception}")
            
            for entry in feed.entries:
                try:
                    title = entry.get('title', 'No Title')
                    link = entry.get('link', '')
                    summary = entry.get('summary', '') or entry.get('description', '')
                    
                    # Parser date
                    date = None
                    if 'published' in entry:
                        date = entry.published
                    elif 'updated' in entry:
                        date = entry.updated
                    
                    author = None
                    if 'author' in entry:
                        author = entry.author
                    
                    article = Article(
                        title=title,
                        url=link,
                        content=summary[:1500],  # Premiers 200 chars du summary
                        source='RSS',
                        date=date,
                        author=author
                    )
                    
                    articles.append(article)
                    logger.debug(f"    ✓ {title[:50]}...")
                
                except Exception as e:
                    self.errors.append({
                        'source': 'RSS',
                        'feed': feed_url,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    continue
            
            logger.info(f"✅ RSS: {len(articles)} articles collectés")
            return articles
        
        except Exception as e:
            logger.error(f"❌ RSS collection failed: {str(e)}")
            self.errors.append({
                'source': 'RSS',
                'feed': feed_url,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return []
    
    # ═══════════════════════════════════════════════════════════════════════
    # YOUTUBE (OPTIONNEL)
    # ═══════════════════════════════════════════════════════════════════════
    
    def collect_from_youtube(self) -> List[Article]:
        """
        Récupère métadonnées vidéos YouTube
        
        NOTE: Nécessite API key YouTube
        Pour tester sans : retourner liste vide
        """
        logger.info("🔄 YouTube (optionnel, non-implémenté dans baseline)")
        return []
    
    # ═══════════════════════════════════════════════════════════════════════
    # COLLECTE TOUS SOURCES
    # ═══════════════════════════════════════════════════════════════════════
    
    def collect_all(self) -> List[Article]:
        """Collecte tous sources activées"""
        logger.info("=" * 70)
        logger.info("📥 COLLECTE ARTICLES - Début")
        logger.info("=" * 70)
        
        self.articles = []
        
        # HackerNews
        if self.config.get('collection', {}).get('hacker_news', {}).get('enabled', True):
            self.articles.extend(self.collect_from_hacker_news())
        
        # RSS
        if self.config.get('collection', {}).get('pycoders_rss', {}).get('enabled', True):
            rss_url = self.config['collection']['pycoders_rss'].get('url')
            if rss_url:
                self.articles.extend(self.collect_from_rss_feed(rss_url))
        
        # YouTube
        if self.config.get('collection', {}).get('youtube', {}).get('enabled', False):
            self.articles.extend(self.collect_from_youtube())
        
                # Wikipedia
        if self.config.get('collection', {}).get('wikipedia', {}).get('enabled', False):
            self.articles.extend(self.collect_from_wikipedia())

        
        logger.info("=" * 70)
        logger.info(f"📊 RÉSUMÉ COLLECTE")
        logger.info(f"   Total articles : {len(self.articles)}")
        logger.info(f"   Total erreurs : {len(self.errors)}")
        logger.info("=" * 70)
        
        return self.articles
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAUVEGARDE DONNÉES
    # ═══════════════════════════════════════════════════════════════════════
    
    def save_to_jsonl(self, filepath: str):
        """Sauvegarde articles en format JSONL"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for article in self.articles:
                    f.write(json.dumps(article.to_dict(), ensure_ascii=False) + '\n')
            logger.info(f"✅ Articles sauvegardés: {filepath}")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {str(e)}")
    
    def save_errors_log(self, filepath: str):
        """Sauvegarde log d'erreurs"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.errors, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Erreurs sauvegardées: {filepath}")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde log: {str(e)}")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN - TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    # Charger config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Collecter
    collector = NewsCollector(config)
    articles = collector.collect_all()
    
    # Sauvegarder
    collector.save_to_jsonl('data/articles_raw.jsonl')
    collector.save_errors_log('data/collection_errors.json')
    
    # Afficher sample
    if articles:
        print("\n📄 Premier article :")
        print(f"  Titre: {articles[0].title}")
        print(f"  URL: {articles[0].url}")
        print(f"  Source: {articles[0].source}")
