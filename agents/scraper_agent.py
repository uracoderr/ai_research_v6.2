"""
Phase 3: fetch and clean the actual article text for the top-ranked
sources, in parallel, stopping as soon as we have "enough" good pages.

THE BIGGEST HIDDEN SPEED BUG IN THE OLD CODE WAS HERE, NOT IN THE LLM
CALLS: the old scraper used `with ThreadPoolExecutor(...) as executor:`.
Exiting a `with` block on an executor calls `shutdown(wait=True)` by
default - which blocks until *every submitted request finishes or times
out*, even after we already had enough successful scrapes and had
logically "moved on". With a 12s per-request timeout and a few slow or
blocked sites in every batch, that alone could silently add 10-20+
seconds of dead waiting to every single run.

The fix: submit all candidates, consume results via `as_completed`, and
the moment we hit the success target, cancel whatever is still in
flight (`cancel_futures=True`) and shut down without waiting.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup

import config
from utils.logger import get_logger

logger = get_logger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
JUNK_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]


def fetch_single_article(article: Dict) -> Dict:
    url = article.get("url", "")
    try:
        res = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=config.SCRAPE_TIMEOUT)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            for tag in soup(JUNK_TAGS):
                tag.extract()
            text = " ".join(p.get_text(strip=True) for p in soup.find_all(["p", "h1", "h2", "h3", "li"]))
            if len(text) > 200:
                snippet = text[: config.SCRAPE_CHAR_LIMIT]
                content = (
                    f"\n\n--- SOURCE: {article.get('source_name')} | "
                    f"SCORE: {article.get('credibility_score')}/10 | URL: {url} ---\n{snippet}\n"
                )
                return {"success": True, "reason": "Success", "content": content, "article": article}
            return {"success": False, "reason": "No_Content", "article": article}
        if res.status_code in (401, 403):
            return {"success": False, "reason": "Blocked_403", "article": article}
        return {"success": False, "reason": f"HTTP_{res.status_code}", "article": article}
    except requests.exceptions.Timeout:
        return {"success": False, "reason": "Timeout", "article": article}
    except requests.exceptions.RequestException:
        return {"success": False, "reason": "Error", "article": article}


def scrape_top_articles(ranked_articles: List[Dict], min_required: int = None) -> Tuple[str, int, List[Dict]]:
    """
    Returns (concatenated_scraped_text, success_count, scraped_source_metadata).
    `scraped_source_metadata` is the list of article dicts that were
    actually, successfully scraped - report_agent uses this to build a
    References section from real data instead of asking the LLM to
    invent citations.
    """
    min_required = min_required or config.SCRAPE_TARGET_SUCCESS
    candidates = ranked_articles
    logger.info("PHASE 3: scraping up to %s candidates (target: %s successes)", len(candidates), min_required)

    scraped_content = ""
    scraped_sources: List[Dict] = []
    stats = {"Requested": len(candidates), "Success": 0, "Blocked_403": 0, "Timeout": 0, "No_Content": 0, "Other_Errors": 0}

    if not candidates:
        return "", 0, []

    executor = ThreadPoolExecutor(max_workers=min(config.SCRAPE_MAX_WORKERS, len(candidates)))
    try:
        futures = {executor.submit(fetch_single_article, a): a for a in candidates}
        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                scraped_content += result["content"]
                scraped_sources.append(result["article"])
                stats["Success"] += 1
                if stats["Success"] >= min_required:
                    break
            else:
                reason = result["reason"]
                stats[reason] = stats.get(reason, 0) + 1
    finally:
        # Don't block on whatever is still in flight once we already have
        # enough good pages - see module docstring. `cancel_futures` needs
        # Python 3.9+; this project targets 3.10+ (see README).
        executor.shutdown(wait=False, cancel_futures=True)

    logger.info("Scraping summary: %s", stats)
    return scraped_content, stats["Success"], scraped_sources
