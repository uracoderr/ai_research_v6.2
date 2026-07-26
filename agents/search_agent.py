"""
Phase 0 + Phase 1: clean up the user's query, then search the web for
candidate sources via Tavily.
"""
from typing import Dict, List

import requests

import config
from utils import cache
from utils.llm_client import LLMError, call_nvidia_api
from utils.logger import get_logger

logger = get_logger(__name__)


def optimize_query(query: str) -> str:
    """Fixes obvious typos in the query. Uses the fast 8B model - this is a
    trivial correction task and never needed 70B-level reasoning."""
    logger.info("PHASE 0: optimizing query: %r", query)
    prompt = (
        "You are a strict query optimizer. Correct any spelling mistakes or typos in this "
        "search query. Return ONLY the exact corrected query string, nothing else - no quotes, "
        f"no intro text, no markdown.\nOriginal: '{query}'"
    )
    try:
        corrected = call_nvidia_api(prompt, max_tokens=50, temperature=0.1, model="fast", retries=1)
        corrected = corrected.strip().strip("'\"")
        if corrected and corrected.lower() != query.lower():
            logger.info("Query auto-corrected to: %r", corrected)
            return corrected
        return query
    except LLMError as e:
        logger.warning("Query optimization skipped (%s); using raw query.", e)
        return query


def fetch_articles(query: str, max_results: int = None, search_depth: str = None) -> List[Dict]:
    """Searches Tavily for candidate sources. Cached for CACHE_TTL_SECONDS so
    repeated/similar topics don't re-pay the network + quota cost."""
    max_results = max_results or config.SEARCH_MAX_RESULTS
    search_depth = search_depth or config.SEARCH_DEPTH
    logger.info("PHASE 1: searching web for %r (depth=%s, max_results=%s)", query, search_depth, max_results)

    cache_key = cache.make_key("search", query, search_depth, str(max_results))
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info("Search cache hit for %r (%s results)", query, len(cached))
        return cached

    if not config.TAVILY_API_KEY:
        logger.error("TAVILY_API_KEY is not configured - cannot search.")
        return []

    payload = {
        "api_key": config.TAVILY_API_KEY,
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
    }
    try:
        response = requests.post(config.TAVILY_URL, json=payload, timeout=config.SEARCH_TIMEOUT)
        response.raise_for_status()
        results = response.json().get("results", [])
        logger.info("Fetched %s raw articles.", len(results))
        if results:
            cache.set(cache_key, results)
        return results
    except requests.exceptions.RequestException as e:
        logger.error("Tavily search error: %s", e)
        return []
