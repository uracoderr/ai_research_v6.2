"""
Phase 2: drop obvious junk/social links, then ask the 70B "quality" model
to score each remaining source's credibility so the best material gets
scraped and used in the report.

Fixes vs the original version:
- the bare `except:` that silently swallowed every possible failure
  (network errors, JSON errors, bad indices) is gone - failures are now
  specific and logged, so a real bug doesn't look identical to "the LLM
  just returned bad JSON"
- `fallback_ranker` no longer crashes on a malformed URL (the original
  did `url.split('/')[2]`, which raises IndexError on anything that
  isn't a full "scheme://host/..." URL)
"""
import json
import re
from typing import Dict, List, Tuple

from utils.llm_client import LLMError, call_nvidia_api
from utils.logger import get_logger

logger = get_logger(__name__)

SOCIAL_DOMAINS = [
    "facebook.com", "instagram.com", "twitter.com", "x.com",
    "youtube.com", "reddit.com", "tiktok.com", "pinterest.com",
    "linkedin.com", "snapchat.com", "tumblr.com", "quora.com",
    "medium.com/m/", "t.me", "telegram.me", "whatsapp.com",
]

# Patterns that suggest garbage third-level / forum / spam domains
JUNK_PATTERNS = [
    r"\.blogspot\.", r"\.wordpress\.com", r"\.weebly\.com", r"\.wixsite\.com",
    r"\.tripod\.com", r"\.angelfire\.com", r"\.geocities\.", r"\.fandom\.com",
    r"\.wikia\.com", r"answers\.yahoo", r"\.ask\.com", r"\.ehow\.com",
    r"\.livejournal\.com", r"\.hubpages\.com", r"\.ezinearticles\.com",
    r"\.selfgrowth\.com", r"clickbait", r"\.buzzfeed\.com", r"\.dailymail\.co",
]

HIGH_TRUST_DOMAINS = [
    ".gov", ".edu", ".ac.uk", ".ac.in", ".ac.jp",
    "reuters.com", "bloomberg.com", "mckinsey.com",
    "nature.com", "who.int", "un.org", "ieee.org",
    "sciencedirect.com", "pubmed.ncbi.nlm.nih.gov", "scholar.google.com",
    "jstor.org", "springer.com", "researchgate.net", "arxiv.org",
    "bbc.com", "nytimes.com", "theguardian.com", "ft.com",
    "wsj.com", "economist.com", "harvard.edu", "mit.edu",
    "stanford.edu", "oxford.ac.uk", "cambridge.org",
    "worldbank.org", "imf.org", "oecd.org", "statista.com",
    "pewresearch.org", "brookings.edu",
]


def _extract_json_array(text: str):
    """
    Robustly extract the first complete JSON array from LLM output.
    Uses bracket-counting so it stops at the first balanced ']' and
    ignores anything that follows — avoids the naive rfind(']') trap
    where trailing LLM commentary causes json.loads to raise 'Extra data'.
    """
    import re as _re
    cleaned = _re.sub(r"```(?:json)?\s*", "", text).strip()
    # Fast path: whole string is already valid JSON
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    # Bracket-counting walk
    start = cleaned.find("[")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(cleaned[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _source_name_from_url(url: str) -> str:
    try:
        return url.split("/")[2].replace("www.", "")
    except IndexError:
        return "Web Source"


def fallback_ranker(articles: List[Dict], top_n: int) -> Tuple[List[Dict], int, int, bool]:
    """Rule-based backup used when the LLM ranking call fails or returns unusable JSON."""
    for a in articles:
        url = a.get("url", "")
        a["credibility_score"] = 8 if any(d in url for d in HIGH_TRUST_DOMAINS) else 5
        a["source_name"] = _source_name_from_url(url)
    ranked = sorted(articles, key=lambda x: x["credibility_score"], reverse=True)
    return ranked[:top_n], 0, 0, False


def _is_junk_url(url: str) -> bool:
    """Returns True for social media, garbage blog farms, and low-quality third-level domains."""
    url_lower = url.lower()
    if any(d in url_lower for d in SOCIAL_DOMAINS):
        return True
    if any(re.search(pat, url_lower) for pat in JUNK_PATTERNS):
        return True
    return False


def filter_and_rank_articles(articles: List[Dict], top_n: int = 16) -> Tuple[List[Dict], int, int, bool]:
    """
    Returns (ranked_articles, social_links_dropped, llm_calls_made, llm_ranking_succeeded).
    """
    logger.info("PHASE 2: ranking %s articles for credibility", len(articles))
    clean_articles = [a for a in articles if not _is_junk_url(a.get("url", ""))]
    social_dropped = len(articles) - len(clean_articles)

    if not clean_articles:
        logger.warning("All %s articles were filtered out as social/junk links.", len(articles))
        return [], social_dropped, 0, False

    input_data = [{"i": i, "u": a.get("url"), "t": (a.get("title") or "")[:80]} for i, a in enumerate(clean_articles)]
    prompt = (
        "Score each source 1-10 for credibility. Return ONLY a raw JSON array: "
        '[{"i": 0, "s": "Source Name", "c": 9}]. '
        f"Data: {json.dumps(input_data)}"
    )

    try:
        raw = call_nvidia_api(
            prompt,
            system="Output strictly valid JSON arrays and nothing else.",
            max_tokens=1500,
            temperature=0.1,
            model="quality",
            retries=2,
        )
        ranked_data = _extract_json_array(raw)
        if not ranked_data:
            raise LLMError("Could not parse a JSON array out of the ranking response.")

        ranked_articles = []
        for item in ranked_data:
            idx = item.get("i")
            if idx is not None and 0 <= idx < len(clean_articles):
                original = clean_articles[idx]
                raw_score = item.get("c", 6)
                try:
                    original["credibility_score"] = int(raw_score)
                except (TypeError, ValueError):
                    original["credibility_score"] = 6
                original["source_name"] = item.get("s") or _source_name_from_url(original.get("url", ""))
                ranked_articles.append(original)

        if not ranked_articles:
            raise LLMError("Ranking response contained no usable items.")

        ranked_articles.sort(key=lambda x: x["credibility_score"], reverse=True)
        logger.info("LLM ranking succeeded: %s articles ranked.", len(ranked_articles))
        return ranked_articles[:top_n], social_dropped, 1, True

    except LLMError as e:
        logger.warning("LLM ranking failed (%s); using rule-based fallback ranker.", e)
        return fallback_ranker(clean_articles, top_n)
