"""
Security helpers.

These close two real vulnerability classes that existed in earlier
versions of this app:

1. PATH TRAVERSAL - user-controlled "topic" strings were used directly
   to build filesystem paths, e.g.
   os.path.join(reports_dir, f"{req.topic}_context.txt")
   A topic like "../../../some_file" could make the app read or write
   outside the reports/ folder. `safe_slug()` fixes this: it strips
   everything except a-z, 0-9, "-" and "_", so a traversal sequence has
   nothing left to traverse with.

2. STORED / REFLECTED XSS - the report shown to the user is built from
   LLM output, and that LLM output is itself influenced by scraped web
   content. A malicious page could contain text designed to make the
   model emit an <script> tag or an onerror= handler, which the old
   frontend then inserted with raw `innerHTML`. `sanitize_html()` runs
   every bit of generated HTML through an allow-list before it reaches
   the browser, so only safe formatting tags ever survive.
"""
import html
import re
from typing import Optional

import bleach
from bs4 import BeautifulSoup

_SLUG_RE = re.compile(r"[^a-z0-9\-_]+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")

MAX_SLUG_LENGTH = 80
MIN_TOPIC_LENGTH = 3
MAX_TOPIC_LENGTH = 300

ALLOWED_TAGS = [
    "h1", "h2", "h3", "h4", "p", "ul", "ol", "li", "strong", "em", "b", "i",
    "a", "table", "thead", "tbody", "tr", "th", "td", "code", "pre",
    "blockquote", "br", "hr", "span",
]
ALLOWED_ATTRS = {"a": ["href", "title"]}
ALLOWED_PROTOCOLS = ["http", "https", "mailto"]


def safe_slug(text: Optional[str], max_length: int = MAX_SLUG_LENGTH) -> str:
    """
    Turn arbitrary user input into a filesystem- and URL-safe slug.
    Guaranteed to never contain "/", "\\", or return "", ".", or "..".
    Used for every filename and folder name derived from user input.
    """
    if not text:
        return "untitled"
    slug = str(text).strip().lower().replace(" ", "_")
    slug = _SLUG_RE.sub("", slug)
    slug = slug.strip("._-")
    if not slug or slug in {".", ".."}:
        slug = "untitled"
    return slug[:max_length] or "untitled"


def validate_topic(topic: Optional[str]) -> str:
    """Raise ValueError on missing/too short/too long input; else return cleaned text."""
    if topic is None:
        raise ValueError("Research topic is required.")
    cleaned = _CONTROL_CHARS_RE.sub("", str(topic)).strip()
    if len(cleaned) < MIN_TOPIC_LENGTH:
        raise ValueError(f"Research topic must be at least {MIN_TOPIC_LENGTH} characters.")
    if len(cleaned) > MAX_TOPIC_LENGTH:
        raise ValueError(f"Research topic must be under {MAX_TOPIC_LENGTH} characters.")
    return cleaned


def sanitize_html(dirty_html: str) -> str:
    """
    Allow-list sanitiser for any HTML that came from the LLM or from
    markdown-rendered LLM output, before it is ever sent to the browser.
    Also forces every surviving link to open safely in a new tab.
    """
    if not dirty_html:
        return ""
    cleaned = bleach.clean(
        dirty_html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRS,
        protocols=ALLOWED_PROTOCOLS,
        strip=True,
    )
    soup = BeautifulSoup(cleaned, "html.parser")
    for a in soup.find_all("a", href=True):
        a["target"] = "_blank"
        a["rel"] = "noopener noreferrer nofollow"
    return str(soup)


def escape_plain_text(text: Optional[str]) -> str:
    """
    For fields that should NEVER contain markup at all (e.g. a podcast
    speaker name or line). HTML-escapes so the raw characters render
    correctly wherever the frontend inserts them.
    """
    return html.escape(str(text) if text is not None else "", quote=True)
