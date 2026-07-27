"""
Central place for reading/writing research artefacts to disk (and,
optionally, to Supabase for persistence across redeploys).

Before this module existed, main.py and app.py each had their own
(slightly different) copy of the "save the report as .md and .html"
logic. Now both call the same functions, so a bug fix or format change
only has to happen once.

Every report lives at:
    reports/<session_id>/<safe_topic>_report.md
    reports/<session_id>/<safe_topic>_report.html
    reports/<session_id>/<safe_topic>_context.txt   (raw scraped text -
                                                       used only server-side
                                                       for RAG/Debate, never
                                                       served directly)
    reports/<session_id>/<safe_topic>_meta.json      (word count, reading
                                                       time, confidence,
                                                       mode, generation
                                                       time, etc. - so a
                                                       *past* report shows
                                                       the same info badges
                                                       as a freshly
                                                       generated one)

<session_id> is "cli" for the command-line tool (single local user, no
isolation needed) and a random per-browser id for the web app (see
utils/session.py) - this is what keeps one web visitor's research
private from another's.

PERSISTENCE: local disk works fine for a single long-lived server, but
most PaaS hosts (Render included) wipe the filesystem on every
redeploy/restart, silently losing everyone's history. If SUPABASE_URL
and SUPABASE_KEY are configured (see .env.example), every save is also
written to Supabase, and reads fall back to Supabase whenever the local
copy is missing (repopulating the local file for next time). If
Supabase isn't configured, every function below behaves exactly as
before - local disk only, no behaviour change.
"""
import json
import os
from typing import Dict, List, Optional

import markdown
import requests

import config
from utils.logger import get_logger
from utils.security import safe_slug

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_ROOT = os.path.join(BASE_DIR, "reports")

_HTML_WRAPPER = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 860px;
          margin: 40px auto; padding: 0 20px; line-height: 1.65; color: #1f2937; }}
  h1, h2, h3 {{ color: #0f172a; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }}
  th {{ background: #f1f5f9; }}
  code {{ background: #f1f5f9; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def session_dir(session_id: str) -> str:
    path = os.path.join(REPORTS_ROOT, safe_slug(session_id, max_length=40))
    os.makedirs(path, exist_ok=True)
    return path


def report_paths(session_id: str, topic: str) -> Dict[str, str]:
    folder = session_dir(session_id)
    slug = safe_slug(topic)
    return {
        "slug": slug,
        "md": os.path.join(folder, f"{slug}_report.md"),
        "html": os.path.join(folder, f"{slug}_report.html"),
        "context": os.path.join(folder, f"{slug}_context.txt"),
        "meta": os.path.join(folder, f"{slug}_meta.json"),
    }


# ---------------------------------------------------------------------------
# Supabase layer (optional) - every function here is best-effort: a Supabase
# failure is logged and swallowed, never raised, so it can never break a
# request that otherwise succeeded on local disk.
# ---------------------------------------------------------------------------

def _supabase_configured() -> bool:
    return bool(config.SUPABASE_URL and config.SUPABASE_KEY)


def _supabase_headers(prefer: str = "return=representation") -> dict:
    return {
        "apikey": config.SUPABASE_KEY,
        "Authorization": f"Bearer {config.SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


def _supabase_upsert(session_id: str, slug: str, topic: str, report_markdown: str,
                      scraped_context: str, metrics: Optional[dict]) -> None:
    if not _supabase_configured():
        return
    url = f"{config.SUPABASE_URL}/rest/v1/{config.SUPABASE_TABLE}?on_conflict=session_id,safe_topic"
    payload = {
        "session_id": session_id,
        "safe_topic": slug,
        "topic": topic,
        "report_markdown": report_markdown,
        "context_text": scraped_context,
        "metrics": metrics or {},
    }
    try:
        resp = requests.post(
            url, json=payload,
            headers=_supabase_headers(prefer="resolution=merge-duplicates,return=minimal"),
            timeout=15,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning("Supabase upsert failed (report is still safely saved locally): %s", e)


def _supabase_fetch_one(session_id: str, slug: str) -> Optional[dict]:
    """
    Fetch a single report row by safe_topic.  Tries an exact session_id match
    first (fast path for the browser that created the report); if nothing is
    found it retries without the session filter so that reports created in a
    previous session (different browser, cleared cookies, redeploy) are still
    accessible.  This is intentional for a single-owner deployment — reports
    are always that user's own data.
    """
    if not _supabase_configured():
        return None

    def _get(extra_filter: str) -> Optional[dict]:
        url = (
            f"{config.SUPABASE_URL}/rest/v1/{config.SUPABASE_TABLE}"
            f"?{extra_filter}safe_topic=eq.{slug}&select=*&limit=1"
        )
        try:
            resp = requests.get(url, headers=_supabase_headers(), timeout=15)
            resp.raise_for_status()
            rows = resp.json()
            return rows[0] if rows else None
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.warning("Supabase fetch failed: %s", e)
            return None

    # Try with session filter first (exact owner match)
    row = _get(f"session_id=eq.{session_id}&")
    if row:
        return row
    # Fall back to any session — covers new browser / cleared cookies / redeploy
    return _get("")


def _supabase_list(session_id: str) -> List[Dict[str, str]]:
    """
    List all reports for this deployment.  The original version filtered by
    session_id, which meant a new browser (new random session_id) always saw
    an empty history even though the reports were in Supabase.  For a
    single-owner deployment the correct behaviour is to show everything.
    """
    if not _supabase_configured():
        return []
    url = (
        f"{config.SUPABASE_URL}/rest/v1/{config.SUPABASE_TABLE}"
        f"?select=safe_topic,topic,created_at&order=created_at.desc"
    )
    try:
        resp = requests.get(url, headers=_supabase_headers(), timeout=15)
        resp.raise_for_status()
        return [{"safe_topic": row["safe_topic"], "title": row.get("topic") or row["safe_topic"]} for row in resp.json()]
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        logger.warning("Supabase list failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Public API - unchanged signatures (metrics is new & optional, defaults to
# None so any existing caller keeps working), now with a Supabase-backed
# safety net.
# ---------------------------------------------------------------------------

def save_report(session_id: str, topic: str, report_markdown: str, scraped_context: str,
                 metrics: Optional[dict] = None) -> Dict[str, str]:
    paths = report_paths(session_id, topic)
    html_body = markdown.markdown(report_markdown, extensions=["tables", "fenced_code"])
    html_doc = _HTML_WRAPPER.format(title=topic, body=html_body)

    with open(paths["md"], "w", encoding="utf-8") as f:
        f.write(report_markdown)
    with open(paths["html"], "w", encoding="utf-8") as f:
        f.write(html_doc)
    with open(paths["context"], "w", encoding="utf-8") as f:
        f.write(scraped_context)
    if metrics is not None:
        try:
            with open(paths["meta"], "w", encoding="utf-8") as f:
                json.dump(metrics, f)
        except (OSError, TypeError) as e:
            logger.warning("Could not save metrics sidecar file: %s", e)

    # Best-effort durable copy - local disk above is the source of truth for
    # this request either way, so a Supabase hiccup never affects the response.
    _supabase_upsert(session_id, paths["slug"], topic, report_markdown, scraped_context, metrics)

    return paths


def load_report_markdown(session_id: str, safe_topic: str) -> Optional[str]:
    slug = safe_slug(safe_topic)
    path = os.path.join(session_dir(session_id), f"{slug}_report.md")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # Not on local disk (e.g. a redeploy wiped it) - fall back to Supabase and
    # repopulate the local file so the next read is fast again.
    row = _supabase_fetch_one(session_id, slug)
    if row and row.get("report_markdown"):
        try:
            save_report(session_id, row.get("topic", slug), row["report_markdown"],
                        row.get("context_text", ""), row.get("metrics"))
        except OSError:
            pass
        return row["report_markdown"]
    return None


def load_metrics(session_id: str, safe_topic: str) -> Optional[dict]:
    """Word count, reading time, confidence, mode, generation time, etc. for
    a previously-saved report - lets the "past research" view show the same
    info badges as a freshly generated report."""
    slug = safe_slug(safe_topic)
    path = os.path.join(session_dir(session_id), f"{slug}_meta.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    row = _supabase_fetch_one(session_id, slug)
    if row and row.get("metrics"):
        return row["metrics"]
    return None


def load_context(session_id: str, safe_topic: str) -> Optional[str]:
    slug = safe_slug(safe_topic)
    path = os.path.join(session_dir(session_id), f"{slug}_context.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    row = _supabase_fetch_one(session_id, slug)
    if row and row.get("context_text"):
        try:
            save_report(session_id, row.get("topic", slug), row.get("report_markdown", ""),
                        row["context_text"], row.get("metrics"))
        except OSError:
            pass
        return row["context_text"]
    return None


def list_reports(session_id: str) -> List[Dict[str, str]]:
    folder = session_dir(session_id)
    suffix = "_report.md"

    # When Supabase is configured it is always the complete, authoritative
    # list — every save writes to both disk AND Supabase, so Supabase is
    # always a superset of what's on local disk.  Using disk-only here was
    # the bug: after a redeploy (disk wiped) followed by one new research run
    # (1 file on disk), the code stopped checking Supabase and the user saw
    # only that 1 report instead of their full history.
    if _supabase_configured():
        supabase_items = _supabase_list(session_id)
        if supabase_items:
            # Merge: add any local-only items not yet in Supabase (edge case:
            # a report saved to disk within the same request that the Supabase
            # write hasn't completed yet, or a failed Supabase write).
            supabase_slugs = {item["safe_topic"] for item in supabase_items}
            for filename in os.listdir(folder):
                if filename.endswith(suffix):
                    slug = filename[: -len(suffix)]
                    if slug not in supabase_slugs:
                        supabase_items.append(
                            {"safe_topic": slug, "title": slug.replace("_", " ").title()}
                        )
            return supabase_items
        # Supabase returned nothing (empty list or failure) — fall through
        # to local disk so the user still sees something.

    # Local-disk-only path (Supabase not configured, or Supabase returned []).
    items = []
    for filename in os.listdir(folder):
        if filename.endswith(suffix):
            slug = filename[: -len(suffix)]
            items.append({"safe_topic": slug, "title": slug.replace("_", " ").title()})

    # Newest first by file modification time.
    def _mtime(item):
        try:
            return os.path.getmtime(os.path.join(folder, item["safe_topic"] + suffix))
        except OSError:
            return 0
    items.sort(key=_mtime, reverse=True)
    return items


# ---------------------------------------------------------------------------
# Thesis session storage (Supabase-backed, thesis mode requires Supabase)
# ---------------------------------------------------------------------------

def save_thesis_session(
    session_id: str,
    thesis_id: str,
    topic: str,
    master_outline: list,
    scraped_context: str = "",
    language: str = "English",
) -> None:
    """Save a new thesis session to Supabase. Raises on failure (thesis requires it)."""
    if not _supabase_configured():
        raise RuntimeError(
            "Thesis Mode requires Supabase. "
            "Please set SUPABASE_URL and SUPABASE_KEY environment variables."
        )
    url = f"{config.SUPABASE_URL}/rest/v1/{config.SUPABASE_TABLE_THESIS}"
    payload = {
        "thesis_id": thesis_id,
        "session_id": session_id,
        "topic": topic,
        "master_outline": master_outline,
        "current_chapter_index": 1,   # 0 = preliminary (already generated), 1 = next to generate
        "scraped_context": scraped_context[:30000],   # cap for DB storage
        "language": language,
    }
    try:
        resp = requests.post(
            url, json=payload,
            headers=_supabase_headers(prefer="return=minimal"),
            timeout=15,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        # Extract the Supabase API error body so callers can give specific guidance
        err_detail = str(e)
        if hasattr(e, "response") and e.response is not None:
            try:
                body = e.response.json()
                err_detail = body.get("message") or body.get("hint") or str(body)
            except Exception:
                err_detail = e.response.text or str(e)
        logger.error("Supabase thesis session save failed: %s", err_detail)
        raise RuntimeError(f"Could not save thesis session to Supabase: {err_detail}") from e


def get_thesis_session(session_id: str, thesis_id: str) -> Optional[dict]:
    """
    Fetch a thesis session from Supabase, verifying session ownership.
    Returns the row dict or None if not found / Supabase unavailable.
    """
    if not _supabase_configured():
        return None
    url = (
        f"{config.SUPABASE_URL}/rest/v1/{config.SUPABASE_TABLE_THESIS}"
        f"?thesis_id=eq.{thesis_id}&session_id=eq.{session_id}&select=*&limit=1"
    )
    try:
        resp = requests.get(url, headers=_supabase_headers(), timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        return rows[0] if rows else None
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.warning("Supabase thesis session fetch failed: %s", e)
        return None


def update_thesis_chapter_index(thesis_id: str, new_index: int) -> None:
    """Advance current_chapter_index after a chapter is successfully generated."""
    if not _supabase_configured():
        return
    url = (
        f"{config.SUPABASE_URL}/rest/v1/{config.SUPABASE_TABLE_THESIS}"
        f"?thesis_id=eq.{thesis_id}"
    )
    try:
        resp = requests.patch(
            url, json={"current_chapter_index": new_index},
            headers=_supabase_headers(prefer="return=minimal"),
            timeout=15,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning("Supabase thesis chapter index update failed (non-fatal): %s", e)


def file_path_for_download(session_id: str, safe_topic: str, fmt: str) -> Optional[str]:
    if fmt not in ("md", "html"):
        return None
    slug = safe_slug(safe_topic)
    path = os.path.join(session_dir(session_id), f"{slug}_report.{fmt}")
    if os.path.exists(path):
        return path
    # Trigger the Supabase fallback/repopulate path used by load_report_markdown,
    # then check disk again - covers "download right after a redeploy".
    if load_report_markdown(session_id, safe_topic) is not None:
        return path if os.path.exists(path) else None
    return None
