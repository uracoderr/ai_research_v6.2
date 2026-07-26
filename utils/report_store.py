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
    if not _supabase_configured():
        return None
    url = (
        f"{config.SUPABASE_URL}/rest/v1/{config.SUPABASE_TABLE}"
        f"?session_id=eq.{session_id}&safe_topic=eq.{slug}&select=*&limit=1"
    )
    try:
        resp = requests.get(url, headers=_supabase_headers(), timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        return rows[0] if rows else None
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.warning("Supabase fetch failed: %s", e)
        return None


def _supabase_list(session_id: str) -> List[Dict[str, str]]:
    if not _supabase_configured():
        return []
    url = (
        f"{config.SUPABASE_URL}/rest/v1/{config.SUPABASE_TABLE}"
        f"?session_id=eq.{session_id}&select=safe_topic,topic,created_at&order=created_at.desc"
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
    items = []
    for filename in os.listdir(folder):
        if filename.endswith(suffix):
            slug = filename[: -len(suffix)]
            items.append({"safe_topic": slug, "title": slug.replace("_", " ").title()})
    # Newest first, by actual file modification time - os.listdir() order is
    # filesystem-dependent and was never a reliable proxy for "most recent".
    items.sort(key=lambda item: os.path.getmtime(os.path.join(folder, item["safe_topic"] + suffix)), reverse=True)

    if not items:
        # Local disk has nothing for this session (e.g. right after a
        # redeploy) - fall back to the durable Supabase copy so history
        # isn't lost from the user's point of view.
        items = _supabase_list(session_id)
    return items


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
