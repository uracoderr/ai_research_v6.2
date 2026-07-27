"""
ThesisPilot web app (FastAPI).

Session model: every browser gets an anonymous, signed session cookie
the first time it visits (see utils/session.py). All research reports
and RAG context are stored per-session on disk, so one visitor never
sees another visitor's research history - the old app used a single
shared reports/ folder exposed via a raw static mount, which leaked
every user's history to every other visitor.

Run with:  uvicorn app:app --reload        (dev)
           uvicorn app:app --host 0.0.0.0 --port 8000   (prod, behind a real ASGI server / Docker)
"""
import asyncio
import io
import os
import time
import traceback
import uuid

import markdown
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from pypdf import PdfReader
import config
from agents.filter_agent import filter_and_rank_articles
from agents.report_agent import (
    challenge_query,
    generate_diagram,
    generate_flashcards,
    generate_podcast_script,
    generate_quiz,
    generate_report,
    generate_slides,
    generate_thesis_chapter,
    generate_thesis_outline,
    generate_thesis_preliminary,
    grade_short_answer,
    humanize_report,
    rag_query,
)
from agents.scraper_agent import scrape_top_articles
from agents.search_agent import fetch_articles, optimize_query
from utils import report_store, session
from utils.llm_client import synthesize_speech
from utils.logger import get_logger, setup_logging
from utils.security import safe_slug, sanitize_html, validate_topic

setup_logging(pretty=False)
logger = get_logger(__name__)

try:
    config.validate_config()
except RuntimeError as e:
    logger.error(str(e))
    # Don't hard-crash a dev server import (e.g. `uvicorn app:app --reload` before
    # .env is filled in) - but a production deployment should fail fast and loud.
    if config.APP_ENV == "production":
        raise

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI(title="ThesisPilot AI Research Agent")

# CORS is only added if the operator explicitly configures extra origins.
# Same-origin browser usage (the normal case - this same app serves the
# frontend) never goes through CORS in the first place, so the secure
# default here is to not add permissive cross-origin/credentialed access.
if config.ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )


def render_template(request: Request, name: str, context: dict = None) -> HTMLResponse:
    """Small compatibility shim: the TemplateResponse(request, name, ctx) vs
    TemplateResponse(name, {"request": request, ...}) signature changed
    across Starlette versions, so support both."""
    context = context or {}
    try:
        return templates.TemplateResponse(request, name, context)
    except TypeError:
        context["request"] = request
        return templates.TemplateResponse(name, context)


def require_access_key(request: Request) -> None:
    """Optional shared-secret gate. A no-op unless SAAS_ACCESS_KEY is set in
    the environment - see .env.example. Not a substitute for real per-user
    auth; see README.md -> 'Scaling this further'."""
    if not config.SAAS_ACCESS_KEY:
        return
    provided = request.headers.get("X-API-Key")
    if provided != config.SAAS_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing access key.")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=300)
    language: str = "english"
    mode: str = config.DEFAULT_REPORT_MODE


class ReportTextRequest(BaseModel):
    report_text: str = Field(..., max_length=20000)
    language: str = "english"


class QuizRequest(BaseModel):
    report_text: str = Field(..., max_length=20000)
    num_questions: int = Field(5, ge=1, le=20)
    language: str = "english"


class FlashcardsRequest(BaseModel):
    report_text: str = Field(..., max_length=20000)
    num_cards: int = Field(10, ge=5, le=20)
    language: str = "english"


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    voice_kind: str = "female"    # "male" | "female"
    accent: str = "american"      # "american" | "indian"


class InteractiveRequest(BaseModel):
    safe_topic: str = Field(..., max_length=120)
    query: str = Field(..., min_length=1, max_length=1000)
    language: str = "english"


class GradeAnswerRequest(BaseModel):
    question: str = Field(..., max_length=500)
    key_points: str = Field("", max_length=500)
    answer: str = Field(..., min_length=1, max_length=2000)
    language: str = "english"


class ThesisStartRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=300)
    language: str = "english"


class ThesisNextChapterRequest(BaseModel):
    thesis_id: str = Field(..., min_length=8, max_length=64)
    chapter_index: int = Field(..., ge=1, le=20)


# ---------------------------------------------------------------------------
# Pages & metadata
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    resp = render_template(request, "index.html", {})
    session.ensure_session(request, resp)
    return resp


@app.get("/api/meta")
async def get_meta(request: Request, response: Response):
    session.ensure_session(request, response)
    modes = [
        {"id": m["id"], "label": m["label"], "description": m["description"]}
        for m in config.REPORT_MODES.values()
    ]
    # Thesis mode uses a completely separate pipeline — append it as a special entry
    modes.append({
        "id": "thesis",
        "label": "📜 Thesis Mode",
        "description": (
            "Auto-generates a Master Outline then builds the thesis chapter by chapter "
            "(~2000-2500 words each). Requires Supabase."
        ),
    })
    return {
        "access_required": bool(config.SAAS_ACCESS_KEY),
        "default_mode": config.DEFAULT_REPORT_MODE,
        "modes": modes,
    }


# ---------------------------------------------------------------------------
# Research pipeline - background job + polling (NOT a single long-lived
# streaming connection)
#
# The old version ran the whole pipeline as one SSE (server-sent events)
# HTTP response. That meant the ENTIRE multi-minute pipeline lived and died
# with a single client connection - the moment a mobile browser backgrounds
# or throttles the tab (which Android Chrome does aggressively to save
# battery), the connection drops and the whole job is lost, surfacing as
# "network error" with no way to recover.
#
# Now: POST /api/research/start kicks off the pipeline as a background
# asyncio task tied to the SERVER's event loop, not to any client
# connection, and returns a job_id immediately. The browser polls
# GET /api/research/status/{job_id} every couple of seconds. If a poll
# fails (tab backgrounded, brief network blip, whatever), the job keeps
# running on the server regardless - the next successful poll just picks
# up wherever the job currently is. Nothing is lost.
# ---------------------------------------------------------------------------
RESEARCH_JOBS: dict = {}
THESIS_JOBS: dict = {}
_background_tasks: set = set()
_JOB_TTL_SECONDS = 3600  # stop tracking jobs older than this (finished or not)

# ---------------------------------------------------------------------------
# Upgrade 4: Lightweight in-memory report cache
# If two users research the exact same topic+mode+language within the TTL
# window, the second request is served instantly from cache — no API calls.
# Capped at MAX_REPORT_CACHE_SIZE entries (evict oldest) to prevent unbounded
# memory growth on the free tier (0.1 vCPU / 512 MB).
# ---------------------------------------------------------------------------
REPORT_CACHE: dict = {}
_REPORT_CACHE_TTL = 21600   # 6 hours
_MAX_REPORT_CACHE_SIZE = 40  # ~40 topics × ~20 KB each ≈ < 1 MB overhead


def _report_cache_key(topic: str, mode: str, language: str) -> str:
    return f"{topic.strip().lower()}|{mode}|{language.strip().lower()}"


def _get_cached_report(key: str):
    entry = REPORT_CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry["cached_at"] > _REPORT_CACHE_TTL:
        REPORT_CACHE.pop(key, None)
        return None
    return entry["result"]


def _set_cached_report(key: str, result: dict) -> None:
    if len(REPORT_CACHE) >= _MAX_REPORT_CACHE_SIZE:
        # Evict the oldest entry
        oldest = min(REPORT_CACHE, key=lambda k: REPORT_CACHE[k]["cached_at"])
        REPORT_CACHE.pop(oldest, None)
    REPORT_CACHE[key] = {"result": result, "cached_at": time.time()}


def _spawn_background(coro) -> None:
    """Fire-and-forget a coroutine on the running event loop, keeping a
    strong reference until it completes so it can't be garbage-collected
    mid-flight (a well-known asyncio footgun with bare create_task calls)."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _prune_old_jobs() -> None:
    cutoff = time.time() - _JOB_TTL_SECONDS
    for jobs_dict in (RESEARCH_JOBS, THESIS_JOBS):
        stale = [jid for jid, job in jobs_dict.items() if job.get("created_at", 0) < cutoff]
        for jid in stale:
            jobs_dict.pop(jid, None)


async def run_research_job(job_id: str, clean_topic: str, mode: str, language: str, session_id: str) -> None:
    """The actual pipeline - identical steps to before, just reporting
    progress into RESEARCH_JOBS instead of yielding SSE events."""
    start_time = time.time()
    llm_calls = 0

    def update(message: str) -> None:
        if job_id in RESEARCH_JOBS:
            RESEARCH_JOBS[job_id]["message"] = message

    try:
        # Upgrade 4: serve from cache if available (same topic + mode + language)
        cache_key = _report_cache_key(clean_topic, mode, language)
        cached = _get_cached_report(cache_key)
        if cached:
            update("⚡ Serving from cache (instant)...")
            await asyncio.sleep(0.3)  # tiny pause so the terminal shows the message
            cached_result = dict(cached)
            cached_result["metrics"] = dict(cached_result.get("metrics", {}))
            cached_result["metrics"]["from_cache"] = True
            cached_result["metrics"]["time_seconds"] = 0
            if job_id in RESEARCH_JOBS:
                RESEARCH_JOBS[job_id].update({"status": "done", "message": "✅ Served from cache!", "result": cached_result})
            return

        def _set_stat(key: str, value) -> None:
            """Update a single phase_stats field without overwriting others."""
            if job_id in RESEARCH_JOBS:
                RESEARCH_JOBS[job_id]["phase_stats"][key] = value

        update("▶ PHASE 0: Understanding your topic...")
        optimized_topic = await asyncio.to_thread(optimize_query, clean_topic)
        llm_calls += 1
        update(f"✨ Query refined to: '{optimized_topic}'")

        update("▶ PHASE 1: Searching the web...")
        raw_articles = await asyncio.to_thread(fetch_articles, optimized_topic)
        if not raw_articles:
            RESEARCH_JOBS[job_id].update({"status": "error", "error": "No articles found for this topic. Try rephrasing it."})
            return
        _set_stat("found", len(raw_articles))
        update(f"✅ Found {len(raw_articles)} candidate sources.")

        update("▶ PHASE 2: Ranking source credibility...")
        ranked_articles, duplicates_removed, filter_calls, llm_success = await asyncio.to_thread(
            filter_and_rank_articles, raw_articles, config.SEARCH_MAX_RESULTS
        )
        llm_calls += filter_calls
        _set_stat("ranked", len(ranked_articles))
        update(f"✅ {len(ranked_articles)} high-quality sources selected.")

        update("▶ PHASE 3: Reading full articles...")
        scraped_data, scraped_count, scraped_sources = await asyncio.to_thread(scrape_top_articles, ranked_articles)
        if scraped_count == 0:
            RESEARCH_JOBS[job_id].update({"status": "error", "error": "Could not read any sources. Please try again."})
            return
        _set_stat("extracted", scraped_count)
        update(f"✅ Extracted content from {scraped_count} sources.")

        num_sections = len(config.REPORT_MODES[mode]["sections"])
        _set_stat("section_total", num_sections)
        update("▶ PHASE 4: Writing your report...")
        stats_dict = {
            "scraped_success": scraped_count,
            "duplicates_removed": duplicates_removed,
            "llm_ranking_success": llm_success,
        }

        def _on_section_done(section_title: str, done: int, total: int) -> None:
            msg = f"✍️ Writing ({done}/{total}): {section_title}…"
            if job_id in RESEARCH_JOBS:
                RESEARCH_JOBS[job_id]["message"] = msg
                RESEARCH_JOBS[job_id]["phase_stats"]["section_done"] = done
                RESEARCH_JOBS[job_id]["phase_stats"]["section_total"] = total
                RESEARCH_JOBS[job_id]["phase_stats"]["section_title"] = section_title

        final_report, meta = await asyncio.to_thread(
            generate_report, optimized_topic, scraped_data, scraped_sources,
            language, stats_dict, mode, _on_section_done
        )
        llm_calls += len(config.REPORT_MODES[mode]["sections"])

        metrics = {
            "time_seconds": round(time.time() - start_time, 1),
            "llm_calls": llm_calls,
            "sources_found": len(raw_articles),
            "sources_used": scraped_count,
            "word_count": meta["word_count"],
            "reading_minutes": meta["reading_minutes"],
            "confidence_score": meta["confidence_score"],
            "mode_label": meta["mode_label"],
            "model_used": meta["model_used"],
        }

        paths = await asyncio.to_thread(
            report_store.save_report, session_id, optimized_topic, final_report, scraped_data, metrics
        )
        metrics["safe_topic"] = paths["slug"]
        metrics["md_download"] = f"/api/report/{paths['slug']}/download/md"
        metrics["html_download"] = f"/api/report/{paths['slug']}/download/html"

        report_html = sanitize_html(markdown.markdown(final_report, extensions=["tables", "fenced_code"]))

        final_result = {"topic": optimized_topic, "report": report_html, "metrics": metrics}

        # Upgrade 4: store in report cache for subsequent identical queries
        _set_cached_report(cache_key, final_result)

        if job_id in RESEARCH_JOBS:
            RESEARCH_JOBS[job_id].update({
                "status": "done",
                "message": "✅ Report ready!",
                "result": final_result,
            })
        logger.info(
            "Research complete: session=%s topic=%r mode=%s time=%.1fs",
            session_id, optimized_topic, mode, metrics["time_seconds"],
        )

    except Exception as e:
        logger.error("Pipeline error (job=%s): %s\n%s", job_id, e, traceback.format_exc())
        # Never expose raw exception text to the client — it may leak internal
        # paths, model names, or stack details. Map known transient failure
        # types to friendly messages; everything else gets a generic fallback.
        err_str = str(e)
        if "timeout" in err_str.lower() or "timed out" in err_str.lower():
            friendly_error = "Research took too long — the AI model timed out. Try a narrower topic or Flash mode."
        elif "rate" in err_str.lower() or "429" in err_str:
            friendly_error = "The AI model is busy right now. Please wait a moment and try again."
        elif "NVIDIA_API_KEY" in err_str or "TAVILY_API_KEY" in err_str:
            friendly_error = "A required API key is not configured. Please contact the administrator."
        elif "No articles found" in err_str:
            friendly_error = err_str  # already a user-facing message
        elif "Could not read any sources" in err_str:
            friendly_error = err_str  # already a user-facing message
        else:
            friendly_error = "Something went wrong during research. Please try a different topic or try again shortly."
        if job_id in RESEARCH_JOBS:
            RESEARCH_JOBS[job_id].update({"status": "error", "error": friendly_error})


# ---------------------------------------------------------------------------
# Thesis pipeline — background job (same polling pattern as research jobs)
# ---------------------------------------------------------------------------
async def run_thesis_job(job_id: str, thesis_id: str, clean_topic: str, language: str, session_id: str) -> None:
    """
    Progressive thesis pipeline:
      1. Web search + scrape (same phases as research pipeline)
      2. Generate Master Thesis Outline (JSON, 7 sections) via LLM
      3. Save outline + scraped context to Supabase thesis_sessions table
      4. Generate Preliminary Pages section immediately
      5. Return result — the frontend renders prelim pages then drives
         chapter generation via POST /api/thesis/next-chapter
    """
    start_time = time.time()

    def update(message: str) -> None:
        if job_id in THESIS_JOBS:
            THESIS_JOBS[job_id]["message"] = message

    try:
        update("▶ PHASE 0: Understanding your topic...")
        optimized_topic = await asyncio.to_thread(optimize_query, clean_topic)
        update(f"✨ Query refined to: '{optimized_topic}'")

        update("▶ PHASE 1: Searching the web for source material...")
        raw_articles = await asyncio.to_thread(fetch_articles, optimized_topic)
        if not raw_articles:
            THESIS_JOBS[job_id].update({"status": "error", "error": "No articles found for this topic. Try rephrasing it."})
            return

        update("▶ PHASE 2: Ranking source credibility...")
        ranked_articles, _, _, _ = await asyncio.to_thread(
            filter_and_rank_articles, raw_articles, config.SEARCH_MAX_RESULTS
        )

        update("▶ PHASE 3: Reading full articles...")
        scraped_data, scraped_count, scraped_sources = await asyncio.to_thread(scrape_top_articles, ranked_articles)
        if scraped_count == 0:
            THESIS_JOBS[job_id].update({"status": "error", "error": "Could not read any sources. Please try again."})
            return

        update("▶ Generating Master Thesis Outline (this may take 20-30 seconds)...")
        master_outline = await asyncio.to_thread(
            generate_thesis_outline, optimized_topic, scraped_data, language
        )
        update(f"✅ Outline ready ({len(master_outline)} sections). Saving to Supabase...")

        # Save thesis session (raises if Supabase not configured)
        await asyncio.to_thread(
            report_store.save_thesis_session,
            session_id, thesis_id, optimized_topic, master_outline, scraped_data, language
        )

        update("✍️ Generating Preliminary Pages...")
        preliminary_md = await asyncio.to_thread(
            generate_thesis_preliminary, optimized_topic, master_outline, scraped_data, language
        )

        header_md = (
            f"# {optimized_topic.title()} — Academic Thesis\n\n"
            f"> 📜 **Thesis Mode** | Master outline generated with {len(master_outline)} sections\n\n"
        )
        full_preliminary_md = header_md + preliminary_md
        preliminary_html = sanitize_html(markdown.markdown(full_preliminary_md, extensions=["tables", "fenced_code"]))

        elapsed = round(time.time() - start_time, 1)
        metrics = {
            "time_seconds": elapsed,
            "llm_calls": 3,
            "sources_found": len(raw_articles),
            "sources_used": scraped_count,
            "word_count": len(full_preliminary_md.split()),
            "reading_minutes": 2,
            "confidence_score": 90,
            "mode_label": "Thesis Mode",
            "model_used": f"NVIDIA Llama-3.1 ({config.MODEL_QUALITY.split('/')[-1] if '/' in config.MODEL_QUALITY else config.MODEL_QUALITY})",
        }

        # Also save preliminary as a regular report so it appears in history
        paths = await asyncio.to_thread(
            report_store.save_report,
            session_id, optimized_topic + " Thesis", full_preliminary_md, scraped_data, metrics
        )
        metrics["safe_topic"] = paths["slug"]
        metrics["md_download"] = f"/api/report/{paths['slug']}/download/md"
        metrics["html_download"] = f"/api/report/{paths['slug']}/download/html"

        final_result = {
            "topic": optimized_topic,
            "report": preliminary_html,
            "metrics": metrics,
            "thesis_id": thesis_id,
            "master_outline": master_outline,
            "current_chapter_index": 1,   # 0 = preliminary done; 1 = first chapter to generate
        }

        if job_id in THESIS_JOBS:
            THESIS_JOBS[job_id].update({
                "status": "done",
                "message": "✅ Preliminary Pages ready! Generate your first chapter below.",
                "result": final_result,
            })
        logger.info(
            "Thesis outline+preliminary complete: session=%s topic=%r sections=%d time=%.1fs",
            session_id, optimized_topic, len(master_outline), elapsed,
        )

    except Exception as e:
        logger.error("Thesis pipeline error (job=%s): %s\n%s", job_id, e, traceback.format_exc())
        err_str = str(e)
        if "timeout" in err_str.lower() or "timed out" in err_str.lower():
            friendly = "Research took too long. Try a narrower topic."
        elif "429" in err_str or "rate" in err_str.lower():
            friendly = "The AI model is busy right now. Please wait a moment and try again."
        elif "Supabase" in err_str or "supabase" in err_str.lower():
            if (
                "does not exist" in err_str.lower()
                or "thesis_sessions" in err_str.lower()
                or "relation" in err_str.lower()
            ):
                friendly = (
                    "Could not save thesis session — the 'thesis_sessions' table is missing in Supabase. "
                    "Please create it using the SQL in replit.md."
                )
            elif not config.SUPABASE_URL or not config.SUPABASE_KEY:
                friendly = "Thesis Mode requires Supabase. Please set SUPABASE_URL and SUPABASE_KEY."
            else:
                friendly = (
                    "Could not save thesis session to Supabase. "
                    "Check that your SUPABASE_URL and SUPABASE_KEY are correct and the thesis_sessions table exists."
                )
        elif "outline" in err_str.lower():
            friendly = "Could not generate a thesis outline. Please try again or rephrase your topic."
        else:
            friendly = "Something went wrong generating the thesis. Please try again."
        if job_id in THESIS_JOBS:
            THESIS_JOBS[job_id].update({"status": "error", "error": friendly})


@app.post("/api/research/start", dependencies=[Depends(require_access_key)])
async def api_start_research(request: Request, payload: ResearchRequest):
    existing_session = session.read_session_id(request)
    session_id = existing_session or uuid.uuid4().hex

    try:
        clean_topic = validate_topic(payload.topic)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    mode = payload.mode if payload.mode in config.REPORT_MODES else config.DEFAULT_REPORT_MODE
    language = (payload.language or "english").capitalize()

    _prune_old_jobs()
    job_id = uuid.uuid4().hex
    RESEARCH_JOBS[job_id] = {
        "status": "running",
        "message": "▶ Starting up...",
        "session_id": session_id,
        "created_at": time.time(),
        "result": None,
        "error": None,
        # Accumulated phase stats — polled separately from the message string
        # so fast-moving early phases aren't missed between 2-second polls.
        "phase_stats": {
            "found": None,
            "ranked": None,
            "extracted": None,
            "section_done": 0,
            "section_total": 0,
            "section_title": None,
        },
    }
    _spawn_background(run_research_job(job_id, clean_topic, mode, language, session_id))

    resp = JSONResponse(content={"job_id": job_id})
    if not existing_session:
        session.set_session_cookie(resp, session_id)
    return resp


@app.get("/api/research/status/{job_id}")
async def api_research_status(job_id: str, request: Request):
    job = RESEARCH_JOBS.get(job_id)
    session_id = session.read_session_id(request)
    # Only the session that started a job can poll it - job_ids are
    # unguessable UUIDs already, this is just defense in depth, consistent
    # with the rest of the app's session-isolation model.
    if not job or job.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Job not found (it may have expired).")
    return {
        "status": job["status"],
        "message": job["message"],
        "result": job["result"],
        "error": job["error"],
        "phase_stats": job.get("phase_stats", {}),
    }


# ---------------------------------------------------------------------------
# Thesis Mode API routes
# ---------------------------------------------------------------------------

@app.post("/api/thesis/start", dependencies=[Depends(require_access_key)])
async def api_start_thesis(request: Request, payload: ThesisStartRequest):
    """
    Starts the thesis pipeline as a background job. Returns a job_id immediately;
    the client polls /api/thesis/status/{job_id} to track progress.
    Requires Supabase to be configured (the master outline must persist between
    chapter generation requests).
    """
    if not report_store._supabase_configured():
        raise HTTPException(
            status_code=400,
            detail=(
                "Thesis Mode requires Supabase. "
                "Please set the SUPABASE_URL and SUPABASE_KEY environment variables "
                "and create the thesis_sessions table (see replit.md for the SQL)."
            ),
        )

    existing_session = session.read_session_id(request)
    session_id = existing_session or uuid.uuid4().hex

    try:
        clean_topic = validate_topic(payload.topic)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    language = (payload.language or "english").capitalize()
    thesis_id = uuid.uuid4().hex

    _prune_old_jobs()
    job_id = uuid.uuid4().hex
    THESIS_JOBS[job_id] = {
        "status": "running",
        "message": "▶ Starting thesis pipeline...",
        "session_id": session_id,
        "created_at": time.time(),
        "result": None,
        "error": None,
    }
    _spawn_background(run_thesis_job(job_id, thesis_id, clean_topic, language, session_id))

    resp = JSONResponse(content={"job_id": job_id})
    if not existing_session:
        session.set_session_cookie(resp, session_id)
    return resp


@app.get("/api/thesis/status/{job_id}")
async def api_thesis_status(job_id: str, request: Request):
    job = THESIS_JOBS.get(job_id)
    session_id = session.read_session_id(request)
    if not job or job.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Thesis job not found (it may have expired).")
    return {
        "status": job["status"],
        "message": job["message"],
        "result": job["result"],
        "error": job["error"],
        "phase_stats": {},
    }


@app.post("/api/thesis/next-chapter", dependencies=[Depends(require_access_key)])
async def api_thesis_next_chapter(request: Request, response: Response, payload: ThesisNextChapterRequest):
    """
    Generates the next chapter of an in-progress thesis.
    Fetches the master_outline and scraped_context from Supabase, generates
    the requested section, advances the chapter index, and returns HTML.
    """
    session_id = session.ensure_session(request, response)

    # Fetch session (verifies ownership via session_id)
    thesis_session = await asyncio.to_thread(
        report_store.get_thesis_session, session_id, payload.thesis_id
    )
    if not thesis_session:
        raise HTTPException(status_code=404, detail="Thesis session not found or has expired.")

    master_outline = thesis_session.get("master_outline") or []
    if not master_outline:
        raise HTTPException(status_code=400, detail="Thesis outline is missing — please start a new thesis.")

    chapter_index = payload.chapter_index
    if chapter_index < 1 or chapter_index >= len(master_outline):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chapter index {chapter_index} (outline has {len(master_outline)} sections)."
        )

    # Prevent skipping chapters (must generate in order)
    stored_index = thesis_session.get("current_chapter_index", 1)
    if chapter_index > stored_index:
        raise HTTPException(
            status_code=400,
            detail=f"Please generate chapters in order. Next expected chapter: {stored_index}."
        )

    scraped_context = thesis_session.get("scraped_context", "")
    topic = thesis_session.get("topic", "")
    language = thesis_session.get("language", "English")

    try:
        chapter_md = await asyncio.to_thread(
            generate_thesis_chapter, topic, master_outline, chapter_index, scraped_context, language
        )
    except Exception as e:
        logger.error("Thesis chapter generation failed (thesis_id=%s, idx=%d): %s", payload.thesis_id, chapter_index, e)
        err_str = str(e)
        if "timeout" in err_str.lower():
            raise HTTPException(status_code=503, detail="Chapter generation timed out. Please try again.")
        raise HTTPException(status_code=500, detail="Could not generate this chapter. Please try again.")

    # Advance the stored chapter index (best-effort; non-fatal if it fails)
    await asyncio.to_thread(
        report_store.update_thesis_chapter_index, payload.thesis_id, chapter_index + 1
    )

    chapter_html = sanitize_html(markdown.markdown(chapter_md, extensions=["tables", "fenced_code"]))

    logger.info(
        "Thesis chapter generated: thesis_id=%s section_index=%d topic=%r",
        payload.thesis_id, chapter_index, topic,
    )
    return {
        "chapter_html": chapter_html,
        "chapter_index": chapter_index,
        "next_index": chapter_index + 1,
        "total_sections": len(master_outline),
    }


# ---------------------------------------------------------------------------
# Interactive tools
# ---------------------------------------------------------------------------
@app.post("/generate-podcast", dependencies=[Depends(require_access_key)])
async def api_podcast(request: Request, response: Response, payload: ReportTextRequest):
    session.ensure_session(request, response)
    try:
        script = await asyncio.to_thread(generate_podcast_script, payload.report_text, payload.language)
        # Sent raw on purpose - see templates/index.html's playPodcastStream().
        # generate_podcast_script() already coerces to str and caps length; the
        # frontend escapes it (only for the on-screen bubble, not for
        # SpeechSynthesisUtterance) right before it touches innerHTML.
        return {"script": script}
    except Exception as e:
        logger.error("Podcast endpoint error: %s", e)
        return JSONResponse(status_code=500, content={"error": "Could not generate the podcast right now."})


@app.post("/generate-diagram", dependencies=[Depends(require_access_key)])
async def api_diagram(request: Request, response: Response, payload: ReportTextRequest):
    session.ensure_session(request, response)
    try:
        mermaid_code = await asyncio.to_thread(generate_diagram, payload.report_text, payload.language)
        return {"mermaid": mermaid_code}
    except Exception as e:
        logger.error("Diagram endpoint error: %s", e)
        return JSONResponse(status_code=500, content={"error": "Could not generate the diagram right now."})


@app.post("/generate-quiz", dependencies=[Depends(require_access_key)])
async def api_quiz(request: Request, response: Response, payload: QuizRequest):
    session.ensure_session(request, response)
    try:
        questions = await asyncio.to_thread(generate_quiz, payload.report_text, payload.num_questions, payload.language)
        return {"questions": questions}
    except Exception as e:
        logger.error("Quiz endpoint error: %s", e)
        return JSONResponse(status_code=500, content={"error": "Could not generate the quiz right now."})


@app.post("/generate-audio", dependencies=[Depends(require_access_key)])
async def api_tts(request: Request, response: Response, payload: TTSRequest):
    """
    Natural-sounding podcast audio via Microsoft Edge's free neural TTS (no
    API key needed - see utils/llm_client.synthesize_speech). Returns raw
    mp3 bytes on success. If TTS_ENABLED=false or the call fails, returns a
    501 - the frontend catches that and falls back to the browser's
    built-in speechSynthesis automatically, so the podcast tool always
    works, just with better voice quality when this succeeds.
    """
    session.ensure_session(request, response)
    accent = payload.accent if payload.accent in config.TTS_VOICES else config.DEFAULT_TTS_ACCENT
    voice_kind = payload.voice_kind if payload.voice_kind in ("male", "female") else "female"
    voice = config.TTS_VOICES[accent][voice_kind]
    audio_bytes = await synthesize_speech(payload.text, voice)
    if audio_bytes is None:
        return JSONResponse(status_code=501, content={"error": "tts_unavailable"})
    return Response(content=audio_bytes, media_type="audio/mpeg")


def _extract_pdf_text(raw_bytes: bytes):
    """Returns (text, total_pages, was_truncated). Only the first
    config.MAX_PDF_PAGES pages are ever read - this app is built for
    reports/articles, not whole books, and without a cap a huge PDF can
    take a long time to parse and produce a response too large to render
    or save reliably."""
    reader = PdfReader(io.BytesIO(raw_bytes))
    total_pages = len(reader.pages)
    pages_to_read = min(total_pages, config.MAX_PDF_PAGES)
    pages = []
    for i in range(pages_to_read):
        try:
            pages.append(reader.pages[i].extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(pages), total_pages, total_pages > config.MAX_PDF_PAGES


@app.post("/api/upload-report", dependencies=[Depends(require_access_key)])
async def api_upload_report(request: Request, file: UploadFile = File(...)):
    """
    Lets a student skip the research pipeline entirely: upload an existing
    report (.pdf/.md/.txt) and immediately use every interactive tool
    (Podcast, Mindmap, RAG, Debate, Quiz, Slides) on it, the same as a
    freshly-researched report.
    """
    existing_session = session.read_session_id(request)
    session_id = existing_session or uuid.uuid4().hex

    filename = file.filename or "uploaded"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("pdf", "md", "txt"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf, .md, or .txt file.")

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    raw_bytes = await file.read(max_bytes + 1)
    if len(raw_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File is too large (max {config.MAX_UPLOAD_MB}MB).")

    notice = None
    try:
        if ext == "pdf":
            text, total_pages, was_truncated = await asyncio.to_thread(_extract_pdf_text, raw_bytes)
            if was_truncated:
                notice = f"This PDF has {total_pages} pages — only the first {config.MAX_PDF_PAGES} were processed."
        else:
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = raw_bytes.decode("latin-1")
    except Exception as e:
        logger.error("Upload extraction failed for %r: %s", filename, e)
        raise HTTPException(status_code=400, detail="Could not read that file. Please try a different one.")

    text = text.strip()
    if len(text) < 50:
        raise HTTPException(
            status_code=400,
            detail="Couldn't find enough readable text in that file. If it's a scanned PDF (photos of pages), text extraction won't work on it.",
        )

    try:
        topic_guess = validate_topic(os.path.splitext(filename)[0][:150] or "Uploaded Report")
    except ValueError:
        topic_guess = "Uploaded Report"
    word_count = len(text.split())
    metrics = {
        "time_seconds": 0,
        "llm_calls": 0,
        "sources_found": 0,
        "sources_used": 0,
        "word_count": word_count,
        "reading_minutes": max(1, round(word_count / 200)),
        "confidence_score": 0,
        "mode_label": "Uploaded Document",
        "model_used": "N/A (uploaded file)",
    }
    paths = await asyncio.to_thread(report_store.save_report, session_id, topic_guess, text, text, metrics)
    report_html = sanitize_html(markdown.markdown(text, extensions=["tables", "fenced_code"]))

    metrics["safe_topic"] = paths["slug"]
    metrics["md_download"] = f"/api/report/{paths['slug']}/download/md"
    metrics["html_download"] = f"/api/report/{paths['slug']}/download/html"

    resp_payload = {
        "status": "done",
        "topic": topic_guess,
        "report": report_html,
        "notice": notice,
        "metrics": metrics,
    }
    resp = JSONResponse(content=resp_payload)
    if not existing_session:
        session.set_session_cookie(resp, session_id)
    logger.info("Uploaded report processed: session=%s file=%r words=%s", session_id, filename, word_count)
    return resp


@app.post("/grade-answer", dependencies=[Depends(require_access_key)])
async def api_grade_answer(request: Request, response: Response, payload: GradeAnswerRequest):
    session.ensure_session(request, response)
    try:
        result = await asyncio.to_thread(
            grade_short_answer, payload.question, payload.key_points, payload.answer, payload.language
        )
        return result
    except Exception as e:
        logger.error("Grading endpoint error: %s", e)
        return JSONResponse(status_code=500, content={"error": "Could not grade this answer right now."})


@app.post("/generate-slides", dependencies=[Depends(require_access_key)])
async def api_slides(request: Request, response: Response, payload: ReportTextRequest):
    session.ensure_session(request, response)
    try:
        slides = await asyncio.to_thread(generate_slides, payload.report_text, payload.language)
        return {"slides": slides}
    except Exception as e:
        logger.error("Slides endpoint error: %s", e)
        return JSONResponse(status_code=500, content={"error": "Could not generate slides right now."})


@app.post("/ask-rag", dependencies=[Depends(require_access_key)])
async def api_ask_rag(request: Request, response: Response, payload: InteractiveRequest):
    session_id = session.ensure_session(request, response)
    context = await asyncio.to_thread(report_store.load_context, session_id, payload.safe_topic)
    if context is None:
        return {"answer": "Context not found for this report. Please regenerate it."}
    try:
        answer = await asyncio.to_thread(rag_query, context, payload.query, payload.language)
        return {"answer": sanitize_html(markdown.markdown(answer))}
    except Exception as e:
        logger.error("RAG endpoint error: %s", e)
        return JSONResponse(status_code=500, content={"error": "Could not answer that right now."})


@app.post("/challenge-report", dependencies=[Depends(require_access_key)])
async def api_challenge(request: Request, response: Response, payload: InteractiveRequest):
    session_id = session.ensure_session(request, response)
    context = await asyncio.to_thread(report_store.load_context, session_id, payload.safe_topic)
    if context is None:
        return {"answer": "Context not found for this report. Please regenerate it."}
    try:
        answer = await asyncio.to_thread(challenge_query, context, payload.query, payload.language)
        return {"answer": sanitize_html(markdown.markdown(answer))}
    except Exception as e:
        logger.error("Challenge endpoint error: %s", e)
        return JSONResponse(status_code=500, content={"error": "Could not process that right now."})


@app.post("/humanize-report", dependencies=[Depends(require_access_key)])
async def api_humanize(request: Request, response: Response, payload: ReportTextRequest):
    session.ensure_session(request, response)
    try:
        humanized = await asyncio.to_thread(humanize_report, payload.report_text, payload.language)
        return {"humanized": humanized}
    except Exception as e:
        logger.error("Humanizer endpoint error: %s", e)
        return JSONResponse(status_code=500, content={"error": "Could not humanize the report right now."})


@app.post("/generate-flashcards", dependencies=[Depends(require_access_key)])
async def api_flashcards(request: Request, response: Response, payload: FlashcardsRequest):
    session.ensure_session(request, response)
    try:
        cards = await asyncio.to_thread(generate_flashcards, payload.report_text, payload.num_cards, payload.language)
        return {"cards": cards}
    except Exception as e:
        logger.error("Flashcards endpoint error: %s", e)
        return JSONResponse(status_code=500, content={"error": "Could not generate flashcards right now."})


# ---------------------------------------------------------------------------
# History & report retrieval (session-scoped - fixes the old multi-tenant leak)
# ---------------------------------------------------------------------------
@app.get("/api/history")
async def get_history(request: Request, response: Response):
    session_id = session.ensure_session(request, response)
    history = await asyncio.to_thread(report_store.list_reports, session_id)
    return {"history": history}


@app.get("/api/report/{safe_topic}")
async def get_past_report(safe_topic: str, request: Request, response: Response):
    session_id = session.ensure_session(request, response)
    md_content = await asyncio.to_thread(report_store.load_report_markdown, session_id, safe_topic)
    if md_content is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    slug = safe_slug(safe_topic)
    report_html = sanitize_html(markdown.markdown(md_content, extensions=["tables", "fenced_code"]))

    metrics = await asyncio.to_thread(report_store.load_metrics, session_id, safe_topic)
    if metrics is not None:
        # Older reports saved before metrics persistence was added won't have
        # this file - metrics stays None for those and the frontend just
        # shows no badges, exactly like before this feature existed.
        metrics = dict(metrics)
        metrics["safe_topic"] = slug
        metrics["md_download"] = f"/api/report/{slug}/download/md"
        metrics["html_download"] = f"/api/report/{slug}/download/html"

    return {
        "topic": slug.replace("_", " ").title(),
        "safe_topic": slug,
        "report": report_html,
        "metrics": metrics,
        "md_download": f"/api/report/{slug}/download/md",
        "html_download": f"/api/report/{slug}/download/html",
    }


@app.get("/api/report/{safe_topic}/download/{fmt}")
async def download_report(safe_topic: str, fmt: str, request: Request):
    """
    Serves the actual .md/.html file for a report, scoped to the caller's
    session. This replaces the old app's `app.mount("/reports", StaticFiles(...))`,
    which exposed every user's report files to every visitor with zero
    access control - the single most serious issue found in this app.
    """
    session_id = session.read_session_id(request)
    if not session_id:
        raise HTTPException(status_code=404, detail="Report not found.")
    if fmt not in ("md", "html"):
        raise HTTPException(status_code=400, detail="Invalid format. Use 'md' or 'html'.")
    path = await asyncio.to_thread(report_store.file_path_for_download, session_id, safe_topic, fmt)
    if not path:
        raise HTTPException(status_code=404, detail="Report not found.")
    media_type = "text/markdown" if fmt == "md" else "text/html"
    return FileResponse(path, media_type=media_type, filename=os.path.basename(path))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=(config.APP_ENV != "production"))
