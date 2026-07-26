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
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="ThesisPilot AI Research Agent")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    return {
        "access_required": bool(config.SAAS_ACCESS_KEY),
        "default_mode": config.DEFAULT_REPORT_MODE,
        "modes": [
            {"id": m["id"], "label": m["label"], "description": m["description"]}
            for m in config.REPORT_MODES.values()
        ],
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
_background_tasks: set = set()
_JOB_TTL_SECONDS = 3600  # stop tracking jobs older than this (finished or not)


def _spawn_background(coro) -> None:
    """Fire-and-forget a coroutine on the running event loop, keeping a
    strong reference until it completes so it can't be garbage-collected
    mid-flight (a well-known asyncio footgun with bare create_task calls)."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _prune_old_jobs() -> None:
    cutoff = time.time() - _JOB_TTL_SECONDS
    stale = [jid for jid, job in RESEARCH_JOBS.items() if job.get("created_at", 0) < cutoff]
    for jid in stale:
        RESEARCH_JOBS.pop(jid, None)


async def run_research_job(job_id: str, clean_topic: str, mode: str, language: str, session_id: str) -> None:
    """The actual pipeline - identical steps to before, just reporting
    progress into RESEARCH_JOBS instead of yielding SSE events."""
    start_time = time.time()
    llm_calls = 0

    def update(message: str) -> None:
        if job_id in RESEARCH_JOBS:
            RESEARCH_JOBS[job_id]["message"] = message

    try:
        update("▶ PHASE 0: Understanding your topic...")
        optimized_topic = await asyncio.to_thread(optimize_query, clean_topic)
        llm_calls += 1
        update(f"✨ Query refined to: '{optimized_topic}'")

        update("▶ PHASE 1: Searching the web...")
        raw_articles = await asyncio.to_thread(fetch_articles, optimized_topic)
        if not raw_articles:
            RESEARCH_JOBS[job_id].update({"status": "error", "error": "No articles found for this topic. Try rephrasing it."})
            return
        update(f"✅ Found {len(raw_articles)} candidate sources.")

        update("▶ PHASE 2: Ranking source credibility...")
        ranked_articles, duplicates_removed, filter_calls, llm_success = await asyncio.to_thread(
            filter_and_rank_articles, raw_articles, config.SEARCH_MAX_RESULTS
        )
        llm_calls += filter_calls
        update(f"✅ {len(ranked_articles)} high-quality sources selected.")

        update("▶ PHASE 3: Reading full articles...")
        scraped_data, scraped_count, scraped_sources = await asyncio.to_thread(scrape_top_articles, ranked_articles)
        if scraped_count == 0:
            RESEARCH_JOBS[job_id].update({"status": "error", "error": "Could not read any sources. Please try again."})
            return
        update(f"✅ Extracted content from {scraped_count} sources.")

        update("▶ PHASE 4: Writing your report...")
        stats_dict = {
            "scraped_success": scraped_count,
            "duplicates_removed": duplicates_removed,
            "llm_ranking_success": llm_success,
        }
        final_report, meta = await asyncio.to_thread(
            generate_report, optimized_topic, scraped_data, scraped_sources, language, stats_dict, mode
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

        if job_id in RESEARCH_JOBS:
            RESEARCH_JOBS[job_id].update({
                "status": "done",
                "message": "✅ Report ready!",
                "result": {"topic": optimized_topic, "report": report_html, "metrics": metrics},
            })
        logger.info(
            "Research complete: session=%s topic=%r mode=%s time=%.1fs",
            session_id, optimized_topic, mode, metrics["time_seconds"],
        )

    except Exception as e:
        logger.error("Pipeline error (job=%s): %s\n%s", job_id, e, traceback.format_exc())
        if job_id in RESEARCH_JOBS:
            RESEARCH_JOBS[job_id].update({"status": "error", "error": str(e)})


@app.post("/api/research/start", dependencies=[Depends(require_access_key)])
@limiter.limit(config.RATE_LIMIT_RESEARCH)
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
    }


# ---------------------------------------------------------------------------
# Interactive tools
# ---------------------------------------------------------------------------
@app.post("/generate-podcast", dependencies=[Depends(require_access_key)])
@limiter.limit(config.RATE_LIMIT_CHAT)
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
@limiter.limit(config.RATE_LIMIT_CHAT)
async def api_diagram(request: Request, response: Response, payload: ReportTextRequest):
    session.ensure_session(request, response)
    try:
        mermaid_code = await asyncio.to_thread(generate_diagram, payload.report_text, payload.language)
        return {"mermaid": mermaid_code}
    except Exception as e:
        logger.error("Diagram endpoint error: %s", e)
        return JSONResponse(status_code=500, content={"error": "Could not generate the diagram right now."})


@app.post("/generate-quiz", dependencies=[Depends(require_access_key)])
@limiter.limit(config.RATE_LIMIT_CHAT)
async def api_quiz(request: Request, response: Response, payload: QuizRequest):
    session.ensure_session(request, response)
    try:
        questions = await asyncio.to_thread(generate_quiz, payload.report_text, payload.num_questions, payload.language)
        return {"questions": questions}
    except Exception as e:
        logger.error("Quiz endpoint error: %s", e)
        return JSONResponse(status_code=500, content={"error": "Could not generate the quiz right now."})


@app.post("/generate-audio", dependencies=[Depends(require_access_key)])
@limiter.limit(config.RATE_LIMIT_TTS)
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
@limiter.limit(config.RATE_LIMIT_RESEARCH)
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
@limiter.limit(config.RATE_LIMIT_CHAT)
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
@limiter.limit(config.RATE_LIMIT_CHAT)
async def api_slides(request: Request, response: Response, payload: ReportTextRequest):
    session.ensure_session(request, response)
    try:
        slides = await asyncio.to_thread(generate_slides, payload.report_text, payload.language)
        return {"slides": slides}
    except Exception as e:
        logger.error("Slides endpoint error: %s", e)
        return JSONResponse(status_code=500, content={"error": "Could not generate slides right now."})


@app.post("/ask-rag", dependencies=[Depends(require_access_key)])
@limiter.limit(config.RATE_LIMIT_CHAT)
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
@limiter.limit(config.RATE_LIMIT_CHAT)
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
@limiter.limit(config.RATE_LIMIT_CHAT)
async def api_humanize(request: Request, response: Response, payload: ReportTextRequest):
    session.ensure_session(request, response)
    try:
        humanized = await asyncio.to_thread(humanize_report, payload.report_text, payload.language)
        return {"humanized": humanized}
    except Exception as e:
        logger.error("Humanizer endpoint error: %s", e)
        return JSONResponse(status_code=500, content={"error": "Could not humanize the report right now."})


@app.post("/generate-flashcards", dependencies=[Depends(require_access_key)])
@limiter.limit(config.RATE_LIMIT_CHAT)
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
    path = await asyncio.to_thread(report_store.file_path_for_download, session_id, safe_topic, fmt)
    if not path:
        raise HTTPException(status_code=404, detail="Report not found.")
    media_type = "text/markdown" if fmt == "md" else "text/html"
    return FileResponse(path, media_type=media_type, filename=os.path.basename(path))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=(config.APP_ENV != "production"))
