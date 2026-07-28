# ThesisPilot — AI Research Agent

An autonomous research pipeline: give it a topic, it searches the web, ranks sources for credibility, reads full articles, and writes a structured, source-cited report. Now with **8 interactive study tools**.

## Stack
- **Backend**: Python 3.12, FastAPI, uvicorn
- **LLM**: NVIDIA NIM (Llama-3.1-70B via `NVIDIA_API_KEY`) — every tool routes through the 70B "quality" model now; the 8B "fast" model is only still used by Flash Mode's guaranteed-speed report sections (see "Model routing" below)
- **Search**: Tavily Search API (via `TAVILY_API_KEY`)
- **TTS**: Microsoft Edge neural voices via `edge-tts` (no API key needed)
- **Frontend**: Single-page Tailwind CSS app (`templates/index.html`)
- **Sessions**: Anonymous signed cookies (`itsdangerous`)

## Running the app

```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```

The workflow "Start application" does this automatically on Replit.

## Required secrets (Replit Secrets tab)

| Secret | Where to get it |
|--------|----------------|
| `NVIDIA_API_KEY` | [build.nvidia.com](https://build.nvidia.com) (free tier available) |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) (free tier available) |
| `SESSION_SECRET` | Already set — signs the session cookie |

## 8 Study Tools

1. 🎙️ **Podcast** — natural 2-host audio via Edge TTS (seamless prefetch buffering)
2. 🧠 **Mindmap** — Mermaid.js flowchart of the report
3. 💬 **AI Tutor** — RAG chat over your report
4. ⚔️ **Viva Mode** — examiner-style follow-up questions (opening question is pre-generated — see below)
5. 📝 **Quiz** — auto-graded MCQ + short-answer questions, always exactly **15 questions** (no longer user-configurable)
6. 📊 **PPT Generator** — slide outline ready to copy
7. ✍️ **Humanizer** — rewrites report in natural, human style
8. 🗂️ **Smart Flashcards** — tap-to-reveal Q&A cards for active recall

## Model routing

Every tool and pipeline phase (search query optimization, source ranking, report writing, and all 8 study tools) uses the 70B "quality" model. The only exception is **Flash Mode**, whose entire purpose is guaranteed speed — it still always uses the 8B "fast" model, in every language, so it stays meaningfully faster than Assignment mode. `MODEL_FAST` in `config.py` is kept defined for this reason.

## Background study-tool pre-generation

Right after the main research report finishes (`run_research_job` in `app.py`), a fire-and-forget background task (`asyncio.create_task`, independent of any client connection — see `_spawn_background`) generates Slides, Mindmap, Flashcards, Quiz, and the opening Viva question concurrently (`agents/report_agent.precompute_study_tools`, via `asyncio.gather(..., return_exceptions=True)` so one failure never blocks the others) and caches the results (`utils/report_store.save_tool_cache`, disk + optional Supabase — see below). The corresponding tool endpoints check this cache first, so opening a tool is instant once pre-generation finishes; a cache miss (older report, still generating, or that one tool failed) just falls back to live generation exactly as before. This currently covers the main research pipeline only — uploaded reports and Thesis Mode still generate tools on demand.

## Language support

All tools dynamically respond in the language you select (English / Hinglish / Hindi). Language is tracked per report and passed to every tool call. The AI Tutor and Viva Mode also detect Hinglish in user queries and respond naturally in kind.

## Key files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI app, all API endpoints |
| `agents/search_agent.py` | Phase 0 (query optimization) + Phase 1 (multi-query Tavily search) |
| `agents/filter_agent.py` | Phase 2 (credibility ranking, deduplication) |
| `agents/scraper_agent.py` | Phase 3 (full-article scraping) |
| `agents/report_agent.py` | Phase 4 (report generation) + all 8 study tool functions |
| `config.py` | All tunables and env var definitions |
| `templates/index.html` | Full frontend (single HTML file) |
| `utils/llm_client.py` | Shared NVIDIA NIM client + edge-tts synthesis |
| `utils/report_store.py` | Per-session report storage (disk + optional Supabase) |
| `utils/session.py` | Anonymous signed cookie sessions |

## Thesis Mode — Supabase setup (required)

Thesis Mode saves the master outline between chapter requests, so it needs Supabase.

**Step 1** — Add two secrets in the Replit Secrets tab:
| Secret | Value |
|--------|-------|
| `SUPABASE_URL` | Your Supabase project URL, e.g. `https://xxxx.supabase.co` |
| `SUPABASE_KEY` | Your Supabase `anon` / service key |

**Step 2** — Run this SQL once in your **Supabase SQL Editor**:

```sql
CREATE TABLE IF NOT EXISTS thesis_sessions (
    thesis_id        TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    topic            TEXT NOT NULL,
    master_outline   JSONB NOT NULL DEFAULT '[]',
    current_chapter_index INTEGER NOT NULL DEFAULT 1,
    scraped_context  TEXT DEFAULT '',
    language         TEXT DEFAULT 'English',
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_thesis_sessions_session_id
    ON thesis_sessions(session_id);
```

The existing `reports` table is unchanged. Thesis Mode is additive — regular research, Flash and Assignment modes all continue to work without Supabase.

## Background tool cache — Supabase table (optional)

Pre-generated Slides/Mindmap/Flashcards/Quiz/first-Viva-question follow the same **optional** dual-write pattern as the main `reports` table (see `utils/report_store.py`): they're always saved to local disk first, and also best-effort mirrored to Supabase if configured, purely so the cache survives a redeploy/restart. Nothing breaks if you skip this — the background pre-generation and instant-load behavior work with local disk alone.

To also persist it to Supabase, run this once in your **Supabase SQL Editor** (uses the same `SUPABASE_URL` / `SUPABASE_KEY` secrets as above):

```sql
CREATE TABLE IF NOT EXISTS report_tools (
    id                   BIGSERIAL PRIMARY KEY,
    session_id           TEXT NOT NULL,
    safe_topic           TEXT NOT NULL,
    slides               JSONB,
    diagram              TEXT,
    flashcards           JSONB,
    quiz                 JSONB,
    first_viva_question  TEXT,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (session_id, safe_topic)
);
```

## User preferences

- Keep the project's existing structure and stack — do not restructure or migrate it.
- Use the existing dark Tailwind UI theme (slate-950 background, teal accents).
