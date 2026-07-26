# 🎓 ThesisPilot — AI Research Agent

An autonomous research pipeline: give it a topic, it searches the web, ranks
sources for credibility, reads the full articles, and writes a structured,
source-cited report. Three report modes: **Flash** (superfast, ~1000 words,
8B model only), **Assignment** (default, ~2800-3200 words, tuned for college
assignments), and **Deep Research** (~3800-4800 words, exhaustive). Don't
have time to research? **Upload an existing report or PDF** instead (up to
~40 pages) and jump straight into the tools below. Research runs as a
background job on the server, so switching apps or locking your phone mid-run
no longer kills it. Includes a CLI and a web app with six interactive tools:
a podcast script generator with natural-sounding, accent-selectable voices
and adjustable playback speed, a mind-map diagram, a RAG chat over your
sources, a "debate the findings" challenge mode, a **Quiz Me / viva-prep
self-test** (you choose how many questions, up to 20), and **one-click slide
outlines** for class presentations. Report history - including word count,
confidence score, and generation time - can optionally persist in Supabase
so it survives redeploys.

This version is a security, performance, and architecture rewrite of the
original prototype — see **[What changed and why](#what-changed-and-why)** below.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your API keys
cp .env.example .env
# then edit .env and fill in TAVILY_API_KEY and NVIDIA_API_KEY

# 3a. Run the CLI
python main.py

# 3b. OR run the web app
uvicorn app:app --reload
# then open http://127.0.0.1:8000
```

Get a free `TAVILY_API_KEY` at [tavily.com](https://tavily.com) and an
`NVIDIA_API_KEY` at [build.nvidia.com](https://build.nvidia.com).

### Run with Docker instead

```bash
cp .env.example .env   # fill in your keys first
docker compose up --build
```

### Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

### Supabase setup (optional - persistent history across redeploys)

Local disk works fine for a single long-lived server, but most hosts (Render
included) wipe the filesystem on every redeploy, silently losing everyone's
research history. To make history durable:

1. Create a free project at [supabase.com](https://supabase.com).
2. In the SQL editor, run:

```sql
create table if not exists reports (
    id bigint generated always as identity primary key,
    session_id text not null,
    safe_topic text not null,
    topic text not null,
    report_markdown text not null,
    context_text text not null,
    metrics jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (session_id, safe_topic)
);
```

If you already created this table before `metrics` existed, just run:
```sql
alter table reports add column if not exists metrics jsonb not null default '{}'::jsonb;
```

3. In Project Settings → API, copy the **Project URL** and the
   **`service_role` key** (server-side only - never expose this key to the
   browser) into `.env`:

```
SUPABASE_URL=https://xxxxxxxx.supabase.co
SUPABASE_KEY=your_service_role_key
```

That's it - every report is now written to both local disk and Supabase.
Reads fall back to Supabase automatically whenever the local copy is
missing (and quietly repopulate it), so a redeploy no longer loses anyone's
history. Leave these two blank to keep using local-disk-only storage -
nothing else changes.

### Natural podcast voices (on by default, no setup needed)

The Podcast tool uses Microsoft Edge's free neural "Read Aloud" voices by
default (via the open-source `edge-tts` library) - no account, no API key,
no cost, nothing to configure. Pick different voices with `TTS_MALE_VOICE`
/ `TTS_FEMALE_VOICE` in `.env` (see `.env.example` for a few good options,
including Indian-English ones). Set `TTS_ENABLED=false` to skip this
entirely and always use the browser's own built-in voice instead.

This works by using the same endpoint the real Edge browser uses
internally - it's not an official, documented Microsoft API, so it could in
principle change or stop working without notice. If that ever happens, the
podcast player automatically falls back to the browser's built-in voice, so
the feature keeps working either way, just with more robotic-sounding audio
until/unless it comes back.

---

## Project structure

```
ai-research-agent-main/
├── main.py                  CLI entry point
├── app.py                   FastAPI web app (session isolation, rate limits, all routes)
├── config.py                every tunable value: models, timeouts, report modes, etc.
├── requirements.txt
├── requirements-dev.txt     [NEW] pytest, for running tests/
├── .env.example             [NEW] every environment variable, documented
├── .gitignore
├── README.md                [NEW] this file
├── LICENSE                  [NEW] MIT
├── Dockerfile                [NEW]
├── docker-compose.yml        [NEW]
│
├── agents/                  the 4-stage research pipeline
│   ├── __init__.py           [NEW] (empty, marks this as a package)
│   ├── search_agent.py        query cleanup + Tavily search
│   ├── filter_agent.py        credibility ranking / junk filtering
│   ├── scraper_agent.py       parallel article scraping
│   └── report_agent.py        section-by-section report synthesis
│
├── utils/                   [NEW FOLDER] shared infrastructure - see below
│   ├── __init__.py            [NEW]
│   ├── logger.py              [NEW] centralised logging (CLI: pretty/Rich, web: plain)
│   ├── security.py            [NEW] path-safe slugs, input validation, HTML sanitising
│   ├── session.py              [NEW] anonymous per-browser session cookies
│   ├── report_store.py         [NEW] one shared place to save/load reports (was duplicated)
│   ├── cache.py                [NEW] simple file cache for repeated search queries
│   └── llm_client.py           [NEW] one shared NVIDIA API client (was duplicated x3)
│
├── templates/
│   └── index.html            web UI (mode selector, access gate, secure downloads)
│
└── tests/                   [NEW FOLDER]
    ├── __init__.py            [NEW]
    └── test_security.py       [NEW] tests for the path-traversal / sanitising fixes
```

**Every new file lives in one of three places: the project root, the new
`utils/` folder, or `tests/`.** `agents/` and `templates/` only contain the
files that already existed, rewritten in place — nothing moved.

Why a new `utils/` folder instead of stuffing this into `agents/`: the four
files in `agents/` are pipeline *stages* (search → filter → scrape → report).
Session handling, sanitisation, logging, storage and the LLM client aren't
pipeline stages — they're plumbing that every stage (and both `main.py` and
`app.py`) needs. Mixing the two made the original `report_agent.py` do LLM
calls, file I/O, *and* report logic all at once.

---

## What changed and why

### 🔴 Security fixes

| Issue | Fix |
|---|---|
| **Path traversal**: `req.topic` was used directly in `os.path.join(...)` for the RAG/debate context file, with no sanitisation. | `utils/security.safe_slug()` strips everything except `a-z0-9-_` before any filename/folder is built. Covered by `tests/test_security.py`. |
| **Multi-tenant data leak**: one shared `reports/` folder, plus `app.mount("/reports", StaticFiles(...))` served it to anyone with zero access control. | Every visitor gets an anonymous, signed, httponly session cookie (`utils/session.py`). Reports live at `reports/<session_id>/...`. The static mount is gone, replaced by `/api/report/{slug}/download/{fmt}`, which checks the requester's session before returning a file. |
| **No rate limiting**: any visitor could trigger unlimited research runs, each one costing real API quota. | `slowapi` rate limits on `/api/research/stream` (default 6/hour/IP) and the chat endpoints (40/hour/IP) — see `RATE_LIMIT_*` in `.env.example`. |
| **Stored XSS risk**: the report shown in the browser is LLM output, which is itself influenced by scraped web content, inserted with raw `innerHTML`. | Every piece of LLM-derived HTML goes through `utils/security.sanitize_html()` (an allow-list sanitiser) before it's sent to the browser. |
| **Fake confidence score**: `avg_credibility` was hardcoded to `8.5` on every single report, regardless of what actually happened. | `report_agent._compute_confidence_score()` averages the *real* credibility scores of the sources that were actually used. |
| **Silent crashes**: a bare `except:` in the ranking logic swallowed every failure type, and the fallback ranker crashed (`IndexError`) on any URL that wasn't a full `scheme://host/...` string. | Specific exception handling throughout (`utils/llm_client.LLMError`), and `_source_name_from_url()` now degrades gracefully instead of crashing. |

### 🟢 Speed: the original took 5+ minutes, this targets ~3 minutes

The single biggest hidden cost wasn't the LLM calls — it was a concurrency bug:

> The scraper used `with ThreadPoolExecutor(...) as executor:`. Exiting a
> `with` block calls `shutdown(wait=True)` by default, which blocks until
> *every* submitted request finishes or times out — even after enough
> successful scrapes had already come in and the code had logically moved on.
>
> Measured in isolation with 10 requests (5 fast, 5 artificially slow) and a
> target of 5 successes: the old pattern took **5.00s** (waiting for the slow
> stragglers anyway); the fixed version (`shutdown(wait=False,
> cancel_futures=True)`) took **0.20s**.

On top of that fix, three more changes attack the real remaining cost —
LLM generation time:

1. **Smaller, mode-based token budgets.** The old code asked for 4 sections
   × 3500 tokens each (≈10,000+ words — closer to a thesis chapter than an
   assignment). The new default **"assignment" mode** targets ~2000-2500
   words with a **3,470-token total budget — a 75% reduction**. A **"deep"**
   mode (7,000 tokens, ~50% reduction from the original) is available for
   anyone who explicitly wants the old exhaustive behaviour.
2. **Model routing.** Lighter, more templated sections (introduction,
   conclusion) route to the fast 8B model; only the two sections that need
   real synthesis from source data use the 70B model. Query typo-correction
   also moved from 70B to 8B — it never needed a large model.
3. **Faster search + tighter scrape timeouts.** Tavily search depth defaults
   to `"basic"` instead of `"advanced"` (we re-scrape every page ourselves
   anyway, so Tavily's only job is finding candidate URLs), and per-article
   scrape timeout dropped from 12s to 8s.

All of these are tunable in `.env` / `config.py` if you want to trade speed
for depth — see `REPORT_MODES` in `config.py`.

*(Honest caveat: total wall-clock time also depends on live NVIDIA API
latency, which is outside this codebase's control. These changes remove the
two things that *were* controllable — a real concurrency bug and roughly
10,000 tokens of unnecessary generation.)*

### 📝 Report quality, tuned for assignments

- **Real references, not hallucinated ones.** The old report never included
  a source list at all. The new "References" section is built in **Python
  from the actual scraped article metadata** — never asked of the LLM — so
  every citation is a real, clickable, verifiable link. Sections are also
  instructed to note `(Source: Name)` inline when citing a specific fact.
- **Word count + reading time** are computed and shown, so you can tell at a
  glance whether a report matches your assignment's length requirement.
- **An academic-integrity note** is included in every report, encouraging
  paraphrasing and citing the original sources rather than submitting the
  generated text directly — this tool is a research accelerator, not a
  substitute for doing the assignment.

### 🧩 Other SaaS-readiness changes

- **One shared LLM client** (`utils/llm_client.py`) instead of three
  duplicated `requests.post(...)` blocks, with proper retries + jittered
  backoff on 429s.
- **One shared report-storage module** (`utils/report_store.py`) instead of
  near-identical save logic in both `main.py` and `app.py`.
- **Structured logging** (`utils/logger.py`) instead of scattering
  `rich`-formatted `console.print()` calls through business logic — those
  print raw ANSI codes into server/Docker logs, which isn't useful. The CLI
  still gets pretty colourised output (via a Rich log handler); the web app
  gets plain, parseable log lines.
- **`config.py` centralises every tunable** — models, timeouts, rate limits,
  report structure — instead of leaving them hardcoded across four files.
- **A small file cache** (`utils/cache.py`) for repeated Tavily searches, so
  two students researching the same trending topic don't both pay full
  latency + quota cost.
- **Optional shared-secret access gate** (`SAAS_ACCESS_KEY`) if you want to
  hand this to a specific class/cohort without building full auth.

---

## Scaling this further

This rewrite makes the app safe and reasonably fast for **many anonymous
users sharing one deployment** — but it deliberately stops short of a full
multi-tenant SaaS with logins and billing, because that's a different, much
bigger project than "fix this codebase." If you want to take it there:

- **Real accounts**: swap `utils/session.py`'s anonymous cookie for a real
  auth provider (Clerk, Auth0, Supabase Auth) and use their user id
  wherever `session_id` is used now.
- **A real database**: move `reports/` off the local filesystem and into
  Postgres/S3 so reports survive redeploys and work across multiple server
  instances.
- **Redis for caching and rate limiting**: `utils/cache.py` and `slowapi`'s
  in-memory limiter both work great for one instance; move to Redis before
  running more than one.
- **Billing**: Stripe (or similar) metering on top of the request counts
  already logged per session.
- **Background jobs**: for very long "deep" mode reports, move generation
  off the request/response cycle entirely (Celery/RQ + a job status
  endpoint) instead of holding a streaming connection open.

None of this is implemented here — it's listed so the next step is a
decision, not a surprise.
