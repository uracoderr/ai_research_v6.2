---
name: ThesisPilot QA fixes + Thesis Mode architecture
description: Key bugs fixed during QA pass and architectural decisions for the Thesis Mode feature
---

# ThesisPilot — QA findings, fixes, and Thesis Mode architecture

## Bugs fixed (original QA pass)

**Why:** Production readiness pass before go-live on 0.1 vCPU / 512 MB Replit host.

### 1. ThreadPoolExecutor over-provisioning (report_agent.py)
`generate_report` used `max_workers=len(sections)` — up to 7 concurrent NVIDIA LLM calls on a tiny host.  
**Fix:** capped to `min(3, len(sections))`. 3 concurrent calls fill the NVIDIA rate-limit window without burning RAM.

### 2. Podcast JSON parsing (report_agent.py)
`generate_podcast_script` used a greedy regex `r"\[\s*\{.*\}\s*\]"` which breaks when the model emits trailing commentary.  
**Fix:** replaced with the same `_extract_json_array()` bracket-counter used everywhere else in the file.

### 3. Raw exception exposure (app.py)
`run_research_job` sent `str(e)` directly as the user-facing error, leaking model names / paths.  
**Fix:** classifies timeout / rate-limit / key errors into friendly messages; generic fallback for everything else.

### 4. setStep animation stacking (index.html)
Rapid poll messages within 300 ms stacked `setTimeout` calls, flashing stale text then overwriting.  
**Fix:** `_stepTimeout` handle — each `setStep` call cancels the previous pending timeout before starting a new one.

### 5. handleChat ID collision (index.html)
All chat "Thinking..." bubbles shared the same DOM id `typing-{boxId}`. Rapid sends left orphaned bubbles.  
**Fix:** `_chatSeq` counter per box; each bubble gets a unique id `typing-{boxId}-{n}`.

### 6. Error state — no recovery path (index.html)
When research errored, the loading screen froze with the error message and no way back.  
**Fix:** hidden `#retry-research-btn` shown on error; spinner hidden; button calls `showNewResearchForm()`.

### 7. Zoom function wrong units (index.html)
`fontSize = val + '%'` meant "100% of parent's 16 px" not "100% of 15 px base", causing a jump.  
**Fix:** `fontSize = (15 * val / 100).toFixed(1) + 'px'` — scales from the 15 px report baseline.

### 8. shareReport uses window.event (index.html)
Deprecated global, fails in Firefox strict mode.  
**Fix:** button passes `this`; function accepts `btn` parameter.

### 9. Podcast prefetch only 1 ahead (index.html)
Single-segment prefetch caused audible gaps on high-latency connections between Host A / B lines.  
**Fix:** prefetch current + next TWO segments per `playNext()` call.

### 10. requirements.txt duplicates
Every package was listed twice; `pypdf` was both pinned and unpinned.  
**Fix:** single clean requirements.txt.

### 11. Python docstring SyntaxWarning
Docstring in `_extract_json_array` had literal `\[` — invalid escape in Python 3.12+.  
**Fix:** escaped to `\\[` in the docstring text.

## How to apply in future sessions
- Port: workflow runs on **5000** (passed via `--port 5000`); `app.py __main__` defaults to 8000 but that path is not used.
- Language: Hinglish is passed as lowercase `"hinglish"` from the form; `_language_instruction` checks `"hinglish" in lang_lower` — case-insensitive, works correctly.
- All 8 tools pull context via `report-content.innerText` (client side) OR `currentTopic` → server `load_context` (RAG/Debate). This is intentional — both paths are correct.
- Supabase is optional for regular modes; local disk is the default and works fine for Replit.

## Tool-specific token / model decisions (tuned for reliability)
- **Podcast**: quality model for Hindi/Hinglish (8B drifts to English in JSON), max_tokens=3500, input cap 12k, 60-line cap, 3 retries.
- **Slides**: quality model always (8B truncates JSON array at 1-2 slides), max_tokens=2800, prompt asks for 10-12 slides with structured outline, slide cap 14.
- **Humanizer**: quality model for non-English, max_tokens=2800, input cap 8k, retries=3.
- **Hindi TTS**: `hi-IN-MadhurNeural` / `hi-IN-SwaraNeural` added as `"hindi"` accent in `config.TTS_VOICES`; frontend auto-selects this accent when `currentLanguage === 'hindi'`.
- **Why quality model matters**: 8B (fast) model reliably fails at sustained non-English output AND long JSON arrays — always use quality for these two scenarios.

---

## Thesis Mode architecture (added later)

**Why:** User requested progressive thesis generation with chapter-by-chapter control, persisted via Supabase.

### Flow
1. User selects "📜 Thesis Mode" from the mode dropdown and submits a topic.
2. Frontend calls `POST /api/thesis/start` (NOT `/api/research/start`).
3. Backend runs the same web-search + scrape pipeline, then calls `generate_thesis_outline()` (70B model, ~7 sections).
4. Outline is saved to Supabase `thesis_sessions` table — **required** (chapter requests must fetch it later).
5. Backend generates Preliminary Pages immediately and returns them in the job result.
6. Frontend shows `#thesis-progress-area` with a "Generate Chapter 1: Introduction →" button.
7. Each chapter click calls `POST /api/thesis/next-chapter` with `thesis_id` + `chapter_index`.
8. Backend fetches outline from Supabase, generates the chapter with strict scope prompt, advances `current_chapter_index`, returns HTML.
9. Frontend appends chapter HTML to `#report-content`, advances button label.

### Key design decisions
- `thesis_sessions` is a **separate Supabase table** from `reports` — do not mix them.
- Thesis mode does NOT write to `RESEARCH_JOBS` — it has its own `THESIS_JOBS` dict and `/api/thesis/status/{job_id}` endpoint.
- `generate_thesis_chapter` system prompt says "Strictly adhere to the provided Master Outline. Do not hallucinate or write content meant for other chapters." — this scope constraint is intentional and must be preserved.
- All thesis chapters use **quality (70B) model only** — no fast-model shortcuts (the 8B model cannot sustain 2000-2500 words of coherent academic prose).
- `scraped_context` is capped at 30 000 chars in Supabase storage.

### Supabase table (thesis_sessions)
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
CREATE INDEX IF NOT EXISTS idx_thesis_sessions_session_id ON thesis_sessions(session_id);
```

### Config
- `config.SUPABASE_TABLE_THESIS` = `"thesis_sessions"` (separate env var from `SUPABASE_TABLE`).
- Thesis mode appears at the END of the modes list from `/api/meta` (hardcoded append, not in `REPORT_MODES`).
