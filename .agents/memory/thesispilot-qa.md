---
name: ThesisPilot QA fixes
description: Key bugs fixed during QA pass and architectural decisions worth preserving
---

# ThesisPilot — QA findings and fixes

## Bugs fixed

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
- Supabase is optional; local disk is the default and works fine for Replit.
