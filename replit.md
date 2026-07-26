# ThesisPilot — AI Research Agent

An autonomous research pipeline: give it a topic, it searches the web, ranks sources for credibility, reads full articles, and writes a structured, source-cited report. Now with **8 interactive study tools**.

## Stack
- **Backend**: Python 3.12, FastAPI, uvicorn
- **LLM**: NVIDIA NIM (Llama-3.1-8B + 70B via `NVIDIA_API_KEY`)
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
4. ⚔️ **Viva Mode** — examiner-style follow-up questions
5. 📝 **Quiz** — auto-graded MCQ + short-answer questions
6. 📊 **PPT Generator** — slide outline ready to copy
7. ✍️ **Humanizer** — rewrites report in natural, human style
8. 🗂️ **Smart Flashcards** — tap-to-reveal Q&A cards for active recall

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

## User preferences

- Keep the project's existing structure and stack — do not restructure or migrate it.
- Use the existing dark Tailwind UI theme (slate-950 background, teal accents).
