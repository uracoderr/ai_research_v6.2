"""
Central configuration for ThesisPilot AI Research Agent.

Every tunable value (models, token budgets, timeouts, rate limits,
report structure) lives here so nothing is hardcoded deep inside the
pipeline. Override any of these with environment variables - see
.env.example for the full list and explanations.
"""
import os
import secrets

from dotenv import load_dotenv
from rich.console import Console
from rich.theme import Theme

load_dotenv()  # loads .env if present; harmless no-op in production where real env vars are injected

# ---------------------------------------------------------------------------
# CLI console theme (used only by main.py)
# ---------------------------------------------------------------------------
custom_theme = Theme({
    "info": "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "step": "bold magenta",
    "highlight": "bold yellow",
    "interactive": "bold cyan",
})
console = Console(theme=custom_theme)

# ---------------------------------------------------------------------------
# API keys (required - see .env.example)
# ---------------------------------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

TAVILY_URL = "https://api.tavily.com/search"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# ---------------------------------------------------------------------------
# App / environment
# ---------------------------------------------------------------------------
APP_ENV = os.getenv("APP_ENV", "development")            # "development" | "production"
SECRET_KEY = os.getenv("SECRET_KEY") or os.getenv("SESSION_SECRET") or secrets.token_hex(32)
SAAS_ACCESS_KEY = os.getenv("SAAS_ACCESS_KEY", "")        # blank = open access (no gate)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Comma-separated list of extra origins allowed to call this API with
# credentials from another domain. Empty by default -> CORS middleware is
# not even added, which is the correct, secure default for the normal
# case of the frontend being served by this same app (same-origin
# requests are never subject to CORS in the first place).
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# ---------------------------------------------------------------------------
# Rate limiting (per client IP) - protects your API budget from abuse
# ---------------------------------------------------------------------------
RATE_LIMIT_RESEARCH = os.getenv("RATE_LIMIT_RESEARCH", "6/hour")
RATE_LIMIT_CHAT = os.getenv("RATE_LIMIT_CHAT", "40/hour")

# ---------------------------------------------------------------------------
# Search (Tavily) tuning
#   "basic" search depth is materially faster than "advanced" and loses
#   little accuracy here since every page gets fully re-scraped by us
#   anyway - Tavily's job is just to find good candidate URLs.
# ---------------------------------------------------------------------------
SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "basic")
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", 16))
SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", 15))

# ---------------------------------------------------------------------------
# Scraping tuning
# ---------------------------------------------------------------------------
SCRAPE_TARGET_SUCCESS = int(os.getenv("SCRAPE_TARGET_SUCCESS", 8))
SCRAPE_TIMEOUT = int(os.getenv("SCRAPE_TIMEOUT", 8))
SCRAPE_MAX_WORKERS = int(os.getenv("SCRAPE_MAX_WORKERS", 16))
SCRAPE_CHAR_LIMIT = int(os.getenv("SCRAPE_CHAR_LIMIT", 6000))          # per-article cap fed to the LLM
REPORT_CONTEXT_CHAR_LIMIT = int(os.getenv("REPORT_CONTEXT_CHAR_LIMIT", 10000))

# Uploaded PDFs longer than this only have their first N pages processed -
# this app is built for reports/articles, not whole books, and without a
# cap a huge PDF can take a long time to parse and produce a response too
# large to render/save reliably.
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", 40))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 15))

# ---------------------------------------------------------------------------
# LLM model routing
#   Light/templated sections (intro, conclusion) go to the fast 8B model;
#   sections that need real synthesis from source data use the 70B model.
#   This dual-routing, combined with smaller per-section token budgets
#   below, is the single biggest lever behind the reduced generation time.
# ---------------------------------------------------------------------------
MODEL_FAST = os.getenv("MODEL_FAST", "meta/llama-3.1-8b-instruct")
MODEL_QUALITY = os.getenv("MODEL_QUALITY", "meta/llama-3.1-70b-instruct")
NVIDIA_API_TIMEOUT = int(os.getenv("NVIDIA_API_TIMEOUT", 80))

# ---------------------------------------------------------------------------
# Result caching (search step only - see utils/cache.py)
# ---------------------------------------------------------------------------
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 21600))  # 6 hours

# ---------------------------------------------------------------------------
# Supabase (optional) - persistent report history
#   Local disk (reports/<session_id>/...) works fine for a single long-lived
#   server, but most PaaS hosts (Render included) wipe the filesystem on
#   every redeploy/restart. If SUPABASE_URL + SUPABASE_KEY are set, every
#   report is also written to a Supabase table, and read back from there if
#   it's ever missing locally - so history survives redeploys. Leave blank
#   to keep using local-disk-only storage (nothing else changes).
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")

# ---------------------------------------------------------------------------
# Cloud text-to-speech (on by default, no API key needed) - for the Podcast tool
#   The browser's built-in speechSynthesis works with zero setup but sounds
#   robotic on most devices. By default, the podcast player instead uses
#   Microsoft Edge's free neural "Read Aloud" voices via the open-source
#   `edge-tts` library - no account, no key, no cost. This works by talking
#   to the same (undocumented) endpoint the real Edge browser uses
#   internally, so it's not an official Microsoft API and could in principle
#   stop working if they change that internal protocol - if that ever
#   happens, synthesize_speech() returns None and the podcast player falls
#   back to the browser's own built-in voice automatically. Set
#   TTS_ENABLED=false to skip it and always use the browser voice.
# ---------------------------------------------------------------------------
TTS_ENABLED = os.getenv("TTS_ENABLED", "true").lower() == "true"
RATE_LIMIT_TTS = os.getenv("RATE_LIMIT_TTS", "300/hour")  # a whole podcast play needs many small calls

# Accent x gender voice matrix for the podcast accent selector. These are
# real, well-established Microsoft neural voice IDs - override any of them
# with env vars if you want different ones.
TTS_VOICES = {
    "american": {
        "male": os.getenv("TTS_US_MALE_VOICE", "en-US-GuyNeural"),
        "female": os.getenv("TTS_US_FEMALE_VOICE", "en-US-AriaNeural"),
    },
    "indian": {
        "male": os.getenv("TTS_IN_MALE_VOICE", "en-IN-PrabhatNeural"),
        "female": os.getenv("TTS_IN_FEMALE_VOICE", "en-IN-NeerjaNeural"),
    },
}
DEFAULT_TTS_ACCENT = os.getenv("DEFAULT_TTS_ACCENT", "american")

# ---------------------------------------------------------------------------
# Report modes
#   "assignment" is the default: ~2500-3000 words, tuned for speed AND for
#   the length/structure a college assignment actually needs - the old
#   default (4 sections x 3500 tokens = a ~10,000-word thesis chapter) was
#   both slower than necessary and a poor fit for the actual use case.
#   "deep" preserves a (still trimmed-down) version of that exhaustive
#   behaviour for anyone who explicitly wants more.
# ---------------------------------------------------------------------------
REPORT_MODES = {
    "assignment": {
        "id": "assignment",
        "label": "Assignment Mode (Fast)",
        "description": "~2800-3200 words, structured for a college assignment. Optimised for speed.",
        "sections": [
            {
                "key": "introduction", "title": "📖 Introduction & Context",
                "instruction": "Introduce the topic clearly: what it is, why it matters, and what this report covers.",
                "model": "fast", "max_tokens": 700, "target_words": "350-420", "temperature": 0.4, "delay": 0.0,
            },
            {
                "key": "background", "title": "🧩 Core Concepts & Background",
                "instruction": "Explain the key concepts and background a reader needs to understand this topic, grounded in the source data.",
                "model": "quality", "max_tokens": 1000, "target_words": "550-620", "temperature": 0.45, "delay": 0.3,
            },
            {
                "key": "applications", "title": "🔍 Current Applications & Real-World Examples",
                "instruction": "Cover concrete current applications, examples, or data points from the source data. This is the most important section.",
                "model": "quality", "max_tokens": 1000, "target_words": "550-620", "temperature": 0.45, "delay": 1.0,
            },
            {
                "key": "challenges", "title": "⚖️ Challenges & Limitations",
                "instruction": "Discuss the main challenges or limitations, grounded in the source data.",
                "model": "quality", "max_tokens": 900, "target_words": "480-550", "temperature": 0.45, "delay": 1.7,
            },
            {
                "key": "trends", "title": "🚀 Future Trends & Opportunities",
                "instruction": "Discuss emerging trends and opportunities related to the topic, grounded in the source data.",
                "model": "quality", "max_tokens": 900, "target_words": "480-550", "temperature": 0.45, "delay": 2.4,
            },
            {
                "key": "conclusion", "title": "✅ Conclusion & Key Takeaways",
                "instruction": "Summarise the report into 4-6 clear takeaways and a short closing paragraph. Do not introduce new facts.",
                "model": "fast", "max_tokens": 600, "target_words": "300-360", "temperature": 0.4, "delay": 0.6,
            },
        ],
    },
    "deep": {
        "id": "deep",
        "label": "Deep Research Mode",
        "description": "~3800-4800 words, exhaustive multi-angle report. Slower (aim ~5-6 min).",
        "sections": [
            {
                "key": "executive", "title": "📊 Executive Summary & Introduction",
                "instruction": "Provide a detailed executive summary and introduction to the topic and its significance.",
                "model": "quality", "max_tokens": 950, "target_words": "550-620", "temperature": 0.5, "delay": 0.0,
            },
            {
                "key": "background", "title": "📚 Historical Context & Background",
                "instruction": "Cover the historical context and background needed to understand the topic in depth.",
                "model": "quality", "max_tokens": 950, "target_words": "550-620", "temperature": 0.5, "delay": 0.5,
            },
            {
                "key": "architecture", "title": "⚙️ Core Technical & Conceptual Deep-Dive",
                "instruction": "Provide a deep-dive analysis of the core technical or conceptual pillars, grounded in the source data.",
                "model": "quality", "max_tokens": 1100, "target_words": "650-720", "temperature": 0.5, "delay": 1.0,
            },
            {
                "key": "market", "title": "🌍 Real-World Applications & Case Studies",
                "instruction": "Cover concrete real-world applications, case studies, and market drivers from the source data.",
                "model": "quality", "max_tokens": 1100, "target_words": "650-720", "temperature": 0.5, "delay": 1.5,
            },
            {
                "key": "risk_matrix", "title": "⚠️ Challenges, Risks & Limitations",
                "instruction": "Contrast opportunities against systemic risks, challenges, and limitations.",
                "model": "quality", "max_tokens": 950, "target_words": "550-620", "temperature": 0.5, "delay": 2.0,
            },
            {
                "key": "roadmap", "title": "🔮 Future Trends & Predictions",
                "instruction": "Provide predictive analysis, emerging trends, and a forward-looking timeline.",
                "model": "quality", "max_tokens": 950, "target_words": "550-620", "temperature": 0.5, "delay": 2.5,
            },
            {
                "key": "conclusion", "title": "✅ Conclusion & Strategic Recommendations",
                "instruction": "Summarise the report's core findings into clear strategic recommendations and takeaways.",
                "model": "quality", "max_tokens": 700, "target_words": "400-450", "temperature": 0.45, "delay": 3.0,
            },
        ],
    },
    "flash": {
        "id": "flash",
        "label": "Flash Mode (Superfast)",
        "description": "~900-1100 words, key highlights only, entirely on the fast 8B model. Best for a quick first look.",
        "sections": [
            {
                "key": "introduction", "title": "📖 Quick Overview",
                "instruction": "Briefly introduce the topic: what it is and why it matters, in a few tight sentences.",
                "model": "fast", "max_tokens": 380, "target_words": "180-220", "temperature": 0.4, "delay": 0.0,
            },
            {
                "key": "analysis", "title": "🔍 Key Highlights",
                "instruction": "Cover the most important facts and current state of the topic using the source data, as concisely as possible.",
                "model": "fast", "max_tokens": 650, "target_words": "380-430", "temperature": 0.45, "delay": 0.15,
            },
            {
                "key": "challenges", "title": "⚖️ Notable Challenges & Trends",
                "instruction": "List the most important challenges and trends briefly, grounded in the source data.",
                "model": "fast", "max_tokens": 480, "target_words": "280-320", "temperature": 0.45, "delay": 0.3,
            },
            {
                "key": "conclusion", "title": "✅ Quick Takeaways",
                "instruction": "Summarise into 3-4 short, punchy takeaways. Do not introduce new facts.",
                "model": "fast", "max_tokens": 320, "target_words": "160-200", "temperature": 0.4, "delay": 0.45,
            },
        ],
    },
}
DEFAULT_REPORT_MODE = os.getenv("DEFAULT_REPORT_MODE", "assignment")


def validate_config() -> None:
    """Call once at startup. Fails loudly and clearly instead of producing confusing errors later."""
    missing = []
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    if not NVIDIA_API_KEY:
        missing.append("NVIDIA_API_KEY")
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your keys before running."
        )
