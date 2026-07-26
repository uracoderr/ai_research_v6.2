"""
Thin, shared client for the NVIDIA NIM chat-completions endpoint.

Before this module existed, three different files (search_agent,
filter_agent, report_agent) each built their own `requests.post(...)`
call with slightly different retry/timeout behaviour, and most errors
were swallowed by bare `except:` blocks. Centralising it here means:

- a fix (better retries, a swapped provider, tighter timeouts) only has
  to happen once
- every caller gets the same jittered backoff on rate limits (429s)
- failures raise a typed LLMError instead of silently returning None or
  triggering an unrelated fallback path, so each agent decides its own
  graceful-degradation story explicitly
"""
import random
import time
from typing import Optional

import requests

import config
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_MAP = {
    "fast": config.MODEL_FAST,
    "quality": config.MODEL_QUALITY,
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, factual research assistant. Output exactly what is "
    "asked for, with no filler, no preamble, and no markdown code fences "
    "unless explicitly requested."
)


class LLMError(RuntimeError):
    """Raised when the LLM API fails after all retries, or is not configured."""


def resolve_model(model: str) -> str:
    """Callers can pass 'fast' / 'quality' (routes to config's model ids) or a raw model id."""
    return MODEL_MAP.get(model, model)


def call_nvidia_api(
    prompt: str,
    system: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 1000,
    temperature: float = 0.4,
    model: str = "fast",
    timeout: Optional[int] = None,
    retries: int = 2,
) -> str:
    """
    Calls the chat-completions endpoint and returns the assistant's text
    content. Raises LLMError if every attempt fails.
    """
    if not config.NVIDIA_API_KEY:
        raise LLMError("NVIDIA_API_KEY is not configured.")

    resolved_model = resolve_model(model)
    timeout = timeout or config.NVIDIA_API_TIMEOUT
    headers = {
        "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(config.NVIDIA_URL, json=payload, headers=headers, timeout=timeout)

            if response.status_code == 429:
                wait = min(8, (2 ** attempt)) + random.uniform(0, 0.5)
                logger.warning(
                    "NVIDIA API rate-limited (attempt %s/%s, model=%s); backing off %.1fs",
                    attempt, retries, resolved_model, wait,
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning("NVIDIA API timeout (attempt %s/%s, model=%s)", attempt, retries, resolved_model)
        except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
            last_error = e
            logger.warning("NVIDIA API error (attempt %s/%s, model=%s): %s", attempt, retries, resolved_model, e)

        if attempt < retries:
            time.sleep(1 + random.uniform(0, 0.5))

    raise LLMError(f"NVIDIA API failed after {retries} attempt(s): {last_error}")


async def synthesize_speech(text: str, voice: str) -> Optional[bytes]:
    """
    Generates natural, human-sounding speech audio (mp3 bytes) using
    Microsoft Edge's free neural "Read Aloud" voices, via the open-source
    `edge-tts` library - no API key, no account, no cost. This exists
    because the browser's own built-in speechSynthesis sounds noticeably
    robotic on most Android/Chrome devices, which is a limitation of the
    device's synthesis engine that no amount of JS-side tuning can fix.

    `edge-tts` works by talking to the same endpoint the real Edge browser
    uses internally for "Read Aloud" - that endpoint isn't an official,
    documented Microsoft API, so it could in principle change or stop
    working without notice. This function never raises: any failure (or
    TTS_ENABLED=false) returns None, and the podcast player falls back to
    the browser's own voice automatically - this feature is a pure upgrade,
    never a hard requirement.
    """
    if not config.TTS_ENABLED or not text.strip():
        return None
    try:
        # Imported lazily (not at module top-level) so that if this optional
        # dependency is ever missing or fails to import for any reason, the
        # rest of the app still boots fine - only this one nice-to-have
        # feature degrades to the browser-voice fallback.
        import edge_tts

        communicate = edge_tts.Communicate(text[:3000], voice)
        audio_chunks = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                audio_chunks.extend(chunk["data"])
        return bytes(audio_chunks) if audio_chunks else None
    except Exception as e:
        logger.warning("Edge TTS failed, caller will fall back to browser voice: %s", e)
        return None
