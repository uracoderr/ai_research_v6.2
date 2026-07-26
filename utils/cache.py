"""
Small file-based cache, currently used for one thing: Tavily search
results (see agents/search_agent.py). If two students research the
same or a very similar topic within the TTL window, the second one
skips a ~3-8s network round trip and doesn't burn extra Tavily quota.

This is intentionally simple - a JSON blob per key on disk, no external
service required. It is NOT meant for multi-instance production
deployments: if you ever scale this app horizontally behind a load
balancer, swap this for Redis. The get/set interface below would stay
identical, so that swap is a one-file change.
"""
import hashlib
import json
import os
import time
from typing import Optional

import config

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")


def _key_to_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{digest}.json")


def make_key(*parts: str) -> str:
    return "|".join(str(p).strip().lower() for p in parts if p)


def get(key: str):
    if not config.CACHE_ENABLED:
        return None
    path = _key_to_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if time.time() - payload.get("_cached_at", 0) > config.CACHE_TTL_SECONDS:
            return None
        return payload.get("data")
    except (json.JSONDecodeError, OSError):
        return None


def set(key: str, data) -> None:
    if not config.CACHE_ENABLED:
        return
    path = _key_to_path(key)
    try:
        import tempfile
        fd, tmp_path = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"_cached_at": time.time(), "data": data}, f)
            os.replace(tmp_path, path)  # atomic on POSIX; best-effort on Windows
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError:
        pass
