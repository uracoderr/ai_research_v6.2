"""
Lightweight, login-free session isolation.

The old app stored every user's reports in one shared reports/ folder
and exposed it via a raw static file mount - meaning any visitor could
list or read any other visitor's research history. There was no concept
of "whose data is this."

This module fixes that without requiring a login: the first time a
browser hits the app, it gets a random, signed, httponly cookie. Every
report and RAG context file is then stored under reports/<session_id>/,
so one visitor can never see another's data, with zero signup friction
for students.

This is intentionally NOT a full user-account system - there is no
password, no email, no way to access your reports from a second device.
For a production SaaS with real logins, subscriptions, and billing,
replace this with a real auth provider (Clerk / Auth0 / Supabase Auth)
plus a database, and use their user id in place of session_id
everywhere below. See README.md -> "Scaling this further".
"""
import uuid
from typing import Optional

from fastapi import Request, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

import config
from utils.security import safe_slug

COOKIE_NAME = "trp_session"
COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 30  # 30 days

_serializer = URLSafeTimedSerializer(config.SECRET_KEY, salt="trp-session")


def _sign(session_id: str) -> str:
    return _serializer.dumps(session_id)


def _verify(signed_value: str) -> Optional[str]:
    try:
        return _serializer.loads(signed_value, max_age=COOKIE_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None


def read_session_id(request: Request) -> Optional[str]:
    """Returns the verified session id from the request's cookie, or None if absent/invalid/expired."""
    raw = request.cookies.get(COOKIE_NAME)
    if not raw:
        return None
    session_id = _verify(raw)
    return safe_slug(session_id, max_length=40) if session_id else None


def set_session_cookie(response: Response, session_id: str) -> None:
    """
    Attach a signed session cookie to a Response object.

    IMPORTANT: this must be called on the exact object your endpoint
    returns. FastAPI does NOT merge cookies set on an injected `response:
    Response` parameter into a different Response object (StreamingResponse,
    TemplateResponse, FileResponse, ...) that your function constructs and
    returns itself - only into responses FastAPI builds for you (e.g. when
    you return a plain dict). Mixing these up is a common, silent bug.
    """
    response.set_cookie(
        key=COOKIE_NAME,
        value=_sign(session_id),
        max_age=COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=(config.APP_ENV == "production"),
    )


def ensure_session(request: Request, response: Response) -> str:
    """
    Convenience for the common case: read the existing session id, or mint
    and attach a new one to `response`. Only correct when `response` is the
    actual object being returned by the endpoint - see set_session_cookie.
    """
    session_id = read_session_id(request)
    if not session_id:
        session_id = uuid.uuid4().hex
        set_session_cookie(response, session_id)
    return session_id
