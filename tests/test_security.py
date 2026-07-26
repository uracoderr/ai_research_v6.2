"""
Minimal smoke tests for the security-critical helpers in utils/security.py.

These don't need network access or API keys, so they're a fast way to
make sure the path-traversal / slug-sanitisation logic keeps working as
the code evolves.

Run with: pytest
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from utils.security import escape_plain_text, safe_slug, validate_topic


def test_safe_slug_basic():
    assert safe_slug("Climate Change 2026") == "climate_change_2026"


def test_safe_slug_blocks_path_traversal():
    slug = safe_slug("../../../etc/passwd")
    assert "/" not in slug
    assert ".." not in slug


def test_safe_slug_blocks_path_traversal_via_session_id():
    # session ids are also run through safe_slug (see utils/session.py) -
    # this must hold even if a cookie were somehow tampered with.
    slug = safe_slug("../../evil-session")
    assert "/" not in slug and ".." not in slug


def test_safe_slug_empty_and_dots_fall_back_to_untitled():
    assert safe_slug("") == "untitled"
    assert safe_slug("....") == "untitled"
    assert safe_slug(None) == "untitled"


def test_safe_slug_length_is_capped():
    long_input = "a" * 500
    assert len(safe_slug(long_input)) <= 80


def test_validate_topic_rejects_blank():
    with pytest.raises(ValueError):
        validate_topic("   ")


def test_validate_topic_rejects_too_long():
    with pytest.raises(ValueError):
        validate_topic("x" * 500)


def test_validate_topic_strips_control_characters():
    cleaned = validate_topic("Hello\x00World")
    assert "\x00" not in cleaned
    assert cleaned == "HelloWorld"


def test_escape_plain_text_neutralises_script_tags():
    escaped = escape_plain_text("<script>alert(1)</script>")
    assert "<script>" not in escaped
    assert "&lt;script&gt;" in escaped
