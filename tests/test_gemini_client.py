"""Tests for the SDK seam (rag/gemini_client.py): auth selection + retry logic.

The real SDK is never called — `_client` / the SDK call are monkeypatched.
"""
import pytest

import rag.gemini_client as gc


@pytest.fixture(autouse=True)
def _fresh_client_cache(monkeypatch):
    gc._client.cache_clear()
    for var in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield
    gc._client.cache_clear()


def test_no_auth_configured_is_a_typed_error():
    with pytest.raises(gc.GeminiError, match="no Gemini auth"):
        gc._client()


def test_vertex_mode_requires_project(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    with pytest.raises(gc.GeminiError, match="GOOGLE_CLOUD_PROJECT"):
        gc._client()


def test_retry_sleeps_through_rate_limits(monkeypatch):
    slept = []
    monkeypatch.setattr(gc, "_sleep", slept.append)

    class RateLimit(Exception):
        code = 429

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimit("quota")
        return "ok"

    assert gc._with_retry(flaky) == "ok"
    assert len(slept) == 2


def test_non_rate_limit_error_fails_fast_and_typed(monkeypatch):
    monkeypatch.setattr(gc, "_sleep", lambda s: pytest.fail("must not sleep"))

    def broken():
        raise RuntimeError("invalid argument")

    with pytest.raises(gc.GeminiError, match="invalid argument"):
        gc._with_retry(broken)
