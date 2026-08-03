"""Tests for the SDK seam (rag/gemini_client.py): auth selection + retry logic.

The real SDK is never called — `_client` / the SDK call are monkeypatched.
"""
import pytest

import rag.gemini_client as gc


@pytest.fixture(autouse=True)
def _fresh_client_cache(monkeypatch):
    gc._reset_clients()
    for var in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield
    gc._reset_clients()


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

    assert gc._with_retry(flaky, op="test", model="m") == "ok"
    assert len(slept) == 2


def test_embed_batch_is_one_call_per_text_order_preserved(monkeypatch):
    # Measured failure this pins: Vertex only embeds ONE content per call; a
    # string list gets silently JOINED. So: one call per text, order kept.
    sent = []

    class FakeModels:
        def embed_content(self, *, model, contents, config):
            sent.append(contents)
            marker = float(len(contents))  # distinguishable per text
            return type("R", (), {"embeddings": [type("E", (), {"values": [marker]})()]})()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(gc, "_client", lambda: FakeClient())
    vectors = gc.embed_batch(["aa", "bbbb", "c"], task_type="RETRIEVAL_DOCUMENT", dims=768)
    assert sorted(sent) == ["aa", "bbbb", "c"]  # one call per text (threaded order varies)
    assert vectors == [[2.0], [4.0], [1.0]]  # results in INPUT order


def test_embed_batch_empty_response_is_typed_error(monkeypatch):
    class FakeModels:
        def embed_content(self, **kwargs):
            return type("R", (), {"embeddings": []})()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(gc, "_client", lambda: FakeClient())
    with pytest.raises(gc.GeminiError, match="no embedding"):
        gc.embed_batch(["a"], task_type="RETRIEVAL_DOCUMENT", dims=768)


def test_non_rate_limit_error_fails_fast_and_typed(monkeypatch):
    monkeypatch.setattr(gc, "_sleep", lambda s: pytest.fail("must not sleep"))

    def broken():
        raise RuntimeError("invalid argument")

    with pytest.raises(gc.GeminiError, match="invalid argument"):
        gc._with_retry(broken, op="test", model="m")


def test_vertex_credentials_resolve_once_and_are_passed_to_every_client(monkeypatch):
    """ADC must resolve ONCE per process, never inside a request.

    google-auth discovers the project by shelling out to the `gcloud` CLI. Inside
    an MCP stdio server that child process never returns, so resolving auth per
    client -- and clients are per-thread -- hangs the server forever (measured:
    search_ipcc hung >300s over stdio, stuck in _run_subprocess_ignore_stderr).
    Resolving once and handing the credentials to every client keeps gcloud out
    of the request path entirely.
    """
    import threading

    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setattr(gc, "_adc_file", lambda: None)  # force the default() branch

    resolved = []
    sentinel = object()

    def fake_default(**kwargs):
        resolved.append(kwargs)
        return sentinel, "test-project"

    captured = []

    def fake_client(**kwargs):
        captured.append(kwargs)
        return object()

    monkeypatch.setattr("google.auth.default", fake_default)
    monkeypatch.setattr("google.genai.Client", fake_client)

    barrier = threading.Barrier(3)

    def build():
        barrier.wait()
        gc._new_client()

    threads = [threading.Thread(target=build) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(resolved) == 1, f"ADC resolved {len(resolved)}x; must be once per process"
    assert len(captured) == 3
    assert all(kw.get("credentials") is sentinel for kw in captured), (
        "every client must receive the pre-resolved credentials, else the SDK "
        "calls load_auth() -> gcloud subprocess on that thread"
    )


def test_adc_file_is_read_directly_never_via_gcloud_subprocess(monkeypatch, tmp_path):
    """When an ADC file exists, google.auth.default() must not be called.

    default() is the function that shells out to `gcloud` for the project. The
    file gives us the same credentials with no subprocess, and the project comes
    from GOOGLE_CLOUD_PROJECT, so there is nothing left for default() to do.
    """
    adc = tmp_path / "application_default_credentials.json"
    adc.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(adc))
    sentinel = object()
    loaded = []

    def boom(**kwargs):
        raise AssertionError("google.auth.default() called - that can shell out to gcloud")

    def fake_load(filename, **kwargs):
        loaded.append(filename)
        return sentinel, None

    monkeypatch.setattr("google.auth.default", boom)
    monkeypatch.setattr("google.auth.load_credentials_from_file", fake_load)

    assert gc._vertex_credentials() is sentinel
    assert loaded == [str(adc)]
