"""Tests for obs/telemetry.py — structural observability at the Gemini seam.

JD-motivated (observability for AI apps: behavior, quality, latency, reliability),
but built our way: every record is measured, costs are labeled estimates, and
the recorder is thread-safe because embeds run in a pool.
"""
import json
import threading

import pytest

from obs.telemetry import Span, estimate_cost_usd, record, reset, snapshot, summarize


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEMETRY_DIR", str(tmp_path))
    reset()
    yield
    reset()


def test_record_lands_in_memory_and_on_disk(tmp_path):
    record(op="generate", model="gemini-2.5-flash", latency_ms=812.5,
           tokens_in=1000, tokens_out=200, retries=0, ok=True)

    [event] = snapshot()
    assert event["op"] == "generate" and event["latency_ms"] == 812.5

    jsonl = list(tmp_path.glob("*.jsonl"))
    assert jsonl, "telemetry must persist to disk, not just memory"
    line = json.loads(jsonl[0].read_text(encoding="utf-8").splitlines()[-1])
    assert line["tokens_in"] == 1000


def test_span_scopes_only_its_own_events():
    record(op="embed", model="m", latency_ms=1, tokens_in=0, tokens_out=0,
           retries=0, ok=True)  # before the span -> excluded
    with Span("report") as span:
        record(op="generate", model="m", latency_ms=100, tokens_in=10,
               tokens_out=5, retries=0, ok=True)
        record(op="embed", model="m", latency_ms=20, tokens_in=0, tokens_out=0,
               retries=1, ok=True)
    assert len(span.events) == 2
    assert span.wall_ms >= 0
    totals = span.summary()
    assert totals["calls"] == 2 and totals["retries"] == 1


def test_cached_events_cost_nothing():
    with Span("report") as span:
        record(op="generate", model="m", latency_ms=1, tokens_in=0, tokens_out=0,
               retries=0, ok=True, cached=True)
    assert span.summary()["cache_hits"] == 1
    assert estimate_cost_usd(span.events) == 0.0


def test_cost_estimate_uses_pricing_and_is_labeled():
    events = [dict(op="generate", model="gemini-2.5-flash", latency_ms=1,
                   tokens_in=1_000_000, tokens_out=0, retries=0, ok=True,
                   cached=False, ts="t")]
    cost = estimate_cost_usd(events)
    assert cost > 0.0  # priced from the (env-overridable) estimate table


def test_summarize_reports_latency_percentiles_and_failures():
    for ms, ok in [(10, True), (20, True), (30, True), (40, False)]:
        record(op="generate", model="m", latency_ms=ms, tokens_in=1,
               tokens_out=1, retries=0, ok=ok)
    stats = summarize(snapshot())
    gen = stats["generate"]
    assert gen["calls"] == 4 and gen["failures"] == 1
    assert gen["p50_ms"] <= gen["p95_ms"]


# --- the seam is the chokepoint: no Gemini call can escape being measured ---

def test_generate_json_records_latency_and_usage(monkeypatch):
    import rag.gemini_client as gc

    class FakeModels:
        def generate_content(self, **kwargs):
            usage = type("U", (), {"prompt_token_count": 123, "candidates_token_count": 45})()
            return type("R", (), {"text": "{}", "usage_metadata": usage})()

    monkeypatch.setattr(gc, "_client", lambda: type("C", (), {"models": FakeModels()})())
    gc.generate_json("prompt", schema={"type": "object"})

    [event] = [e for e in snapshot() if e["op"] == "generate"]
    assert event["ok"] and event["tokens_in"] == 123 and event["tokens_out"] == 45
    assert event["model"] == gc.GENERATE_MODEL


def test_failed_call_is_recorded_before_raising(monkeypatch):
    import rag.gemini_client as gc

    class FakeModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("boom (not a rate limit)")

    monkeypatch.setattr(gc, "_client", lambda: type("C", (), {"models": FakeModels()})())
    with pytest.raises(gc.GeminiError):
        gc.generate_json("prompt", schema={"type": "object"})

    [event] = snapshot()
    assert event["ok"] is False  # failures are data too


def test_recorder_is_thread_safe():
    def worker():
        for _ in range(50):
            record(op="embed", model="m", latency_ms=1, tokens_in=1,
                   tokens_out=0, retries=0, ok=True)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(snapshot()) == 400
