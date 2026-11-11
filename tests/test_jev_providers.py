"""Tests for Jev providers: TypeSafe direct (the default) and the Experiential
Labs gateway, which serves the same request schema with its own limits.

Written before the code they test. Every request is answered by pytest-httpx.
"""
from __future__ import annotations

import json

import pytest

import rag.jev_client as jc
import rag.jev_triage as tri
from obs import telemetry
from tests.jev_fakes import (
    FAKE_KEY,
    GATEWAY,
    GATEWAY_URL,
    JEV_URL,
    FakeClock,
    choice,
    gateway_reply,
    noul,
    reply,
    triage_reply,
    use_gateway,
)

QUESTIONS = {"hot": {"type": "noul", "instructions": "Is `text` about hot weather?"}}
STATE = {"text": "It will be 45 degrees in Delhi this week."}


# --- choosing the provider -----------------------------------------------------

def test_typesafe_direct_is_the_default(monkeypatch):
    monkeypatch.delenv("JEV_BASE_URL", raising=False)
    provider = jc.get_provider()
    assert provider.name == "typesafe" and provider.base_url == "https://api.typesafe.ai"
    assert provider.default_model == "jev-1.14.0"
    assert provider.max_request_tokens == 64_000


@pytest.mark.parametrize("raw", [GATEWAY, GATEWAY + "/", f"  {GATEWAY}  "])
def test_the_gateway_is_chosen_by_jev_base_url(monkeypatch, raw):
    monkeypatch.setenv("JEV_BASE_URL", raw)
    provider = jc.get_provider()
    assert provider.name == "experientiallabs"
    # the gateway grants only the alias (HTTP 403 model_not_granted for jev-1.14.0)
    assert provider.default_model == "jev-latest"
    assert provider.max_request_tokens == 32_000  # its context_window_tokens


def test_an_unknown_base_url_is_refused_not_followed(monkeypatch, httpx_mock):
    """The target stays a vetted constant: config picks one, it cannot add one."""
    monkeypatch.setenv("CRG_JEV", "triage")
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    monkeypatch.setenv("JEV_BASE_URL", "https://evil.example.com")
    with pytest.raises(jc.JevConfigError):
        jc.get_provider()
    assert tri.triage("How hot will it get?").decision == "unavailable"
    assert httpx_mock.get_requests() == []


def test_the_gateway_client_posts_to_the_gateway_with_the_alias(monkeypatch, httpx_mock):
    use_gateway(monkeypatch)
    httpx_mock.add_response(url=GATEWAY_URL, json=gateway_reply({"hot": noul(0.98)}))

    result = jc.get_client().ask(STATE, QUESTIONS, op="test")

    [request] = httpx_mock.get_requests()
    assert str(request.url) == GATEWAY_URL
    assert json.loads(request.content)["model"] == "jev-latest"
    assert result.noul("hot") == pytest.approx(0.98)


def test_jev_api_key_wins_and_typesafe_api_key_is_the_fallback(monkeypatch, httpx_mock):
    httpx_mock.add_response(url=JEV_URL, json=reply({"hot": noul(0.9)}), is_reusable=True)
    monkeypatch.setenv("TYPESAFE_API_KEY", "fallback-key")
    jc.get_client().ask(STATE, QUESTIONS, op="test", use_cache=False)
    monkeypatch.setenv("JEV_API_KEY", "preferred-key")
    jc.get_client().ask(STATE, QUESTIONS, op="test", use_cache=False)

    auth = [r.headers["Authorization"] for r in httpx_mock.get_requests()]
    assert auth == ["Bearer fallback-key", "Bearer preferred-key"]


def test_switching_provider_rebuilds_the_shared_client(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    first = jc.get_client()
    monkeypatch.setenv("JEV_BASE_URL", GATEWAY)
    second = jc.get_client()
    assert second is not first and second.provider.name == "experientiallabs"


# --- the gateway's reply ---------------------------------------------------------

def test_the_gateway_reply_parses_and_its_reported_cost_is_used(monkeypatch, httpx_mock):
    use_gateway(monkeypatch)
    httpx_mock.add_response(json=gateway_reply({"hot": noul(0.98)}, cost=0.0),
                            headers={"x-request-id": "req_9"})

    result = jc.get_client().ask(STATE, QUESTIONS, op="test")

    assert result.cost_usd == 0.0  # what the gateway reports, not the list price
    assert result.usage.reported_cost_usd == 0.0
    assert result.request_id == "req_9" and result.upstream == "typesafe"
    [event] = [e for e in telemetry.snapshot() if e["op"] == "jev:test"]
    assert event["cost_usd"] == 0.0
    # an alias has no price-table entry, but a reported cost is not an unknown cost
    assert telemetry.unpriced_models([event]) == []
    assert telemetry.estimate_cost_usd([event]) == 0.0


def test_without_a_reported_cost_the_list_price_applies(monkeypatch, httpx_mock):
    use_gateway(monkeypatch)
    httpx_mock.add_response(json=gateway_reply({"hot": noul(0.98)}, cost=None))

    result = jc.get_client().ask(STATE, QUESTIONS, op="test")

    assert result.usage.reported_cost_usd is None
    assert result.cost_usd == pytest.approx(1011 * 0.042 / 1e6)


def test_the_decision_id_stands_in_when_there_is_no_request_id_header(monkeypatch, httpx_mock):
    use_gateway(monkeypatch)
    httpx_mock.add_response(json=gateway_reply({"hot": noul(0.5)}))

    assert jc.get_client().ask(STATE, QUESTIONS, op="test").request_id == "decision_0123"


def test_a_nonsense_reported_cost_is_ignored_not_trusted(monkeypatch, httpx_mock):
    use_gateway(monkeypatch)
    httpx_mock.add_response(json=gateway_reply({"hot": noul(0.5)}, cost=-3))

    assert jc.get_client().ask(STATE, QUESTIONS, op="test").usage.reported_cost_usd is None


def test_triage_works_through_the_gateway(monkeypatch, httpx_mock):
    use_gateway(monkeypatch)
    payload = triage_reply(topic={"assessed_hazard": 1.0}, about=0.93,
                           hazard={"heatwave": 1.0}, other={"unrelated": 0.5, "drought": 0.5})
    payload.update({"id": "decision_1", "provider": "typesafe", "model": "jev-latest"})
    payload["usage"].update({"cost": 0.0, "is_byok": False})
    httpx_mock.add_response(url=GATEWAY_URL, json=payload)

    reading = tri.triage("How hot will it get in Delhi next week?")

    assert reading.decision == "hint" and reading.hazard_hint == "heatwave"
    assert reading.model == "jev-latest" and reading.cost_usd == 0.0


# --- per-provider request size -----------------------------------------------------

def _big_request() -> tuple[dict, dict]:
    """About 20k estimated tokens of state and two questions of about 7k each:
    under every per-question limit, over the gateway's 32k per request."""
    state = {"text": "x" * 40_000}
    questions = {
        name: {"type": "choice", "instructions": "Which?", "criteria": {"a": "y" * 14_000, "b": "z"}}
        for name in ("q1", "q2")
    }
    return state, questions


def test_the_gateways_smaller_context_is_enforced_locally(monkeypatch, httpx_mock):
    state, questions = _big_request()
    use_gateway(monkeypatch)
    with pytest.raises(jc.JevRequestError, match="32,000-token"):
        jc.get_client().ask(state, questions, op="test")
    assert httpx_mock.get_requests() == []


def test_the_same_request_fits_typesafe_direct(monkeypatch, httpx_mock):
    state, questions = _big_request()
    httpx_mock.add_response(json=reply({"q1": choice({"a": 1.0, "b": 0.0}),
                                        "q2": choice({"a": 1.0, "b": 0.0})}))
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)

    jc.get_client().ask(state, questions, op="test")

    assert len(httpx_mock.get_requests()) == 1


# --- pacing, for plans with a per-minute request cap ------------------------------------

def test_pacing_spaces_requests_and_is_not_counted_as_latency(httpx_mock):
    clock = FakeClock()
    client = jc.JevClient(FAKE_KEY, sleep=clock.sleep, clock=clock, rng=lambda: 0.0,
                          min_interval_s=7.5)
    httpx_mock.add_response(json=reply({"hot": noul(0.9)}), is_reusable=True)
    try:
        first = client.ask(STATE, QUESTIONS, op="test", use_cache=False)
        second = client.ask(STATE, QUESTIONS, op="test", use_cache=False)
    finally:
        client.close()

    assert clock.sleeps == [pytest.approx(7.5)]  # the second waited for its slot
    assert first.latency_ms == 0.0 and second.latency_ms == 0.0  # the wait is not Jev time


def test_pacing_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    monkeypatch.setenv("CRG_JEV_MIN_INTERVAL_S", "7.5")
    assert jc.get_client().min_interval_s == 7.5


def test_a_rate_limit_without_retry_after_waits_at_least_one_paced_slot(httpx_mock):
    clock = FakeClock()
    client = jc.JevClient(FAKE_KEY, sleep=clock.sleep, clock=clock, rng=lambda: 0.0,
                          min_interval_s=7.5)
    httpx_mock.add_response(status_code=429)
    httpx_mock.add_response(json=reply({"hot": noul(0.9)}))
    try:
        result = client.ask(STATE, QUESTIONS, op="test",
                            policy=jc.RetryPolicy(budget_s=30.0))
    finally:
        client.close()

    assert result.retries == 1
    assert max(clock.sleeps) >= 7.5  # not the 0.25 s default first backoff


def test_list_models_reads_the_gateways_openai_style_list(monkeypatch, httpx_mock):
    use_gateway(monkeypatch)
    httpx_mock.add_response(url=f"{GATEWAY}/v1/models", json={"object": "list", "data": [
        {"id": "jev-latest", "object": "model", "context_window_tokens": 32000}]})

    assert [m["name"] for m in jc.get_client().list_models()] == ["jev-latest"]
