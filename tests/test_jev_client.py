"""Tests for rag/jev_client.py: the one seam to TypeSafe's Jev API.

What matters, in order: Jev being down never raises anything but a typed
JevError the callers catch; the key never leaves the Authorization header; a
retry never overruns the time budget; and every call, live, cached or failed,
lands in telemetry. Every request is answered by pytest-httpx; nothing here
reaches the network.
"""
from __future__ import annotations

import json
import logging

import httpx
import pytest

import rag.jev_client as jc
from obs import telemetry
from tests.jev_fakes import FAKE_KEY, JEV_URL, FakeClock, choice, noul, reply

QUESTIONS = {
    "hot": {"type": "noul", "instructions": "Is `text` about hot weather?"},
    "kind": {"type": "choice", "instructions": "Which kind?",
             "criteria": {"b_second": "B", "a_first": "A"}},
}
STATE = {"text": "It will be 45 degrees in Delhi this week."}
GOOD = reply({"hot": noul(0.97), "kind": choice({"b_second": 0.9, "a_first": 0.1})})


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def client(clock) -> jc.JevClient:
    c = jc.JevClient(FAKE_KEY, sleep=clock.sleep, clock=clock, rng=lambda: 0.0)
    yield c
    c.close()


def _ask(client: jc.JevClient, **kw) -> jc.JevResult:
    return client.ask(STATE, QUESTIONS, op="test", **kw)


# --- the flag ----------------------------------------------------------------

@pytest.mark.parametrize(("value", "mode", "triage", "verify"), [
    (None, "off", False, False),
    ("off", "off", False, False),
    ("triage", "triage", True, False),
    ("verify", "verify", False, True),
    ("all", "all", True, True),
    (" ALL ", "all", True, True),
])
def test_flag_resolves_each_documented_mode(monkeypatch, value, mode, triage, verify):
    if value is not None:
        monkeypatch.setenv("CRG_JEV", value)
    assert jc.jev_mode() == mode
    assert jc.triage_enabled() is triage
    assert jc.verify_enabled() is verify


def test_an_unknown_flag_value_is_off_and_says_so(monkeypatch, caplog):
    monkeypatch.setenv("CRG_JEV", "everything")
    with caplog.at_level(logging.WARNING):
        assert jc.jev_mode() == "off"
    assert "CRG_JEV" in caplog.text and not jc.triage_enabled()


# --- request shape -------------------------------------------------------------

def test_request_is_the_documented_post_with_a_bearer_key(client, httpx_mock):
    httpx_mock.add_response(url=JEV_URL, method="POST", json=GOOD)

    _ask(client)

    [request] = httpx_mock.get_requests()
    assert str(request.url) == JEV_URL
    assert request.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert request.headers["Content-Type"] == "application/json"
    body = json.loads(request.content)
    assert body == {"model": "jev-1.14.0", "state": STATE, "questions": QUESTIONS}


def test_choice_option_order_survives_onto_the_wire(client, httpx_mock):
    """Jev leans toward the first option, so the order sent is part of the question."""
    httpx_mock.add_response(json=GOOD)

    _ask(client)

    sent = json.loads(httpx_mock.get_requests()[0].content)
    assert list(sent["questions"]["kind"]["criteria"]) == ["b_second", "a_first"]


def test_the_model_is_a_pinned_version_and_overridable(monkeypatch):
    assert jc.DEFAULT_MODEL == "jev-1.14.0"  # a version, never the moving alias
    monkeypatch.setenv("CRG_JEV_MODEL", "jev-1.13.0")
    c = jc.JevClient(FAKE_KEY)
    try:
        assert c.model == "jev-1.13.0"
    finally:
        c.close()


# --- response parsing ----------------------------------------------------------

def test_reply_is_parsed_into_typed_answers(client, httpx_mock):
    httpx_mock.add_response(json=GOOD, headers={"x-typesafe-request-id": "req_123"})

    result = _ask(client)

    assert result.noul("hot") == pytest.approx(0.97)
    kind = result.choice("kind")
    assert kind.choice == "b_second" and kind.p("a_first") == pytest.approx(0.1)
    assert result.model == "jev-1.14.0"
    assert result.usage == jc.Usage(input_tokens=420, output_tokens=30)
    assert result.request_id == "req_123"
    assert result.cost_usd == pytest.approx(420 * 0.042 / 1e6)
    assert not result.cached and result.retries == 0


@pytest.mark.parametrize("broken", [
    {"model": "jev-1.14.0", "usage": {"input_tokens": 1, "output_tokens": 0}},  # no answers
    reply({"hot": noul(0.9)}),  # a question left unanswered
    reply({"hot": {"type": "choice", "choice": "x", "confidence": 1, "probabilities": {}},
           "kind": choice({"b_second": 1.0, "a_first": 0.0})}),  # wrong answer type
    reply({"hot": noul(1.7), "kind": choice({"b_second": 1.0, "a_first": 0.0})}),  # p > 1
    reply({"hot": noul(True), "kind": choice({"b_second": 1.0, "a_first": 0.0})}),  # a bool
    reply({"hot": noul(0.5), "kind": choice({"b_second": 0.5, "zzz": 0.5})}),  # unknown option
])
def test_a_reply_that_breaks_the_schema_is_a_typed_error_and_not_retried(
    client, httpx_mock, broken
):
    httpx_mock.add_response(json=broken)

    with pytest.raises(jc.JevResponseError):
        _ask(client)
    assert len(httpx_mock.get_requests()) == 1


def test_a_non_json_reply_is_a_response_error(client, httpx_mock):
    httpx_mock.add_response(text="<html>oops</html>")

    with pytest.raises(jc.JevResponseError, match="not JSON"):
        _ask(client)


# --- retries, backoff and the time budget -----------------------------------------

def test_overload_then_success_retries_with_backoff(client, clock, httpx_mock):
    httpx_mock.add_response(status_code=529)
    httpx_mock.add_response(json=GOOD)

    result = _ask(client)

    assert result.retries == 1
    assert clock.sleeps == [pytest.approx(0.25)]  # first backoff step, no jitter (rng=0)
    assert len(httpx_mock.get_requests()) == 2


def test_rate_limit_honours_retry_after_ms_then_retry_after(client, clock, httpx_mock):
    httpx_mock.add_response(status_code=429, headers={"retry-after-ms": "150"})
    httpx_mock.add_response(status_code=429, headers={"Retry-After": "1"})
    httpx_mock.add_response(json=GOOD)

    result = _ask(client, policy=jc.RetryPolicy(budget_s=5.0))

    assert clock.sleeps == [pytest.approx(0.15), pytest.approx(1.0)]
    assert result.retries == 2


def test_a_timeout_is_retried_like_an_outage(client, httpx_mock):
    httpx_mock.add_exception(httpx.ReadTimeout("slow"))
    httpx_mock.add_response(json=GOOD)

    assert _ask(client).retries == 1


def test_persistent_server_errors_give_up_as_unavailable(client, httpx_mock):
    for _ in range(3):
        httpx_mock.add_response(status_code=503)

    with pytest.raises(jc.JevUnavailable, match="after 3 attempts"):
        _ask(client)
    assert len(httpx_mock.get_requests()) == 3  # the initial try plus max_retries=2


def test_a_server_wait_longer_than_the_budget_is_not_slept_through(client, clock, httpx_mock):
    httpx_mock.add_response(status_code=429, headers={"Retry-After": "30"})

    with pytest.raises(jc.JevUnavailable, match="budget"):
        _ask(client)
    assert clock.sleeps == []  # gave up at once rather than sleeping 30 s past the budget


@pytest.mark.parametrize("status", [400, 413, 415, 422])
def test_a_request_the_api_refuses_is_not_retried(client, httpx_mock, status):
    httpx_mock.add_response(status_code=status, json={"detail": "nope"})

    with pytest.raises(jc.JevRequestError) as raised:
        _ask(client)
    assert raised.value.status == status
    assert len(httpx_mock.get_requests()) == 1


def test_a_validation_error_never_echoes_the_rejected_input(client, httpx_mock):
    """A 422 body carries an `input` preview, which can be the user's own question."""
    httpx_mock.add_response(status_code=422, json={"detail": [{
        "loc": ["body", "state"], "msg": "bad", "type": "value_error",
        "input": "MY PRIVATE QUESTION ABOUT DELHI",
    }]})

    with pytest.raises(jc.JevRequestError) as raised:
        _ask(client)
    assert "PRIVATE" not in raised.value.reason
    assert "value_error" in raised.value.reason and "body.state" in raised.value.reason
    assert "PRIVATE" not in json.dumps(telemetry.snapshot())


# --- the circuit breaker -------------------------------------------------------

@pytest.mark.parametrize("status", [401, 402, 403])
def test_a_refused_key_opens_the_circuit_so_later_calls_cost_no_round_trip(
    client, clock, httpx_mock, status
):
    httpx_mock.add_response(status_code=status, json={"detail": {
        "error_type": "authentication_error", "message": "Cannot authenticate"}})

    with pytest.raises(jc.JevRequestError):
        _ask(client)
    with pytest.raises(jc.JevUnavailable, match="circuit open"):
        _ask(client)
    assert len(httpx_mock.get_requests()) == 1

    clock.now += jc.FATAL_COOLDOWN_S + 1
    httpx_mock.add_response(json=GOOD)
    assert _ask(client).noul("hot") == pytest.approx(0.97)  # it tries again after the pause


def test_repeated_outages_open_the_circuit_for_a_short_cooldown(client, clock, httpx_mock):
    no_retry = jc.RetryPolicy(max_retries=0)
    for _ in range(jc.OUTAGE_FAILURES_TO_OPEN):
        httpx_mock.add_response(status_code=529)
    for _ in range(jc.OUTAGE_FAILURES_TO_OPEN):
        with pytest.raises(jc.JevUnavailable):
            _ask(client, policy=no_retry)

    with pytest.raises(jc.JevUnavailable, match="circuit open"):
        _ask(client, policy=no_retry)
    assert len(httpx_mock.get_requests()) == jc.OUTAGE_FAILURES_TO_OPEN

    clock.now += jc.OUTAGE_COOLDOWN_S + 1
    httpx_mock.add_response(json=GOOD)
    assert _ask(client, policy=no_retry).retries == 0


# --- the cache -----------------------------------------------------------------

def test_an_identical_request_is_served_from_cache_for_free(client, httpx_mock):
    httpx_mock.add_response(json=GOOD)

    first = _ask(client)
    second = _ask(client)

    assert len(httpx_mock.get_requests()) == 1
    assert second.cached and second.cost_usd == 0.0
    assert second.answers == first.answers


def test_a_reordered_option_list_is_a_different_request(client, httpx_mock):
    httpx_mock.add_response(json=GOOD, is_reusable=True)
    reordered = {**QUESTIONS, "kind": {**QUESTIONS["kind"],
                                       "criteria": {"a_first": "A", "b_second": "B"}}}

    _ask(client)
    client.ask(STATE, reordered, op="test")

    assert len(httpx_mock.get_requests()) == 2


def test_cache_entries_expire(client, clock, httpx_mock):
    httpx_mock.add_response(json=GOOD, is_reusable=True)

    _ask(client)
    clock.now += 3601
    _ask(client)

    assert len(httpx_mock.get_requests()) == 2


def test_a_failure_is_never_cached(client, httpx_mock):
    httpx_mock.add_response(status_code=400, json={"detail": "nope"})
    httpx_mock.add_response(json=GOOD)

    with pytest.raises(jc.JevRequestError):
        _ask(client)
    assert _ask(client).noul("hot") == pytest.approx(0.97)


def test_use_cache_false_always_calls(client, httpx_mock):
    httpx_mock.add_response(json=GOOD, is_reusable=True)

    _ask(client, use_cache=False)
    _ask(client, use_cache=False)

    assert len(httpx_mock.get_requests()) == 2


# --- documented request limits, checked before anything is sent --------------------

def test_a_body_over_32_mib_is_refused_locally(client, monkeypatch):
    monkeypatch.setattr(jc, "MAX_BODY_BYTES", 1000)
    with pytest.raises(jc.JevRequestError, match="32 MiB"):
        client.ask({"text": "x" * 2000}, QUESTIONS, op="test")


def test_a_non_finite_number_is_refused_locally(client):
    with pytest.raises(jc.JevRequestError, match="non-finite"):
        client.ask({"value": float("nan")}, QUESTIONS, op="test")


def test_nesting_deeper_than_64_levels_is_refused_locally(client):
    deep: dict = {}
    node = deep
    for _ in range(70):
        node["x"] = {}
        node = node["x"]
    with pytest.raises(jc.JevRequestError, match="nests deeper"):
        client.ask(deep, QUESTIONS, op="test")


def test_a_state_that_may_overflow_the_context_is_refused_locally(client):
    with pytest.raises(jc.JevRequestError, match="context"):
        client.ask({"text": "word " * 30_000}, QUESTIONS, op="test")


def test_a_request_needs_a_question(client):
    with pytest.raises(jc.JevRequestError):
        client.ask(STATE, {}, op="test")


# --- telemetry -----------------------------------------------------------------

def test_a_live_call_is_recorded_priced_and_on_disk(client, httpx_mock, tmp_path):
    httpx_mock.add_response(json=GOOD)

    _ask(client)

    [event] = [e for e in telemetry.snapshot() if e["op"] == "jev:test"]
    assert event["ok"] and not event["cached"]
    assert event["model"] == "jev-1.14.0" and event["tokens_in"] == 420
    assert telemetry.unpriced_models([event]) == []
    # the rollup rounds to six decimals, so one call's $0.0000176 reads as $0.000018
    assert telemetry.estimate_cost_usd([event]) == pytest.approx(420 * 0.042 / 1e6, abs=1e-6)
    sink = list((tmp_path / "telemetry").glob("*.jsonl"))
    assert sink and "jev:test" in sink[0].read_text(encoding="utf-8")


def test_a_failure_is_recorded_with_its_reason(client, httpx_mock):
    for _ in range(3):
        httpx_mock.add_response(status_code=529)

    with pytest.raises(jc.JevUnavailable):
        _ask(client)

    [event] = [e for e in telemetry.snapshot() if e["op"] == "jev:test"]
    assert not event["ok"] and event["retries"] == 2
    assert "HTTP 529" in event["detail"]


def test_a_cache_hit_is_recorded_as_cached(client, httpx_mock):
    httpx_mock.add_response(json=GOOD)

    _ask(client)
    _ask(client)

    events = [e for e in telemetry.snapshot() if e["op"] == "jev:test"]
    assert [e["cached"] for e in events] == [False, True]
    assert telemetry.estimate_cost_usd(events[1:]) == 0.0


# --- the key -------------------------------------------------------------------

def test_the_key_never_appears_in_repr_telemetry_or_logs(client, httpx_mock, caplog, tmp_path):
    httpx_mock.add_response(status_code=401, json={"detail": {"error_type": "authentication_error"}})

    with caplog.at_level(logging.DEBUG), pytest.raises(jc.JevRequestError) as raised:
        _ask(client)

    assert FAKE_KEY not in repr(client)
    assert FAKE_KEY not in str(raised.value)
    assert FAKE_KEY not in caplog.text
    assert FAKE_KEY not in json.dumps(telemetry.snapshot())
    for path in (tmp_path / "telemetry").glob("*.jsonl"):
        assert FAKE_KEY not in path.read_text(encoding="utf-8")


def test_no_key_is_a_config_error(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(jc.JevConfigError):
        jc.get_client()
    with pytest.raises(jc.JevConfigError):
        jc.JevClient("  ")


def test_the_shared_client_follows_a_rotated_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_KEY)
    first = jc.get_client()
    assert jc.get_client() is first
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_KEY + "-rotated")
    assert jc.get_client() is not first


def test_list_models_reads_the_models_endpoint(client, httpx_mock):
    httpx_mock.add_response(url="https://api.typesafe.ai/v1/models", json={"models": [
        {"name": "jev-latest", "description": "General", "release_date": "2026-10-14"}]})

    assert [m["name"] for m in client.list_models()] == ["jev-latest"]


# --- review findings, pinned ----------------------------------------------------

def test_a_reply_that_trickles_in_is_cut_at_the_budget(client, clock, httpx_mock):
    """httpx's read timeout restarts per chunk; the client must hold the total."""
    from pytest_httpx import IteratorStream

    def trickle():
        body = json.dumps(GOOD).encode()
        for i in range(0, len(body), 8):
            clock.sleep(1.0)  # each chunk arrives a second after the last
            yield body[i:i + 8]

    httpx_mock.add_response(stream=IteratorStream(trickle()))

    with pytest.raises(jc.JevUnavailable, match="still arriving"):
        _ask(client, policy=jc.RetryPolicy(budget_s=2.5))
    assert clock.now <= 2.5 + 1.0  # stopped within one chunk of the deadline


def test_a_key_an_http_header_cannot_carry_is_refused_without_repeating_it():
    with pytest.raises(jc.JevConfigError) as raised:
        jc.JevClient("abc\r\ndef")
    assert "abc" not in str(raised.value)


def test_reply_values_never_reach_an_error_reason(client, httpx_mock):
    secret = "MY PRIVATE QUESTION ABOUT DELHI"
    httpx_mock.add_response(json=reply({
        "hot": {"type": secret}, "kind": choice({"b_second": 1.0, "a_first": 0.0})}))
    httpx_mock.add_response(json=reply({
        "hot": noul(0.5),
        "kind": {"type": "choice", "choice": secret, "confidence": 1.0,
                 "probabilities": {"b_second": 1.0, "a_first": 0.0}}}))

    for _ in range(2):
        with pytest.raises(jc.JevResponseError) as raised:
            _ask(client, use_cache=False)
        assert secret not in raised.value.reason
    assert secret not in json.dumps(telemetry.snapshot())


def test_an_unexpected_http_error_is_still_a_typed_jev_error(client, httpx_mock):
    httpx_mock.add_exception(httpx.DecodingError("bad gzip"))

    with pytest.raises(jc.JevError):
        _ask(client)


def test_a_malformed_error_body_is_reduced_to_its_status(client, httpx_mock):
    httpx_mock.add_response(status_code=422, json={"detail": [{"loc": "not-a-list",
                                                               "type": "x"}]})
    with pytest.raises(jc.JevRequestError) as raised:
        _ask(client)
    assert raised.value.reason.startswith("HTTP 422")


@pytest.mark.parametrize("payload", [
    reply({"hot": noul(0.9), "kind": choice({"b_second": 1.0, "a_first": 0.0})},
          input_tokens=420.0),  # an integral float is still a count
    {"model": "", "answers": {"hot": noul(0.9), "kind": choice({"b_second": 1.0,
                                                               "a_first": 0.0})},
     "usage": {"input_tokens": 3}},  # empty model name and no output_tokens: both allowed
])
def test_replies_the_spec_allows_are_accepted(client, httpx_mock, payload):
    httpx_mock.add_response(json=payload)

    result = _ask(client)

    assert result.model  # an empty name falls back to the requested model
    assert result.noul("hot") == pytest.approx(0.9)


def test_a_score_outside_its_rubric_is_rejected():
    questions = {"s": {"type": "score", "criteria": ["low", "high"]}}
    for bad in (float("nan"), 2.5, -0.1):
        payload = reply({"s": {"type": "score", "score": bad, "confidence": 0.5,
                               "legend": {"0": "low", "1": "high"},
                               "probabilities": {"0": 0.5, "1": 0.5}}})
        with pytest.raises(jc.JevResponseError):
            jc.parse_answers(payload, questions)
