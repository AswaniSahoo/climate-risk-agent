"""The spend guard (rag/spend_guard.py) and its one enforcement point,
rag/gemini_client.py. No test reaches Gemini: every SDK call is a fake, and
every counter is the in-process one unless a test hands in another.
"""
from __future__ import annotations

import json
import logging

import pytest

import rag.gemini_client as gc
from rag import spend_guard as sg

FLASH = "gemini-2.5-flash"


class _FakeModels:
    """Answers generate calls with a fixed usage, counting what was sent."""

    def __init__(self, tokens_in: int = 1000, tokens_out: int = 400, thoughts: int = 600):
        self.calls = 0
        self.embeds = 0
        self._usage = type("U", (), {"prompt_token_count": tokens_in,
                                     "candidates_token_count": tokens_out,
                                     "thoughts_token_count": thoughts})()

    def generate_content(self, **kwargs):
        self.calls += 1
        return type("R", (), {"text": "{}", "usage_metadata": self._usage})()

    def embed_content(self, **kwargs):
        self.embeds += 1
        emb = type("E", (), {"values": [0.0, 1.0]})()
        return type("R", (), {"embeddings": [emb]})()


@pytest.fixture
def fake(monkeypatch) -> _FakeModels:
    models = _FakeModels()
    monkeypatch.setattr(gc, "_client", lambda: type("C", (), {"models": models})())
    return models


def _spent() -> int:
    return sg.spent_today_micro_usd()


# --- the budget, in money --------------------------------------------------

def test_the_defaults_are_the_documented_safe_ones(monkeypatch):
    for name in ("CRG_DAILY_MODEL_BUDGET_USD", "CRG_SESSION_FRESH_REPORTS",
                 "CRG_API_RATE_LIMIT", "CRG_API_RATE_WINDOW_S"):
        monkeypatch.delenv(name, raising=False)

    assert sg.daily_budget_micro_usd() == 150_000  # US$0.15 a UTC day
    assert sg.session_fresh_report_limit() == 5
    assert sg.api_rate_limit() == (10, 60)


@pytest.mark.parametrize("raw", ["nonsense", "-1", "nan", "inf", ""])
def test_a_bad_value_keeps_the_default_never_unlimited(monkeypatch, raw):
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", raw)

    assert sg.daily_budget_micro_usd() == 150_000


def test_off_disables_the_budget_and_zero_refuses_every_call(monkeypatch, fake):
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "off")
    gc.generate_json("p", schema={})
    assert fake.calls == 1 and _spent() == 0

    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "0")
    with pytest.raises(gc.ModelBudgetExceeded):
        gc.generate_json("p", schema={})
    assert fake.calls == 1  # refused before anything was sent


def test_a_call_starts_only_if_its_worst_case_fits_in_the_day(monkeypatch, fake):
    """The reservation is the prompt at two characters a token plus the full
    default thinking budget and a visible answer, at the list price."""
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "0.03")
    reservation = sg.reserve_generate(FLASH, "x" * 2000)

    # 1000 prompt tokens x $0.30 + (8192 + 2048) output tokens x $2.50, in micro-USD
    assert reservation.micro_usd == 300 + 25_600
    with pytest.raises(sg.BudgetRefused, match="daily model budget"):
        sg.reserve_generate(FLASH, "x" * 2000)  # 51,800 > 30,000
    assert _spent() == 25_900  # the refused reservation was given back


def test_a_finished_call_is_charged_what_its_usage_says_was_billed(monkeypatch, fake):
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "1")
    gc.generate_json("x" * 2000, schema={})

    # 1000 in x $0.30 + (400 visible + 600 thinking) x $2.50: thinking bills as output
    assert _spent() == 300 + 2_500


def test_the_budget_stops_calls_mid_day_and_says_why(monkeypatch, fake, caplog):
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "0.03")
    gc.generate_json("p", schema={})  # reserves ~25,600, settles to 2,800
    caplog.set_level(logging.WARNING, logger="rag.gemini_client")
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "0.01")

    with pytest.raises(gc.ModelBudgetExceeded) as refused:
        gc.generate_json("p", schema={})

    assert isinstance(refused.value, gc.GeminiError)  # every caller's degraded path
    assert "resets at 00:00 UTC" in str(refused.value)
    assert "CRG_DAILY_MODEL_BUDGET_USD" in caplog.text  # the operator learns the knob
    assert fake.calls == 1


def test_each_utc_day_has_its_own_budget(monkeypatch, fake):
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "0.03")
    monkeypatch.setattr(sg, "_now", lambda: 1_794_700_000.0)  # 2026-11-14 UTC
    sg.reserve_generate(FLASH, "p")
    with pytest.raises(sg.BudgetRefused):
        sg.reserve_generate(FLASH, "p")

    monkeypatch.setattr(sg, "_now", lambda: 1_794_700_000.0 + 86_400)
    sg.reserve_generate(FLASH, "p")  # a new day, a new budget


def test_a_rate_limited_call_is_given_back_and_others_keep_the_prompt(monkeypatch):
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "1")
    monkeypatch.setattr(gc, "_sleep", lambda s: None)

    def rate_limited(**kwargs):
        raise type("ClientError", (Exception,), {"code": 429})("429 RESOURCE_EXHAUSTED")

    models = type("M", (), {"generate_content": staticmethod(rate_limited)})()
    monkeypatch.setattr(gc, "_client", lambda: type("C", (), {"models": models})())
    with pytest.raises(gc.GeminiError):
        gc.generate_json("x" * 2000, schema={})
    assert _spent() == 0  # a 429 generated nothing

    def broken(**kwargs):
        raise RuntimeError("connection reset after the request was sent")

    models = type("M", (), {"generate_content": staticmethod(broken)})()
    monkeypatch.setattr(gc, "_client", lambda: type("C", (), {"models": models})())
    with pytest.raises(gc.GeminiError):
        gc.generate_json("x" * 2000, schema={})
    assert _spent() == 300  # 1000 prompt tokens x $0.30: maybe billed; no output was


def test_a_call_that_never_left_the_process_costs_nothing(monkeypatch):
    """No credentials: the client is never built, so nothing can be billed, and
    the reader keeps seeing the auth error, not a budget one."""
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "1")
    for var in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    gc._reset_clients()

    with pytest.raises(gc.GeminiError, match="no Gemini auth"):
        gc.generate_json("p", schema={})
    assert _spent() == 0


def test_embeddings_are_charged_up_front_for_the_whole_batch(monkeypatch, fake):
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "1")
    gc.embed_batch(["x" * 40, "y" * 20], task_type="RETRIEVAL_QUERY", dims=2)

    # (20 + 10) tokens at two characters each, x $0.20 per million (Gemini Embedding 2)
    assert _spent() == 6 and fake.embeds == 2


def test_an_unpriced_model_is_refused_while_the_budget_is_on(monkeypatch):
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "1")
    with pytest.raises(sg.BudgetRefused, match="No price is configured"):
        sg.reserve_generate("gemini-0-unpriced", "p")

    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "off")
    sg.reserve_generate("gemini-0-unpriced", "p")  # off means off


# --- the startup probe stays a health check --------------------------------

def test_the_embedding_probe_is_exempt_and_cached(monkeypatch, fake):
    """A spent budget must not read as an embedding misconfiguration in the
    UI's banner, and a stream of new sessions must not mean a stream of calls."""
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "0")

    assert gc.embedding_available(dims=2) == (True, f"dense OK [{gc.EMBED_MODEL}]")
    assert gc.embedding_available(dims=2)[0] is True
    assert fake.embeds == 1  # the second session read the cached result


# --- scopes: a pause switch that follows the work --------------------------

def test_a_paused_scope_refuses_without_touching_the_counter(monkeypatch, fake):
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "1")
    with sg.call_scope(blocked="paused for this session") as scope:
        with pytest.raises(gc.ModelBudgetExceeded, match="paused for this session"):
            gc.generate_json("p", schema={})
    assert scope.calls == 0 and fake.calls == 0 and _spent() == 0


def test_a_scope_counts_the_calls_made_inside_it(monkeypatch, fake):
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "off")
    with sg.call_scope() as scope:
        gc.generate_json("p", schema={})
        gc.embed_batch(["a"], task_type="RETRIEVAL_QUERY", dims=2)
    assert scope.calls == 2
    assert sg.current_scope() is None


def test_the_scope_reaches_into_langgraph_nodes():
    """LangGraph runs nodes on its own executor; the pause must still apply."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class S(TypedDict, total=False):
        seen: str

    def node(state: S) -> dict:
        try:
            sg.reserve_generate(FLASH, "p")
        except sg.BudgetRefused as exc:
            return {"seen": str(exc)}
        return {"seen": "not paused"}

    builder = StateGraph(S)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile()

    with sg.call_scope(blocked="paused"):
        assert graph.invoke({})["seen"] == "paused"


# --- the shared counter: Upstash over REST, with a local fallback ----------

def test_the_shared_counter_is_one_atomic_incrby_and_expire(httpx_mock):
    from tools.cache_backend import UpstashRedisCache

    httpx_mock.add_response(url="https://db.example/multi-exec",
                            json=[{"result": 42}, {"result": 1}])
    counter = sg.UpstashCounter(UpstashRedisCache("https://db.example", "tok"))

    assert counter.add("crg:v1:spend:2026-11-11", 40, 172_800) == 42
    [request] = httpx_mock.get_requests()
    assert json.loads(request.content) == [
        ["INCRBY", "crg:v1:spend:2026-11-11", 40],
        ["EXPIRE", "crg:v1:spend:2026-11-11", 172_800],
    ]
    assert request.headers["Authorization"] == "Bearer tok"


def test_an_unreachable_shared_counter_falls_back_to_this_process(httpx_mock, caplog):
    from tools.cache_backend import UpstashRedisCache

    httpx_mock.add_response(url="https://db.example/multi-exec", status_code=401)
    clock = [0.0]
    counter = sg.FallbackCounter(
        sg.UpstashCounter(UpstashRedisCache("https://db.example", "tok")),
        sg.LocalCounter(), retry_after_s=60.0, clock=lambda: clock[0],
    )

    assert counter.add("k", 5, 60) == 5
    assert counter.add("k", 5, 60) == 10  # inside the cool-down: no second request
    assert len(httpx_mock.get_requests()) == 1
    assert "counting in this process" in caplog.text
    assert "tok" not in caplog.text  # the token never reaches a log line

    clock[0] = 61.0
    httpx_mock.add_response(url="https://db.example/multi-exec",
                            json=[{"result": 7}, {"result": 1}])
    assert counter.add("k", 5, 60) == 7  # back on the shared counter


def test_a_discarded_transaction_is_an_unavailable_counter(httpx_mock):
    from tools.cache_backend import CacheUnavailable, UpstashRedisCache

    httpx_mock.add_response(url="https://db.example/multi-exec",
                            json={"error": "ERR unsupported command"})
    with pytest.raises(CacheUnavailable):
        UpstashRedisCache("https://db.example", "tok").incrby_with_ttl("k", 1, 60)


# --- the per-client request limit used by the API ---------------------------

def test_each_client_gets_its_own_window(monkeypatch):
    monkeypatch.setenv("CRG_API_RATE_LIMIT", "2")
    monkeypatch.setenv("CRG_API_RATE_WINDOW_S", "60")
    monkeypatch.setattr(sg, "_now", lambda: 1_794_700_010.0)

    assert sg.client_rate_limited("api", "203.0.113.7") is None
    assert sg.client_rate_limited("api", "203.0.113.7") is None
    assert sg.client_rate_limited("api", "203.0.113.7") == 10  # until the window ends
    assert sg.client_rate_limited("api", "198.51.100.9") is None


def test_the_request_limit_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("CRG_API_RATE_LIMIT", "off")
    assert all(sg.client_rate_limited("api", "x") is None for _ in range(100))
