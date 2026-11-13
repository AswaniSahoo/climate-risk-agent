"""Jev wired into the product: scope guard, free-text front door, graph and API.

The properties that matter more than any single answer:
1. CRG_JEV off (the default) makes zero Jev calls, whatever else is configured;
2. stage 1's lexical verdict is final, with Jev on or off;
3. Jev failing is exactly today's behaviour, plus a reason in telemetry;
4. Jev's output only ever adds a refusal, a hazard reading or an annotation;
   it never removes a refusal, a citation or the citation validation.

Every request is answered by pytest-httpx, dispatched by host, so the forecast
and Jev mocks cannot answer each other's requests.
"""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

import agent.graph as graph_mod
import agent.nl as nl_mod
import api.app as api_app
import rag.scope as scope_mod
from agent.contracts import Citation, Hazard, RiskLevel, RiskReport
from agent.graph import run_agent
from agent.nl import parse_query, run_agent_nl, triage_query
from rag.answer import CitedAnswer, answer_with_guard
from rag.chunk import Chunk
from rag.scope import scope_verdict
from tests.jev_fakes import FAKE_KEY, enable_jev, relation_reply, triage_reply

SILENT_OFF_TOPIC = "What percentage of GDP will the world lose by 2100?"
# No hazard word in it, for the parser or for stage 1: "temperatures" can be hot or
# cold, so only a reader such as Jev can say this one is about heat.
SILENT_HEAT = "How high will temperatures climb in Delhi next week?"

REFUSE_POLICY = triage_reply(
    topic={"policy_or_economics": 0.97, "climate_science": 0.03}, about=0.02,
    hazard={"heatwave": 0.5, "wind": 0.5}, other={"policy": 0.95, "unrelated": 0.05},
)
HINT_HEAT = triage_reply(
    topic={"assessed_hazard": 0.96, "climate_science": 0.04}, about=0.97,
    hazard={"heatwave": 0.93, "wind": 0.07},
)
SCIENCE_DEFER = triage_reply(
    topic={"climate_science": 0.95, "assessed_hazard": 0.05}, about=0.1,
    hazard={"heatwave": 0.4, "extreme_precip": 0.3, "wind": 0.3},
)


@pytest.fixture(autouse=True)
def _clean_scope_cache():
    scope_mod.clear_stage2_cache()
    yield
    scope_mod.clear_stage2_cache()


def _jev_requests(httpx_mock) -> list[httpx.Request]:
    return [r for r in httpx_mock.get_requests() if r.url.host == "api.typesafe.ai"]


# --- 1. off means off ------------------------------------------------------------

def test_flag_off_makes_zero_jev_calls_even_with_a_key_set(httpx_mock, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_KEY)  # a key alone must not turn anything on

    for question in (SILENT_OFF_TOPIC, SILENT_HEAT, "Will temperature extremes rise in Delhi?"):
        parse_query(question)
        scope_verdict(question)
        triage_query(question)

    assert httpx_mock.get_requests() == []
    assert parse_query(SILENT_HEAT).triage is None  # the parse is exactly what it was


# --- 2. stage 1 stays first and final ------------------------------------------------

def test_a_lexical_refusal_is_final_and_jev_is_never_asked(httpx_mock, monkeypatch):
    enable_jev(monkeypatch)

    decision = scope_verdict("Has South Asia seen a significant increase in drought?")

    assert decision.out_of_scope == "drought" and decision.stage == "lexical"
    assert httpx_mock.get_requests() == []


def test_a_lexically_supported_question_never_reaches_jev_in_the_guard(httpx_mock, monkeypatch):
    enable_jev(monkeypatch)

    decision = scope_verdict("Will heavy precipitation intensify in South Asia?")

    assert decision.out_of_scope is None and decision.stage == "lexical"
    assert httpx_mock.get_requests() == []


# --- the silent bucket ---------------------------------------------------------------

def test_jev_refuses_a_confident_off_topic_question_in_the_silent_bucket(httpx_mock, monkeypatch):
    enable_jev(monkeypatch)
    httpx_mock.add_response(json=REFUSE_POLICY)

    decision = scope_verdict(SILENT_OFF_TOPIC)

    assert decision.stage == "jev"
    assert decision.out_of_scope == "climate policy and economics"
    assert decision.jev is not None and decision.jev.decision == "refuse"


def test_a_jev_refusal_on_the_rag_path_skips_gemini(httpx_mock, monkeypatch):
    enable_jev(monkeypatch)
    httpx_mock.add_response(json=REFUSE_POLICY)

    def no_gemini(*_a, **_kw):
        raise AssertionError("Gemini must not be called for a refused question")

    monkeypatch.setattr("rag.answer.generate_json", no_gemini)
    chunk = Chunk(chunk_id="IPCC_AR6_WGI_SPM.pdf#p9#0", source="IPCC_AR6_WGI_SPM.pdf",
                  page=9, text="some excerpt")

    answer = answer_with_guard(SILENT_OFF_TOPIC, [chunk])

    assert answer.abstain and "climate policy and economics" in answer.abstain_reason


def test_jev_deferring_falls_through_to_stage_two_as_configured(httpx_mock, monkeypatch):
    enable_jev(monkeypatch)
    httpx_mock.add_response(json=SCIENCE_DEFER)

    decision = scope_verdict("What is the best estimate of equilibrium climate sensitivity?")

    assert decision.out_of_scope is None and decision.hazard_hint is None
    assert decision.stage == "lexical"  # stage 2 off: no verdict, exactly as with Jev off
    assert decision.jev is not None and decision.jev.decision == "defer"


# --- 3. failure is today's behaviour -------------------------------------------------

def test_jev_down_is_todays_behaviour_and_says_why(httpx_mock, monkeypatch):
    enable_jev(monkeypatch)
    monkeypatch.setenv("CRG_JEV_TRIAGE_BUDGET_S", "0.05")  # keep the test fast
    httpx_mock.add_response(status_code=529, is_reusable=True)

    parsed = parse_query(SILENT_HEAT)

    assert parsed.hazard is None and parsed.out_of_scope is None  # as with Jev off
    assert parsed.triage is not None
    assert parsed.triage.jev_status.startswith("unavailable")
    from obs import telemetry
    failures = [e for e in telemetry.snapshot() if e["op"] == "jev:triage" and not e["ok"]]
    assert failures and "529" in failures[0]["detail"]


def test_a_bad_key_costs_one_round_trip_then_fails_fast(httpx_mock, monkeypatch):
    enable_jev(monkeypatch)
    httpx_mock.add_response(status_code=401, json={"detail": {"error_type": "authentication_error"}})

    first = parse_query(SILENT_HEAT)
    second = parse_query("Will it pour in Mumbai this week?")

    assert first.hazard is None and second.hazard is None
    assert len(_jev_requests(httpx_mock)) == 1
    assert second.triage is not None and "circuit open" in second.triage.jev_status


# --- the free-text front door --------------------------------------------------------

def test_a_paraphrase_the_parser_misses_becomes_a_hazard_with_its_probability(
    httpx_mock, monkeypatch
):
    enable_jev(monkeypatch)
    httpx_mock.add_response(json=HINT_HEAT)

    parsed = parse_query(SILENT_HEAT)

    assert parsed.hazard is Hazard.HEATWAVE and parsed.place == "Delhi"
    assert parsed.horizon_days == 7  # the horizon stays lexical
    assert parsed.triage.hazard_source == "jev"
    assert parsed.triage.hazard_probability == pytest.approx(0.93)
    assert parsed.triage.scope_stage == "jev" and parsed.triage.jev_model == "jev-1.14.0"


def test_a_lexical_hazard_needs_no_jev_call_but_says_where_it_came_from(httpx_mock, monkeypatch):
    enable_jev(monkeypatch)

    parsed = parse_query("How risky are heatwaves in Rourkela over the next 10 days?")

    assert parsed.hazard is Hazard.HEATWAVE
    assert parsed.triage.hazard_source == "lexical" and parsed.triage.jev_status == "not called"
    assert httpx_mock.get_requests() == []


def test_in_scope_by_vocabulary_but_unnamed_hazard_asks_jev_which_one(httpx_mock, monkeypatch):
    """Stage 1 matches "temperature extremes"; the front door's patterns do not."""
    enable_jev(monkeypatch)
    httpx_mock.add_response(json=triage_reply(
        topic={"climate_science": 0.6, "assessed_hazard": 0.4}, about=0.91,
        hazard={"heatwave": 0.95, "wind": 0.05},
    ))

    parsed = parse_query("Will temperature extremes rise in Delhi next week?")

    assert parsed.hazard is Hazard.HEATWAVE and parsed.triage.hazard_source == "jev"


def test_in_scope_by_vocabulary_jev_cannot_refuse(httpx_mock, monkeypatch):
    enable_jev(monkeypatch)
    httpx_mock.add_response(json=REFUSE_POLICY)

    parsed = parse_query("Will temperature extremes rise in Delhi next week?")

    assert parsed.out_of_scope is None  # stage 1 let it in; Jev may only name a hazard
    assert parsed.hazard is None  # and here it named none, so: today's refusal


def test_run_agent_nl_carries_the_triage_onto_the_report(httpx_mock, monkeypatch):
    enable_jev(monkeypatch)
    httpx_mock.add_response(json=HINT_HEAT)
    captured = {}

    def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return RiskReport(location="Delhi", hazard=kwargs["hazard"], horizon_days=7,
                          confidence=0.5, risk_level=RiskLevel.LOW, summary="stub")

    from tools.ar6_regions import AR6Region
    from tools.geocode import GeoLocation

    monkeypatch.setattr(nl_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(nl_mod, "geocode", lambda place: GeoLocation(
        name=place, latitude=28.6, longitude=77.2, country="India", admin1="Delhi"))
    monkeypatch.setattr(nl_mod, "region_for", lambda lat, lon: AR6Region(acronym="SAS",
                                                                         name="South Asia"))

    report = run_agent_nl(SILENT_HEAT, use_climatology=False)

    assert captured["hazard"] is Hazard.HEATWAVE
    assert report.triage is not None and report.triage.hazard_source == "jev"


def test_an_instant_refusal_carries_the_triage_too(httpx_mock, monkeypatch):
    enable_jev(monkeypatch)
    httpx_mock.add_response(json=REFUSE_POLICY)

    report = run_agent_nl(SILENT_OFF_TOPIC)

    assert report.refusal.startswith("climate policy and economics is not a supported hazard")
    assert report.triage.scope_stage == "jev"


def test_a_marine_heatwave_is_refused_by_the_front_door_with_jev_off():
    """The heat pattern matches the "heatwave" inside it; stage 1's rule must win."""
    parsed = parse_query("What is the marine heatwave risk near Perth next week?")

    assert parsed.hazard is None and parsed.out_of_scope == "marine heatwave"


# --- triage outcome and the API -------------------------------------------------------

def test_triage_query_states_the_refusal_query_would_give(httpx_mock, monkeypatch):
    enable_jev(monkeypatch)
    httpx_mock.add_response(json=REFUSE_POLICY)

    outcome = triage_query(SILENT_OFF_TOPIC)
    report = run_agent_nl(SILENT_OFF_TOPIC)  # same question: served from the client cache

    assert outcome.refusal == report.refusal
    assert len(_jev_requests(httpx_mock)) == 1


def test_the_triage_endpoint_answers_without_running_the_pipeline(httpx_mock, monkeypatch):
    enable_jev(monkeypatch)
    httpx_mock.add_response(json=HINT_HEAT)
    monkeypatch.setattr(api_app, "run_agent_nl", lambda *a, **kw: pytest.fail("no pipeline"))

    response = TestClient(api_app.app).post("/triage", json={"query": SILENT_HEAT})

    assert response.status_code == 200
    body = response.json()
    assert body["triage"]["hazard"] == "heatwave" and body["triage"]["refusal"] is None
    assert body["triage"]["triage"]["hazard_probability"] == pytest.approx(0.93)
    assert body["telemetry"]["calls"] == 1 and body["telemetry"]["tokens_in"] == 420


def test_the_triage_endpoint_is_key_gated_like_query(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    client = TestClient(api_app.app)

    assert client.post("/triage", json={"query": "x"}).status_code == 401
    ok = client.post("/triage", json={"query": "heat in Delhi"}, headers={"x-api-key": "secret"})
    assert ok.status_code == 200 and ok.json()["triage"]["hazard"] == "heatwave"


# --- the graph: citation support, annotation only --------------------------------------

_FORECAST = {
    "latitude": 22.26, "longitude": 84.85, "timezone": "Asia/Kolkata",
    "daily_units": {"time": "iso8601", "precipitation_sum": "mm", "temperature_2m_max": "°C",
                    "wind_speed_10m_max": "km/h", "wind_gusts_10m_max": "km/h"},
    "daily": {"time": ["2026-07-02", "2026-07-03", "2026-07-04"],
              "precipitation_sum": [14.2, 0.1, 80.0], "temperature_2m_max": [46.0, 44.0, 40.0],
              "wind_speed_10m_max": [70.0, 45.0, 30.0], "wind_gusts_10m_max": [95.0, 60.0, 42.0]},
}
_CHUNKS = (
    Chunk(chunk_id="IPCC_AR6_WGI_Chapter11.pdf#p124#2", source="IPCC_AR6_WGI_Chapter11.pdf",
          page=124, text="East Central Asia (ECA): median increase of more than 3.5°C."),
    Chunk(chunk_id="IPCC_AR6_WGI_SPM.pdf#p16#0", source="IPCC_AR6_WGI_SPM.pdf",
          page=16, text="Glaciers are projected to continue to lose mass."),
)
_ANSWER = CitedAnswer(
    answer="Hot extremes in East Central Asia are projected to intensify by more than 3.5°C.",
    citations=[c.chunk_id for c in _CHUNKS], abstain=False,
    allowed_ids=[c.chunk_id for c in _CHUNKS],
)


class _FakeRetriever:
    def retrieve(self, question, top_k):
        return list(_CHUNKS)[:top_k]


@pytest.fixture
def grounded_graph(monkeypatch, tmp_path):
    from tools.cache_backend import DiskCache, JsonCache

    monkeypatch.setattr("tools.forecast._forecast_cache",
                        lambda: JsonCache("forecast", backend=DiskCache(tmp_path / "fc")))
    monkeypatch.setattr(graph_mod, "_ipcc_retriever", lambda: _FakeRetriever())
    monkeypatch.setattr(graph_mod, "answer_with_guard", lambda q, chunks, **_: _ANSWER)


def _dispatch(jev_answer):
    """Open-Meteo gets the canned forecast; TypeSafe gets `jev_answer(request)`."""
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.typesafe.ai":
            return jev_answer(request)
        return httpx.Response(200, json=_FORECAST)
    return handle


def _supports_only_the_ch11_page(request: httpx.Request) -> httpx.Response:
    passage = json.loads(request.content)["state"]["passage"]
    supports = 0.96 if "3.5°C" in passage else 0.03
    return httpx.Response(200, json=relation_reply(supports))


def _run(**kw) -> RiskReport:
    return run_agent(location="Rourkela", latitude=22.26, longitude=84.85,
                     hazard=Hazard.HEATWAVE, horizon_days=3, **kw)


def test_verify_annotates_each_cited_page_and_changes_nothing_else(
    grounded_graph, httpx_mock, monkeypatch
):
    httpx_mock.add_callback(_dispatch(lambda r: httpx.Response(500)), is_reusable=True)
    baseline = _run()  # flag off: no Jev at all
    assert baseline.evidence is None
    assert _jev_requests(httpx_mock) == []

    enable_jev(monkeypatch, "verify")
    httpx_mock.add_callback(_dispatch(_supports_only_the_ch11_page), is_reusable=True)
    report = _run()

    by_page = {e.locator: e for e in report.evidence.citations}
    assert by_page["p124"].strength == "strong" and by_page["p16"].strength == "weak"
    assert report.evidence.status == "ok" and report.evidence.unsupported_statements == []
    # annotation only: the citations, the band and the confidence are untouched
    assert report.citations == baseline.citations
    assert Citation(source="IPCC_AR6_WGI_SPM.pdf", locator="p16") in report.citations
    assert report.risk_level is baseline.risk_level
    assert report.confidence == baseline.confidence


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_verify_failing_ships_the_same_report_marked_unavailable(
    grounded_graph, httpx_mock, monkeypatch
):
    enable_jev(monkeypatch, "verify")
    monkeypatch.setenv("CRG_JEV_VERIFY_BUDGET_S", "0.05")
    httpx_mock.add_callback(_dispatch(lambda r: httpx.Response(529)), is_reusable=True)

    report = _run()

    assert report.evidence.status == "unavailable"
    assert {e.strength for e in report.evidence.citations} == {"unchecked"}
    assert len(report.citations) == 2 and report.risk_level is RiskLevel.SEVERE


def test_triage_mode_does_not_verify(grounded_graph, httpx_mock, monkeypatch):
    enable_jev(monkeypatch, "triage")
    httpx_mock.add_callback(_dispatch(lambda r: httpx.Response(500)), is_reusable=True)

    report = _run()

    assert report.evidence is None and _jev_requests(httpx_mock) == []
