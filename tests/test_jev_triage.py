"""Tests for rag/jev_triage.py: Jev's reading of a question, turned into a decision by code.

The decision rule is pure, so most of these build a reply and check what the
thresholds make of it. The few that send a request answer it with pytest-httpx.
"""
from __future__ import annotations

import json

import pytest

import rag.jev_client as jc
import rag.jev_triage as tri
from tests.jev_fakes import FAKE_KEY, FakeClock, enable_jev, triage_reply


def _result(**kw) -> jc.JevResult:
    """Parse a fake reply exactly the way the client does."""
    payload = triage_reply(**kw)
    model, answers, usage = jc.parse_answers(payload, tri.build_questions())
    return jc.JevResult(model=model, answers=answers, usage=usage, latency_ms=120.0)


# --- the question set ------------------------------------------------------------

def test_one_request_asks_four_typed_questions():
    questions = tri.build_questions()
    assert {name: q["type"] for name, q in questions.items()} == {
        "topic": "choice", "about_assessed_hazard": "noul",
        "hazard": "choice", "other_topic": "choice",
    }
    # in-scope topics first: Jev's documented lean to the first option then
    # favours deferring over refusing
    assert list(questions["topic"]["criteria"])[:2] == ["assessed_hazard", "climate_science"]
    assert list(questions["hazard"]["criteria"]) == ["heatwave", "extreme_precip", "wind"]


def test_the_topic_order_can_be_reversed_for_the_order_check_but_not_corrupted():
    reversed_order = tuple(reversed(tri.TOPIC_OPTIONS))
    questions = tri.build_questions(topic_order=reversed_order)
    assert tuple(questions["topic"]["criteria"]) == reversed_order
    with pytest.raises(ValueError):
        tri.build_questions(topic_order=("assessed_hazard",))


def test_every_other_topic_names_a_refusal():
    assert all(name for _, name in tri.OTHER_TOPIC_OPTIONS.values())
    assert set(tri.HAZARD_HINT_NAME) == set(tri.HAZARD_OPTIONS)


def test_the_question_text_is_whitespace_collapsed_and_capped():
    assert tri.clean_question("  how   hot\n\nwill it get?  ") == "how hot will it get?"
    assert len(tri.clean_question("x" * 5000)) == tri.QUESTION_CHARS


# --- the decision rule -----------------------------------------------------------

def test_a_confident_other_hazard_refuses_and_names_it():
    reading = tri.decide(_result(
        topic={"other_hazard": 0.95, "climate_science": 0.05}, about=0.04,
        hazard={"heatwave": 0.6, "wind": 0.4}, other={"wildfire": 0.9, "drought": 0.1},
    ))
    assert reading.decision == "refuse"
    assert reading.out_of_scope == "wildfire / fire weather"
    assert reading.hazard_hint is None


def test_refusal_mass_is_summed_across_the_three_refusal_topics():
    """0.45 policy + 0.40 unrelated is a clear refusal even though no single topic is."""
    reading = tri.decide(_result(
        topic={"policy_or_economics": 0.45, "unrelated": 0.40, "climate_science": 0.15},
        about=0.05, hazard={"heatwave": 1.0}, other={"policy": 0.8, "unrelated": 0.2},
    ))
    assert reading.decision == "refuse"
    assert reading.refuse_probability == pytest.approx(0.85)
    assert reading.out_of_scope == "climate policy and economics"


def test_an_unsure_name_falls_back_to_a_generic_one():
    reading = tri.decide(_result(
        topic={"other_hazard": 0.9, "assessed_hazard": 0.1}, about=0.1,
        hazard={"wind": 1.0}, other={"drought": 0.4, "landslide": 0.35, "earthquake": 0.25},
    ))
    assert reading.decision == "refuse" and reading.out_of_scope == "an unsupported hazard"


def test_no_refusal_when_the_companion_noul_disagrees():
    reading = tri.decide(_result(
        topic={"other_hazard": 0.9, "assessed_hazard": 0.1}, about=0.55,
        hazard={"wind": 1.0}, other={"tropical_cyclone": 1.0},
    ))
    assert reading.decision == "defer"


def test_a_confident_paraphrase_of_our_hazard_is_a_hint():
    reading = tri.decide(_result(
        topic={"assessed_hazard": 0.96, "climate_science": 0.04}, about=0.97,
        hazard={"heatwave": 0.93, "wind": 0.07},
    ))
    assert reading.decision == "hint"
    assert reading.hazard_hint == "heatwave" and reading.hazard_probability == pytest.approx(0.93)


def test_a_hint_needs_all_three_signals():
    weak_hazard = tri.decide(_result(
        topic={"assessed_hazard": 0.96, "climate_science": 0.04}, about=0.97,
        hazard={"heatwave": 0.55, "wind": 0.45},
    ))
    weak_topic = tri.decide(_result(
        topic={"assessed_hazard": 0.6, "climate_science": 0.4}, about=0.97,
        hazard={"heatwave": 0.95, "wind": 0.05},
    ))
    weak_noul = tri.decide(_result(
        topic={"assessed_hazard": 0.96, "climate_science": 0.04}, about=0.5,
        hazard={"heatwave": 0.95, "wind": 0.05},
    ))
    assert {weak_hazard.decision, weak_topic.decision, weak_noul.decision} == {"defer"}


def test_climate_science_is_neither_refused_nor_routed():
    reading = tri.decide(_result(
        topic={"climate_science": 0.97, "assessed_hazard": 0.03}, about=0.08,
        hazard={"heatwave": 0.4, "extreme_precip": 0.3, "wind": 0.3},
    ))
    assert reading.decision == "defer"


def test_when_stage_one_already_let_it_in_jev_may_name_a_hazard_but_never_refuse():
    refusal_shaped = _result(
        topic={"other_hazard": 0.99, "assessed_hazard": 0.01}, about=0.01,
        hazard={"wind": 1.0}, other={"drought": 1.0},
    )
    science_shaped = _result(
        topic={"climate_science": 0.9, "assessed_hazard": 0.1}, about=0.92,
        hazard={"heatwave": 0.94, "wind": 0.06},
    )
    assert tri.decide(refusal_shaped, may_refuse=False).decision == "defer"
    hinted = tri.decide(science_shaped, may_refuse=False)
    assert hinted.decision == "hint" and hinted.hazard == "heatwave"


def test_the_reading_carries_its_numbers_and_cost():
    reading = tri.decide(_result(
        topic={"assessed_hazard": 0.96, "climate_science": 0.04}, about=0.97,
        hazard={"wind": 0.91, "heatwave": 0.09}, input_tokens=1000,
    ))
    assert reading.model == "jev-1.14.0" and reading.input_tokens == 1000
    assert reading.cost_usd == pytest.approx(1000 * 0.042 / 1e6)
    assert "jev hint" in reading.detail and "wind:0.91" in reading.detail


# --- one request, never an exception ----------------------------------------------

def test_triage_sends_only_the_question_as_state(httpx_mock):
    httpx_mock.add_response(json=triage_reply(
        topic={"assessed_hazard": 0.96, "climate_science": 0.04}, about=0.97,
        hazard={"heatwave": 0.93, "wind": 0.07},
    ))
    client = jc.JevClient(FAKE_KEY)
    try:
        reading = tri.triage("  How hot will it get   in Delhi next week? ", client=client)
    finally:
        client.close()

    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body["state"] == {"question": "How hot will it get in Delhi next week?"}
    assert reading.decision == "hint" and reading.status == "ok"


def test_triage_never_raises_when_jev_is_down(httpx_mock, monkeypatch):
    clock = FakeClock()
    for _ in range(3):
        httpx_mock.add_response(status_code=529)
    client = jc.JevClient(FAKE_KEY, sleep=clock.sleep, clock=clock, rng=lambda: 0.0)
    try:
        reading = tri.triage("Is the smoke season getting longer?", client=client)
    finally:
        client.close()

    assert reading.decision == "unavailable"
    assert reading.status.startswith("unavailable:") and "529" in reading.status


def test_triage_without_a_key_is_unavailable_not_an_error(monkeypatch):
    monkeypatch.setenv("CRG_JEV", "triage")  # flag on, key missing: a deploy mistake
    reading = tri.triage("Is the smoke season getting longer?")
    assert reading.decision == "unavailable" and "TYPESAFE_API_KEY" in reading.status


def test_an_empty_question_is_not_sent(httpx_mock, monkeypatch):
    enable_jev(monkeypatch)
    assert tri.triage("   ").decision == "defer"
    assert httpx_mock.get_requests() == []


@pytest.mark.parametrize("raw", ["0,8", "high", "1.5", "-0.2", "nan"])
def test_a_malformed_threshold_falls_back_instead_of_raising(monkeypatch, caplog, raw):
    monkeypatch.setenv("CRG_JEV_REFUSE_P", raw)
    assert tri.env_probability("CRG_JEV_REFUSE_P", 0.8) == 0.8
    assert "CRG_JEV_REFUSE_P" in caplog.text


def test_a_valid_threshold_override_is_used(monkeypatch):
    monkeypatch.setenv("CRG_JEV_REFUSE_P", "0.9")
    assert tri.env_probability("CRG_JEV_REFUSE_P", 0.8) == 0.9
