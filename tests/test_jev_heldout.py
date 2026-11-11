"""Tests for evals/run_jev_heldout.py: the one-time held-out triage check sends only
what the shipped path sends, never runs live by accident, and scores the pass
conditions fixed in its docstring."""
from __future__ import annotations

import json

import httpx
import pytest

import evals.run_jev_heldout as ho


def _clean_env(monkeypatch, tmp_path, *, key: bool = True):
    from tests.jev_fakes import FAKE_KEY

    for name in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GEMINI_API_KEY",
                 "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    if key:
        monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_KEY)
    monkeypatch.setattr(ho, "RESULTS_DIR", tmp_path)


def _fake_jev(sent: list[str]):
    """A stand-in TypeSafe that records every question it is asked."""
    from tests.jev_fakes import triage_reply

    def handle(request):
        question = json.loads(request.content)["state"]["question"]
        sent.append(question)
        lowered = question.lower()
        if any(word in lowered for word in ("gdp", "pledge", "insurance", "cost", "economic")):
            reply = triage_reply(topic={"policy_or_economics": 0.95, "climate_science": 0.05},
                                 about=0.03, hazard={"heatwave": 1.0}, other={"policy": 1.0})
        elif any(word in lowered for word in ("hot", "warm", "temperature")):
            reply = triage_reply(topic={"assessed_hazard": 0.97, "climate_science": 0.03},
                                 about=0.96, hazard={"heatwave": 0.94, "wind": 0.06})
        else:
            reply = triage_reply(topic={"climate_science": 0.9, "assessed_hazard": 0.1},
                                 about=0.2, hazard={"wind": 0.5, "heatwave": 0.5})
        return httpx.Response(200, json=reply)

    return handle


def test_labels_follow_the_dev_eval_rules():
    rows = ho.held_out_rows()
    assert len(rows) == 105
    refuse = {r["slice"] for r in rows if r["should_refuse"]}
    assert refuse == {"out_of_scope_hazard", "out_of_corpus"}
    assert not any(r["should_refuse"] for r in rows if r["slice"] == "premise_injection")


def test_without_the_flag_nothing_is_sent(tmp_path, monkeypatch, httpx_mock):
    _clean_env(monkeypatch, tmp_path)

    assert ho.main([]) == 0

    assert httpx_mock.get_requests() == []
    artifact = json.loads(next(tmp_path.glob("jev-triage-test-*.json")).read_text("utf-8"))
    assert artifact["exposure_spent"] is False and "gates" not in artifact
    assert artifact["sha256_ok"] is True
    report = next(tmp_path.glob("jev-triage-test-*.md")).read_text("utf-8")
    assert "\u2014" not in report and "\u2013" not in report  # house style: no em or en dashes


def test_a_live_run_sends_only_what_the_shipped_path_sends(tmp_path, monkeypatch, httpx_mock):
    _clean_env(monkeypatch, tmp_path)
    sent: list[str] = []
    httpx_mock.add_callback(_fake_jev(sent), is_reusable=True)

    assert ho.main(["--spend-exposure", "--max-rpm", "0"]) == 0

    artifact = json.loads(next(tmp_path.glob("jev-triage-test-*.json")).read_text("utf-8"))
    rows = {r["id"]: r for r in artifact["rows"]}
    would_send = {r["question"] for r in rows.values() if r["sent_to_jev"]}
    assert sent and set(sent) <= would_send  # nothing stage 1 or the parser settles leaves
    assert set(sent) == would_send
    assert artifact["exposure_spent"] is True
    assert set(artifact["gates"]) == {"1_no_new_false_refusal", "2_no_wrong_hazard", "3_stability"}
    # three asks per question (the run, options reversed, a second uncached run)
    assert artifact["jev"]["live_calls"] == 3 * len(would_send)
    assert artifact["order_check"]["n"] == len(would_send)
    report = next(tmp_path.glob("jev-triage-test-*.md")).read_text("utf-8")
    assert "Pass conditions (fixed before the run)" in report


def test_a_second_live_run_is_refused_unless_asked_for(tmp_path, monkeypatch, httpx_mock):
    _clean_env(monkeypatch, tmp_path)
    (tmp_path / "jev-triage-test-2026-01-01.json").write_text(
        json.dumps({"exposure_spent": True}), encoding="utf-8")

    assert ho.main(["--spend-exposure", "--max-rpm", "0"]) == 3
    assert httpx_mock.get_requests() == []


def test_the_gates_read_off_the_scores():
    artifact = {
        "scope": {"stage1": {"false_refusals": ["A"]},
                  "shipped": {"false_refusals": ["A", "B"]}},
        "hazard": {"front_door": {"wrong": []}},
        "order_check": {"n": 10, "decision_flips": ["X"]},
        "repeats": {"n": 10, "unstable": ["X", "Y"]},
    }
    gates = ho.gates(artifact)
    assert gates["1_no_new_false_refusal"] == {"met": False, "new_false_refusals": ["B"]}
    assert gates["2_no_wrong_hazard"]["met"] is True
    assert gates["3_stability"]["met"] is False  # 8 of 10 kept across runs
    assert gates["3_stability"]["kept_with_options_reversed"] == pytest.approx(0.9)


def test_a_run_leaves_no_environment_behind(tmp_path, monkeypatch, httpx_mock):
    import os

    import evals.run_jev_eval as dev

    for name in dev._SCOPED_ENV:
        monkeypatch.delenv(name, raising=False)
    _clean_env(monkeypatch, tmp_path, key=False)

    ho.main([])

    assert not [name for name in dev._SCOPED_ENV if name in os.environ]
