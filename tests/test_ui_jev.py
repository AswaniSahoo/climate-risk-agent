"""The UI with Jev (adr/0002, docs/jev-ui-integration.md), offline.

With CRG_JEV off the page is the page it always was. With triage on, Ask reads
the question first: a refusal is shown at once with no run behind it, a hazard
Jev named is shown as Jev's reading with any other hazard one click away, and
the run answers exactly that reading. With verify on, each citation carries
its evidence label and nothing else about the report moves.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

import agent.graph as graph_mod
import agent.location as location_mod
import agent.nl as nl_mod
import tools.climatology as climatology_mod
from agent.contracts import (
    Citation,
    CitationEvidence,
    EvidenceCheck,
    Hazard,
    RiskLevel,
    RiskReport,
)
from agent.location import ResolvedPlace
from tests.jev_fakes import enable_jev, triage_reply

# Absolute: since Streamlit 1.66 a relative path is "resolved against the Python
# file that calls AppTest.from_file()" (tests/), not the working directory.
_APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")
STATEMENT = "Hot extremes have become more frequent. Heatwaves will stop by 2030."
# A heat question no hazard word names, so with triage on only Jev can read it.
SILENT_HEAT = "How high will temperatures climb in Delhi next week?"


def _report(**overrides) -> RiskReport:
    base = dict(
        location="Delhi, India", hazard=Hazard.HEATWAVE, horizon_days=7, confidence=0.6,
        risk_level=RiskLevel.HIGH,
        summary=f"Peak daily max temperature of 44.0 °C over 7 days. IPCC AR6: {STATEMENT}",
        citations=[Citation(source="IPCC_AR6_WGI_Chapter11.pdf", locator="p124"),
                   Citation(source="IPCC_AR6_WGI_SPM.pdf", locator="p9")],
    )
    base.update(overrides)
    return RiskReport(**base)


_EVIDENCE = EvidenceCheck(
    status="ok", model="jev-latest",
    citations=[
        CitationEvidence(source="IPCC_AR6_WGI_Chapter11.pdf", locator="p124", chunk_id="c#p124#0",
                         supports=0.97, contradicts=0.01, strength="strong"),
        CitationEvidence(source="IPCC_AR6_WGI_SPM.pdf", locator="p9", chunk_id="s#p9#0",
                         supports=0.02, contradicts=0.91, strength="contradicted"),
    ],
    unsupported_statements=["Heatwaves will stop by 2030."],
)


@pytest.fixture
def calls(monkeypatch):
    """Stubs the agent, the climatology and the geocoder; records what was run."""
    import streamlit as st

    seen: dict[str, list] = {"nl": [], "graph": []}

    def fake_nl(query, **kwargs):
        seen["nl"].append({"query": query, **kwargs})
        return _report(evidence=_EVIDENCE)

    def fake_graph(**kwargs):
        seen["graph"].append(kwargs)
        return _report(hazard=kwargs["hazard"], location=kwargs["location"], evidence=_EVIDENCE)

    monkeypatch.setattr(nl_mod, "run_agent_nl", fake_nl)
    monkeypatch.setattr(graph_mod, "run_agent", fake_graph)
    monkeypatch.setattr(
        climatology_mod, "climatology_hazard_stat",
        lambda *a, **k: (_ for _ in ()).throw(climatology_mod.ClimatologyError("offline test")),
    )
    monkeypatch.setattr(
        location_mod, "resolve_place",
        lambda query: ResolvedPlace(name="Delhi", country="India", latitude=28.61, longitude=77.21),
    )
    monkeypatch.setattr(st, "map", lambda *a, **k: None)
    return seen


def _fake_jev(request):
    """A stand-in Jev: heat for a weather question, policy for a money question."""
    question = json.loads(request.content)["state"]["question"].lower()
    if "gdp" in question:
        reply = triage_reply(topic={"policy_or_economics": 0.96, "climate_science": 0.04},
                             about=0.02, hazard={"heatwave": 1.0}, other={"policy": 1.0})
    else:
        reply = triage_reply(topic={"assessed_hazard": 0.98, "climate_science": 0.02},
                             about=0.97, hazard={"heatwave": 0.93, "wind": 0.07})
    return httpx.Response(200, json=reply)


def _ask(question: str) -> AppTest:
    """Type the question and press Ask, in one rerun, as a person does."""
    at = AppTest.from_file(_APP, default_timeout=30)
    at.run()
    at.text_input(key="nl_query").input(question)
    at.button(key="ask_btn").click().run()
    return at


def _panels(at: AppTest) -> list:
    """The run's progress panels. AppTest also lists every icon'd expander as a Status,
    so a panel is the block wearing an icon only st.status sets (as in test_ui.py)."""
    return [s for s in at.status if s.icon in {"spinner", ":material/check:", ":material/error:"}]


def _text(at: AppTest) -> str:
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption]
                    + [e.value for e in at.error])


def test_with_the_flag_off_nothing_of_jev_shows_or_runs(calls, httpx_mock):
    at = _ask(SILENT_HEAT)

    assert not at.exception
    assert httpx_mock.get_requests() == []
    assert "How your question was read" not in _text(at)
    assert "Supports the statement" not in _text(at)  # the report carries evidence; unshown
    assert calls["nl"] and "parsed" not in calls["nl"][0]  # today's call, unchanged


def test_a_question_jev_reads_shows_the_reading_and_runs_it(calls, monkeypatch, httpx_mock):
    enable_jev(monkeypatch, "triage")
    httpx_mock.add_callback(_fake_jev, is_reusable=True)

    at = _ask(SILENT_HEAT)

    assert not at.exception
    assert len(httpx_mock.get_requests()) == 1  # one reading, reused by the run
    text = _text(at)
    assert "How your question was read" in text
    assert "Heatwave" in text and "next 7 days" in text and "Delhi" in text
    assert "93% probability that it is about heat" in text
    assert at.sidebar.selectbox(key="hazard_input").value == Hazard.HEATWAVE
    assert any("Jev read it as heat" in c.value for c in at.sidebar.caption)
    [run] = calls["nl"]
    assert run["parsed"].hazard is Hazard.HEATWAVE  # the run answers the reading shown
    assert {b.label for b in at.button if b.key and b.key.startswith("reassess_")} == {
        "Assess as extreme rain instead", "Assess as wind instead"}


def test_one_click_assesses_the_same_point_as_another_hazard(calls, monkeypatch, httpx_mock):
    enable_jev(monkeypatch, "triage")
    httpx_mock.add_callback(_fake_jev, is_reusable=True)
    at = _ask(SILENT_HEAT)

    at.button(key="reassess_wind").click().run()

    assert not at.exception
    [manual] = calls["graph"]
    assert manual["hazard"] is Hazard.WIND
    assert manual["latitude"] == pytest.approx(28.61) and manual["horizon_days"] == 7
    assert at.sidebar.selectbox(key="hazard_input").value == Hazard.WIND


def test_an_off_topic_question_is_refused_at_once_with_no_run(calls, monkeypatch, httpx_mock):
    enable_jev(monkeypatch, "triage")
    httpx_mock.add_callback(_fake_jev, is_reusable=True)

    at = _ask("How much GDP will Mumbai lose by 2050?")

    assert not at.exception
    assert calls["nl"] == [] and calls["graph"] == []  # no pipeline behind the refusal
    assert not _panels(at)  # and no progress panel
    assert any("climate policy and economics is not a supported hazard" in e.value
               for e in at.error)
    assert "before any forecast" in _text(at) and "Jev read the question" in _text(at)


def test_a_lexical_refusal_is_instant_too_and_asks_jev_nothing(calls, monkeypatch, httpx_mock):
    enable_jev(monkeypatch, "triage")

    at = _ask("What is the wildfire risk in Sydney next week?")

    assert not at.exception
    assert httpx_mock.get_requests() == []
    assert calls["nl"] == [] and not _panels(at)
    assert any("wildfire" in e.value for e in at.error)
    assert "Jev read the question" not in _text(at)


def test_a_question_with_no_place_runs_on_the_sidebar_point(calls, monkeypatch, httpx_mock):
    enable_jev(monkeypatch, "triage")

    at = _ask("How risky are heatwaves over the next 10 days?")

    assert not at.exception
    assert calls["nl"] == []
    [run] = calls["graph"]
    assert (run["hazard"], run["horizon_days"]) == (Hazard.HEATWAVE, 10)
    assert run["latitude"] == pytest.approx(22.26)  # the seeded sidebar point, Rourkela
    assert "no place in the question" in _text(at)


def test_with_verify_on_each_citation_carries_its_label_and_nothing_else_moves(
    calls, monkeypatch
):
    monkeypatch.setenv("CRG_JEV", "verify")

    at = _ask("How risky are heatwaves in Delhi over the next 7 days?")

    assert not at.exception
    text = _text(at)
    assert "Supports the statement" in text and "May contradict the statement" in text
    assert text.count("cra-cite-src") == 2  # both citations, none dropped
    assert "cra-unsupported" in text and "No cited page clearly supports this sentence" in text
    assert "Evidence labels" in text
    assert at.metric[0].value == "60%"  # the confidence is the report's own


def test_no_button_carries_an_icon(calls, monkeypatch, httpx_mock):
    """Streamlit writes a button's Material icon as text with no aria-hidden, so a
    screen reader would read "thermostat" as part of the button's name."""
    enable_jev(monkeypatch, "all")
    httpx_mock.add_callback(_fake_jev, is_reusable=True)

    at = _ask(SILENT_HEAT)

    assert not at.exception
    assert at.button and all(not b.proto.icon for b in at.button)


def test_a_reading_is_never_shown_as_certain(calls, monkeypatch, httpx_mock):
    """Jev's probability is a reading's, not a verdict's: 0.999 shows as "over 99%"."""
    enable_jev(monkeypatch, "triage")
    httpx_mock.add_callback(
        lambda request: httpx.Response(200, json=triage_reply(
            topic={"assessed_hazard": 1.0}, about=0.99, hazard={"heatwave": 0.999})),
        is_reusable=True,
    )

    at = _ask(SILENT_HEAT)

    assert not at.exception
    assert "over 99% probability" in _text(at)
    assert "100%" not in _text(at)
