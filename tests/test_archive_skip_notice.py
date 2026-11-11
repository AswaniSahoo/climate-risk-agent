"""When the archive budget or its breaker says no, the report still runs and says why.

The refusal is an ArchiveBudgetExceeded carrying one sentence for the reader
(tools/archive_budget.py). Both front doors pass it to the agent, the agent
puts it in the report's numbers line, and the UI shows that line beside the
band, so a reader on either path learns that ERA5 was skipped and when it
comes back, instead of a generic failure.
"""
import pytest
from streamlit.testing.v1 import AppTest

import agent.graph as graph_mod
import agent.nl as nl_mod
import tools.climatology as climatology_mod
from agent.contracts import Hazard, RiskLevel
from tools.geocode import GeoLocation

SENTENCE = ("ERA5 history skipped for this place: today's free data allowance for new places "
            "is used up. It resets at 00:00 UTC.")

CANNED = {
    "latitude": 52.52, "longitude": 13.40, "timezone": "Europe/Berlin", "daily_units": {},
    "daily": {"time": ["2026-07-02", "2026-07-03"], "precipitation_sum": [1.0, 0.0],
              "temperature_2m_max": [30.0, 31.0], "wind_speed_10m_max": [10.0, 12.0],
              "wind_gusts_10m_max": [20.0, 25.0]},
}


@pytest.fixture(autouse=True)
def _offline_ipcc(monkeypatch):
    from rag.corpus import CorpusError

    def no_corpus():
        raise CorpusError("offline test: no corpus")

    monkeypatch.setattr(graph_mod, "_ipcc_retriever", no_corpus)


def test_the_agent_puts_the_sentence_in_the_numbers_line(httpx_mock):
    httpx_mock.add_response(json=CANNED)

    report = graph_mod.run_agent("Berlin, Western and Central Europe (WCE)", 52.52, 13.40,
                                 Hazard.HEATWAVE, horizon_days=2, climatology_note=SENTENCE)

    numbers = report.summary.partition(" IPCC AR6: ")[0]
    assert SENTENCE in numbers
    assert report.hazard_stats == []


def test_a_report_without_a_note_is_unchanged(httpx_mock):
    httpx_mock.add_response(json=CANNED)

    report = graph_mod.run_agent("Berlin", 52.52, 13.40, Hazard.HEATWAVE, horizon_days=2)

    assert "ERA5 history skipped" not in report.summary


def test_the_ask_path_hands_the_sentence_to_the_agent(monkeypatch):
    captured = {}

    def refuse(*args, **kwargs):
        raise climatology_mod.ArchiveBudgetExceeded(SENTENCE)

    def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return graph_mod.RiskReport(location=kwargs["location"], hazard=kwargs["hazard"],
                                    horizon_days=kwargs["horizon_days"], confidence=0.5,
                                    risk_level=RiskLevel.LOW, summary="stub")

    monkeypatch.setattr(nl_mod, "climatology_hazard_stat", refuse)
    monkeypatch.setattr(nl_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(nl_mod, "geocode", lambda place: GeoLocation(
        name=place, latitude=52.52, longitude=13.40, country="Germany"))

    nl_mod.run_agent_nl("How risky are heatwaves in Berlin over the next 7 days?")

    assert captured["climatology_note"] == SENTENCE
    assert captured["hazard_stat"] is None


def test_the_manual_path_renders_the_sentence_beside_the_band(monkeypatch):
    import streamlit as st

    captured = {}

    def refuse(*args, **kwargs):
        raise climatology_mod.ArchiveBudgetExceeded(SENTENCE)

    def fake_run_agent(**kwargs):
        captured.update(kwargs)
        note = kwargs.get("climatology_note") or ""
        return graph_mod.RiskReport(
            location="Rourkela, South Asia (SAS)", hazard=kwargs["hazard"],
            horizon_days=kwargs["horizon_days"], confidence=0.5, risk_level=RiskLevel.LOW,
            summary=f"Peak daily max temperature of 42.0 °C over 7 days. {note}".strip())

    monkeypatch.setattr(climatology_mod, "climatology_hazard_stat", refuse)
    monkeypatch.setattr(graph_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(st, "map", lambda *a, **k: None)

    at = AppTest.from_file("ui/app.py", default_timeout=30)
    at.run()
    at.sidebar.button(key="assess").set_value(True)
    at.run()

    assert not at.exception
    assert captured["climatology_note"] == SENTENCE
    assert any(SENTENCE in c.value for c in at.caption)  # beside the band, in the numbers line
    # One plain sentence, not the generic failure notice on top of it.
    assert not any("Climatology unavailable" in w.value for w in at.warning)
