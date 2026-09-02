"""Streamlit UI smoke tests (streamlit.testing.AppTest — headless, offline).

The agent + climatology are monkeypatched: these pin that the UI renders every
RiskReport path honestly (risk, citations, refusal), not the agent logic.
"""
from datetime import datetime, timezone

import pytest
from streamlit.testing.v1 import AppTest

import agent.graph as graph_mod
import agent.location as location_mod
import tools.climatology as climatology_mod
from agent.contracts import (
    Citation,
    DataProvenance,
    Hazard,
    RiskDriver,
    RiskLevel,
    RiskReport,
)

_APP = "ui/app.py"


def _report(**overrides) -> RiskReport:
    base = dict(
        location="Rourkela, India",
        hazard=Hazard.HEATWAVE,
        horizon_days=7,
        confidence=0.6,
        risk_level=RiskLevel.HIGH,
        summary="Peak daily max temperature of 42.0 °C over 7 days.",
        drivers=[RiskDriver(factor="temperature", detail="max daily 42.0 °C")],
        citations=[Citation(source="IPCC_AR6_WGI_Chapter11.pdf", locator="p124")],
        provenance=[DataProvenance(
            source="Open-Meteo", url="https://api.open-meteo.com/v1/forecast",
            retrieved_at=datetime.now(timezone.utc), params={},
        )],
    )
    base.update(overrides)
    return RiskReport(**base)


@pytest.fixture
def stubbed(monkeypatch):
    def fake_run_agent(**kwargs):
        return _report(hazard=kwargs["hazard"], horizon_days=kwargs["horizon_days"])

    monkeypatch.setattr(graph_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        climatology_mod, "climatology_hazard_stat",
        lambda *a, **k: (_ for _ in ()).throw(climatology_mod.ClimatologyError("offline test")),
    )


def _run_clicked(at: AppTest) -> AppTest:
    at.run()
    at.sidebar.button(key="assess").set_value(True)  # by key: the sidebar has example buttons too
    return at.run()


def test_report_path_renders_risk_and_citations(stubbed):
    at = _run_clicked(AppTest.from_file(_APP, default_timeout=30))

    assert not at.exception
    rendered = " ".join(el.value for el in at.markdown) + " ".join(s.value for s in at.subheader)
    assert "HIGH" in rendered
    assert "IPCC_AR6_WGI_Chapter11.pdf" in rendered
    assert "p124" in rendered


def test_manual_coordinates_reach_the_agent_unchanged(stubbed, monkeypatch):
    """Any point on Earth: a southern + western pair must arrive with its signs."""
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return _report(hazard=kwargs["hazard"], horizon_days=kwargs["horizon_days"])

    monkeypatch.setattr(graph_mod, "run_agent", capture)
    monkeypatch.setattr(location_mod, "region_for", lambda lat, lon: None)

    at = AppTest.from_file(_APP, default_timeout=30)
    at.session_state["lat_input"] = -33.90  # Sydney
    at.session_state["lon_input"] = -70.67  # deliberately both negative
    _run_clicked(at)  # the seeded name belongs to other coordinates -> dropped

    assert captured["latitude"] == pytest.approx(-33.90)
    assert captured["longitude"] == pytest.approx(-70.67)
    assert "33.9000°S" in captured["location"] and "70.6700°W" in captured["location"]


def test_point_with_no_ar6_region_shows_the_notice_and_still_runs(stubbed, monkeypatch):
    monkeypatch.setattr(location_mod, "region_for", lambda lat, lon: None)

    at = AppTest.from_file(_APP, default_timeout=30)
    at.session_state["lat_input"] = 0.0  # mid-Pacific
    at.session_state["lon_input"] = -140.0
    _run_clicked(at)  # the seeded name belongs to other coordinates -> dropped

    assert not at.exception
    assert any("IPCC AR6 land region" in i.value for i in at.info)
    assert at.subheader  # the report still rendered — no region is not a refusal


def test_refusal_path_renders_as_refusal_not_risk(stubbed, monkeypatch):
    monkeypatch.setattr(
        graph_mod, "run_agent",
        lambda **kw: _report(risk_level=None, citations=[], drivers=[],
                             summary="", confidence=0.0,
                             refusal="wind risk is not supported yet"),
    )
    at = _run_clicked(AppTest.from_file(_APP, default_timeout=30))

    assert not at.exception
    assert any("Refused" in e.value for e in at.error)
    assert not at.subheader  # no risk badge on a refusal
