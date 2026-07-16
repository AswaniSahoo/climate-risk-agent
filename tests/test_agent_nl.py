"""Tests for run_agent_nl — free text through parse -> geocode -> region -> agent.

All seams monkeypatched at the agent.nl module level (offline, deterministic).
Every failure mode must produce a typed REFUSAL report, never an exception and
never a guessed answer.
"""
import pytest

import agent.nl as nl_mod
from agent.contracts import Hazard, RiskLevel, RiskReport
from agent.nl import run_agent_nl
from tools.ar6_regions import AR6Region
from tools.geocode import GeocodeError, GeoLocation


def _stub_report(**overrides) -> RiskReport:
    base = dict(
        location="Rourkela, South Asia (SAS)", hazard=Hazard.HEATWAVE,
        horizon_days=10, confidence=0.55, risk_level=RiskLevel.LOW, summary="stub",
    )
    base.update(overrides)
    return RiskReport(**base)


@pytest.fixture
def happy_seams(monkeypatch):
    captured = {}

    def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return _stub_report(horizon_days=kwargs["horizon_days"])

    monkeypatch.setattr(nl_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        nl_mod, "geocode",
        lambda place: GeoLocation(name=place, latitude=22.25, longitude=84.88,
                                  country="India", admin1="Odisha"),
    )
    monkeypatch.setattr(
        nl_mod, "region_for",
        lambda lat, lon: AR6Region(acronym="SAS", name="South Asia"),
    )
    return captured


def test_full_query_reaches_agent_with_region_labelled_location(happy_seams):
    report = run_agent_nl("How risky are heatwaves in Rourkela over the next 10 days?")

    assert report.refusal is None
    assert happy_seams["hazard"] is Hazard.HEATWAVE
    assert happy_seams["horizon_days"] == 10
    assert happy_seams["latitude"] == pytest.approx(22.25)
    # region label in the corpus's vocabulary -> deterministic table-row retrieval
    assert "South Asia (SAS)" in happy_seams["location"]
    assert "Rourkela" in happy_seams["location"]


def test_ocean_or_unknown_region_degrades_to_place_name(happy_seams, monkeypatch):
    monkeypatch.setattr(nl_mod, "region_for", lambda lat, lon: None)

    run_agent_nl("heatwave risk in Rourkela next week")

    assert "Rourkela" in happy_seams["location"]
    assert "(SAS)" not in happy_seams["location"]


def test_out_of_scope_hazard_refuses_with_the_named_hazard(happy_seams):
    report = run_agent_nl("wildfire risk in Sydney next week")

    assert report.refusal is not None and "wildfire" in report.refusal
    assert report.hazard is None and report.risk_level is None
    assert not happy_seams  # the agent must never have been called


def test_unidentifiable_hazard_refuses_typed(happy_seams):
    report = run_agent_nl("what's the weather like in Paris")
    assert report.refusal is not None and "hazard" in report.refusal.lower()
    assert not happy_seams


def test_missing_place_refuses_typed(happy_seams):
    report = run_agent_nl("how bad will heatwaves get next week")
    assert report.refusal is not None and "location" in report.refusal.lower()
    assert not happy_seams


def test_geocode_failure_refuses_typed(happy_seams, monkeypatch):
    def boom(place):
        raise GeocodeError(f"no match for place name: {place!r}")

    monkeypatch.setattr(nl_mod, "geocode", boom)
    report = run_agent_nl("heatwave risk in Xyzzyville next week")

    assert report.refusal is not None and "Xyzzyville" in report.refusal
    assert not happy_seams
