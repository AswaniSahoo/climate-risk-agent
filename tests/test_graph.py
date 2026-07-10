"""Tests for the 3-node LangGraph agent (agent/graph.py).

run_agent wires plan -> call get_forecast -> synthesize into one call that
returns a RiskReport. HTTP is mocked (pytest-httpx), so these are offline and
deterministic. The wind case proves the refusal path short-circuits BEFORE any
forecast call.
"""
import pytest

from agent.contracts import Hazard, RiskLevel, RiskReport
from agent.graph import run_agent
from tools.climatology import build_hazard_stat

# ~20 years of (illustrative) annual-max 2m_temperature in Kelvin.
_HEAT_MAXIMA = [
    309.6, 310.5, 308.6, 311.0, 307.9, 312.1, 309.0, 310.2, 308.4, 311.5,
    309.8, 307.5, 310.9, 308.1, 312.4, 309.3, 310.7, 308.8, 311.2, 309.5,
]

CANNED = {
    "latitude": 22.26,
    "longitude": 84.85,
    "timezone": "Asia/Kolkata",
    "daily_units": {
        "time": "iso8601",
        "precipitation_sum": "mm",
        "temperature_2m_max": "°C",
        "wind_speed_10m_max": "km/h",
    },
    "daily": {
        "time": ["2026-07-02", "2026-07-03", "2026-07-04"],
        "precipitation_sum": [14.2, 0.1, 80.0],     # max 80mm  -> HIGH precip band
        "temperature_2m_max": [46.0, 44.0, 40.0],   # max 46°C  -> SEVERE heat band
        "wind_speed_10m_max": [70.0, 45.0, 30.0],   # max 70km/h -> HIGH wind band
    },
}


def test_extreme_precip_query_yields_high_risk_report(httpx_mock):
    httpx_mock.add_response(json=CANNED)

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.EXTREME_PRECIP, horizon_days=3,
    )

    assert isinstance(report, RiskReport)
    assert report.refusal is None
    assert report.risk_level is RiskLevel.HIGH
    assert report.provenance and report.provenance[0].source == "Open-Meteo"


def test_heatwave_query_yields_severe_risk_report(httpx_mock):
    httpx_mock.add_response(json=CANNED)

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.HEATWAVE, horizon_days=3,
    )

    assert report.hazard is Hazard.HEATWAVE
    assert report.risk_level is RiskLevel.SEVERE


def test_wind_query_yields_high_risk_report(httpx_mock):
    httpx_mock.add_response(json=CANNED)

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.WIND, horizon_days=3,
    )

    assert report.refusal is None
    assert report.risk_level is RiskLevel.HIGH  # 70 km/h
    assert report.drivers[0].factor == "wind"
    assert "m/s" in report.drivers[0].detail  # km/h converted for ERA5 comparability


def test_unsupported_hazard_takes_refusal_path_without_forecast(monkeypatch):
    # Simulate a hazard with no data path. No httpx mock on purpose: if the graph
    # wrongly calls get_forecast, this test fails.
    monkeypatch.setattr("agent.graph._ANSWERABLE", {Hazard.HEATWAVE})

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.WIND, horizon_days=3,
    )

    assert report.refusal is not None
    assert report.risk_level is None


def test_heatwave_report_includes_injected_hazard_stats(httpx_mock):
    httpx_mock.add_response(json=CANNED)
    hs = build_hazard_stat(
        list(range(2003, 2023)), _HEAT_MAXIMA, hazard=Hazard.HEATWAVE,
        latitude=22.26, longitude=84.85, timezone="Asia/Kolkata",
    )

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.HEATWAVE, horizon_days=3, hazard_stat=hs,
    )

    assert report.hazard_stats and report.hazard_stats[0].n_years == len(_HEAT_MAXIMA)
    assert report.confidence > 0.3  # climatology grounding beats the raw heuristic
    assert "return level" in report.summary.lower()

