"""Tests for the 3-node LangGraph agent (agent/graph.py).

run_agent wires plan -> call get_forecast -> synthesize into one call that
returns a RiskReport. HTTP is mocked (pytest-httpx), so these are offline and
deterministic. The wind case proves the refusal path short-circuits BEFORE any
forecast call.
"""
import pytest

from agent.contracts import Hazard, RiskLevel, RiskReport
from agent.graph import run_agent

CANNED = {
    "latitude": 22.26,
    "longitude": 84.85,
    "timezone": "Asia/Kolkata",
    "daily_units": {"time": "iso8601", "precipitation_sum": "mm", "temperature_2m_max": "°C"},
    "daily": {
        "time": ["2026-07-02", "2026-07-03", "2026-07-04"],
        "precipitation_sum": [14.2, 0.1, 80.0],   # max 80mm -> HIGH precip band
        "temperature_2m_max": [46.0, 44.0, 40.0],  # max 46°C -> SEVERE heat band
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


def test_wind_query_takes_refusal_path_without_forecast():
    # No httpx mock added on purpose: if the graph wrongly calls get_forecast,
    # risk_level would be set and this test fails.
    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.WIND, horizon_days=3,
    )

    assert report.refusal is not None
    assert report.risk_level is None
