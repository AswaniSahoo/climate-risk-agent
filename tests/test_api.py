"""Tests for the FastAPI service (api/app.py).

Uses fastapi.testclient.TestClient — sync client, but it drives the async app
underneath via ASGI transport, so no separate async test setup is needed.

Agent + climatology calls are monkeypatched at the `api.app` MODULE level
(not `agent.graph` / `tools.climatology`) because that's the name api/app.py
actually looks up at call time — patching the origin module would leave
api.app's already-imported reference untouched.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.app as api_app
from agent.contracts import Hazard, RiskLevel, RiskReport
from tools.climatology import ClimatologyError


def _stub_report(hazard: Hazard = Hazard.HEATWAVE) -> RiskReport:
    return RiskReport(
        location="Test City",
        hazard=hazard,
        horizon_days=7,
        confidence=0.75,
        risk_level=RiskLevel.MODERATE,
        summary="stub report",
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(api_app.app)


def _request_body(**overrides) -> dict:
    body = {
        "location": "Test City",
        "latitude": 12.34,
        "longitude": 56.78,
        "hazard": "heatwave",
        "horizon_days": 7,
        "use_climatology": True,
    }
    body.update(overrides)
    return body


def test_report_happy_path(client, monkeypatch):
    monkeypatch.setattr(api_app, "run_agent", lambda *a, **kw: _stub_report())
    monkeypatch.setattr(api_app, "climatology_hazard_stat", lambda *a, **kw: object())

    response = client.post("/report", json=_request_body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["report"]["risk_level"] == "moderate"
    assert "wall_ms" in payload["telemetry"]


def test_report_value_error_is_422(client, monkeypatch):
    def _raise_value_error(*a, **kw):
        raise ValueError("latitude 999.0 outside valid range [-90, 90]")

    monkeypatch.setattr(api_app, "climatology_hazard_stat", lambda *a, **kw: object())
    monkeypatch.setattr(api_app, "run_agent", _raise_value_error)

    response = client.post("/report", json=_request_body(latitude=999.0))

    assert response.status_code == 422
    assert "latitude" in response.json()["detail"]


def test_report_climatology_failure_still_returns_report(client, monkeypatch):
    def _raise_climatology_error(*a, **kw):
        raise ClimatologyError("archive unavailable")

    monkeypatch.setattr(api_app, "climatology_hazard_stat", _raise_climatology_error)
    monkeypatch.setattr(api_app, "run_agent", lambda *a, **kw: _stub_report())

    response = client.post("/report", json=_request_body())

    assert response.status_code == 200
    assert response.json()["report"]["summary"] == "stub report"


def test_report_requires_api_key_when_set(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setattr(api_app, "run_agent", lambda *a, **kw: _stub_report())
    monkeypatch.setattr(api_app, "climatology_hazard_stat", lambda *a, **kw: object())

    missing = client.post("/report", json=_request_body())
    wrong = client.post("/report", json=_request_body(), headers={"x-api-key": "nope"})
    right = client.post("/report", json=_request_body(), headers={"x-api-key": "secret"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert right.status_code == 200


def test_metrics_requires_api_key_when_set(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")

    missing = client.get("/metrics")
    right = client.get("/metrics", headers={"x-api-key": "secret"})

    assert missing.status_code == 401
    assert right.status_code == 200


def test_healthz_always_open_no_key_required(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_returns_dict(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)
