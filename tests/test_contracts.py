"""Tests for the RiskReport output contract (agent/contracts.py).

These tests ARE the spec: they define what a valid climate-risk report is
before any schema code exists.
"""
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agent.contracts import (
    Citation,
    DataProvenance,
    Hazard,
    RiskDriver,
    RiskLevel,
    RiskReport,
)


def _provenance() -> DataProvenance:
    return DataProvenance(
        source="Open-Meteo",
        url="https://api.open-meteo.com/v1/forecast",
        retrieved_at=datetime(2026, 6, 20, tzinfo=timezone.utc),
        params={"latitude": 22.26, "longitude": 84.85},
    )


def _valid_report(**overrides) -> RiskReport:
    """A fully-populated, non-refusal report; overrides let each test poke one field."""
    data = dict(
        location="Rourkela",
        hazard=Hazard.EXTREME_PRECIP,
        horizon_days=7,
        risk_level=RiskLevel.HIGH,
        summary="Heavy rain expected; elevated flood risk.",
        drivers=[RiskDriver(factor="precipitation", detail="max daily 80mm")],
        citations=[Citation(source="IPCC AR6 WG1", locator="Ch.11")],
        provenance=[_provenance()],
        confidence=0.4,
    )
    data.update(overrides)
    return RiskReport(**data)


def test_valid_report_builds_with_expected_fields():
    report = _valid_report()
    assert report.risk_level is RiskLevel.HIGH
    assert report.hazard is Hazard.EXTREME_PRECIP
    assert report.refusal is None
    assert report.drivers[0].factor == "precipitation"


def test_risk_level_rejects_unknown_value():
    with pytest.raises(ValidationError):
        _valid_report(risk_level="catastrophic")


def test_hazard_rejects_unknown_value():
    with pytest.raises(ValidationError):
        _valid_report(hazard="meteor")


def test_supported_hazards_match_master_plan():
    assert {h.value for h in Hazard} == {"heatwave", "extreme_precip", "wind"}


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_confidence_must_be_between_0_and_1(bad):
    with pytest.raises(ValidationError):
        _valid_report(confidence=bad)


def test_horizon_days_must_be_positive():
    with pytest.raises(ValidationError):
        _valid_report(horizon_days=0)


def test_json_round_trip_preserves_data():
    report = _valid_report()
    raw = report.model_dump_json()
    restored = RiskReport.model_validate_json(raw)
    assert restored == report


def test_refusal_report_is_valid_without_risk():
    report = RiskReport(
        location="Rourkela",
        hazard=Hazard.EXTREME_PRECIP,
        horizon_days=7,
        confidence=0.0,
        refusal="Out of scope: only flood hazard is supported.",
    )
    assert report.refusal is not None
    assert report.risk_level is None
    assert report.drivers == []


def test_refusal_cannot_also_assert_risk_level():
    with pytest.raises(ValidationError):
        RiskReport(
            location="Rourkela",
            hazard=Hazard.EXTREME_PRECIP,
            horizon_days=7,
            confidence=0.0,
            risk_level=RiskLevel.HIGH,
            refusal="contradictory: refusing but also asserting a level",
        )


def test_non_refusal_requires_risk_level():
    with pytest.raises(ValidationError):
        RiskReport(
            location="Rourkela",
            hazard=Hazard.EXTREME_PRECIP,
            horizon_days=7,
            confidence=0.2,
            summary="no level given",
        )
