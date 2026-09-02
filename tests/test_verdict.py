"""Tests for agent/verdict.py — composed confidence.

Confidence must be composed from what actually grounds the report, not a
constant. Severity banding moved to agent/risk_bands.py and is tested in
tests/test_risk_bands.py.
"""
import pytest

from agent.verdict import compose_confidence
from tools.hazard_stats import Representativeness

# --- confidence composition ---

def test_confidence_forecast_only_stays_low():
    assert compose_confidence(representativeness=None, ipcc_cited=False) == 0.3


def test_confidence_climbs_with_grounding_quality():
    point = compose_confidence(
        representativeness=Representativeness.POINT_INTERPOLATED_REANALYSIS,
        ipcc_cited=False,
    )
    regional = compose_confidence(
        representativeness=Representativeness.REGIONAL_GRID_SIGNAL, ipcc_cited=False,
    )
    assert 0.3 < regional < point  # better representativeness -> more confidence


def test_not_representative_climatology_adds_nothing():
    assert compose_confidence(
        representativeness=Representativeness.NOT_REPRESENTATIVE, ipcc_cited=False,
    ) == 0.3


def test_ipcc_citation_bumps_and_ceiling_holds():
    top = compose_confidence(
        representativeness=Representativeness.STATION_CALIBRATED, ipcc_cited=True,
    )
    assert top <= 0.75  # honest ceiling: no skill-aware verification layer yet
    assert compose_confidence(representativeness=None, ipcc_cited=True) == pytest.approx(0.4)
