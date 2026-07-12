"""Tests for agent/verdict.py — the GEV-grounded risk verdict + composed confidence.

This is the evaluator-gap-#1 fix: risk_level must come from where the forecast
peak sits on the location's return-level curve (location-relative severity),
not from hardcoded absolute thresholds; confidence must be composed from what
actually grounds the report, not a constant.
"""
import pytest

from agent.contracts import RiskLevel
from agent.verdict import compose_confidence, level_from_return_periods
from tools.hazard_stats import Representativeness, ReturnLevel

# A return-level curve: 10-yr=100, 50-yr=140, 100-yr=160 (any unit).
_CURVE = [
    ReturnLevel(return_period_years=10, level=100.0),
    ReturnLevel(return_period_years=50, level=140.0),
    ReturnLevel(return_period_years=100, level=160.0),
]


@pytest.mark.parametrize(
    ("peak", "expected"),
    [
        (50.0, RiskLevel.LOW),        # well inside decadal experience
        (99.9, RiskLevel.LOW),        # just under the 10-yr level
        (100.0, RiskLevel.MODERATE),  # at the 10-yr level -> unusual
        (139.9, RiskLevel.MODERATE),
        (140.0, RiskLevel.HIGH),      # at the 50-yr level -> rare
        (159.9, RiskLevel.HIGH),
        (160.0, RiskLevel.SEVERE),    # at/above the 100-yr level -> record-class
        (500.0, RiskLevel.SEVERE),
    ],
)
def test_level_from_return_period_position(peak, expected):
    assert level_from_return_periods(peak, _CURVE) is expected


def test_unordered_curve_is_handled():
    shuffled = [_CURVE[2], _CURVE[0], _CURVE[1]]
    assert level_from_return_periods(150.0, shuffled) is RiskLevel.HIGH


def test_missing_periods_raise_typed_error():
    with pytest.raises(ValueError, match="return periods"):
        level_from_return_periods(1.0, [ReturnLevel(return_period_years=7, level=5.0)])


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
