"""Tests for agent/risk_bands.py: severity as local rarity, not an absolute number.

The band must come from where the forecast peak lands on the location's own GEV
return-level curve (2 / 10 / 50-year edges), the explanation must carry the
numbers it was decided on, and the absolute-cutoff fallback must announce itself
so nobody mistakes it for a location-relative verdict.
"""
import math

import pytest

from agent.contracts import Hazard, RiskLevel
from agent.risk_bands import band_from_fixed_thresholds, band_from_return_period
from tools.hazard_stats import (
    HazardStat,
    Representativeness,
    ReturnLevel,
    TrendInfo,
)

# 2-yr=10, 10-yr=20, 50-yr=30, 100-yr=40 (any unit; here mm).
_LEVELS = {2: 10.0, 10: 20.0, 50: 30.0, 100: 40.0}


def _stat(levels: dict[int, float] | None = None, **overrides) -> HazardStat:
    levels = _LEVELS if levels is None else levels
    fields = dict(
        variable="precipitation_sum",
        statistic_definition="annual maximum of ERA5 daily precipitation total",
        unit="mm", source="test", model="era5", native_resolution_deg=0.25,
        captures_diurnal_peak=True, timezone="Asia/Kolkata",
        latitude=22.26, longitude=84.85, n_years=63,
        record_start_year=1960, record_end_year=2022, record_max=max(levels.values()),
        return_levels=[
            ReturnLevel(return_period_years=t, level=v) for t, v in levels.items()
        ],
        is_bias_corrected=False,
        representativeness=Representativeness.POINT_INTERPOLATED_REANALYSIS,
        interpretation="test fixture",
    )
    fields.update(overrides)
    return HazardStat(**fields)


# --- band edges -------------------------------------------------------------

@pytest.mark.parametrize(
    ("peak", "expected"),
    [
        (0.0, RiskLevel.LOW),         # nothing like an extreme here
        (9.99, RiskLevel.LOW),        # just under the 2-yr level
        (10.0, RiskLevel.MODERATE),   # AT the 2-yr level -> a notable year
        (19.99, RiskLevel.MODERATE),
        (20.0, RiskLevel.HIGH),       # AT the 10-yr level -> decadal-plus
        (29.99, RiskLevel.HIGH),
        (30.0, RiskLevel.SEVERE),     # AT the 50-yr level -> rare
        (39.99, RiskLevel.SEVERE),
        (40.0, RiskLevel.SEVERE),     # at/above the top fitted level
        (1000.0, RiskLevel.SEVERE),
    ],
)
def test_every_band_edge_is_inclusive_at_the_lower_bound(peak, expected):
    level, _ = band_from_return_period(peak, _stat())
    assert level is expected


@pytest.mark.parametrize(
    ("peak", "expected_period", "expected_level"),
    [
        (15.0, 4, RiskLevel.MODERATE),  # halfway between the 2- and 10-yr levels
        (25.0, 22, RiskLevel.HIGH),     # halfway between the 10- and 50-yr levels
        (35.0, 71, RiskLevel.SEVERE),   # halfway between the 50- and 100-yr levels
    ],
)
def test_return_period_is_interpolated_in_log_space(peak, expected_period, expected_level):
    # Halfway in LEVEL is not halfway in years: the curve is a straight line in
    # log(T), so 15 mm reads as a 1-in-4-year event, not a 1-in-6-year one.
    level, explanation = band_from_return_period(peak, _stat())

    assert level is expected_level
    assert f"1-in-{expected_period}-year event" in explanation


def test_interpolated_period_matches_the_gumbel_formula():
    _, explanation = band_from_return_period(24.0, _stat())
    expected = math.exp(math.log(10) + 0.4 * (math.log(50) - math.log(10)))

    assert f"1-in-{expected:.0f}-year event" in explanation


def test_explanation_names_the_bracketing_levels():
    _, explanation = band_from_return_period(25.0, _stat())

    assert "Forecast peak 25 mm" in explanation
    assert "between the 10- and 50-year levels (20.0 / 30.0 mm)" in explanation


def test_below_the_smallest_level_says_ordinary_year_not_a_return_period():
    # Nothing is extrapolated off the bottom of the fitted curve.
    level, explanation = band_from_return_period(5.0, _stat())

    assert level is RiskLevel.LOW
    assert "below the 2-year level (10.0 mm)" in explanation
    assert "1-in-" not in explanation


def test_at_or_above_the_top_level_is_called_record_class():
    level, explanation = band_from_return_period(60.0, _stat())

    assert level is RiskLevel.SEVERE
    assert "at or above the 100-year level (40.0 mm)" in explanation
    assert "record-class" in explanation


def test_curve_order_does_not_matter():
    shuffled = {50: 30.0, 2: 10.0, 100: 40.0, 10: 20.0}

    assert band_from_return_period(25.0, _stat(shuffled))[0] is RiskLevel.HIGH


def test_missing_band_edge_raises_instead_of_guessing():
    # A curve without the 2-yr level cannot tell LOW from MODERATE, so it must
    # fail loudly rather than silently pick one.
    with pytest.raises(ValueError, match="required return periods"):
        band_from_return_period(25.0, _stat({10: 20.0, 50: 30.0, 100: 40.0}))


def test_collapsed_levels_do_not_break_the_bracket_search():
    # A degenerate fit whose 10- and 50-yr levels coincide still has to produce a
    # band: the value falls into the next bracket up rather than dividing by zero.
    level, explanation = band_from_return_period(
        20.0, _stat({2: 10.0, 10: 20.0, 50: 20.0, 100: 40.0})
    )

    assert level is RiskLevel.SEVERE  # 20 mm is already the 50-yr level here
    assert "between the 50- and 100-year levels (20.0 / 40.0 mm)" in explanation


# --- effective (trend-adjusted) levels ---------------------------------------

def test_significant_trend_says_the_levels_are_effective_at_that_year():
    stat = _stat(trend=TrendInfo(slope_per_decade=2.5, p_value=0.003,
                                 significant=True, evaluated_at_year=2022))

    _, explanation = band_from_return_period(25.0, stat)

    assert "on levels effective at 2022" in explanation


def test_insignificant_trend_leaves_the_sentence_unqualified():
    stat = _stat(trend=TrendInfo(slope_per_decade=0.4, p_value=0.61,
                                 significant=False, evaluated_at_year=None))

    _, explanation = band_from_return_period(25.0, stat)

    assert "effective at" not in explanation


# --- the ungrounded fallback -------------------------------------------------

@pytest.mark.parametrize(
    ("hazard", "peak", "expected"),
    [
        (Hazard.HEATWAVE, 34.9, RiskLevel.LOW),
        (Hazard.HEATWAVE, 35.0, RiskLevel.MODERATE),
        (Hazard.HEATWAVE, 39.9, RiskLevel.MODERATE),
        (Hazard.HEATWAVE, 40.0, RiskLevel.HIGH),
        (Hazard.HEATWAVE, 44.9, RiskLevel.HIGH),
        (Hazard.HEATWAVE, 45.0, RiskLevel.SEVERE),
        (Hazard.EXTREME_PRECIP, 19.9, RiskLevel.LOW),
        (Hazard.EXTREME_PRECIP, 20.0, RiskLevel.MODERATE),
        (Hazard.EXTREME_PRECIP, 49.9, RiskLevel.MODERATE),
        (Hazard.EXTREME_PRECIP, 50.0, RiskLevel.HIGH),
        (Hazard.EXTREME_PRECIP, 99.9, RiskLevel.HIGH),
        (Hazard.EXTREME_PRECIP, 100.0, RiskLevel.SEVERE),
        (Hazard.WIND, 39.9, RiskLevel.LOW),
        (Hazard.WIND, 40.0, RiskLevel.MODERATE),
        (Hazard.WIND, 61.9, RiskLevel.MODERATE),
        (Hazard.WIND, 62.0, RiskLevel.HIGH),
        (Hazard.WIND, 87.9, RiskLevel.HIGH),
        (Hazard.WIND, 88.0, RiskLevel.SEVERE),
    ],
)
def test_fallback_keeps_the_day_one_cutoffs_exactly(hazard, peak, expected):
    level, _ = band_from_fixed_thresholds(peak, hazard)
    assert level is expected


def test_fallback_explanation_admits_it_is_not_location_relative():
    _, explanation = band_from_fixed_thresholds(41.0, Hazard.HEATWAVE)

    assert "no ERA5 climatology is available for this location" in explanation
    assert "fixed absolute cutoffs rather than local rarity" in explanation
    assert "peak daily maximum temperature 41 °C grades high (40 to 45 °C)" in explanation


def test_fallback_range_text_covers_the_open_ended_bands():
    _, lowest = band_from_fixed_thresholds(10.0, Hazard.EXTREME_PRECIP)
    _, highest = band_from_fixed_thresholds(250.0, Hazard.EXTREME_PRECIP)

    assert "below 20 mm" in lowest
    assert "at or above 100 mm" in highest


def test_fallback_reports_the_caller_supplied_reason():
    _, explanation = band_from_fixed_thresholds(
        70.0, Hazard.WIND, reason="the fitted variable is not the forecast quantity"
    )

    assert "(the fitted variable is not the forecast quantity)" in explanation
    assert "sustained 10 m wind 70 km/h" in explanation
