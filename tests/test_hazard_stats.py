"""Tests for the GEV hazard-statistics core (tools/hazard_stats.py).

These assert *mathematical properties* of extreme-value stats — true regardless
of the exact fitted parameters — so we can write them before the implementation.
The fit is deterministic (same input maxima -> same params), so no randomness.
"""
import pytest

from tools.hazard_stats import HazardStat, hazard_stat, return_level, return_period

# 30 years of (illustrative) annual-maximum daily precipitation, mm.
ANNUAL_MAXIMA = [
    45.0, 52.0, 38.0, 61.0, 49.0, 55.0, 42.0, 70.0, 48.0, 53.0,
    58.0, 44.0, 66.0, 50.0, 47.0, 63.0, 51.0, 40.0, 57.0, 46.0,
    54.0, 68.0, 43.0, 59.0, 49.0, 62.0, 41.0, 56.0, 50.0, 64.0,
]


def test_return_level_increases_with_return_period():
    # A 100-year event must be at least as extreme as a 10-year event.
    assert return_level(ANNUAL_MAXIMA, 100) > return_level(ANNUAL_MAXIMA, 10)


def test_return_period_round_trips_return_level():
    # period(level(T)) == T, exactly, by definition of the quantile.
    level_100yr = return_level(ANNUAL_MAXIMA, 100)
    assert return_period(ANNUAL_MAXIMA, level_100yr) == pytest.approx(100, rel=1e-6)


def test_rarer_value_has_longer_return_period():
    # A bigger extreme is rarer -> longer return period.
    assert return_period(ANNUAL_MAXIMA, 80) > return_period(ANNUAL_MAXIMA, 55)


def test_hazard_stat_builds_typed_result():
    stat = hazard_stat(
        ANNUAL_MAXIMA, variable="precipitation", latitude=22.26, longitude=84.85
    )
    assert isinstance(stat, HazardStat)
    assert stat.variable == "precipitation"
    assert stat.years_of_data == len(ANNUAL_MAXIMA)
    assert [rl.return_period_years for rl in stat.return_levels] == [10, 50, 100]
    levels = [rl.level for rl in stat.return_levels]
    assert levels == sorted(levels)  # rarer event -> higher level
    assert levels[-1] > levels[0]

