"""Tests for the Open-Meteo Archive hazard source (tools/climatology.py).

Pure `annual_maxima` + `build_hazard_stat` are tested directly; the network fetch
is mocked with pytest-httpx, so everything is offline and deterministic.
"""
import pytest

from agent.contracts import Hazard
from tools.climatology import (
    ClimatologyError,
    annual_maxima,
    build_hazard_stat,
    climatology_hazard_stat,
)
from tools.hazard_stats import HazardStat, Representativeness


@pytest.fixture(autouse=True)
def _clear_climatology_cache():
    """climatology_hazard_stat is lru_cached; clear it so mocked calls stay isolated."""
    climatology_hazard_stat.cache_clear()
    yield


CANNED = {
    "timezone": "Asia/Kolkata",
    "daily_units": {"time": "iso8601", "temperature_2m_max": "°C"},
    "daily": {
        "time": ["2000-05-15", "2000-06-15", "2001-05-15", "2001-06-15", "2002-05-15", "2002-06-15"],
        "temperature_2m_max": [44.0, 41.0, 45.0, 43.0, 46.0, 42.0],
    },
}


def test_annual_maxima_reduces_per_year_and_skips_none():
    times = ["2000-05-01", "2000-06-01", "2001-05-01", "2001-06-01"]
    values = [40.0, None, 39.0, 44.0]

    years, maxima = annual_maxima(times, values)

    assert years == [2000, 2001]
    assert maxima == [40.0, 44.0]


def test_build_hazard_stat_is_provenanced_and_honest():
    years = list(range(1990, 2020))
    maxima = [40.0 + (i % 7) for i in range(len(years))]

    stat = build_hazard_stat(
        years, maxima, hazard=Hazard.HEATWAVE, latitude=22.26, longitude=84.85,
        timezone="Asia/Kolkata",
    )

    assert isinstance(stat, HazardStat)
    assert stat.variable == "temperature_2m_max"
    assert stat.unit == "°C"
    assert stat.captures_diurnal_peak is True
    assert stat.is_bias_corrected is False
    assert stat.representativeness is Representativeness.POINT_INTERPOLATED_REANALYSIS
    assert stat.record_max == max(maxima)
    assert (stat.record_start_year, stat.record_end_year) == (1990, 2019)
    assert [rl.return_period_years for rl in stat.return_levels] == [10, 50, 100]


def test_wind_uses_gust_variable_with_lower_bound_caveat():
    stat = build_hazard_stat(
        [2000, 2001], [80.0, 95.0], hazard=Hazard.WIND, latitude=13.08, longitude=80.27,
        timezone="Asia/Kolkata",
    )
    assert stat.variable == "wind_gusts_10m_max"
    assert stat.unit == "km/h"
    assert "lower bound" in stat.interpretation.lower()


def test_climatology_hazard_stat_parses_archive(httpx_mock):
    httpx_mock.add_response(json=CANNED)

    stat = climatology_hazard_stat(22.26, 84.85, Hazard.HEATWAVE, start_year=2000, end_year=2002)

    assert stat.n_years == 3
    assert stat.record_max == 46.0
    assert stat.timezone == "Asia/Kolkata"
    assert stat.variable == "temperature_2m_max"


def test_climatology_hazard_stat_raises_on_http_error(httpx_mock):
    httpx_mock.add_response(status_code=500)

    with pytest.raises(ClimatologyError):
        climatology_hazard_stat(22.26, 84.85, Hazard.HEATWAVE)
