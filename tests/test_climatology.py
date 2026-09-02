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
    # The 2-year level is a risk-band edge (agent/risk_bands.py), so it is fitted
    # alongside the rarer ones rather than extrapolated at report time.
    assert [rl.return_period_years for rl in stat.return_levels] == [2, 10, 50, 100]


def test_wind_uses_gust_variable_with_lower_bound_caveat():
    stat = build_hazard_stat(
        [2000, 2001], [80.0, 95.0], hazard=Hazard.WIND, latitude=13.08, longitude=80.27,
        timezone="Asia/Kolkata",
    )
    assert stat.variable == "wind_gusts_10m_max"
    assert stat.unit == "km/h"
    assert "lower bound" in stat.interpretation.lower()


def _build(years, maxima, **kw):
    return build_hazard_stat(
        years, maxima, hazard=Hazard.HEATWAVE, latitude=22.26, longitude=84.85,
        timezone="Asia/Kolkata", **kw,
    )


def test_trending_series_reports_effective_levels(monkeypatch):
    monkeypatch.setattr("tools.climatology._TREND_N_BOOT", 20)  # keep test fast
    years = list(range(1960, 2023))
    # strong deterministic warming (0.8 °C/decade) + bounded wiggle
    maxima = [38.0 + 0.08 * i + (i % 7) * 0.4 for i in range(len(years))]

    stat = _build(years, maxima)

    assert stat.trend is not None
    assert stat.trend.significant is True
    assert stat.trend.p_value < 0.05
    assert stat.trend.slope_per_decade == pytest.approx(0.8, abs=0.3)
    assert stat.trend.evaluated_at_year == 2022
    # effective levels carry the bootstrap band too
    assert all(rl.ci_low is not None for rl in stat.return_levels)
    # effective 10-yr level at 2022 must sit near the END of the warmed series,
    # far above the stationary whole-period fit would put it
    ten_year = next(rl for rl in stat.return_levels if rl.return_period_years == 10)
    assert ten_year.level > max(maxima) - 3.0


def test_flat_series_keeps_stationary_levels_but_reports_the_test():
    years = list(range(1960, 2023))
    maxima = [40.0 + (i % 7) for i in range(len(years))]  # zero trend

    stat = _build(years, maxima)

    assert stat.trend is not None
    assert stat.trend.significant is False
    assert stat.trend.evaluated_at_year is None  # stationary levels reported
    assert all(rl.ci_low is not None for rl in stat.return_levels)


def test_short_series_skips_trend_test():
    years = list(range(2000, 2010))  # 10 years < _MIN_YEARS_FOR_TREND
    maxima = [40.0 + (i % 3) for i in range(len(years))]

    stat = _build(years, maxima)

    assert stat.trend is None


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


# --- Cache v2: the fitted statistic, not just the raw series ----------------
#
# The 1960-2022 ERA5 record is static, so a (location, hazard) statistic never
# changes. Caching the FIT is what matters: the measured cost is ~42 s for the
# archive fetch plus 12.9 s for the GEV fit + trend test + bootstrap, and a
# raw-series cache would still pay the second half.

def _fit_cache_for(tmp_path):
    from tools.cache_backend import DiskCache, JsonCache

    return JsonCache("hazard_fit", backend=DiskCache(tmp_path))


def test_hazard_fit_cache_makes_a_repeat_skip_the_archive(httpx_mock, tmp_path, monkeypatch):
    import tools.climatology as clim

    monkeypatch.setattr(clim, "_fit_cache", lambda: _fit_cache_for(tmp_path))
    httpx_mock.add_response(json=CANNED)  # exactly ONE archive response is registered

    first = clim.climatology_hazard_stat(
        22.26, 84.85, Hazard.HEATWAVE, start_year=2000, end_year=2002
    )
    # Drop the in-process memo so ONLY the persistent cache can serve run two —
    # that is the tier a Cloud Run cold start actually has.
    clim.climatology_hazard_stat.cache_clear()
    second = clim.climatology_hazard_stat(
        22.26, 84.85, Hazard.HEATWAVE, start_year=2000, end_year=2002
    )

    assert len(httpx_mock.get_requests()) == 1  # fetched once, served twice
    assert second == first


def test_hazard_fit_cache_key_rounds_coordinates_to_two_decimals():
    import tools.climatology as clim

    args = (Hazard.HEATWAVE, 1960, 2022, (10, 50, 100))
    # ~1.1 km apart, far inside one ~25 km ERA5 cell: the same fit, one entry.
    assert clim._fit_cache_key(22.2601, 84.85, *args) == clim._fit_cache_key(22.2649, 84.85, *args)
    assert clim._fit_cache_key(22.26, 84.85, *args) != clim._fit_cache_key(22.30, 84.85, *args)


def test_hazard_fit_cache_key_tracks_bootstrap_count_and_fitting_code(monkeypatch):
    import tools.climatology as clim

    args = (22.26, 84.85, Hazard.HEATWAVE, 1960, 2022, (10, 50, 100))
    base = clim._fit_cache_key(*args)

    monkeypatch.setattr(clim, "_N_BOOT", 42)  # narrower CIs are a different statistic
    assert clim._fit_cache_key(*args) != base

    monkeypatch.undo()
    monkeypatch.setattr(clim, "_code_fingerprint", lambda: "0000deadbeef")
    assert clim._fit_cache_key(*args) != base  # edit the GEV code -> refit, no stale numbers


# --- the fit-cache fingerprint: the code that PRODUCES the numbers ----------

def test_the_code_fingerprint_covers_this_module_too(tmp_path, monkeypatch):
    """The digest used to hash hazard_stats.py and gev_trend.py only, so editing
    the assembly in THIS module (which ERA5 variable a hazard reads, the
    trend-vs-stationary rule, the minimum years for a trend fit) left every
    cached HazardStat in place while the numbers behind it had changed."""
    import tools.climatology as clim

    assert "climatology.py" in clim._FINGERPRINT_SOURCES

    for name in clim._FINGERPRINT_SOURCES:  # a working copy we can edit
        (tmp_path / name).write_bytes((clim._SOURCE_DIR / name).read_bytes())
    monkeypatch.setattr(clim, "_SOURCE_DIR", tmp_path)

    clim._code_fingerprint.cache_clear()
    before = clim._code_fingerprint()

    edited = tmp_path / "climatology.py"
    edited.write_bytes(edited.read_bytes() + b"# a change to the fitting code")
    clim._code_fingerprint.cache_clear()
    after = clim._code_fingerprint()

    assert before != after
    clim._code_fingerprint.cache_clear()  # leave the real digest cached for others

