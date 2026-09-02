"""Tests for the get_forecast tool (tools/forecast.py).

We feed a CANNED Open-Meteo response via pytest-httpx, so these tests never
touch the live network: fast, offline, deterministic. They check that our
parser turns Open-Meteo's JSON into a clean, typed ForecastResult.
"""
from datetime import date

import pytest

from tools.forecast import ForecastError, ForecastResult, get_forecast

# A fixed response shaped exactly like the real Open-Meteo API (shape verified live).
CANNED = {
    "latitude": 22.26,
    "longitude": 84.85,
    "timezone": "Asia/Kolkata",
    "daily_units": {
        "time": "iso8601",
        "precipitation_sum": "mm",
        "temperature_2m_max": "°C",
        "wind_speed_10m_max": "km/h",
        "wind_gusts_10m_max": "km/h",
    },
    "daily": {
        "time": ["2026-07-02", "2026-07-03", "2026-07-04"],
        "precipitation_sum": [14.2, 0.1, 55.0],
        "temperature_2m_max": [35.1, 36.4, 33.0],
        "wind_speed_10m_max": [70.0, 45.0, 30.0],
        "wind_gusts_10m_max": [95.0, 60.0, 42.0],  # a gust always tops the sustained speed
    },
}


def test_get_forecast_parses_daily_series(httpx_mock):
    httpx_mock.add_response(json=CANNED)

    result = get_forecast(latitude=22.26, longitude=84.85, horizon_days=3)

    assert isinstance(result, ForecastResult)
    assert result.precipitation_sum == [14.2, 0.1, 55.0]
    assert result.temperature_2m_max[2] == 33.0
    assert result.wind_speed_10m_max == [70.0, 45.0, 30.0]  # km/h
    assert result.wind_gusts_10m_max == [95.0, 60.0, 42.0]  # km/h
    assert result.time[0] == date(2026, 7, 2)
    assert result.timezone == "Asia/Kolkata"


def test_request_asks_for_gusts_because_the_climatology_is_a_gust_fit(httpx_mock):
    # The ERA5 wind hazard is fitted on wind_gusts_10m_max. Grading a sustained
    # -wind forecast against a gust return-level curve understates the risk, so
    # the outbound request must carry the gust variable.
    httpx_mock.add_response(json=CANNED)

    get_forecast(latitude=22.26, longitude=84.85, horizon_days=3)

    daily = httpx_mock.get_requests()[0].url.params["daily"].split(",")
    assert "wind_gusts_10m_max" in daily
    assert "wind_speed_10m_max" in daily  # still fetched, for display


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Retry backoff must not actually sleep during tests."""
    monkeypatch.setattr("tools.forecast._SLEEP", lambda _s: None)


def test_get_forecast_retries_transient_5xx_then_succeeds(httpx_mock):
    # Open-Meteo throws intermittent 503s (seen live) — a transient blip on the
    # agent's core input must not sink the whole report.
    httpx_mock.add_response(status_code=503)
    httpx_mock.add_response(json=CANNED)

    result = get_forecast(latitude=22.26, longitude=84.85, horizon_days=3)

    assert result.temperature_2m_max[2] == 33.0
    assert len(httpx_mock.get_requests()) == 2  # retried once


def test_get_forecast_does_not_retry_client_error(httpx_mock):
    # A 400 is our bug (bad params) — retrying wastes time and money, so fail fast.
    httpx_mock.add_response(status_code=400)

    with pytest.raises(ForecastError):
        get_forecast(latitude=22.26, longitude=84.85, horizon_days=3)

    assert len(httpx_mock.get_requests()) == 1  # no retry on 4xx


def test_get_forecast_raises_after_exhausting_retries(httpx_mock):
    for _ in range(3):  # every one of the 3 bounded attempts sees a 503
        httpx_mock.add_response(status_code=503)

    with pytest.raises(ForecastError):
        get_forecast(latitude=22.26, longitude=84.85, horizon_days=3)

    assert len(httpx_mock.get_requests()) == 3  # bounded, not infinite


# --- Cache v2: a short TTL, because a forecast is not a static record -------

def _forecast_cache_for(tmp_path):
    from tools.cache_backend import DiskCache, JsonCache

    return JsonCache("forecast", backend=DiskCache(tmp_path))


def test_forecast_cache_serves_a_repeat_without_a_second_request(
    httpx_mock, tmp_path, monkeypatch
):
    import tools.forecast as fc

    monkeypatch.setattr(fc, "_forecast_cache", lambda: _forecast_cache_for(tmp_path))
    httpx_mock.add_response(json=CANNED)  # ONE response for TWO calls

    first = fc.get_forecast(latitude=22.26, longitude=84.85, horizon_days=3)
    second = fc.get_forecast(latitude=22.26, longitude=84.85, horizon_days=3)

    assert len(httpx_mock.get_requests()) == 1
    assert second == first


def test_forecast_cache_expires_after_an_hour(httpx_mock, tmp_path, monkeypatch):
    import tools.cache_backend as backend_mod
    import tools.forecast as fc

    clock = [1000.0]
    monkeypatch.setattr(backend_mod, "_now", lambda: clock[0])
    monkeypatch.setattr(fc, "_forecast_cache", lambda: _forecast_cache_for(tmp_path))
    httpx_mock.add_response(json=CANNED, is_reusable=True)

    fc.get_forecast(latitude=22.26, longitude=84.85, horizon_days=3)
    clock[0] += fc._FORECAST_TTL_S + 1
    fc.get_forecast(latitude=22.26, longitude=84.85, horizon_days=3)

    # A day-old forecast served as today's is worse than no cache at all.
    assert len(httpx_mock.get_requests()) == 2


def test_forecast_cache_key_separates_location_horizon_and_variables(monkeypatch):
    import tools.forecast as fc

    base = fc._forecast_cache_key(22.26, 84.85, 3)

    assert fc._forecast_cache_key(22.26, 84.85, 7) != base  # different horizon
    assert fc._forecast_cache_key(22.30, 84.85, 3) != base  # different place
    assert fc._forecast_cache_key(22.2649, 84.85, 3) == base  # same 2-dp cell

    monkeypatch.setattr(fc, "_DAILY_VARS", "temperature_2m_max")
    assert fc._forecast_cache_key(22.26, 84.85, 3) != base  # asking for fewer variables


def test_forecast_errors_are_never_cached(httpx_mock, tmp_path, monkeypatch):
    import tools.forecast as fc

    monkeypatch.setattr(fc, "_forecast_cache", lambda: _forecast_cache_for(tmp_path))
    httpx_mock.add_response(status_code=400)  # our bad params: fail fast, cache nothing

    with pytest.raises(ForecastError):
        fc.get_forecast(latitude=22.26, longitude=84.85, horizon_days=3)

    httpx_mock.add_response(json=CANNED)
    result = fc.get_forecast(latitude=22.26, longitude=84.85, horizon_days=3)

    assert result.temperature_2m_max[2] == 33.0  # the retry went live, not to a cached failure
    assert len(httpx_mock.get_requests()) == 2
