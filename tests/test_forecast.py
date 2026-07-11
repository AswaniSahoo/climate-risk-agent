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
    },
    "daily": {
        "time": ["2026-07-02", "2026-07-03", "2026-07-04"],
        "precipitation_sum": [14.2, 0.1, 55.0],
        "temperature_2m_max": [35.1, 36.4, 33.0],
        "wind_speed_10m_max": [70.0, 45.0, 30.0],
    },
}


def test_get_forecast_parses_daily_series(httpx_mock):
    httpx_mock.add_response(json=CANNED)

    result = get_forecast(latitude=22.26, longitude=84.85, horizon_days=3)

    assert isinstance(result, ForecastResult)
    assert result.precipitation_sum == [14.2, 0.1, 55.0]
    assert result.temperature_2m_max[2] == 33.0
    assert result.wind_speed_10m_max == [70.0, 45.0, 30.0]  # km/h
    assert result.time[0] == date(2026, 7, 2)
    assert result.timezone == "Asia/Kolkata"


def test_get_forecast_raises_forecasterror_on_http_error(httpx_mock):
    httpx_mock.add_response(status_code=500)

    with pytest.raises(ForecastError):
        get_forecast(latitude=22.26, longitude=84.85, horizon_days=3)
