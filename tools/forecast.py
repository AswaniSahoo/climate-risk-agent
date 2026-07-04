"""get_forecast: the agent's first sense organ.

Calls Open-Meteo (free, no API key) for a lat/lon and returns the next few days
of rain + heat as a typed ForecastResult. The raw API hands back a loose JSON
blob; we convert it once, here, so the rest of the system only ever sees clean,
guaranteed fields.
"""
from __future__ import annotations

from datetime import date

import httpx
from pydantic import BaseModel

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


class ForecastError(RuntimeError):
    """Raised when the Open-Meteo request fails (network error or bad status)."""


class ForecastResult(BaseModel):
    """Typed daily forecast series for one location."""

    latitude: float
    longitude: float
    timezone: str
    time: list[date]
    precipitation_sum: list[float]  # mm per day
    temperature_2m_max: list[float]  # °C per day


def get_forecast(
    latitude: float, longitude: float, horizon_days: int = 7
) -> ForecastResult:
    """Fetch a daily forecast (precipitation + max temp) for the next N days."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "precipitation_sum,temperature_2m_max",
        "forecast_days": horizon_days,
        "timezone": "auto",
    }
    try:
        response = httpx.get(OPEN_METEO_URL, params=params, timeout=10)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ForecastError(f"Open-Meteo request failed: {exc}") from exc

    data = response.json()
    daily = data["daily"]
    return ForecastResult(
        latitude=data["latitude"],
        longitude=data["longitude"],
        timezone=data["timezone"],
        time=daily["time"],
        precipitation_sum=daily["precipitation_sum"],
        temperature_2m_max=daily["temperature_2m_max"],
    )
