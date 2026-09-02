"""get_forecast: the agent's first sense organ.

Calls Open-Meteo (free, no API key) for a lat/lon and returns the next few days
of rain, heat and wind as a typed ForecastResult. The raw API hands back a loose
JSON blob; we convert it once, here, so the rest of the system only ever sees
clean, guaranteed fields.
"""
from __future__ import annotations

import hashlib
import time
from datetime import date
from functools import lru_cache

import httpx
from pydantic import BaseModel

from tools.cache_backend import JsonCache
from tools.forecast_skill import ForecastSkill

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# The variable set is part of the cache identity: ask for fewer variables and
# the cached ForecastResult would be missing fields the caller expects.
# `wind_gusts_10m_max` is here because the ERA5 wind climatology is fitted on
# GUSTS: comparing a sustained-wind forecast against a gust return-level curve
# understates the risk, so we fetch the quantity the curve actually describes.
_DAILY_VARS = (
    "precipitation_sum,temperature_2m_max,wind_speed_10m_max,wind_gusts_10m_max"
)

# The forecast is the agent's core input: a transient Open-Meteo blip (503s are
# common there, and timeouts happen) must not sink the whole report, so we retry
# with a short exponential backoff. 4xx is our own bad request — fail fast.
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 0.5
_SLEEP = time.sleep  # module attr so tests can neutralize the backoff


class ForecastError(RuntimeError):
    """Raised when the Open-Meteo request fails (network error or bad status)."""


def _is_transient(exc: httpx.HTTPError) -> bool:
    """A transient failure is worth retrying; a 4xx (our bad params) is not."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)  # timeouts, connect/protocol errors


class ForecastResult(BaseModel):
    """Typed daily forecast series for one location."""

    latitude: float
    longitude: float
    timezone: str
    time: list[date]
    precipitation_sum: list[float]  # mm per day
    temperature_2m_max: list[float]  # °C per day
    # Both wind fields are km/h (the Open-Meteo default for wind_speed_10m_max and
    # wind_gusts_10m_max on the forecast AND archive endpoints). Sustained speed is
    # kept for display; the gust is what the ERA5 gust climatology is fitted on.
    wind_speed_10m_max: list[float]  # km/h per day, sustained 10 m wind
    wind_gusts_10m_max: list[float]  # km/h per day, 10 m gust

    # How much this forecast is worth at this horizon, from the measured skill
    # table. Attached by the agent's forecast node (agent/graph.py), NOT here:
    # the block is per-HAZARD (heat reads the temperature row, wind the gust
    # row) while this fetch and its cache key are hazard-blind, so baking it in
    # would serve one hazard's skill to the next hazard asking for the same
    # city and horizon. None whenever a ForecastResult comes straight from the
    # tool or the MCP server, and whenever the table has no row for the variable.
    skill: ForecastSkill | None = None


# Unlike the ERA5 climatology, a forecast is PERISHABLE: Open-Meteo refreshes
# its model runs through the day, so an hour is the longest a cached copy stays
# honest. This TTL is not there to save a day of requests — it absorbs the
# repeat click and the second Cloud Run replica asking for the same city.
_FORECAST_TTL_S = 3600


@lru_cache(maxsize=1)
def _forecast_cache() -> JsonCache:
    """The forecast namespace (memoised; tests swap this for a tmp_path one)."""
    return JsonCache("forecast")


def _forecast_cache_key(latitude: float, longitude: float, horizon_days: int) -> str:
    # 2 dp ≈ 1.1 km — finer than any daily forecast product resolves, so two
    # clicks on the same city share one entry.
    raw = f"{round(latitude, 2)}|{round(longitude, 2)}|{horizon_days}|{_DAILY_VARS}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_forecast(
    latitude: float, longitude: float, horizon_days: int = 7
) -> ForecastResult:
    """Fetch a daily forecast for the next N days.

    Series returned: precipitation total, max temperature, max sustained 10 m
    wind and max 10 m gust.
    """
    from tools.validation import validate_coordinates, validate_horizon

    validate_coordinates(latitude, longitude)
    validate_horizon(horizon_days)

    cache = _forecast_cache()
    cache_key = _forecast_cache_key(latitude, longitude, horizon_days)
    cached = cache.get_model(cache_key, ForecastResult)
    if cached is not None:
        return cached

    params: dict[str, str | int | float] = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": _DAILY_VARS,
        "forecast_days": horizon_days,
        "timezone": "auto",
    }
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = httpx.get(OPEN_METEO_URL, params=params, timeout=10)
            response.raise_for_status()
            break
        except httpx.HTTPError as exc:
            if not _is_transient(exc) or attempt == _MAX_ATTEMPTS - 1:
                raise ForecastError(f"Open-Meteo request failed: {exc}") from exc
            _SLEEP(_BACKOFF_BASE_S * 2**attempt)

    data = response.json()
    daily = data["daily"]
    result = ForecastResult(
        latitude=data["latitude"],
        longitude=data["longitude"],
        timezone=data["timezone"],
        time=daily["time"],
        precipitation_sum=daily["precipitation_sum"],
        temperature_2m_max=daily["temperature_2m_max"],
        wind_speed_10m_max=daily["wind_speed_10m_max"],
        wind_gusts_10m_max=daily["wind_gusts_10m_max"],
    )
    # Only a SUCCESS reaches this line — every failure path raises above, so a
    # bad response can never be cached and replayed for an hour.
    cache.set_model(cache_key, result, ttl_s=_FORECAST_TTL_S)
    return result
