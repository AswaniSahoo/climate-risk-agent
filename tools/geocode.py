"""Geocoding: place name -> coordinates, via the Open-Meteo geocoding API.

Same posture as the other tools: hardcoded host (SSRF pin), typed error,
lru_cache (place names don't move — repeat lookups are free and can't hammer
the free tier). First match wins: Open-Meteo ranks by population/relevance,
and the report echoes the resolved name + country so a wrong pick is visible,
never silent.
"""
from __future__ import annotations

from functools import lru_cache

import httpx
from pydantic import BaseModel

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


class GeocodeError(RuntimeError):
    """Raised when a place name cannot be resolved to coordinates."""


class GeoLocation(BaseModel):
    """One resolved place: what the user said, made spatial and auditable."""

    name: str
    latitude: float
    longitude: float
    country: str = ""
    admin1: str = ""  # state / province, for disambiguation in the report


@lru_cache(maxsize=256)
def geocode(place: str) -> GeoLocation:
    """Resolve a place name to its best-match location (typed error on failure)."""
    try:
        response = httpx.get(
            GEOCODE_URL, params={"name": place, "count": 1, "format": "json"}, timeout=10
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise GeocodeError(f"geocoding request failed: {exc}") from exc

    results = response.json().get("results") or []
    if not results:
        raise GeocodeError(f"no match for place name: {place!r}")
    top = results[0]
    return GeoLocation(
        name=top["name"],
        latitude=float(top["latitude"]),
        longitude=float(top["longitude"]),
        country=top.get("country", ""),
        admin1=top.get("admin1", ""),
    )
