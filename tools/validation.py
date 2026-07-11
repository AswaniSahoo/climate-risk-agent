"""Input validation at the tool boundary (Security Tier 1).

Every network tool validates caller input BEFORE building a request: coordinate
ranges (garbage-in guard) and horizon caps (denial-of-wallet guard). Raising
here keeps invalid values out of URLs, caches, and provider quotas.
"""
from __future__ import annotations

MAX_FORECAST_DAYS = 16  # Open-Meteo forecast API upper bound


def validate_coordinates(latitude: float, longitude: float) -> None:
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"latitude {latitude} outside valid range [-90, 90]")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError(f"longitude {longitude} outside valid range [-180, 180]")


def validate_horizon(horizon_days: int, *, max_days: int = MAX_FORECAST_DAYS) -> None:
    if not 1 <= horizon_days <= max_days:
        raise ValueError(f"horizon {horizon_days} outside valid range [1, {max_days}]")
