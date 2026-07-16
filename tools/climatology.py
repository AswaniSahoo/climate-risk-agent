"""Historical climate extremes from the Open-Meteo Archive (ERA5 reanalysis).

Replaces the WeatherBench2 zarr path. ERA5 **daily** maxima, point-interpolated to
the requested location, so the statistic sits at the right place and captures the
diurnal peak — the two things the coarse 6-hourly WB2 grid got wrong (see
docs/hazard-data-source.md for the measured comparison).

Split like get_forecast: pure `annual_maxima` + `build_hazard_stat` (unit-tested
offline) and the `climatology_hazard_stat` network edge (mocked in tests).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import httpx

from agent.contracts import Hazard
from tools.hazard_stats import HazardStat, Representativeness, return_levels_with_ci

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_ERA5_RESOLUTION_DEG = 0.25


class ClimatologyError(RuntimeError):
    """Raised when the Open-Meteo Archive request fails."""


@dataclass(frozen=True)
class _HazardVar:
    daily_var: str
    unit: str
    statistic_definition: str
    interpretation: str


# ERA5 underestimates sharp convective gusts, so wind is stated as a lower bound.
_HAZARD_VARS: dict[Hazard, _HazardVar] = {
    Hazard.HEATWAVE: _HazardVar(
        daily_var="temperature_2m_max",
        unit="°C",
        statistic_definition="annual maximum of ERA5 daily-maximum 2 m air temperature",
        interpretation=(
            "Point-interpolated ERA5 daily-max temperature (~25 km); captures the diurnal "
            "peak at the requested location but is not bias-corrected to a station, so it "
            "may modestly under-represent a true local extreme in heterogeneous terrain."
        ),
    ),
    Hazard.EXTREME_PRECIP: _HazardVar(
        daily_var="precipitation_sum",
        unit="mm",
        statistic_definition="annual maximum of ERA5 daily precipitation total",
        interpretation=(
            "Point-interpolated ERA5 daily precipitation (~25 km); reanalysis smooths local "
            "convective peaks, so treat as a regional-scale estimate, not a gauge value."
        ),
    ),
    Hazard.WIND: _HazardVar(
        daily_var="wind_gusts_10m_max",
        unit="km/h",
        statistic_definition="annual maximum of ERA5 daily-maximum 10 m wind gust",
        interpretation=(
            "Point-interpolated ERA5 daily-max 10 m wind gust (~25 km); ERA5 underestimates "
            "sharp convective downburst gusts, so this is a lower bound on the true hazard."
        ),
    ),
}


def annual_maxima(
    times: Sequence[str], values: Sequence[float | None]
) -> tuple[list[int], list[float]]:
    """Reduce a daily series to one maximum per year, skipping missing days.

    Returns (years, maxima), both sorted ascending by year.
    """
    by_year: dict[int, float] = {}
    for iso_date, value in zip(times, values):
        if value is None:
            continue
        year = int(iso_date[:4])
        by_year[year] = value if year not in by_year else max(by_year[year], value)
    years = sorted(by_year)
    return years, [by_year[y] for y in years]


def build_hazard_stat(
    years: Sequence[int],
    maxima: Sequence[float],
    *,
    hazard: Hazard,
    latitude: float,
    longitude: float,
    timezone: str,
    return_periods: Sequence[int] = (10, 50, 100),
) -> HazardStat:
    """Assemble a fully-provenanced HazardStat from annual maxima (pure; no network)."""
    cfg = _HAZARD_VARS[hazard]
    return HazardStat(
        variable=cfg.daily_var,
        statistic_definition=cfg.statistic_definition,
        unit=cfg.unit,
        source="Open-Meteo Archive (ERA5 reanalysis)",
        model="era5",
        native_resolution_deg=_ERA5_RESOLUTION_DEG,
        captures_diurnal_peak=True,
        timezone=timezone,
        latitude=latitude,
        longitude=longitude,
        n_years=len(maxima),
        record_start_year=years[0],
        record_end_year=years[-1],
        record_max=max(maxima),
        # 90% bootstrap band on every level: a 100-yr estimate from ~60 maxima
        # has real sampling noise, and the report must say how much.
        return_levels=return_levels_with_ci(maxima, return_periods),
        is_bias_corrected=False,
        representativeness=Representativeness.POINT_INTERPOLATED_REANALYSIS,
        interpretation=cfg.interpretation,
    )


@lru_cache(maxsize=256)
def climatology_hazard_stat(
    latitude: float,
    longitude: float,
    hazard: Hazard,
    *,
    start_year: int = 1960,
    end_year: int = 2022,
    return_periods: tuple[int, ...] = (10, 50, 100),
) -> HazardStat:
    """Live: fetch ERA5 daily history from the Open-Meteo Archive → GEV → HazardStat.

    Cached (lru_cache): historical archive data is static, so the fitted statistic
    for a (location, hazard) never changes — repeat calls are instant and make no
    network request (also a denial-of-wallet guard on the free Archive tier).
    """
    from tools.validation import validate_coordinates

    validate_coordinates(latitude, longitude)
    cfg = _HAZARD_VARS[hazard]
    params: dict[str, str | int | float] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "daily": cfg.daily_var,
        "models": "era5",
        "timezone": "auto",  # local-day boundaries (matters for daily precip totals)
    }
    try:
        response = httpx.get(ARCHIVE_URL, params=params, timeout=60)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ClimatologyError(f"Open-Meteo Archive request failed: {exc}") from exc

    data = response.json()
    daily = data["daily"]
    years, maxima = annual_maxima(daily["time"], daily[cfg.daily_var])
    return build_hazard_stat(
        years,
        maxima,
        hazard=hazard,
        latitude=latitude,
        longitude=longitude,
        timezone=data.get("timezone", "auto"),
        return_periods=return_periods,
    )
