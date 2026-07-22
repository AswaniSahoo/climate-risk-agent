"""Historical climate extremes from the Open-Meteo Archive (ERA5 reanalysis).

Replaces the WeatherBench2 zarr path. ERA5 **daily** maxima, point-interpolated to
the requested location, so the statistic sits at the right place and captures the
diurnal peak — the two things the coarse 6-hourly WB2 grid got wrong (see
docs/hazard-data-source.md for the measured comparison).

Split like get_forecast: pure `annual_maxima` + `build_hazard_stat` (unit-tested
offline) and the `climatology_hazard_stat` network edge (mocked in tests).
"""
from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx

from agent.contracts import Hazard
from tools.gev_trend import GevTrendFit, fit_gev_trend, trend_return_levels
from tools.hazard_stats import (
    HazardStat,
    Representativeness,
    ReturnLevel,
    TrendInfo,
    return_levels_with_ci,
)

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


# Below this many annual maxima a trend fit is statistically meaningless
# (4 free parameters on a handful of points); we keep the stationary fit.
_MIN_YEARS_FOR_TREND = 20

# 200 bootstrap refits keeps the first (uncached) call ~7 s; the 90% band is
# stable well below that.
_TREND_N_BOOT = 200


def _reported_levels(
    fit: GevTrendFit,
    years: Sequence[int],
    maxima: Sequence[float],
    return_periods: Sequence[int],
) -> tuple[list[ReturnLevel], TrendInfo]:
    """Decide WHICH return levels the report carries, given the trend verdict.

    Standard practice (Katz et al. 2002 "effective return level"; extRemes;
    NEVA): when the likelihood-ratio test prefers the drifting-location model,
    report levels EVALUATED AT THE LATEST YEAR — "today's 100-year event", not
    the 1960–2022 average. When the trend is statistically noise, injecting it
    would fabricate drift, so the stationary fit stays — but the TrendInfo
    always carries the slope and p-value, so the report shows the test RAN.
    """
    trend = TrendInfo(
        slope_per_decade=fit.slope_per_decade,
        p_value=fit.p_value,
        significant=fit.significant,
        evaluated_at_year=years[-1] if fit.significant else None,
    )
    if fit.significant:
        levels = trend_return_levels(
            fit, at=years[-1], return_periods=return_periods, n_boot=_TREND_N_BOOT,
        )
    else:
        levels = return_levels_with_ci(maxima, return_periods)
    return levels, trend


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
    if len(maxima) >= _MIN_YEARS_FOR_TREND:
        fit = fit_gev_trend(maxima, years)
        levels, trend = _reported_levels(fit, years, maxima, return_periods)
    else:
        levels, trend = return_levels_with_ci(maxima, return_periods), None
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
        return_levels=levels,
        trend=trend,
        is_bias_corrected=False,
        representativeness=Representativeness.POINT_INTERPOLATED_REANALYSIS,
        interpretation=cfg.interpretation,
    )


_log = logging.getLogger(__name__)

# Disk cache for the fitted GEV stat. The 1960–2022 ERA5 record is STATIC, so a
# (location, hazard) statistic never changes — persisting it across process
# restarts turns every cold-start (Cloud Run scale events) repeat query from a
# ~20 s archive fetch + bootstrap into an instant disk read. The in-process
# lru_cache below still handles same-instance repeats; this survives restarts.
_STAT_CACHE_DIR = Path(os.environ.get("CLIMATOLOGY_CACHE_DIR", "data/cache/climatology"))


def _stat_cache_enabled() -> bool:
    # Hermetic tests: never touch the shared on-disk cache during pytest (mirrors
    # the corpus-download guard in ui/app.py), so network-mocked tests stay exact.
    return not os.environ.get("PYTEST_CURRENT_TEST")


def _stat_cache_key(
    latitude: float,
    longitude: float,
    hazard: Hazard,
    start_year: int,
    end_year: int,
    return_periods: tuple[int, ...],
) -> str:
    # ~4 dp ≈ 11 m, well inside the ERA5 ~25 km grid cell, so nearby coords that
    # resolve to the same cell still share a cache entry.
    raw = (
        f"{round(latitude, 4)}|{round(longitude, 4)}|{hazard.value}|"
        f"{start_year}|{end_year}|{tuple(return_periods)}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _stat_cache_load(key: str) -> HazardStat | None:
    if not _stat_cache_enabled():
        return None
    path = _STAT_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return HazardStat.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — a corrupt/stale cache file must never break a request
        _log.warning("climatology cache read failed (%s) — refetching", exc)
        return None


def _stat_cache_store(key: str, stat: HazardStat) -> None:
    if not _stat_cache_enabled():
        return
    try:
        _STAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (_STAT_CACHE_DIR / f"{key}.json").write_text(
            stat.model_dump_json(), encoding="utf-8"
        )
    except OSError as exc:
        _log.warning("climatology cache write failed (%s) — continuing", exc)


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

    Two cache tiers, because the historical archive is static so the fitted
    statistic for a (location, hazard) never changes:
    - in-process `lru_cache`: instant repeats within one running instance;
    - on-disk cache (`_STAT_CACHE_DIR`): survives restarts / cold starts, so a
      re-scheduled Cloud Run container skips the ~20 s archive fetch + bootstrap.
    Both also act as denial-of-wallet guards on the free Archive tier.
    """
    from tools.validation import validate_coordinates

    validate_coordinates(latitude, longitude)

    cache_key = _stat_cache_key(
        latitude, longitude, hazard, start_year, end_year, return_periods
    )
    cached = _stat_cache_load(cache_key)
    if cached is not None:
        return cached

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
    stat = build_hazard_stat(
        years,
        maxima,
        hazard=hazard,
        latitude=latitude,
        longitude=longitude,
        timezone=data.get("timezone", "auto"),
        return_periods=return_periods,
    )
    _stat_cache_store(cache_key, stat)
    return stat
