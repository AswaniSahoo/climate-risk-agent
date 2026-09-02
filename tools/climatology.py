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
from tools.cache_backend import JsonCache
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

# Bootstrap refits per fit. MEASURED cost/precision tradeoff on a 63-year
# series (2026-07-23): n=300 -> 27.7 s, 100yr CI 195-330; n=150 -> 12.9 s,
# CI 193-319 (within a few percent); n=50 -> 5.3 s but CI 186-280, visibly
# too narrow in the tail. So 150 is the floor that preserves the band.
# Default stays at the higher value so published numbers keep full precision;
# an interactive deploy opts into the cheaper one via CRG_BOOTSTRAP_N.
#
# CRG_BOOTSTRAP_N (env, read ONCE at import) overrides both counts at the same
# time. It is part of the fit-cache key, so lowering it does not read back
# entries fitted at the default — it refits them. Documented in .env.example,
# README.md and DEPLOY.md; scripts/prewarm.py prints the resolved values.
_N_BOOT = int(os.environ.get("CRG_BOOTSTRAP_N", "300"))
_TREND_N_BOOT = int(os.environ.get("CRG_BOOTSTRAP_N", "200"))


def bootstrap_settings() -> dict[str, int]:
    """The bootstrap sizes this process resolved, so a run can print them."""
    return {"n_boot": _N_BOOT, "trend_n_boot": _TREND_N_BOOT}


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
        levels = return_levels_with_ci(maxima, return_periods, n_boot=_N_BOOT)
    return levels, trend


def build_hazard_stat(
    years: Sequence[int],
    maxima: Sequence[float],
    *,
    hazard: Hazard,
    latitude: float,
    longitude: float,
    timezone: str,
    # The 2-year level is a band edge in agent/risk_bands.py (below it, a peak is
    # what an ordinary year does here), so it is fitted like any other level
    # rather than extrapolated off the bottom of the curve.
    return_periods: Sequence[int] = (2, 10, 50, 100),
) -> HazardStat:
    """Assemble a fully-provenanced HazardStat from annual maxima (pure; no network)."""
    cfg = _HAZARD_VARS[hazard]
    if len(maxima) >= _MIN_YEARS_FOR_TREND:
        fit = fit_gev_trend(maxima, years)
        levels, trend = _reported_levels(fit, years, maxima, return_periods)
    else:
        levels, trend = return_levels_with_ci(maxima, return_periods, n_boot=_N_BOOT), None
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

# Persistent cache for the FITTED stat (Cache v2 — see tools/cache_backend.py).
# The 1960–2022 ERA5 record is STATIC, so a (location, hazard) statistic never
# changes, and the expensive half is the FIT, not the fetch: ~42 s for a cold
# archive request PLUS 12.9 s of GEV fit + trend test + bootstrap at n_boot=150
# (measured). Caching the finished HazardStat skips both. Since v2 the entry
# goes through the shared backend, so on Cloud Run — where every replica has
# its own ephemeral disk — a warm entry survives a redeploy and is visible to
# the other replica instead of dying with the container.
_FIT_TTL_S = 365 * 24 * 3600  # the record is static; the TTL is hygiene, not staleness


# Every file whose CONTENT decides the numbers in a cached HazardStat. This
# module is in the list because it owns the pieces the other two do not: which
# ERA5 variable each hazard reads, the trend-vs-stationary rule in
# `_reported_levels`, the minimum years for a trend fit, and the assembly in
# `build_hazard_stat`. Editing any of those changes the statistic while
# hazard_stats.py and gev_trend.py sit byte-identical, which is exactly the
# stale entry this fingerprint exists to retire.
_FINGERPRINT_SOURCES = ("hazard_stats.py", "gev_trend.py", "climatology.py")
_SOURCE_DIR = Path(__file__).resolve().parent  # module attr so tests can redirect it


@lru_cache(maxsize=1)
def _code_fingerprint() -> str:
    """Identity of the code that PRODUCES the numbers, hashed from its source.

    Same trick as rag/corpus.py's chunk-cache fingerprint: edit the GEV fit, the
    trend test or the assembly here and every cached statistic invalidates
    itself, so nobody can forget to bump a version constant and ship stale
    return levels.
    """
    digest = hashlib.sha256()
    for name in _FINGERPRINT_SOURCES:
        digest.update((_SOURCE_DIR / name).read_bytes())
    return digest.hexdigest()[:16]


@lru_cache(maxsize=1)
def _fit_cache() -> JsonCache:
    """The hazard-fit namespace (memoised; tests swap this for a tmp_path one)."""
    return JsonCache("hazard_fit")


def _fit_cache_key(
    latitude: float,
    longitude: float,
    hazard: Hazard,
    start_year: int,
    end_year: int,
    return_periods: tuple[int, ...],
) -> str:
    # 2 dp ≈ 1.1 km, far inside the ERA5 ~25 km grid cell: two requests for the
    # same city collapse onto one entry instead of paying one fit each. n_boot
    # is in the key because a narrower bootstrap is a DIFFERENT statistic.
    raw = (
        f"{round(latitude, 2)}|{round(longitude, 2)}|{hazard.value}|"
        f"{start_year}|{end_year}|{tuple(return_periods)}|"
        f"{_N_BOOT}|{_TREND_N_BOOT}|{_code_fingerprint()}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@lru_cache(maxsize=256)
def climatology_hazard_stat(
    latitude: float,
    longitude: float,
    hazard: Hazard,
    *,
    start_year: int = 1960,
    end_year: int = 2022,
    return_periods: tuple[int, ...] = (2, 10, 50, 100),
) -> HazardStat:
    """Live: fetch ERA5 daily history from the Open-Meteo Archive → GEV → HazardStat.

    Two cache tiers, because the historical archive is static so the fitted
    statistic for a (location, hazard) never changes:
    - in-process `lru_cache`: instant repeats within one running instance;
    - the shared backend (`tools/cache_backend.py`): Upstash Redis when it is
      configured, local disk otherwise. It survives restarts AND crosses
      replicas, so a re-scheduled Cloud Run container skips the ~42 s archive
      fetch and the 12.9 s bootstrap instead of paying them again.
    Both also act as denial-of-wallet guards on the free Archive tier.
    """
    from tools.validation import validate_coordinates

    validate_coordinates(latitude, longitude)

    cache = _fit_cache()
    cache_key = _fit_cache_key(
        latitude, longitude, hazard, start_year, end_year, return_periods
    )
    cached = cache.get_model(cache_key, HazardStat)
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
    cache.set_model(cache_key, stat, ttl_s=_FIT_TTL_S)
    return stat
