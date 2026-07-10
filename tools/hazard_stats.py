"""GEV hazard statistics — the domain moat.

Turns a series of annual maxima (e.g. each year's wettest day or hottest day)
into extreme-value statistics: return levels and return periods, via a
Generalized Extreme Value (GEV) fit — the distribution Extreme Value Theory
prescribes for block maxima.

This is the math core. The ERA5/WeatherBench2 data feed that produces the
annual maxima is a separate layer (wired next).
"""
from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np
from pydantic import BaseModel
from scipy.stats import genextreme


def _fit(annual_maxima: Sequence[float]) -> tuple[float, float, float]:
    """Fit a GEV to the annual maxima; returns (shape c, loc, scale)."""
    c, loc, scale = genextreme.fit(np.asarray(annual_maxima, dtype=float))
    return float(c), float(loc), float(scale)


def return_level(annual_maxima: Sequence[float], return_period_years: float) -> float:
    """The value expected to be exceeded once every `return_period_years` years.

    e.g. return_level(maxima, 100) is the "100-year" event magnitude.
    """
    c, loc, scale = _fit(annual_maxima)
    quantile = 1.0 - 1.0 / return_period_years
    return float(genextreme.ppf(quantile, c, loc, scale))


def return_period(annual_maxima: Sequence[float], value: float) -> float:
    """Average number of years between events at least as extreme as `value`."""
    c, loc, scale = _fit(annual_maxima)
    exceedance_prob = 1.0 - float(genextreme.cdf(value, c, loc, scale))
    if exceedance_prob <= 0.0:
        return float("inf")
    return 1.0 / exceedance_prob


class ReturnLevel(BaseModel):
    """One (return period → level) pair, e.g. the 100-year event magnitude."""

    return_period_years: int
    level: float


class Representativeness(str, Enum):
    """How faithfully the statistic represents the requested point's TRUE extreme.

    The honest label is the whole point: a reanalysis point value is not a station
    observation, and a coarse/aliased grid value is not a local extreme at all.
    """

    STATION_CALIBRATED = "station_calibrated"  # bias-corrected to observations
    POINT_INTERPOLATED_REANALYSIS = "point_interpolated_reanalysis"  # ERA5 ~25km, point-interp, no station cal
    REGIONAL_GRID_SIGNAL = "regional_grid_signal"  # a named coarse grid cell
    NOT_REPRESENTATIVE = "not_representative"  # wrong cell / temporally aliased


class HazardStat(BaseModel):
    """Typed hazard statistics for one variable at one location, carrying the
    provenance a reader needs to judge how far to trust the numbers.

    `record_max` sits next to `return_levels` on purpose: if a 100-year level is
    at or below the observed record, the tail is degenerate and the number lies.
    """

    variable: str
    statistic_definition: str
    unit: str
    source: str
    model: str
    native_resolution_deg: float
    captures_diurnal_peak: bool
    timezone: str
    latitude: float
    longitude: float
    n_years: int
    record_start_year: int
    record_end_year: int
    record_max: float
    return_levels: list[ReturnLevel]
    is_bias_corrected: bool
    representativeness: Representativeness
    interpretation: str


def return_levels(
    annual_maxima: Sequence[float], return_periods: Sequence[int] = (10, 50, 100)
) -> list[ReturnLevel]:
    """Fit a GEV once and report the return level for each requested period."""
    return [
        ReturnLevel(return_period_years=int(t), level=return_level(annual_maxima, t))
        for t in return_periods
    ]
