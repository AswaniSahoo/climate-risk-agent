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

import numpy as np
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
