"""Tests for bootstrap confidence intervals on GEV return levels.

Evaluator gap #3: a 100-year level fitted from ~60 annual maxima is a point
estimate with real sampling noise — reporting it without a band overclaims
precision. Parametric bootstrap: resample from the FITTED distribution, refit,
take percentile bands. Seeded -> deterministic tests.
"""
import numpy as np
import pytest
from scipy.stats import genextreme

from tools.hazard_stats import ReturnLevel, return_levels_with_ci

# Synthetic maxima from a known GEV (c=-0.1, loc=40, scale=2) — heat-like, °C.
_RNG = np.random.default_rng(7)
_MAXIMA_60 = list(genextreme.rvs(-0.1, loc=40.0, scale=2.0, size=60, random_state=_RNG))
_MAXIMA_20 = _MAXIMA_60[:20]


def test_ci_brackets_the_point_estimate():
    levels = return_levels_with_ci(_MAXIMA_60, (10, 50, 100), n_boot=200, seed=0)
    for r in levels:
        assert r.ci_low is not None and r.ci_high is not None
        assert r.ci_low <= r.level <= r.ci_high
        assert r.ci_low < r.ci_high  # a real band, not a collapsed point


def test_rarer_events_have_wider_bands():
    levels = {r.return_period_years: r for r in
              return_levels_with_ci(_MAXIMA_60, (10, 100), n_boot=200, seed=0)}
    width = lambda r: r.ci_high - r.ci_low
    assert width(levels[100]) > width(levels[10])  # extrapolating further = less certain


def test_more_data_shrinks_the_band():
    w = lambda maxima: (lambda r: r.ci_high - r.ci_low)(
        return_levels_with_ci(maxima, (100,), n_boot=200, seed=0)[0]
    )
    assert w(_MAXIMA_60) < w(_MAXIMA_20)


def test_seeded_bootstrap_is_deterministic():
    a = return_levels_with_ci(_MAXIMA_60, (50,), n_boot=100, seed=42)[0]
    b = return_levels_with_ci(_MAXIMA_60, (50,), n_boot=100, seed=42)[0]
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_return_level_without_ci_still_validates():
    r = ReturnLevel(return_period_years=100, level=46.0)  # backward compatible
    assert r.ci_low is None and r.ci_high is None
