"""Tests for the non-stationary (trend-aware) GEV fit.

All synthetic + seeded: we generate annual maxima from a KNOWN shifting GEV and
check the fit recovers the trend, the likelihood-ratio test separates trend
from noise, and effective return levels move the right way.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import genextreme

from tools.gev_trend import GevTrendFit, fit_gev_trend, trend_return_levels

YEARS = np.arange(1960, 2023)  # 63 "years", like the real ERA5 window

# Known truth: 0.3 °C/decade warming in the location parameter.
TRUE_SLOPE = 0.03
TRUE_C = 0.1  # scipy convention (c = -xi)
TRUE_SIGMA = 1.5


def _trending_maxima(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    loc = 38.0 + TRUE_SLOPE * (YEARS - YEARS.mean())
    return genextreme.rvs(TRUE_C, loc=loc, scale=TRUE_SIGMA,
                          size=YEARS.size, random_state=rng)


def _flat_maxima(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return genextreme.rvs(TRUE_C, loc=38.0, scale=TRUE_SIGMA,
                          size=YEARS.size, random_state=rng)


def test_fit_recovers_known_trend():
    fit = fit_gev_trend(_trending_maxima(), YEARS)
    assert fit.slope == pytest.approx(TRUE_SLOPE, abs=0.02)
    assert fit.sigma == pytest.approx(TRUE_SIGMA, abs=0.5)


def test_trending_data_is_significant():
    fit = fit_gev_trend(_trending_maxima(), YEARS)
    assert fit.p_value < 0.05
    assert fit.lr_statistic > 0.0


def test_flat_data_is_not_significant():
    fit = fit_gev_trend(_flat_maxima(), YEARS)
    assert fit.p_value > 0.05


def test_nonstationary_likelihood_never_worse_than_stationary():
    # Nested models: the trend fit has one extra free parameter, so its
    # log-likelihood must be >= the stationary one (lr >= 0 after clamping).
    for maxima in (_trending_maxima(), _flat_maxima()):
        fit = fit_gev_trend(maxima, YEARS)
        assert fit.lr_statistic >= 0.0


def test_loc_at_evaluates_the_trend_line():
    fit = fit_gev_trend(_trending_maxima(), YEARS)
    assert fit.loc_at(2022) > fit.loc_at(1960)
    # slope consistency: difference over 62 years == slope * 62
    span = fit.loc_at(2022) - fit.loc_at(1960)
    assert span == pytest.approx(fit.slope * 62, rel=1e-9)


def test_effective_levels_at_latest_year_exceed_earliest():
    fit = fit_gev_trend(_trending_maxima(), YEARS)
    now = trend_return_levels(fit, at=2022, return_periods=(10, 100))
    then = trend_return_levels(fit, at=1960, return_periods=(10, 100))
    for level_now, level_then in zip(now, then):
        assert level_now.level > level_then.level
        assert level_now.return_period_years == level_then.return_period_years


def test_effective_levels_carry_bootstrap_ci():
    fit = fit_gev_trend(_trending_maxima(), YEARS)
    levels = trend_return_levels(fit, at=2022, return_periods=(10, 100),
                                 n_boot=50, seed=0)
    for rl in levels:
        assert rl.ci_low is not None and rl.ci_high is not None
        assert rl.ci_low <= rl.level <= rl.ci_high


def test_ci_off_by_default():
    fit = fit_gev_trend(_trending_maxima(), YEARS)
    levels = trend_return_levels(fit, at=2022, return_periods=(10,))
    assert levels[0].ci_low is None and levels[0].ci_high is None


def test_fit_is_deterministic():
    a = fit_gev_trend(_trending_maxima(), YEARS)
    b = fit_gev_trend(_trending_maxima(), YEARS)
    assert a.slope == b.slope and a.p_value == b.p_value


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        fit_gev_trend([1.0, 2.0, 3.0], YEARS)


def test_fit_exposes_slope_per_decade():
    fit = fit_gev_trend(_trending_maxima(), YEARS)
    assert fit.slope_per_decade == pytest.approx(fit.slope * 10)


def test_result_type():
    fit = fit_gev_trend(_trending_maxima(), YEARS)
    assert isinstance(fit, GevTrendFit)
