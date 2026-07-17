"""Non-stationary GEV: location parameter drifts linearly with a covariate.

Why: the stationary fit in hazard_stats.py treats 1960 and 2022 as draws from
the SAME distribution. Under warming that understates today's risk. Here the
GEV location is mu(x) = mu0 + slope * (x - x_mean) with x = year, fitted by
maximum likelihood, and a likelihood-ratio test against the nested stationary
fit says whether the trend is signal (p < 0.05) or noise.

Convention: scipy's shape `c` throughout (c = -xi in the Coles textbook
parameterization). All densities/quantiles go through scipy.stats.genextreme,
so the two fits are directly comparable and support violations surface as
-inf log-likelihoods the optimizer walks away from.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, genextreme

from tools.hazard_stats import ReturnLevel, _fit


@dataclass(frozen=True)
class GevTrendFit:
    """A fitted non-stationary GEV plus the trend-vs-noise verdict.

    `mu0` is the location AT THE MEAN covariate (fit is centered for numerical
    conditioning); `loc_at(x)` maps back to any year.
    """

    c: float          # scipy shape (= -xi)
    mu0: float        # location at x_mean
    slope: float      # location change per covariate unit (per year)
    sigma: float      # scale
    x_mean: float
    lr_statistic: float  # 2 * (ll_trend - ll_stationary), clamped at 0
    p_value: float       # chi-squared(1) tail probability of lr_statistic
    covariate: tuple[float, ...]  # the grid the fit was conditioned on (for bootstrap)

    @property
    def slope_per_decade(self) -> float:
        return self.slope * 10.0

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def loc_at(self, x: float) -> float:
        return self.mu0 + self.slope * (x - self.x_mean)


def _mle(
    y: np.ndarray, xc: np.ndarray, x0: Sequence[float]
) -> tuple[np.ndarray, float]:
    """Maximize the drifting-location GEV likelihood from start point `x0`
    (= [c, mu0, slope, log_sigma]); returns (params, log-likelihood).

    Nelder-Mead: derivative-free, robust for the GEV's bounded-support
    likelihood, and 4 parameters is tiny. log_sigma keeps scale positive.
    """

    def nll(params: np.ndarray) -> float:
        c, mu0, slope, log_sigma = params
        sigma = float(np.exp(log_sigma))
        ll = genextreme.logpdf(y, c, loc=mu0 + slope * xc, scale=sigma).sum()
        return float(-ll) if np.isfinite(ll) else float("inf")

    result = minimize(
        nll, x0=np.asarray(x0, dtype=float), method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000},
    )
    return result.x, -float(result.fun)


def fit_gev_trend(
    annual_maxima: "Sequence[float] | np.ndarray",
    covariate: "Sequence[float] | np.ndarray",
) -> GevTrendFit:
    """MLE fit of a GEV with linearly drifting location, plus LR test.

    Warm-started from the stationary fit — its optimum is a point in this
    model's parameter space at slope = 0, so the optimizer starts on the
    likelihood ridge instead of in the void.
    """
    y = np.asarray(annual_maxima, dtype=float)
    x = np.asarray(covariate, dtype=float)
    if y.size != x.size:
        raise ValueError(f"maxima ({y.size}) and covariate ({x.size}) length mismatch")
    x_mean = float(x.mean())
    xc = x - x_mean

    c0, loc0, scale0 = _fit(y)
    stationary_ll = float(genextreme.logpdf(y, c0, loc=loc0, scale=scale0).sum())

    params, trend_ll = _mle(y, xc, [c0, loc0, 0.0, np.log(scale0)])
    c, mu0, slope, log_sigma = params

    # Nested models: clamp tiny negative LR from optimizer tolerance to 0.
    lr = max(0.0, 2.0 * (trend_ll - stationary_ll))
    return GevTrendFit(
        c=float(c), mu0=float(mu0), slope=float(slope),
        sigma=float(np.exp(log_sigma)), x_mean=x_mean,
        lr_statistic=lr, p_value=float(chi2.sf(lr, df=1)),
        covariate=tuple(float(v) for v in x),
    )


def trend_return_levels(
    fit: GevTrendFit,
    *,
    at: float,
    return_periods: Sequence[int] = (10, 50, 100),
    n_boot: int = 0,
    seed: int = 0,
    alpha: float = 0.10,
) -> list[ReturnLevel]:
    """Return levels of the fitted GEV EVALUATED at covariate `at`.

    `at=2022` gives "effective present-day" levels — what the T-year event
    magnitude is NOW, given the fitted drift. n_boot > 0 adds a (1 - alpha)
    parametric-bootstrap band; 0 keeps the fast point estimate.
    """
    loc = fit.loc_at(at)
    quantiles = {int(t): 1.0 - 1.0 / t for t in return_periods}
    point = {
        t: float(genextreme.ppf(q, fit.c, loc=loc, scale=fit.sigma))
        for t, q in quantiles.items()
    }
    if n_boot <= 0:
        return [
            ReturnLevel(return_period_years=t, level=level)
            for t, level in point.items()
        ]

    # Simulate on the SAME covariate grid the fit was conditioned on — the
    # slope's uncertainty depends on the grid's spread, so a proxy grid would
    # mis-state the band. Refits warm-start at the parent optimum (each
    # resample's optimum is nearby), skipping the inner stationary fit.
    x_grid = np.asarray(fit.covariate, dtype=float)
    xc = x_grid - fit.x_mean
    parent = [fit.c, fit.mu0, fit.slope, np.log(fit.sigma)]
    at_c = at - fit.x_mean
    rng = np.random.default_rng(seed)
    boot: dict[int, list[float]] = {t: [] for t in quantiles}
    for _ in range(n_boot):
        sample = genextreme.rvs(
            fit.c, loc=fit.mu0 + fit.slope * xc, scale=fit.sigma,
            size=x_grid.size, random_state=rng,
        )
        (bc, bmu0, bslope, blog_sigma), _ = _mle(sample, xc, parent)
        bloc = bmu0 + bslope * at_c
        for t, q in quantiles.items():
            boot[t].append(
                float(genextreme.ppf(q, bc, loc=bloc, scale=float(np.exp(blog_sigma))))
            )

    lo_pct, hi_pct = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    out = []
    for t in quantiles:
        lo, hi = np.percentile(boot[t], [lo_pct, hi_pct])
        out.append(ReturnLevel(
            return_period_years=t, level=point[t],
            ci_low=float(min(lo, point[t])), ci_high=float(max(hi, point[t])),
        ))
    return out
