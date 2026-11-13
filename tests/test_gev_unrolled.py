"""The unrolled GEV likelihood must be scipy's own, bit for bit.

tools/hazard_stats.py and tools/gev_trend.py fit through a likelihood that runs
the elementwise operations scipy 1.17.1 runs for genextreme, at about a quarter
of the cost per call (74 to 92 us down to 18 to 20 us on 63 annual maxima,
measured 2026-11-12). That is only safe while every number it hands the
optimizer is the float scipy hands it, so these tests hold the two side by
side: on the shipped Berlin series, on synthetic series with fixed seeds (some
built to sit on the edge of the GEV support), and on the likelihood itself at
parameters inside, on and outside the support.

A scipy release that changes those internals fails here, loudly. Production
does not drift meanwhile: `unrolled_active` sends every fit through scipy's own
path on any scipy but the pinned one, until this file passes and the pin moves.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy
from scipy.stats import genextreme

import tools.gev_trend as gt
import tools.hazard_stats as hs
from agent.contracts import Hazard
from tools import climatology as clim

SEED_FILE = Path(__file__).resolve().parents[1] / "tools" / "cache_seed" / "climatology.json"
PERIODS = (2, 10, 50, 100)
YEARS = list(range(1960, 2023))


@pytest.fixture
def scipy_path(monkeypatch):
    """Every fit through scipy's own likelihood, as on an unpinned scipy."""
    monkeypatch.setattr(hs, "unrolled_active", lambda: False)


def _both(fn, monkeypatch):
    """`fn()` through the unrolled likelihood, then through scipy's own."""
    if not hs.unrolled_active():
        pytest.skip(f"scipy {scipy.__version__} is not the pinned {hs.UNROLLED_FOR_SCIPY}: every "
                    "fit takes scipy's own path here, so there is no second path to compare")
    fast = fn()
    with monkeypatch.context() as m:
        m.setattr(hs, "unrolled_active", lambda: False)
        slow = fn()
    return fast, slow


def _synthetic() -> dict[str, np.ndarray]:
    """Fixed-seed series that push the fit to its awkward places."""
    rng = np.random.default_rng(20261112)
    bounded = genextreme.rvs(0.45, loc=35.0, scale=1.5, size=63, random_state=rng)
    outlier = bounded.copy()
    outlier[np.argmax(outlier)] += 25.0  # far past the upper end the rest imply
    return {
        "gumbel": genextreme.rvs(0.0, loc=30.0, scale=2.0, size=63, random_state=rng),
        "heavy_tail": genextreme.rvs(-0.35, loc=40.0, scale=8.0, size=63, random_state=rng),
        "bounded_tail": bounded,
        "outlier_past_the_end": outlier,
        "ties": np.round(genextreme.rvs(0.1, loc=30.0, scale=3.0, size=63, random_state=rng)),
        "tiny_spread": genextreme.rvs(0.2, loc=35.0, scale=0.02, size=63, random_state=rng),
        "drifting": (genextreme.rvs(0.1, loc=30.0, scale=1.0, size=63, random_state=rng)
                     + 0.06 * (np.arange(63) - 31)),
    }


SYNTHETIC = _synthetic()


def _berlin_rows() -> list[dict]:
    return json.loads(SEED_FILE.read_text(encoding="utf-8"))["entries"]


def _same(a: float, b: float) -> bool:
    """Equal as floats, NaN equal to NaN, and the same sign on a zero."""
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    return a == b and np.signbit(a) == np.signbit(b)


# --- the pin --------------------------------------------------------------------

# uv.lock resolves scipy 1.17.1 below Python 3.12 and 1.18.0 from 3.12 on. The
# Dockerfile and CI run 3.11, so that is where a scipy bump must fail loudly; on a
# newer local Python the other scipy is expected and production never runs it.
@pytest.mark.skipif(sys.version_info >= (3, 12),
                    reason="uv.lock resolves another scipy from Python 3.12 on; the image and CI run 3.11")
def test_the_unrolled_likelihood_is_pinned_to_the_scipy_it_was_verified_on():
    assert scipy.__version__ == hs.UNROLLED_FOR_SCIPY, (
        f"scipy is {scipy.__version__}, the unrolled GEV likelihood was verified against "
        f"{hs.UNROLLED_FOR_SCIPY}. Until this file passes on {scipy.__version__} and "
        "UNROLLED_FOR_SCIPY moves to it, every fit takes scipy's own (slower) path. "
        "Moving the pin retires every shipped fit key: run "
        "`python -m scripts.precompute_climatology --refit` and check the values are unchanged.")


@pytest.fixture
def fresh_guard():
    hs.unrolled_active.cache_clear()
    yield
    hs.unrolled_active.cache_clear()


def test_another_scipy_takes_scipys_own_path_and_says_so_once(fresh_guard, monkeypatch, caplog):
    monkeypatch.setattr(scipy, "__version__", "1.99.0")
    data = SYNTHETIC["gumbel"]

    with caplog.at_level(logging.WARNING, logger="tools.hazard_stats"):
        assert hs.unrolled_active() is False
        assert hs.unrolled_active() is False
        fitted = hs._fit(data)

    assert fitted == tuple(float(v) for v in genextreme.fit(data))
    warnings = [r for r in caplog.records if "1.99.0" in r.getMessage()]
    assert len(warnings) == 1 and hs.UNROLLED_FOR_SCIPY in warnings[0].getMessage()


# --- the likelihood itself, value for value --------------------------------------

_DATA = {
    "berlin_heat": np.asarray(next(r for r in _berlin_rows() if r["hazard"] == "heatwave")
                              ["series"]["maxima"], dtype=float),
    "on_the_upper_end": np.array([0.3, 1.0, 1.7, 2.0, 0.0, -1.5]),  # 2.0 = 1/c for c = 0.5
    **SYNTHETIC,
}
_THETAS = [
    (0.3, 30.0, 2.0), (0.45, 34.0, 1.0), (-0.4, 45.0, 3.0), (-0.35, 40.0, 8.0),
    (0.0, 30.0, 2.0), (-0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.5, 0.0, 1.0), (2.0, 30.0, 0.5),
    (1e-300, 30.0, 2.0), (-1e-300, 30.0, 2.0), (0.2, 35.0, 0.02),
    (0.2, 30.0, 0.0), (0.2, 30.0, -1.0), (np.nan, 30.0, 2.0), (np.inf, 30.0, 2.0),
    (0.2, np.nan, 2.0), (0.2, 30.0, np.inf),
]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scale 0, NaN loc: on purpose
@pytest.mark.parametrize("name", sorted(_DATA))
def test_the_penalized_likelihood_is_scipys_at_every_test_point(name):
    data = _DATA[name]
    for theta in _THETAS:
        params = np.array(theta, dtype=float)
        assert _same(hs._gev_penalized_nnlf(params, data),
                     genextreme._penalized_nnlf(params, data)), (name, theta)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scale 0, NaN loc: on purpose
@pytest.mark.parametrize("name", sorted(_DATA))
def test_the_drifting_log_likelihood_is_scipys_at_every_test_point(name):
    y = _DATA[name]
    xc = np.arange(y.size, dtype=float) - (y.size - 1) / 2
    for c, mu0, scale in _THETAS:
        for slope in (0.0, 0.05, -0.3):
            loc = mu0 + slope * xc
            assert _same(gt._gev_loglik(y, c, loc, scale),
                         genextreme.logpdf(y, c, loc=loc, scale=scale).sum()), (name, c, mu0, scale)


def test_the_awkward_points_really_are_awkward():
    """The grid above must reach the branches that matter: points past either end
    of the support, the exact upper end (c x = 1), c = 0 and c = 1."""
    reached = set()
    for name, data in _DATA.items():
        for c, loc, scale in _THETAS:
            if not (np.isfinite(c) and np.isfinite(loc) and np.isfinite(scale)) or scale <= 0:
                continue
            x = (data - loc) / scale
            a, b = hs._gev_support(np.asarray(c))
            if np.any(x > b):
                reached.add("past the upper end")
            if np.any(x < a):
                reached.add("below the lower end")
            if c != 0 and np.any(c * x == 1):
                reached.add("exactly on the upper end")
            if c == 0:
                reached.add("c = 0")
            if c == 1 and np.any(x == 1):
                reached.add("c = 1 at x = 1")
    assert reached == {"past the upper end", "below the lower end", "exactly on the upper end",
                       "c = 0", "c = 1 at x = 1"}


# --- whole fits, bit for bit ----------------------------------------------------

@pytest.mark.parametrize("name", sorted(SYNTHETIC))
def test_a_stationary_fit_and_its_band_are_scipys(name, monkeypatch):
    data = SYNTHETIC[name]

    assert hs._fit(data) == tuple(float(v) for v in genextreme.fit(data))
    fast, slow = _both(lambda: hs.return_levels_with_ci(data, PERIODS, n_boot=12), monkeypatch)
    assert fast == slow


@pytest.mark.parametrize("name", sorted(SYNTHETIC))
def test_a_drifting_fit_and_its_band_are_scipys(name, monkeypatch):
    data = SYNTHETIC[name]

    def trend():
        fit = gt.fit_gev_trend(data, YEARS)
        return fit, gt.trend_return_levels(fit, at=YEARS[-1], return_periods=PERIODS, n_boot=12)

    fast, slow = _both(trend, monkeypatch)
    assert fast == slow


def test_the_fixtures_include_a_significant_trend_and_a_fit_that_leaves_the_support():
    """Without these the two tests above would never exercise the drifting
    bootstrap a real warming series takes, or the penalty branch."""
    assert gt.fit_gev_trend(SYNTHETIC["drifting"], YEARS).significant
    if not hs.unrolled_active():
        pytest.skip("the penalty-branch spy sits on the unrolled likelihood, which this scipy "
                    "never calls")

    seen = []
    real = hs._gev_penalized_nnlf

    def spy(theta, data):
        c, loc, scale = theta
        if np.isfinite(c) and scale > 0:
            a, b = hs._gev_support(np.asarray(c))
            x = (data - loc) / scale
            seen.append(bool(np.any((x < a) | (x > b))))
        return real(theta, data)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(hs, "_gev_penalized_nnlf", spy)
        hs._fit(SYNTHETIC["outlier_past_the_end"])
    assert any(seen), "the optimizer never tried a point outside the support"


def test_berlin_bootstrap_refits_are_scipys_one_by_one(monkeypatch):
    """The first 40 bootstrap refits of each shipped Berlin series, refit by refit."""
    for row in _berlin_rows():
        s = row["series"]
        data = np.asarray(s["maxima"], dtype=float)
        fit = gt.fit_gev_trend(data, s["years"])
        rng = np.random.default_rng(0)
        if fit.significant:
            xc = np.asarray(fit.covariate) - fit.x_mean
            parent = [fit.c, fit.mu0, fit.slope, np.log(fit.sigma)]
            for _ in range(40):
                sample = genextreme.rvs(fit.c, loc=fit.mu0 + fit.slope * xc, scale=fit.sigma,
                                        size=xc.size, random_state=rng)
                fast, slow = _both(lambda: gt._mle(sample, xc, parent), monkeypatch)
                assert fast[1] == slow[1] and fast[0].tobytes() == slow[0].tobytes()
        else:
            c, loc, scale = genextreme.fit(data)
            for _ in range(40):
                sample = genextreme.rvs(c, loc=loc, scale=scale, size=data.size, random_state=rng)
                fast, slow = _both(lambda: hs._fit(sample), monkeypatch)
                assert fast == slow


@pytest.mark.parametrize("row", _berlin_rows(), ids=lambda row: f"berlin-{row['hazard']}")
def test_the_shipped_berlin_fits_come_out_exactly_as_shipped(row):
    s = row["series"]
    stat = clim.build_hazard_stat(s["years"], s["maxima"], hazard=Hazard(row["hazard"]),
                                  latitude=row["latitude"], longitude=row["longitude"],
                                  timezone=s["timezone"], return_periods=PERIODS)

    assert json.loads(stat.model_dump_json()) == row["value"]
