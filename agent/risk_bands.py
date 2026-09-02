"""Risk bands: how RARE is this forecast peak *here*, not how big is the number.

Why this replaces the Day-1 cutoffs: a fixed threshold ("40 °C = high") is wrong
twice over. It ignores the place (45 °C is an ordinary May afternoon in Rourkela
and a national emergency in Berlin) and it ignores the 60+ years of ERA5
climatology this system already fits for every query. The defensible question is
where the forecast peak lands on THIS location's GEV return-level curve, which is
how catastrophe risk is priced.

Bands, by the return period of the forecast peak:

    T <  2 yr        -> LOW       an ordinary year does this
    2 <= T < 10 yr   -> MODERATE  a notable year
    10 <= T < 50 yr  -> HIGH      a decadal-plus event
    T >= 50 yr       -> SEVERE    rare, approaching record class

Between two fitted levels the return period is interpolated linearly in log(T),
the Gumbel plotting scale on which a GEV return-level curve is a straight line
for the Gumbel case and near-straight across one bracket. Nothing is
extrapolated past the ends of the fitted curve: below the smallest fitted level
the band is read off that edge and the sentence says so.

`band_from_return_period` reads `hazard_stat.return_levels` as given, which is
already the EFFECTIVE curve (evaluated at the latest year) whenever the
non-stationarity test found a significant trend, see
`tools/climatology.py::_reported_levels`. The explanation names that year so a
reader knows the levels are "today's", not a 1960-2022 average.

`band_from_fixed_thresholds` is the ungrounded fallback for a location with no
fitted climatology. It keeps the Day-1 absolute cutoffs and says in its own
explanation that they are absolute, so an absolute band can never be mistaken
for a location-relative one.
"""
from __future__ import annotations

from agent.contracts import Hazard, RiskLevel
from tools.hazard_stats import HazardStat, ReturnLevel

# (upper edge in return-period years, band below that edge). Anything at or
# beyond the last edge is SEVERE.
_BAND_EDGES: tuple[tuple[float, RiskLevel], ...] = (
    (2.0, RiskLevel.LOW),
    (10.0, RiskLevel.MODERATE),
    (50.0, RiskLevel.HIGH),
)

# The curve must pin every band edge, otherwise a peak below the smallest fitted
# level is ambiguous between two bands. Fitted by tools/climatology.py.
_REQUIRED_PERIODS: tuple[int, ...] = (2, 10, 50)

# The Day-1 absolute cutoffs, per hazard: (quantity graded, unit, ascending
# (cutoff, band-below-cutoff) pairs). Unchanged numbers, moved here so the
# fallback lives next to the thing it falls back from.
_FIXED_THRESHOLDS: dict[Hazard, tuple[str, str, tuple[tuple[float, RiskLevel], ...]]] = {
    Hazard.HEATWAVE: (
        "peak daily maximum temperature",
        "°C",
        ((35.0, RiskLevel.LOW), (40.0, RiskLevel.MODERATE), (45.0, RiskLevel.HIGH)),
    ),
    Hazard.EXTREME_PRECIP: (
        "peak daily precipitation total",
        "mm",
        ((20.0, RiskLevel.LOW), (50.0, RiskLevel.MODERATE), (100.0, RiskLevel.HIGH)),
    ),
    # Beaufort-inspired, and calibrated on SUSTAINED 10 m wind, not on gusts:
    # a gust is routinely 1.4-1.6x the sustained speed, so grading a gust on
    # these numbers would inflate an ordinary breezy day to HIGH.
    Hazard.WIND: (
        "peak daily maximum sustained 10 m wind",
        "km/h",
        ((40.0, RiskLevel.LOW), (62.0, RiskLevel.MODERATE), (88.0, RiskLevel.HIGH)),
    ),
}

_NO_CLIMATOLOGY = "no ERA5 climatology is available for this location"


def _band_for_period(return_period_years: float) -> RiskLevel:
    """The band a given return period falls in (see the module docstring table)."""
    for edge, level in _BAND_EDGES:
        if return_period_years < edge:
            return level
    return RiskLevel.SEVERE


def _interpolated_period(value: float, low: ReturnLevel, high: ReturnLevel) -> float:
    """Return period of `value` between two fitted levels, linear in log(T).

    The caller only ever passes a bracket satisfying `low.level <= value <
    high.level`, so the span is strictly positive and the fraction lands in
    [0, 1): no division guard and no extrapolation are possible here.

    Written as a ratio power rather than exp(log-interpolation) because the two
    agree mathematically but not in floating point: exp(log(50)) is 49.999...,
    one ULP short, which silently banded a peak sitting EXACTLY on the 50-year
    level as HIGH instead of SEVERE. This form returns the anchor's period
    exactly when the value equals a fitted level, which is where the band edges
    are, so the edge cases are the ones it gets exactly right.
    """
    fraction = (value - low.level) / (high.level - low.level)
    ratio = high.return_period_years / low.return_period_years
    return low.return_period_years * ratio**fraction


def _effective_note(hazard_stat: HazardStat) -> str:
    """Say when the curve is the effective (trend-adjusted) one, else say nothing."""
    trend = hazard_stat.trend
    if trend is not None and trend.significant and trend.evaluated_at_year is not None:
        return f" on levels effective at {trend.evaluated_at_year}"
    return ""


def band_from_return_period(
    value: float, hazard_stat: HazardStat
) -> tuple[RiskLevel, str]:
    """Band a forecast peak by its position on this location's return-level curve.

    `value` must be the same quantity and unit as `hazard_stat.variable` (the
    caller is responsible for that check). Returns the band and a one-sentence
    explanation carrying the numbers it was decided on.

    Raises ValueError if the curve does not pin every band edge.
    """
    curve = sorted(hazard_stat.return_levels, key=lambda r: r.return_period_years)
    have = {r.return_period_years for r in curve}
    missing = [t for t in _REQUIRED_PERIODS if t not in have]
    if missing:
        raise ValueError(f"return-level curve lacks required return periods: {missing}")

    unit = hazard_stat.unit
    basis = _effective_note(hazard_stat)
    lowest, highest = curve[0], curve[-1]

    if value < lowest.level:
        return RiskLevel.LOW, (
            f"Forecast peak {value:g} {unit} is below the "
            f"{lowest.return_period_years}-year level ({lowest.level:.1f} {unit}) "
            f"for this location{basis}, within an ordinary year's range."
        )
    if value >= highest.level:
        return RiskLevel.SEVERE, (
            f"Forecast peak {value:g} {unit} is at or above the "
            f"{highest.return_period_years}-year level ({highest.level:.1f} {unit}) "
            f"for this location{basis}, a record-class event."
        )

    # For FINITE levels a bracket always exists here: value >= curve[0].level
    # forces the next level down the chain to be <= value until one exceeds it,
    # and value < curve[-1].level guarantees one does — monotone or not. A
    # NON-FINITE level (a GEV fit that failed to converge and produced NaN)
    # breaks that argument, because every comparison against NaN is False: the
    # two guards above both fall through and no pair matches. `next()` with no
    # default would then raise a bare StopIteration — no message, and silently
    # swallowed into an early stop if this ever runs inside a generator. Name
    # the broken curve instead.
    bracket = next(
        ((a, b) for a, b in zip(curve, curve[1:]) if a.level <= value < b.level), None
    )
    if bracket is None:
        raise ValueError(
            f"return-level curve does not bracket {value:g} {unit}: levels must be "
            f"finite and rise with return period, got "
            f"{[(r.return_period_years, r.level) for r in curve]}"
        )
    low, high = bracket
    period = _interpolated_period(value, low, high)
    return _band_for_period(period), (
        f"Forecast peak {value:g} {unit} sits between the "
        f"{low.return_period_years}- and {high.return_period_years}-year levels "
        f"({low.level:.1f} / {high.level:.1f} {unit}) for this location{basis}, "
        f"about a 1-in-{period:.0f}-year event."
    )


def _threshold_range_text(
    level: RiskLevel, unit: str, edges: tuple[tuple[float, RiskLevel], ...]
) -> str:
    """Human-readable cutoff range for the band that was assigned."""
    for index, (cutoff, banded) in enumerate(edges):
        if banded is level:
            if index == 0:
                return f"below {cutoff:g} {unit}"
            return f"{edges[index - 1][0]:g} to {cutoff:g} {unit}"
    return f"at or above {edges[-1][0]:g} {unit}"


def band_from_fixed_thresholds(
    value: float, hazard: Hazard, *, reason: str = _NO_CLIMATOLOGY
) -> tuple[RiskLevel, str]:
    """Fallback band from the Day-1 absolute cutoffs, when the curve cannot be used.

    `reason` says WHY the location-relative path was unavailable; it defaults to
    the common case (no climatology fitted). The explanation always states that
    the band is absolute, never location-relative.
    """
    quantity, unit, edges = _FIXED_THRESHOLDS[hazard]
    level = RiskLevel.SEVERE
    for cutoff, banded in edges:
        if value < cutoff:
            level = banded
            break
    return level, (
        f"Climatology-conditioned banding unavailable ({reason}), so this band "
        f"comes from fixed absolute cutoffs rather than local rarity: "
        f"{quantity} {value:g} {unit} grades {level.value} "
        f"({_threshold_range_text(level, unit, edges)})."
    )
