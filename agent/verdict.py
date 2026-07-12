"""The risk verdict: severity from the return-level curve, confidence composed.

Why (evaluator gap #1): absolute thresholds ("50 mm = HIGH") are wrong twice —
they ignore the location (46 °C is a normal summer day in Rourkela and a
catastrophe in Berlin) and they ignore the 60+ years of ERA5 climatology this
system fits. The defensible severity of a forecast peak is HOW RARE it is at
that location, which is exactly what a GEV return-level curve encodes
(insurance / cat-risk framing):

    peak <  10-yr level  -> LOW       (within recent-decadal experience)
    peak >= 10-yr level  -> MODERATE  (unusual)
    peak >= 50-yr level  -> HIGH      (rare)
    peak >= 100-yr level -> SEVERE    (record-class)

Confidence is composed from what actually grounds the report instead of a
constant: forecast-only reports stay at 0.3; climatology raises it by how
representative the statistic is of the true local extreme; a cited IPCC answer
adds a little. Hard ceiling 0.75 — nothing verifies forecast *skill* yet, so
the system never claims near-certainty.
"""
from __future__ import annotations

from agent.contracts import RiskLevel
from tools.hazard_stats import Representativeness, ReturnLevel

_REQUIRED_PERIODS = (10, 50, 100)
_PERIOD_TO_LEVEL = {100: RiskLevel.SEVERE, 50: RiskLevel.HIGH, 10: RiskLevel.MODERATE}

# How much trust each representativeness grade adds over the 0.3 forecast base.
_REPRESENTATIVENESS_BONUS = {
    Representativeness.STATION_CALIBRATED: 0.35,
    Representativeness.POINT_INTERPOLATED_REANALYSIS: 0.25,
    Representativeness.REGIONAL_GRID_SIGNAL: 0.15,
    Representativeness.NOT_REPRESENTATIVE: 0.0,
}

_BASE_CONFIDENCE = 0.3
_IPCC_BONUS = 0.1
_CEILING = 0.75


def level_from_return_periods(peak: float, curve: list[ReturnLevel]) -> RiskLevel:
    """Map a forecast peak to a RiskLevel by its position on the return-level curve."""
    by_period = {r.return_period_years: r.level for r in curve}
    missing = [t for t in _REQUIRED_PERIODS if t not in by_period]
    if missing:
        raise ValueError(f"curve lacks required return periods: {missing}")
    for period in sorted(_PERIOD_TO_LEVEL, reverse=True):
        if peak >= by_period[period]:
            return _PERIOD_TO_LEVEL[period]
    return RiskLevel.LOW


def compose_confidence(
    *, representativeness: Representativeness | None, ipcc_cited: bool
) -> float:
    """Compose report confidence from its actual grounding (see module docstring)."""
    confidence = _BASE_CONFIDENCE
    if representativeness is not None:
        confidence += _REPRESENTATIVENESS_BONUS[representativeness]
    if ipcc_cited:
        confidence += _IPCC_BONUS
    return round(min(confidence, _CEILING), 2)
