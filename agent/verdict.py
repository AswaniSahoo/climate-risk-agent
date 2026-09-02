"""Composed confidence: how much the report is allowed to trust itself.

Severity is not here. `agent/risk_bands.py` owns it, on the return period of the
forecast peak at that location (2 / 10 / 50-year edges, interpolated between
fitted levels and returned with an explanation sentence).

Confidence composition is live. It is composed from what actually grounds the
report instead of a constant: the forecast contributes 0.3 scaled by its
MEASURED skill at that horizon; climatology raises it by how representative the
statistic is of the true local extreme; a cited IPCC answer adds a little. Hard
ceiling 0.75, because none of these three is a verification against what
actually happened, so the system never claims near-certainty.
"""
from __future__ import annotations

from tools.hazard_stats import Representativeness

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


def compose_confidence(
    *,
    representativeness: Representativeness | None,
    ipcc_cited: bool,
    forecast_skill_weight: float | None = None,
) -> float:
    """Compose report confidence from its actual grounding.

    THE FORMULA, in full:

        confidence = min(0.75, 0.3 * w + representativeness_bonus + 0.1 * [IPCC cited])
        w = min(extreme_hit_rate[1..L]) / extreme_hit_rate[1]   (`forecast_skill_weight`)

    where L is the forecast lead day the report is graded at (its horizon,
    clamped to the measured day 7), and the bonuses are 0.35 station-calibrated
    / 0.25 point-interpolated reanalysis / 0.15 regional grid signal / 0.0 not
    representative.

    WHY w AT ALL. The forecast used to contribute a flat 0.3 whether the peak
    was predicted for tomorrow or for day 12. It is now scaled by how often the
    model actually caught the local extreme at that lead time, measured against
    real archived runs (tools/forecast_skill_table.json): temperature detection
    falls from 85% at day 1 to 47% at day 7, so a day-7 heat report keeps 0.556
    of the forecast evidence a day-1 one gets.

    Two properties hold by construction, and both are tested. w <= 1 with
    equality at day 1, so nothing scores higher than it did before this change.
    And w is non-increasing in the horizon (see `_confidence_weight`), so for a
    fixed hazard confidence can only fall as the horizon grows.

    `forecast_skill_weight=None` means no measured row for this variable: the
    forecast keeps its old flat 0.3 rather than being penalised for a gap in
    our own measurement.
    """
    weight = 1.0 if forecast_skill_weight is None else forecast_skill_weight
    confidence = _BASE_CONFIDENCE * weight
    if representativeness is not None:
        confidence += _REPRESENTATIVENESS_BONUS[representativeness]
    if ipcc_cited:
        confidence += _IPCC_BONUS
    return round(min(confidence, _CEILING), 2)
