"""Read the measured forecast-skill table: how wrong the forecast usually is.

WHY THIS EXISTS
---------------
The agent already answers "how extreme is this?" from 60+ years of ERA5. It
does not yet answer "how much should you trust the forecast that produced it?".
A 41 C peak predicted 2 days out and the same peak predicted 12 days out are
different claims, and today the report presents them with identical confidence.

This module is the read side of the fix. `scripts/measure_forecast_skill.py`
measures, against real archived model runs, how far the forecast typically
lands from the model's own final analysis at each lead time, and freezes the
result into `tools/forecast_skill_table.json`. Here we just load it and answer
one question:

    skill_for("temperature_2m_max", horizon_days=6)
    -> SkillEntry(mae=..., bias=..., rmse=..., extreme_hit_rate=..., ...)

HOW THIS WEIGHTS CONFIDENCE
---------------------------
The consumer is the confidence field on the agent's RiskReport, in two concrete
ways. Reading 2 is LIVE (`forecast_skill` at the bottom of this module builds
the block the report carries, and `agent.verdict.compose_confidence` multiplies
the forecast term by its `confidence_weight`); reading 1 is not built yet.

1. Widen the severity band by the measured error. Today the report places a
   forecast peak on the location's GEV return-level curve as a single point. A
   day-6 heat peak carries a measured MAE of about 2 C, so the honest statement
   is that the peak lands in a band roughly the width of that MAE, and the
   severity should be the range of return levels that band spans, not the point
   estimate. `mae` (typical size of the miss) and `bias` (which way the model
   leans) are the two numbers that set that band.

2. Cap confidence by the measured extreme-hit rate. `extreme_hit_rate` is the
   probability of detection for the worst 5% of days at the measured cities: if
   the forecast only catches half the genuinely extreme days at that horizon,
   the report should not say "high confidence" about an extreme call at that
   horizon, whatever the model's prose sounds like. This is the number that
   turns a hedge phrase into a measured one.

Both readings degrade gracefully past the archive: the API only publishes lead
offsets 1-7, while `tools/forecast.py` accepts horizons up to 16 days.
`skill_for` clamps anything beyond 7 to the day-7 row and sets
`extrapolated=True`, so a caller can say "at least this uncertain, and we did
not measure this far out" instead of silently inventing a day-14 number.

READ `provenance` BEFORE QUOTING ANY OF THESE NUMBERS
------------------------------------------------------
Two caveats live in the table itself and both change how the numbers should be
described. The reference series is the model's own day-0 run, not observations,
so these are run-to-run consistency errors and a lower bound on true forecast
error. And `hourly_repeat_fraction` flags where the numbers reflect a coarser
model output resolution (GFS goes 3-hourly past forecast hour 120) rather than
predictability. `scripts/measure_forecast_skill.py` documents both in full.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

TABLE_PATH = Path(__file__).resolve().parent / "forecast_skill_table.json"

# The Previous Model Runs API publishes lead-time offsets 1-7 only. Horizons
# past this are answered from the day-7 row and flagged, never interpolated.
MAX_MEASURED_LEAD_DAY = 7


class SkillTableError(RuntimeError):
    """The skill table is missing, malformed, or lacks the requested hazard."""


class SkillEntry(BaseModel):
    """Measured forecast error for one hazard at one horizon."""

    hazard: str
    unit: str
    requested_horizon_days: int
    lead_day: int = Field(description="the table row actually used, always 1..7")
    extrapolated: bool = Field(
        description="True when the horizon ran past the measured archive and was clamped to day 7"
    )

    mae: float = Field(description="mean absolute error, in `unit`")
    bias: float = Field(description="mean signed error; positive means the forecast runs high")
    rmse: float = Field(description="root mean squared error, in `unit`")
    extreme_hit_rate: float | None = Field(
        default=None,
        description="fraction of days above the location's 95th percentile that the forecast also "
        "placed above it; None when too few extreme days were measured to be meaningful",
    )
    hourly_repeat_fraction: float | None = Field(
        default=None,
        description="fraction of consecutive hours holding an identical value; a jump signals the "
        "underlying model switched to coarser output at this lead, not a skill cliff",
    )

    n_days: int
    n_extreme_days: int
    n_cities: int


class SkillTable(BaseModel):
    """The committed table plus the provenance that makes it quotable."""

    schema_version: int
    provenance: dict[str, Any]
    hazards: dict[str, Any]


@lru_cache(maxsize=4)
def load_skill_table(path: Path = TABLE_PATH) -> SkillTable:
    """Load and validate the committed table. Cached: the file never changes at runtime."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SkillTableError(
            f"no skill table at {path}; run scripts/measure_forecast_skill.py to build it"
        ) from exc
    except json.JSONDecodeError as exc:
        raise SkillTableError(f"skill table at {path} is not valid JSON: {exc}") from exc

    try:
        return SkillTable(**raw)
    except Exception as exc:  # pydantic ValidationError, or a missing top-level key
        raise SkillTableError(f"skill table at {path} has an unexpected shape: {exc}") from exc


def available_hazards(path: Path = TABLE_PATH) -> list[str]:
    """Hazard keys the table can answer for, in table order."""
    return list(load_skill_table(path).hazards)


def skill_for(
    hazard: str, horizon_days: int, path: Path = TABLE_PATH
) -> SkillEntry:
    """Measured error for `hazard` at `horizon_days` out.

    Horizons past the measured archive (day 7) are answered from the day-7 row
    with `extrapolated=True` rather than extrapolated numerically. Error at long
    lead flattens towards climatological spread rather than growing without
    bound, so the day-7 row is a floor on the uncertainty at day 14, not an
    estimate of it — the flag exists so a caller states that rather than hiding it.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")

    table = load_skill_table(path)
    block = table.hazards.get(hazard)
    if block is None:
        raise SkillTableError(
            f"hazard {hazard!r} is not in the skill table; have: {', '.join(table.hazards)}"
        )

    lead_day = min(horizon_days, MAX_MEASURED_LEAD_DAY)
    row = block["lead_days"].get(str(lead_day))
    if row is None or row.get("mae") is None:
        raise SkillTableError(f"skill table has no measured row for {hazard!r} at lead day {lead_day}")

    return SkillEntry(
        hazard=hazard,
        unit=block["unit"],
        requested_horizon_days=horizon_days,
        lead_day=lead_day,
        extrapolated=horizon_days > MAX_MEASURED_LEAD_DAY,
        mae=row["mae"],
        bias=row["bias"],
        rmse=row["rmse"],
        extreme_hit_rate=row.get("extreme_hit_rate"),
        hourly_repeat_fraction=row.get("hourly_repeat_fraction"),
        n_days=row["n_days"],
        n_extreme_days=row["n_extreme_days"],
        n_cities=row["n_cities"],
    )


# --------------------------------------------------------------------------- #
# The report-facing block: measured skill as the agent quotes and weights it
# --------------------------------------------------------------------------- #

# Display labels for report prose. Explicit maps rather than string
# prettification: an unrecognised key falls back to itself instead of having a
# display name invented for it.
_MODEL_LABELS = {"gfs_global": "GFS Global"}
_VARIABLE_LABELS = {
    "temperature_2m_max": "temperature",
    "precipitation_sum": "precipitation",
    "wind_speed_10m_max": "wind speed",
    "wind_gusts_10m_max": "wind gust",
}


def _confidence_weight(block: dict[str, Any], lead_day: int) -> float:
    """How much the forecast half of the evidence is worth at `lead_day`; 1.0 at day 1.

        weight = min(extreme_hit_rate[1..lead_day]) / extreme_hit_rate[1]

    The hit rate is the fraction of genuinely extreme days (above the location's
    own 95th percentile) that the forecast also called extreme, so the ratio is
    "how much of day-1's detection survives at this lead". It is <= 1 by
    construction, because day 1 is itself inside the window being minimised over.

    RUNNING MINIMUM, not the raw row, because a rate that ticks back up is
    sampling noise on ~450 extreme days, never a forecast that improved with
    lead time: precipitation measures 1.7% at day 6 and 3.5% at day 7. The
    running minimum makes the weight, and so the confidence built on it,
    monotone non-increasing in horizon, which is the property the report must
    not violate. An unmeasured rate anywhere in the window returns 1.0: a
    missing measurement is a reason to leave the old weighting alone, never a
    reason to invent a penalty.
    """
    rates: list[float] = []
    for day in range(1, lead_day + 1):
        row = block["lead_days"].get(str(day)) or {}
        rate = row.get("extreme_hit_rate")
        if rate is None:
            return 1.0
        rates.append(float(rate))
    if not rates or rates[0] <= 0.0:
        return 1.0
    return round(min(rates) / rates[0], 4)


class ForecastSkill(BaseModel):
    """Measured skill for the one variable a report is graded on, at its horizon.

    Rides on `ForecastResult.skill` and drives two things: the sentence in the
    report's `forecast_skill` driver, and the lead-day-aware forecast term in
    `agent.verdict.compose_confidence`.
    """

    variable: str
    unit: str
    requested_horizon_days: int
    lead_day: int = Field(description="the measured row actually used, always 1..7")
    extrapolated: bool = Field(
        description="True when the horizon ran past the measured archive and was clamped to day 7"
    )
    mae: float = Field(description="mean absolute error at this lead, in `unit`")
    extreme_hit_rate: float | None = Field(
        default=None,
        description="fraction of days above the location's 95th percentile that the forecast "
        "also placed above it; None when too few extreme days were measured",
    )
    confidence_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="forecast evidence weight relative to day 1 (see _confidence_weight); "
        "carried in the payload so the composed confidence can be recomputed by hand",
    )
    source: str = Field(description="model, city count and date range, from the table provenance")

    def detail(self) -> str:
        """One sentence for the `forecast_skill` driver, with the measured number in it."""
        label = _VARIABLE_LABELS.get(self.variable, self.variable)
        if self.extreme_hit_rate is None:
            return (
                f"Day-{self.lead_day} {label} forecasts carry a measured error of "
                f"{self.mae:.2f} {self.unit} ({self.source}), but too few extreme days were "
                f"measured to score the extreme call, so forecast evidence keeps full weight."
            )
        pct = round(self.extreme_hit_rate * 100)
        if self.extrapolated:
            return (
                f"A {self.requested_horizon_days}-day horizon is past the measured archive, "
                f"which stops at day 7, so the day-7 figure is reused: {label} forecasts hit "
                f"the local extreme {pct}% of the time ({self.source}), and skill does not "
                f"improve with lead time, so forecast evidence is down-weighted to "
                f"{self.confidence_weight:.2f} of its day-1 value."
            )
        if self.confidence_weight >= 1.0:
            baseline = (
                "the day-1 baseline"
                if self.lead_day == 1
                else "no worse than the day-1 baseline"
            )
            return (
                f"Day-{self.lead_day} {label} forecasts hit the local extreme {pct}% of the "
                f"time ({self.source}), {baseline}, so forecast evidence carries full weight."
            )
        return (
            f"Day-{self.lead_day} {label} forecasts hit the local extreme {pct}% of the time "
            f"({self.source}), so forecast evidence is down-weighted to "
            f"{self.confidence_weight:.2f} of its day-1 value."
        )


def _source_sentence(provenance: dict[str, Any]) -> str:
    """Quotable provenance: "GFS Global vs its own day-0 run, 13 cities, 2024-2025"."""
    model = provenance.get("model", "unknown model")
    label = _MODEL_LABELS.get(model, model)
    date_range = provenance.get("date_range") or {}
    start, end = str(date_range.get("start", ""))[:4], str(date_range.get("end", ""))[:4]
    cities = provenance.get("city_count")
    parts = [f"{label} vs its own day-0 run"]
    if cities:
        parts.append(f"{cities} cities")
    if start:
        parts.append(start if start == end else f"{start}-{end}")
    return ", ".join(parts)


def forecast_skill(variable: str, horizon_days: int, path: Path | None = None) -> ForecastSkill:
    """The skill block a report carries for `variable` at `horizon_days` out.

    `path` resolves to the committed table at CALL time, not at import time, so
    a test can point the whole system at a synthetic table by monkeypatching
    `TABLE_PATH`. Raises `SkillTableError` when the table has no row for the
    variable: the caller then ships without a skill block and confidence falls
    back to its flat, lead-blind form.
    """
    table_path = TABLE_PATH if path is None else path
    entry = skill_for(variable, horizon_days, path=table_path)
    table = load_skill_table(table_path)
    return ForecastSkill(
        variable=variable,
        unit=entry.unit,
        requested_horizon_days=entry.requested_horizon_days,
        lead_day=entry.lead_day,
        extrapolated=entry.extrapolated,
        mae=entry.mae,
        extreme_hit_rate=entry.extreme_hit_rate,
        confidence_weight=_confidence_weight(table.hazards[variable], entry.lead_day),
        source=_source_sentence(table.provenance),
    )
