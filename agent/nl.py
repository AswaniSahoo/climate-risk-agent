"""Natural-language front door: free text -> typed, deterministic query parts.

Lexical by design, like the scope guard: classification that runs before any
LLM cannot be prompt-injected, costs zero tokens, and fails TYPED (hazard=None)
instead of guessing. An LLM classifier can be added behind this later; the
deterministic layer stays first.
"""
from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from agent.contracts import Hazard, RiskReport
from agent.graph import run_agent
from rag.scope import out_of_scope_hazard
from tools.ar6_regions import region_for
from tools.climatology import ClimatologyError, climatology_hazard_stat
from tools.geocode import GeocodeError, geocode

_log = logging.getLogger(__name__)

# Per-hazard vocabulary. First match in enum order wins; compound queries that
# mention one of OUR hazards stay in scope (same policy as rag/scope).
_HAZARD_PATTERNS: dict[Hazard, re.Pattern[str]] = {
    Hazard.HEATWAVE: re.compile(
        r"\bheat ?waves?\b|\bextreme heat\b|\bhot extremes?\b|\bheat\b|\bhot spells?\b",
        re.IGNORECASE,
    ),
    Hazard.EXTREME_PRECIP: re.compile(
        r"\brain\w*\b|\bprecipitation\b|\bmonsoons?\b|\bdownpours?\b|\bwet spells?\b",
        re.IGNORECASE,
    ),
    Hazard.WIND: re.compile(r"\bwinds?\b|\bgusts?\b", re.IGNORECASE),
}

_MAX_HORIZON = 16  # Open-Meteo forecast limit; tools/validation re-enforces

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# "in/near/for/at/around <Proper Name>", stopping before time phrases and punctuation
_PLACE = re.compile(
    r"\b(?:in|near|for|at|around|over)\s+"
    r"([A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*)"
)


class ParsedQuery(BaseModel):
    """What the deterministic layer could extract — Nones are honest gaps."""

    hazard: Hazard | None = None
    out_of_scope: str | None = None  # named unsupported hazard, for the refusal text
    place: str | None = None
    horizon_days: int = 7


def _extract_hazard(text: str) -> tuple[Hazard | None, str | None]:
    for hazard, pattern in _HAZARD_PATTERNS.items():
        if pattern.search(text):
            return hazard, None
    return None, out_of_scope_hazard(text)


def _extract_horizon(text: str) -> int:
    lowered = text.lower()
    match = re.search(r"next\s+(\d+)\s+days?", lowered)
    if match:
        return min(int(match.group(1)), _MAX_HORIZON)
    match = re.search(r"next\s+(\w+)\s+weeks?", lowered)
    if match and match.group(1) in _WORD_NUMBERS:
        return min(_WORD_NUMBERS[match.group(1)] * 7, _MAX_HORIZON)
    match = re.search(r"(\w+)\s+days?", lowered)
    if match and match.group(1) in _WORD_NUMBERS:
        return min(_WORD_NUMBERS[match.group(1)], _MAX_HORIZON)
    if re.search(r"\bnext week\b|\bthis week\b", lowered):
        return 7
    if re.search(r"\bfortnight\b|\btwo weeks\b", lowered):
        return 14
    if re.search(r"\btomorrow\b", lowered):
        return 2  # today + tomorrow: the peak the user asked about is day 2
    return 7


def _extract_place(text: str) -> str | None:
    candidates = [m.group(1).strip() for m in _PLACE.finditer(text)]
    for candidate in candidates:
        # drop trailing sentence-case words that are time/filler, keep proper names
        cleaned = re.sub(r"\s+(?:Next|This|Over|During|The)\b.*$", "", candidate).strip("?.!, ")
        if cleaned:
            return cleaned
    return None


def parse_query(text: str) -> ParsedQuery:
    """Parse free text into typed parts; every miss is a None, never a guess."""
    hazard, out_of_scope = _extract_hazard(text)
    return ParsedQuery(
        hazard=hazard,
        out_of_scope=out_of_scope,
        place=_extract_place(text),
        horizon_days=_extract_horizon(text),
    )


def _nl_refusal(query: str, reason: str, *, horizon_days: int = 7) -> RiskReport:
    return RiskReport(
        location=query, hazard=None, horizon_days=horizon_days,
        confidence=0.0, refusal=reason,
    )


def run_agent_nl(query: str, *, use_climatology: bool = True) -> "RiskReport":
    """Free text -> RiskReport: parse -> geocode -> AR6 region -> the agent.

    Every unresolvable step returns a typed REFUSAL report (the contract's
    explicit out-of-scope output) — the front door never guesses a hazard,
    a place, or coordinates.
    """
    parsed = parse_query(query)
    if parsed.out_of_scope:
        return _nl_refusal(
            query,
            f"{parsed.out_of_scope} is not a supported hazard "
            f"(supported: heatwave, extreme precipitation, wind).",
            horizon_days=parsed.horizon_days,
        )
    if parsed.hazard is None:
        return _nl_refusal(
            query,
            "could not identify a supported hazard in the question "
            "(supported: heatwave, extreme precipitation, wind).",
            horizon_days=parsed.horizon_days,
        )
    if parsed.place is None:
        return _nl_refusal(
            query, "could not identify a location in the question.",
            horizon_days=parsed.horizon_days,
        )

    try:
        located = geocode(parsed.place)
    except GeocodeError as exc:
        return _nl_refusal(query, f"could not resolve the location: {exc}",
                           horizon_days=parsed.horizon_days)

    region = region_for(located.latitude, located.longitude)
    if region is not None:
        # the corpus's own region vocabulary -> deterministic table-row retrieval
        location_label = f"{located.name}, {region.label}"
    else:
        location_label = f"{located.name}, {located.country}".rstrip(", ")

    hazard_stat = None
    if use_climatology:
        try:
            hazard_stat = climatology_hazard_stat(
                located.latitude, located.longitude, parsed.hazard
            )
        except ClimatologyError as exc:
            _log.warning("climatology unavailable (%s) — proceeding without it", exc)

    return run_agent(
        location=location_label,
        latitude=located.latitude,
        longitude=located.longitude,
        hazard=parsed.hazard,
        horizon_days=parsed.horizon_days,
        hazard_stat=hazard_stat,
    )
