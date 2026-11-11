"""Natural-language front door: free text -> typed, deterministic query parts.

Lexical by design, like the scope guard: classification that runs before any
LLM cannot be prompt-injected, costs zero tokens, and fails TYPED (hazard=None)
instead of guessing. The deterministic layer stays first.

With CRG_JEV=triage|all, TypeSafe's Jev reads what the lexical layer cannot
(adr/0002): a question with no hazard words at all goes through the scope
guard's Jev stage, and a question the guard ruled in scope whose hazard these
patterns cannot name ("temperature extremes") gets one Jev reading that may
only name one of our three hazards. The parse then carries a `QueryTriage`
saying who named the hazard and how firmly, so the reader sees that a report
answers an interpretation. The horizon and the place stay lexical: dates and
numbers are a documented Jev weakness, and a place string goes to the geocoder.
"""
from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from agent.contracts import Hazard, QueryTriage, RiskReport
from agent.graph import run_agent
from agent.progress import CLIMATOLOGY, LOCATE, OnStep, StepStatus, emit_step, track
from rag.jev_client import triage_enabled
from rag.scope import ScopeDecision, has_supported_signal, out_of_scope_hazard, scope_verdict
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

# Scope stage 2 reports a hazard by its policy NAME; the front door needs the
# enum. Only used when the lexical patterns above found nothing at all.
_HINT_TO_HAZARD: dict[str, Hazard] = {
    "heatwave": Hazard.HEATWAVE,
    "extreme precipitation": Hazard.EXTREME_PRECIP,
    "wind": Hazard.WIND,
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
    # How the hazard or the refusal was decided. None unless CRG_JEV includes
    # triage, so with the flag off the parse is exactly what it always was.
    triage: QueryTriage | None = None


def _triage_info(
    *, source: str | None, decision: ScopeDecision | None, reading=None
) -> QueryTriage:
    """The reader-facing account of one parse. `reading` is a rag.jev_triage.JevTriage."""
    stage = decision.stage if decision is not None else "lexical"
    if reading is None:
        return QueryTriage(hazard_source=source, scope_stage=stage)  # type: ignore[arg-type]
    return QueryTriage(
        hazard_source=source,  # type: ignore[arg-type]
        hazard_probability=reading.hazard_probability if source == "jev" else None,
        scope_stage=stage,
        jev_status=reading.status,
        jev_model=reading.model,
        detail=reading.detail,
    )


def _extract_hazard(text: str) -> tuple[Hazard | None, str | None, QueryTriage | None]:
    """(hazard, out_of_scope_name, triage): the lexical patterns first, then the guard.

    When a second reader is on (Jev, or scope stage 2) and recovers a hazard
    from a paraphrase the patterns miss ("storms getting stronger" -> wind),
    that hint becomes the parsed hazard instead of a "could not identify a
    supported hazard" refusal. With both off there is no hint and no triage
    record, so this is the same function as before.
    """
    jev_on = triage_enabled()
    # A marine heatwave contains "heatwave", which the heat pattern below would
    # read as OUR hazard and send to a land heat report. rag/scope.py tests it
    # first for exactly this reason (a false answer the held-out set caught);
    # the front door has to honour the same rule before its own patterns run.
    if out_of_scope_hazard(text) == "marine heatwave":
        info = _triage_info(source=None, decision=ScopeDecision(out_of_scope="marine heatwave"))
        return None, "marine heatwave", info if jev_on else None
    for matched, pattern in _HAZARD_PATTERNS.items():
        if pattern.search(text):
            return matched, None, _triage_info(source="lexical", decision=None) if jev_on else None

    decision = scope_verdict(text)
    if decision.out_of_scope:
        info = _triage_info(source=None, decision=decision, reading=decision.jev)
        return None, decision.out_of_scope, info if jev_on else None
    hazard = _HINT_TO_HAZARD.get(decision.hazard_hint or "")
    if hazard is not None:
        info = _triage_info(source=decision.stage, decision=decision, reading=decision.jev)
        return hazard, None, info if jev_on else None
    if not jev_on:
        return None, None, None

    reading = decision.jev
    if reading is None and has_supported_signal(text):
        # Stage 1 ruled the question in scope by its own vocabulary, but these
        # patterns cannot say WHICH hazard ("temperature extremes"). Jev may name
        # one of our three; it may not refuse what stage 1 has let in.
        from rag.jev_triage import HAZARD_HINT_NAME, triage

        reading = triage(text, may_refuse=False)
        if reading.decision == "hint" and reading.hazard is not None:
            hazard = _HINT_TO_HAZARD[HAZARD_HINT_NAME[reading.hazard]]
            return hazard, None, _triage_info(source="jev", decision=decision, reading=reading)
    return None, None, _triage_info(source=None, decision=decision, reading=reading)


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
    hazard, out_of_scope, triage = _extract_hazard(text)
    return ParsedQuery(
        hazard=hazard,
        out_of_scope=out_of_scope,
        place=_extract_place(text),
        horizon_days=_extract_horizon(text),
        triage=triage,
    )


_SUPPORTED_TEXT = "(supported: heatwave, extreme precipitation, wind)"


def refusal_reason(parsed: ParsedQuery) -> str | None:
    """What the front door refuses with, decided from the parse alone, or None.

    One function so that `/triage` shows the user exactly the sentence `/query`
    would refuse with, before any geocoding or forecast has run.
    """
    if parsed.out_of_scope:
        return f"{parsed.out_of_scope} is not a supported hazard {_SUPPORTED_TEXT}."
    if parsed.hazard is None:
        return f"could not identify a supported hazard in the question {_SUPPORTED_TEXT}."
    if parsed.place is None:
        return "could not identify a location in the question."
    return None


class TriageOutcome(BaseModel):
    """The front door's reading of a question, without running the report.

    What a client can show the moment a question is typed: the parsed hazard,
    horizon and place to pre-fill its controls, and either the refusal the
    full run would give or `None`. Costs no geocoding, forecast or Gemini call;
    at most one Jev request, and none when the lexical parser settles it.
    """

    query: str
    hazard: Hazard | None = None
    horizon_days: int = 7
    place: str | None = None
    refusal: str | None = None
    triage: QueryTriage | None = None


def triage_query(text: str) -> TriageOutcome:
    """Parse `text` and say what `/query` would do with it, before doing any of it."""
    parsed = parse_query(text)
    return TriageOutcome(
        query=text,
        hazard=parsed.hazard,
        horizon_days=parsed.horizon_days,
        place=parsed.place,
        refusal=refusal_reason(parsed),
        triage=parsed.triage,
    )


def _nl_refusal(
    query: str, reason: str, *, horizon_days: int = 7, triage: QueryTriage | None = None
) -> RiskReport:
    return RiskReport(
        location=query, hazard=None, horizon_days=horizon_days,
        confidence=0.0, refusal=reason, triage=triage,
    )


def run_agent_nl(
    query: str,
    *,
    use_climatology: bool = True,
    on_step: OnStep = None,
    parsed: ParsedQuery | None = None,
) -> "RiskReport":
    """Free text -> RiskReport: parse -> geocode -> AR6 region -> the agent.

    Every unresolvable step returns a typed REFUSAL report (the contract's
    explicit out-of-scope output) — the front door never guesses a hazard,
    a place, or coordinates.

    `on_step` (optional) is told what is happening while it happens. The two
    steps this function owns (resolving the place and fitting the climatology)
    run BEFORE the graph is invoked, so they are reported here; the graph's own
    nodes are reported by `run_agent`, into the same callback.

    `parsed` (optional) is `parse_query(query)` made earlier, by a client that
    showed the reading before running it (the UI with CRG_JEV=triage). The run
    then answers exactly the reading the person saw, and asks Jev nothing twice.
    """
    parsed = parsed if parsed is not None else parse_query(query)
    reason = refusal_reason(parsed)
    if reason is not None:
        return _nl_refusal(
            query, reason, horizon_days=parsed.horizon_days, triage=parsed.triage
        )
    assert parsed.hazard is not None and parsed.place is not None  # refusal_reason checked both

    try:
        with track(on_step, LOCATE):
            located = geocode(parsed.place)
            region = region_for(located.latitude, located.longitude)
    except GeocodeError as exc:
        return _nl_refusal(query, f"could not resolve the location: {exc}",
                           horizon_days=parsed.horizon_days, triage=parsed.triage)

    if region is not None:
        # the corpus's own region vocabulary -> deterministic table-row retrieval
        location_label = f"{located.name}, {region.label}"
    else:
        location_label = f"{located.name}, {located.country}".rstrip(", ")

    hazard_stat = None
    if use_climatology:
        try:
            with track(on_step, CLIMATOLOGY):
                hazard_stat = climatology_hazard_stat(
                    located.latitude, located.longitude, parsed.hazard
                )
        except ClimatologyError as exc:
            _log.warning("climatology unavailable (%s) — proceeding without it", exc)
    else:
        emit_step(on_step, CLIMATOLOGY, StepStatus.SKIPPED, detail="ERA5 grounding off")

    report = run_agent(
        location=location_label,
        latitude=located.latitude,
        longitude=located.longitude,
        hazard=parsed.hazard,
        horizon_days=parsed.horizon_days,
        hazard_stat=hazard_stat,
        on_step=on_step,
    )
    if parsed.triage is not None:
        report = report.model_copy(update={"triage": parsed.triage})
    return report
