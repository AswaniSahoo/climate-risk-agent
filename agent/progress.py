"""Progress vocabulary for one agent run: the event contract and the words.

The agent knows WHICH node it is on. A person staring at a blank page needs
something else: what that node is doing, how long it should take, and whether
the answer came back from a cache. That translation is pure text and pure
arithmetic, so it lives here rather than inside either the producer or the
renderer:

- `agent/graph.py` PRODUCES `StepEvent`s (`run_agent(..., on_step=...)`),
- `agent/nl.py` adds the two steps that happen before the graph is invoked,
- `ui/app.py` RENDERS them into an `st.status` container,

and all three import this module. It sits under `agent/` and not `ui/` for two
reasons: the graph must never import from the UI package, and `ui/app.py`
deliberately keeps langgraph off its boot path. This module is Pydantic plus
the standard library, so importing it at boot costs nothing.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class StepStatus(str, Enum):
    """What happened to a step. `skipped` is a real outcome, not an absence:
    a refused question never reaches the forecast, and saying so is honest."""

    STARTED = "started"
    FINISHED = "finished"
    SKIPPED = "skipped"
    FAILED = "failed"


class StepEvent(BaseModel):
    """One thing the agent started, finished, skipped or failed.

    `seconds` is measured wall time for a finished/failed step and 0.0 for the
    other two. `detail` carries the short badge text (a cache hit, a missing
    regional projection); the human sentence for a failure comes from
    `error_sentence`, because only the caller holds the exception.
    """

    node: str
    label: str
    status: StepStatus
    seconds: float = 0.0
    detail: str = ""


#: What a progress consumer looks like. `None` means "nobody is watching", which
#: every emitter here accepts, so no caller needs an `if` around its own work.
OnStep = Optional[Callable[[StepEvent], None]]


# The two steps that happen OUTSIDE the graph. The point is resolved and the
# climatology is fitted before `run_agent` is invoked (the UI does it on the
# "Assess risk" path, `agent/nl.py` on the "Ask" path), so whoever does that
# work emits these two itself: same event type, same words, one status panel.
LOCATE = "locate"
CLIMATOLOGY = "climatology"

# The graph's own nodes in the order they run (see agent/graph.py:_build_graph).
GRAPH_NODES: tuple[str, ...] = ("plan", "call", "research", "project", "synthesize")

# The full user-facing order. Climatology precedes the forecast because that is
# the true order of execution: the fitted HazardStat is an ARGUMENT to
# run_agent, so it has to exist before the graph starts.
STEP_ORDER: tuple[str, ...] = (LOCATE, CLIMATOLOGY, *GRAPH_NODES)

_LABELS: dict[str, str] = {
    LOCATE: "Resolving location",
    CLIMATOLOGY: "Fitting 60 years of ERA5 extremes",
    "plan": "Checking the question is in scope",
    "call": "Fetching the forecast (Open-Meteo)",
    "research": "Searching IPCC AR6 (hybrid BM25 + dense)",
    "project": "Reading Chapter 12 projections for your region",
    "synthesize": "Writing the cited report",
}

# One line each, in the second person, saying what the step buys the reader.
# The climatology line carries the expected-wait hint because that step is the
# only one that can take minutes, and only ever on a first visit.
_MEANINGS: dict[str, str] = {
    LOCATE: (
        "Turns the place into coordinates and finds the AR6 reference region "
        "that contains the point."
    ),
    CLIMATOLOGY: (
        "First visit to a place: about 1 to 2 minutes (60+ years of ERA5 daily "
        "extremes, fetched and fitted). Repeat visits: instant, from the cache."
    ),
    "plan": (
        "Three hazards have a real data path: heat, extreme precipitation, "
        "wind. Anything else is refused rather than guessed."
    ),
    "call": (
        "Live daily rainfall, maximum temperature and wind gusts for the point. "
        "Cached for one hour."
    ),
    "research": (
        "Retrieves AR6 pages, then Gemini writes an answer that is allowed to "
        "cite only those pages."
    ),
    "project": (
        "Pulls the AR6 Chapter 12 sentence for this point's reference region, "
        "verbatim. Absent for an ocean point or an unassessed region."
    ),
    "synthesize": (
        "Places the forecast peak on this location's own return-level curve, "
        "then assembles drivers, citations and confidence. No model call here."
    ),
}

# Which cache namespace each step reads, so a hit can be badged. The names are
# the ones the JsonCache instances are constructed with, and telemetry records
# them as op `cache:<namespace>` (tools/cache_backend.py:_record).
STEP_CACHE_NAMESPACE: dict[str, str] = {
    "call": "forecast",  # tools/forecast.py:_forecast_cache
    CLIMATOLOGY: "hazard_fit",  # tools/climatology.py:_fit_cache
    "research": "answers",  # rag/answer_cache.py
}


def step_label(node: str, *, horizon_days: int | None = None) -> str:
    """The heading a person reads for this step.

    Only the forecast label varies, and only when the horizon is known. A
    made-up default would be a number the reader could not trust.
    """
    if node == "call" and horizon_days is not None:
        return f"Fetching the {horizon_days}-day forecast (Open-Meteo)"
    return _LABELS.get(node, node)


def step_meaning(node: str) -> str:
    """One line explaining what this step is actually doing (may be empty)."""
    return _MEANINGS.get(node, "")


def cache_badge(records: Sequence[Mapping[str, object]], node: str) -> str:
    """`cached (redis)` / `cached (disk)` when this step was served from cache.

    `records` are raw telemetry events (obs/telemetry.py) emitted while the step
    ran. A cache MISS is recorded too, with `cached=False`, so the flag is what
    separates the two, not the presence of a record.
    """
    namespace = STEP_CACHE_NAMESPACE.get(node)
    if namespace is None:
        return ""
    op = f"cache:{namespace}"
    for event in records:
        if event.get("op") == op and event.get("cached"):
            return f"cached ({event.get('model') or 'unknown'})"
    return ""


def finished_detail(
    node: str,
    result: Mapping[str, object] | None,
    records: Sequence[Mapping[str, object]],
) -> str:
    """The short badge text for a finished step: cache hit, and any absence.

    An absent Chapter 12 projection and an out-of-scope refusal are outcomes the
    reader has to see at the step that produced them, not infer from the report.
    """
    parts: list[str] = []
    badge = cache_badge(records, node)
    if badge:
        parts.append(badge)
    result = result or {}
    if node == "project" and result.get("projection_note"):
        parts.append("no regional projection for this point")
    if node == "plan" and result.get("report") is not None:
        parts.append("out of scope: refused")
    return " · ".join(parts)


def format_elapsed(seconds: float) -> str:
    """Human elapsed time: 0.02 s · 1.4 s · 42 s · 1 m 22 s.

    Precision shrinks as the number grows, because nobody reading a 90-second
    wait cares about the hundredths, and a 0.02 s cache hit is invisible at
    one decimal place.
    """
    seconds = max(seconds, 0.0)
    if seconds < 1:
        return f"{seconds:.2f} s"
    if seconds < 10:
        return f"{seconds:.1f} s"
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes} m {rest:02d} s"


def emit_step(
    on_step: OnStep,
    node: str,
    status: StepStatus,
    *,
    seconds: float = 0.0,
    detail: str = "",
    horizon_days: int | None = None,
) -> None:
    """Send one event, if anyone is listening. A no-op when `on_step` is None."""
    if on_step is None:
        return
    on_step(
        StepEvent(
            node=node,
            label=step_label(node, horizon_days=horizon_days),
            status=status,
            seconds=seconds,
            detail=detail,
        )
    )


@contextmanager
def track(on_step: OnStep, node: str, *, horizon_days: int | None = None) -> Iterator[None]:
    """Time one step that happens OUTSIDE the graph and report it.

    Geocoding and the ERA5 fit both run before `run_agent` is invoked, so they
    cannot come from LangGraph's task stream, but a reader should not be able
    to tell which of the steps in the panel came from where. A failure is
    reported and then re-raised: the caller owns what the reader is told.

    The telemetry import is local so this module stays free to import at boot.
    """
    from obs import telemetry

    mark = len(telemetry.snapshot())
    started = time.perf_counter()
    emit_step(on_step, node, StepStatus.STARTED, horizon_days=horizon_days)
    try:
        yield
    except BaseException as exc:
        emit_step(
            on_step,
            node,
            StepStatus.FAILED,
            seconds=time.perf_counter() - started,
            detail=type(exc).__name__,
            horizon_days=horizon_days,
        )
        raise
    emit_step(
        on_step,
        node,
        StepStatus.FINISHED,
        seconds=time.perf_counter() - started,
        detail=cache_badge(telemetry.snapshot()[mark:], node),
        horizon_days=horizon_days,
    )


_NETWORK = (
    "A network call to an upstream data service failed. That is usually "
    "temporary. Try again in a moment."
)
_BAD_INPUT = (
    "Those inputs are not a valid request. Check the latitude, longitude and "
    "forecast horizon in the sidebar."
)

# Exception class name -> one plain sentence, ending in what the reader can do.
_ERROR_SENTENCES: dict[str, str] = {
    "GeocodeError": (
        "That place could not be resolved. Check the spelling, or type a "
        "latitude and longitude in the sidebar instead."
    ),
    "ForecastError": (
        "The forecast service (Open-Meteo) did not answer. That is an upstream "
        "outage, not a problem with your request. Try again in a minute."
    ),
    "ClimatologyError": (
        "The 60-year ERA5 history for this point could not be fetched, so the "
        "forecast cannot be graded against this location's own extremes. Try "
        "again, or untick the ERA5 grounding box to run without it."
    ),
    "AnswerError": (
        "The cited-answer step could not produce an answer it was able to "
        "ground, so it asserted nothing. Try again, or ask about another place."
    ),
    "EmbeddingError": (
        "The embedding service is unavailable, so IPCC retrieval cannot run at "
        "full strength and citations may be missing. Try again later."
    ),
    "CorpusError": (
        "The IPCC AR6 corpus is not available on this machine, so no citation "
        "can be produced. Reload the page to let it download once."
    ),
    "ValidationError": _BAD_INPUT,
    "ValueError": _BAD_INPUT,
    "HTTPStatusError": _NETWORK,
    "TimeoutException": _NETWORK,
    "TransportError": _NETWORK,
    "HTTPError": _NETWORK,
}

_FALLBACK_SENTENCE = (
    "Something went wrong while building this report. The full error is in the "
    "server log. Try again, or pick another place."
)


def _gemini_sentence(message: str) -> str:
    """Split GeminiError by cause: no credentials, no quota, or anything else.

    All three arrive as the same class (rag/gemini_client.py raises one type on
    purpose), and the three have completely different answers for the reader,
    so the message is the only thing left to key on.
    """
    lowered = message.lower()
    if "no gemini auth configured" in lowered or "google_cloud_project" in lowered:
        return (
            "No Gemini credentials are configured, so the cited IPCC answer "
            "cannot be written. Set GEMINI_API_KEY (or Vertex AI) and reload."
        )
    if any(
        term in lowered
        for term in ("rate-limit", "resource_exhausted", "quota", "429")
    ):
        return (
            "The Gemini quota is used up. Wait a few minutes for the quota "
            "window to reset, then run it again."
        )
    return (
        "The language-model call failed, so this report has no cited IPCC "
        "answer. Try again in a moment."
    )


def error_sentence(exc: BaseException) -> str:
    """One plain sentence for a person, ending in what they can do about it.

    Matched on the exception's class NAMES along its MRO rather than on imported
    classes, so this module stays clear of the tool and RAG import graph (the UI
    imports it at boot, before anything heavy) and so it is directly testable
    with plain stand-ins. Never shows a traceback: the caller logs that.
    """
    names = [cls.__name__ for cls in type(exc).__mro__]
    if "GeminiError" in names:
        return _gemini_sentence(str(exc))
    for name in names:
        if name in _ERROR_SENTENCES:
            return _ERROR_SENTENCES[name]
    return _FALLBACK_SENTENCE
