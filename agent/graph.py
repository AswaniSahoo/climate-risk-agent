r"""The agent: a 5-node LangGraph that fills a RiskReport for one query.

Flow:  START -> plan -> (call -> research -> project -> synthesize) -> END
                    \----------------- refusal ------------------/

- plan       : is this hazard answerable with our data? if not, write a refusal.
- call       : run get_forecast, put the ForecastResult on the shared state.
- research   : IPCC AR6 RAG — retrieve + cited LLM answer for this hazard/region.
               Loud, non-fatal: offline/no-LLM degrades to a citation-less report.
- project    : the AR6 Ch.12 climatic impact-driver projection for the AR6
               reference region containing the point (rag/cid.py — retrieval +
               regex, no LLM). Absent for an ocean point or an unassessed
               region/hazard pair, and the report says so.
- synthesize : turn forecast (+ optional ERA5 climatology + IPCC answer) into a
               RiskReport with page-level Citations. Severity is the forecast
               peak's return period on the location's own GEV curve
               (agent/risk_bands.py: 2 / 10 / 50-year band edges); absolute
               cutoffs survive only as the fallback for a location with no
               fitted climatology, and say so in the report. Confidence is
               composed from the report's actual grounding (agent/verdict.py),
               with the forecast term scaled by the MEASURED skill at that
               horizon (tools/forecast_skill.py), so the same peak predicted a
               week out is worth less than one predicted tomorrow.

Every node reads/writes one shared `AgentState` (the "clipboard"). Nodes depend on
the state shape, not on each other — which is what lets this grow into the
MASTER-PLAN's 4 parallel agents by *adding nodes*, not rewiring.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.contracts import (
    Citation,
    DataProvenance,
    Hazard,
    ProjectedChange,
    RiskDriver,
    RiskReport,
)
from agent.progress import (
    GRAPH_NODES,
    OnStep,
    StepStatus,
    emit_step,
    finished_detail,
)
from agent.risk_bands import band_from_fixed_thresholds, band_from_return_period
from agent.verdict import compose_confidence
from obs import telemetry
from rag.answer import AnswerError, CitedAnswer, answer_with_guard
from rag.chunk import Chunk
from rag.cid import projected_change_for
from rag.corpus import CorpusError, load_corpus_chunks
from rag.embed import EmbeddingError
from rag.gemini_client import GeminiError
from rag.retrieve import HybridRetriever
from tools.ar6_regions import region_for
from tools.forecast import OPEN_METEO_URL, ForecastResult, get_forecast
from tools.forecast_skill import SkillTableError, forecast_skill
from tools.hazard_stats import HazardStat

_log = logging.getLogger(__name__)

# Hazards we can actually answer today (have a data path in get_forecast).
_ANSWERABLE = {Hazard.HEATWAVE, Hazard.EXTREME_PRECIP, Hazard.WIND}
# Open-Meteo reports wind in km/h on BOTH the forecast and the archive endpoint
# (verified in the docs), so the forecast-vs-climatology comparison is km/h to
# km/h. m/s is shown alongside because wind hazard literature quotes m/s.
_KMH_TO_MS = 1.0 / 3.6
# Which forecast variable each hazard's peak metric comes from — must equal the
# fitted HazardStat.variable for the GEV band to be a like-for-like comparison.
_FORECAST_VARIABLE = {
    Hazard.HEATWAVE: "temperature_2m_max",
    Hazard.EXTREME_PRECIP: "precipitation_sum",
    Hazard.WIND: "wind_gusts_10m_max",  # gust vs the ERA5 GUST fit, like for like
}
# A/B-measured on the frozen set: k=8 admits table-header chunks (GWL column
# labels), fixing column-ambiguity refusals — matrix 33/11/1/0, false_answer 0.
_IPCC_TOP_K = 8


class AgentState(TypedDict, total=False):
    """The shared clipboard passed between nodes."""

    location: str
    latitude: float
    longitude: float
    hazard: Hazard
    horizon_days: int
    forecast: Optional[ForecastResult]
    hazard_stat: Optional[HazardStat]  # optional ERA5 climatology, injected by caller
    ipcc_answer: Optional[CitedAnswer]  # research node output (None if degraded)
    ipcc_chunks: list[Chunk]  # what research retrieved — needed to map chunk_id -> page
    projected_change: Optional[ProjectedChange]  # project node output (None if nothing found)
    projection_note: str  # why there is no projection, when there is none
    report: Optional[RiskReport]


def plan(state: AgentState) -> dict:
    """Scope guardrail: refuse hazards we have no data path for."""
    hazard = state["hazard"]
    if hazard not in _ANSWERABLE:
        return {
            "report": RiskReport(
                location=state["location"],
                hazard=hazard,
                horizon_days=state["horizon_days"],
                confidence=0.0,
                refusal=f"{hazard.value} risk is not supported yet (no data path).",
            )
        }
    return {}


def call(state: AgentState) -> dict:
    """Fetch live weather for the requested location, and how much it is worth.

    The skill block is attached HERE rather than inside get_forecast because it
    is per-hazard (heat reads the temperature row, wind the gust row) while the
    fetch and its cache key are hazard-blind: baking it into the cached
    ForecastResult would serve one hazard's skill to the next hazard asking for
    the same city and horizon. The hazard -> variable map already lives in this
    module, so this is also the only place that mapping is written down.

    Missing skill is non-fatal and loud: the report ships without the block and
    confidence falls back to its old flat forecast term.
    """
    forecast = get_forecast(
        latitude=state["latitude"],
        longitude=state["longitude"],
        horizon_days=state["horizon_days"],
    )
    variable = _FORECAST_VARIABLE[state["hazard"]]
    try:
        skill = forecast_skill(variable, state["horizon_days"])
    except (SkillTableError, ValueError) as exc:
        _log.warning(
            "no measured forecast skill for %s (%s), confidence keeps the flat forecast term",
            variable,
            exc,
        )
        return {"forecast": forecast}
    return {"forecast": forecast.model_copy(update={"skill": skill})}


@lru_cache(maxsize=1)
def _ipcc_retriever() -> HybridRetriever:
    """Build (once) the measured hybrid retriever over the on-disk IPCC corpus."""
    return HybridRetriever.build(list(load_corpus_chunks()))


@lru_cache(maxsize=1)
def _answer_cache():
    """Shared cache: repeat (question, evidence) pairs cost zero tokens.

    No directory argument, so it takes the process-wide backend from the
    environment (Redis when configured, disk otherwise) instead of a folder
    that dies with the replica.
    """
    from rag.answer_cache import AnswerCache

    return AnswerCache()


# Fused vocabulary per hazard: AR6 table terms ("hot extremes", "heavy
# precipitation" — BM25 anchors) + real-world risk-screening phrasing
# ("extreme heat", "extreme rainfall" — dense bridges the paraphrase).
_IPCC_PHRASES = {
    Hazard.HEATWAVE: "hot extremes and extreme heat",
    Hazard.EXTREME_PRECIP: "heavy precipitation and extreme rainfall",
    Hazard.WIND: "mean wind speed and wind extremes",
}


def _ipcc_question(hazard: Hazard, location: str) -> str:
    """Build the IPCC retrieval question for this hazard + location.

    "intensity and frequency" and "global warming levels" are the exact terms
    the AR6 regional assessment-table rows use — they pull the projection rows,
    not just prose. Location passes through as-is (city -> AR6 region mapping
    is a later, measurable upgrade).
    """
    return (
        f"How are {_IPCC_PHRASES[hazard]} projected to change in intensity "
        f"and frequency over {location} at higher global warming levels?"
    )


def research(state: AgentState) -> dict:
    """Ground the report in IPCC AR6: retrieve + cited answer for this hazard.

    Loud, non-fatal by design: no corpus / no LLM auth / answer failure prints
    a warning and the report ships without citations — degraded, never silent,
    never fabricated.
    """
    question = _ipcc_question(state["hazard"], state["location"])
    try:
        chunks = _ipcc_retriever().retrieve(question, top_k=_IPCC_TOP_K)
        answer = answer_with_guard(question, chunks, cache=_answer_cache())
    except (CorpusError, AnswerError, GeminiError) as exc:
        _log.warning("IPCC grounding unavailable (%s) — report ships without citations", exc)
        return {}
    return {"ipcc_answer": answer, "ipcc_chunks": chunks}


_NO_REGION_NOTE = (
    "This location is outside the AR6 land reference regions (ocean), so no "
    "regional AR6 Ch.12 projection applies."
)
_NOT_FOUND_NOTE = (
    "No AR6 Ch.12 regional projection for this hazard was found in the corpus."
)


def project(state: AgentState) -> dict:
    """Attach the cited AR6 Ch.12 CID projection for this location's AR6 region.

    Guarded on both sides: an ocean point has no AR6 land region, and a region
    with no qualifying Ch.12 sentence yields nothing. Either way the forecast /
    hazard path is untouched and the report carries a note saying which it was —
    an absent section is stated, never silently dropped and never invented.
    """
    region = region_for(state["latitude"], state["longitude"])
    if region is None:
        _log.info("no AR6 land region for this point — skipping the Ch.12 projection")
        return {"projection_note": _NO_REGION_NOTE}
    try:
        change = projected_change_for(region, state["hazard"], retriever=_ipcc_retriever())
    except (CorpusError, EmbeddingError, GeminiError) as exc:
        _log.warning("AR6 Ch.12 projection unavailable (%s) — report ships without it", exc)
        return {"projection_note": _NOT_FOUND_NOTE}
    if change is None:
        return {"projection_note": _NOT_FOUND_NOTE}
    return {"projected_change": change}


def synthesize(state: AgentState) -> dict:
    """Turn the forecast into a structured, provenanced RiskReport."""
    forecast = state["forecast"]
    hazard = state["hazard"]
    assert forecast is not None  # guaranteed by the graph path (plan -> call -> here)

    # `metric` is the quantity compared against the fitted climatology curve.
    # `absolute_metric` is what the Day-1 absolute cutoffs were calibrated on;
    # the two differ for wind only (gust vs sustained speed).
    if hazard is Hazard.EXTREME_PRECIP:
        metric = absolute_metric = max(forecast.precipitation_sum)
        driver = RiskDriver(factor="precipitation", detail=f"max daily {metric} mm")
        summary = f"Peak daily rainfall of {metric} mm over {state['horizon_days']} days."
    elif hazard is Hazard.HEATWAVE:
        metric = absolute_metric = max(forecast.temperature_2m_max)
        driver = RiskDriver(factor="temperature", detail=f"max daily {metric} °C")
        summary = f"Peak daily max temperature of {metric} °C over {state['horizon_days']} days."
    else:  # Hazard.WIND — graded on the GUST, which is what the ERA5 fit describes
        metric = max(forecast.wind_gusts_10m_max)
        absolute_metric = max(forecast.wind_speed_10m_max)
        metric_ms = metric * _KMH_TO_MS
        driver = RiskDriver(
            factor="wind",
            detail=(
                f"max daily gust {metric} km/h ({metric_ms:.1f} m/s), "
                f"max daily sustained {absolute_metric} km/h"
            ),
        )
        summary = (
            f"Peak daily wind gust of {metric} km/h ({metric_ms:.1f} m/s), "
            f"peak sustained wind {absolute_metric} km/h, "
            f"over {state['horizon_days']} days."
        )

    provenance = DataProvenance(
        source="Open-Meteo",
        url=OPEN_METEO_URL,
        retrieved_at=datetime.now(timezone.utc),
        params={
            "latitude": state["latitude"],
            "longitude": state["longitude"],
            "horizon_days": state["horizon_days"],
        },
    )
    drivers = [driver]
    hazard_stats: list[HazardStat] = []
    stat = state.get("hazard_stat")
    if stat is not None:
        hazard_stats = [stat]
        levels_txt = ", ".join(
            f"{r.return_period_years}yr={round(r.level, 1)}" for r in stat.return_levels
        )
        # Non-stationary verdict changes what the numbers MEAN, so the text
        # must say which regime was fitted (effective at latest year vs
        # stationary), and that the trend test ran either way.
        trend = stat.trend
        if trend is not None and trend.significant:
            clim_detail = (
                f"{stat.n_years}-yr ERA5 non-stationary GEV ({stat.variable}), "
                f"levels effective at {trend.evaluated_at_year}"
            )
            summary += (
                f" ERA5 effective return levels at {trend.evaluated_at_year} "
                f"({levels_txt}; warming trend {trend.slope_per_decade:+.1f} "
                f"{stat.unit}/decade, p={trend.p_value:.3f})."
            )
        else:
            clim_detail = f"{stat.n_years}-yr ERA5 GEV ({stat.variable})"
            if trend is not None:
                clim_detail += f", no significant trend (p={trend.p_value:.2f})"
            summary += f" ERA5 return levels ({levels_txt})."
        drivers.append(RiskDriver(factor="climatology", detail=clim_detail))

    # Severity = where the forecast peak lands on THIS location's return-level
    # curve. The absolute cutoffs only survive as the fallback, and the
    # explanation that ships with the band always says which of the two it is.
    if stat is not None and stat.variable == _FORECAST_VARIABLE[hazard]:
        try:
            level, severity_basis = band_from_return_period(metric, stat)
        except ValueError as exc:
            # A curve that does not pin every band edge (a caller-supplied stat
            # fitted at other return periods, say) is a banding problem, not a
            # report-killing one: fall through to the absolute cutoffs and let
            # the explanation carry the reason the reader needs.
            level, severity_basis = band_from_fixed_thresholds(
                absolute_metric, hazard, reason=str(exc)
            )
    elif stat is not None:
        # A caller-injected statistic for a different quantity: real climatology,
        # but not comparable, so say that rather than "no climatology".
        level, severity_basis = band_from_fixed_thresholds(
            absolute_metric,
            hazard,
            reason=(
                f"the fitted climatology variable {stat.variable} is not the "
                f"forecast quantity {_FORECAST_VARIABLE[hazard]}"
            ),
        )
    else:
        level, severity_basis = band_from_fixed_thresholds(absolute_metric, hazard)
    drivers.append(RiskDriver(factor="severity_basis", detail=severity_basis))

    # How much the forecast behind that severity is worth at this horizon. The
    # number is measured, not asserted, and it is the same number that scales
    # the forecast term of the confidence below.
    skill = forecast.skill
    if skill is not None:
        drivers.append(RiskDriver(factor="forecast_skill", detail=skill.detail()))

    citations: list[Citation] = []
    answer = state.get("ipcc_answer")
    if answer is not None and not answer.abstain:
        by_id = {c.chunk_id: c for c in state.get("ipcc_chunks", [])}
        # one Citation per (source, page): several chunks of one table row must
        # not inflate the citation list
        per_page: dict[tuple[str, int], Citation] = {}
        for chunk_id in answer.citations:
            c = by_id[chunk_id]
            per_page[(c.source, c.page)] = Citation(source=c.source, locator=f"p{c.page}")
        citations = list(per_page.values())
        summary += f" IPCC AR6: {answer.answer}"

    # The Ch.12 projection is verbatim corpus text, so it goes into the summary
    # quoted; its absence is stated in words rather than left to be noticed.
    projected = state.get("projected_change")
    if projected is not None:
        summary += (
            f' AR6 Ch.12 projection for {projected.region_name} '
            f'({projected.region_acronym}): "{projected.statement}"'
        )
    else:
        summary += " " + state.get("projection_note", _NOT_FOUND_NOTE)

    confidence = compose_confidence(
        representativeness=stat.representativeness if stat is not None else None,
        ipcc_cited=bool(citations),
        forecast_skill_weight=skill.confidence_weight if skill is not None else None,
    )

    report = RiskReport(
        location=state["location"],
        hazard=hazard,
        horizon_days=state["horizon_days"],
        risk_level=level,
        summary=summary,
        drivers=drivers,
        citations=citations,
        provenance=[provenance],
        hazard_stats=hazard_stats,
        projected_change=projected,
        confidence=confidence,
    )
    return {"report": report}


def _route_after_plan(state: AgentState) -> str:
    """If plan already produced a (refusal) report, skip to the end."""
    return "end" if state.get("report") is not None else "call"


def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("plan", plan)
    builder.add_node("call", call)
    builder.add_node("research", research)
    builder.add_node("project", project)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", _route_after_plan, {"call": "call", "end": END})
    builder.add_edge("call", "research")
    builder.add_edge("research", "project")
    builder.add_edge("project", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


_GRAPH = _build_graph()


def _run_with_events(state: AgentState, on_step: OnStep, horizon_days: int) -> dict:
    """Invoke the graph, reporting each node as it starts and finishes.

    LangGraph's `tasks` stream mode is what makes this possible without touching
    a single node: "Emit events when tasks start and finish, including their
    results and errors" (Pregel.stream docstring, langgraph 1.2.6). Measured on
    this graph, the start chunk is yielded BEFORE the node body runs, so the
    panel can show a step as running rather than only after the fact.

    `values` rides along on the same stream because `Pregel.invoke` is
    documented as returning "the latest output" of a `values` stream, so keeping
    the last `values` chunk is therefore the same final state `invoke` returns,
    which is what preserves run_agent's exact return value.

    Cache hits are read off telemetry rather than guessed: every JsonCache read
    records `cache:<namespace>` with the tier that served it, so the events
    recorded between a node's start and finish say whether it paid or not.
    """
    final: dict = dict(state)
    started_at: dict[str, float] = {}
    telemetry_mark: dict[str, int] = {}
    finished: set[str] = set()
    reported_failure: set[str] = set()
    running: Optional[str] = None

    def emit(node: str, status: StepStatus, seconds: float = 0.0, detail: str = "") -> None:
        emit_step(
            on_step, node, status, seconds=seconds, detail=detail, horizon_days=horizon_days
        )

    try:
        for mode, chunk in _GRAPH.stream(state, stream_mode=["tasks", "values"]):
            if mode == "values":
                final = chunk
                continue
            node = chunk["name"]
            if "result" not in chunk and "error" not in chunk:  # task START chunk
                running = node
                started_at[node] = time.perf_counter()
                telemetry_mark[node] = len(telemetry.snapshot())
                emit(node, StepStatus.STARTED)
                continue
            seconds = time.perf_counter() - started_at.get(node, time.perf_counter())
            records = telemetry.snapshot()[telemetry_mark.get(node, 0) :]
            running = None
            error = chunk.get("error")
            if error is not None:
                reported_failure.add(node)
                emit(node, StepStatus.FAILED, seconds, type(error).__name__)
                continue  # the stream itself raises next; do not swallow it here
            finished.add(node)
            emit(node, StepStatus.FINISHED, seconds, finished_detail(node, chunk.get("result"), records))
    except BaseException as exc:
        # A node that raised outright gets no finish chunk at all, so the panel
        # would otherwise leave it spinning forever. The exception still
        # propagates: the caller decides what the reader is told.
        if running is not None and running not in reported_failure:
            emit(
                running,
                StepStatus.FAILED,
                time.perf_counter() - started_at.get(running, time.perf_counter()),
                type(exc).__name__,
            )
        raise

    # A refusal short-circuits plan -> END, so the remaining nodes never ran.
    # Saying "skipped" is the honest rendering; silence would read as a hang.
    for node in GRAPH_NODES:
        if node not in finished and node not in reported_failure:
            emit(node, StepStatus.SKIPPED)
    return final


def run_agent(
    location: str,
    latitude: float,
    longitude: float,
    hazard: Hazard,
    horizon_days: int = 7,
    hazard_stat: Optional[HazardStat] = None,
    on_step: OnStep = None,
) -> RiskReport:
    """Run the agent end-to-end and return the RiskReport.

    Pass `hazard_stat` (from tools.climatology.climatology_hazard_stat) to ground
    the report in ERA5 GEV climatology and raise its confidence.

    Pass `on_step` to be told what the agent is doing while it does it: it is
    called with one `agent.progress.StepEvent` per node start / finish / skip /
    failure. It is a callback rather than a second generator entry point because
    a callback threads through `run_agent_nl` in one line and leaves this
    function's return value untouched by construction; a generator would need a
    parallel entry point for the NL path and a StopIteration dance to hand the
    report back. Omitting it takes the untouched `invoke` path, so the API, MCP
    and eval callers are byte-for-byte unaffected.
    """
    state: AgentState = {
        "location": location,
        "latitude": latitude,
        "longitude": longitude,
        "hazard": hazard,
        "horizon_days": horizon_days,
    }
    if hazard_stat is not None:
        state["hazard_stat"] = hazard_stat
    if on_step is None:
        final = _GRAPH.invoke(state)
    else:
        final = _run_with_events(state, on_step, horizon_days)
    return final["report"]
