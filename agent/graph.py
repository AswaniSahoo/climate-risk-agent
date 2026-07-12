r"""The agent: a 4-node LangGraph that fills a RiskReport for one query.

Flow:  START -> plan -> (call -> research -> synthesize) -> END
                    \------------ refusal -------------/

- plan       : is this hazard answerable with our data? if not, write a refusal.
- call       : run get_forecast, put the ForecastResult on the shared state.
- research   : IPCC AR6 RAG — retrieve + cited LLM answer for this hazard/region.
               Loud, non-fatal: offline/no-LLM degrades to a citation-less report.
- synthesize : turn forecast (+ optional ERA5 climatology + IPCC answer) into a
               RiskReport with page-level Citations.

Every node reads/writes one shared `AgentState` (the "clipboard"). Nodes depend on
the state shape, not on each other — which is what lets this grow into the
MASTER-PLAN's 4 parallel agents by *adding nodes*, not rewiring.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.contracts import (
    Citation,
    DataProvenance,
    Hazard,
    RiskDriver,
    RiskLevel,
    RiskReport,
)
from rag.answer import AnswerError, CitedAnswer, answer_with_guard
from rag.chunk import Chunk
from rag.corpus import CorpusError, load_corpus_chunks
from rag.gemini_client import GeminiError
from rag.retrieve import HybridRetriever
from tools.forecast import ForecastResult, get_forecast, OPEN_METEO_URL
from tools.hazard_stats import HazardStat

# Hazards we can actually answer today (have a data path in get_forecast).
_ANSWERABLE = {Hazard.HEATWAVE, Hazard.EXTREME_PRECIP, Hazard.WIND}
_KMH_TO_MS = 1.0 / 3.6  # Open-Meteo reports km/h; ERA5 return levels are m/s.
_DAY1_CONFIDENCE = 0.3  # low on purpose: crude heuristic, no climatology yet.
_CLIMATOLOGY_CONFIDENCE = 0.6  # bumped when ERA5 GEV climatology grounds the report.
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
    report: Optional[RiskReport]


def _precip_level(max_mm: float) -> RiskLevel:
    """Day-1 flood/precip heuristic on max daily precipitation (mm)."""
    if max_mm < 20:
        return RiskLevel.LOW
    if max_mm < 50:
        return RiskLevel.MODERATE
    if max_mm < 100:
        return RiskLevel.HIGH
    return RiskLevel.SEVERE


def _heat_level(max_c: float) -> RiskLevel:
    """Day-1 heatwave heuristic on max daily temperature (°C)."""
    if max_c < 35:
        return RiskLevel.LOW
    if max_c < 40:
        return RiskLevel.MODERATE
    if max_c < 45:
        return RiskLevel.HIGH
    return RiskLevel.SEVERE


def _wind_level(max_kmh: float) -> RiskLevel:
    """Day-1 wind heuristic on max daily 10 m wind speed (km/h), Beaufort-inspired."""
    if max_kmh < 40:
        return RiskLevel.LOW
    if max_kmh < 62:
        return RiskLevel.MODERATE
    if max_kmh < 88:
        return RiskLevel.HIGH
    return RiskLevel.SEVERE


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
    """Fetch live weather for the requested location."""
    forecast = get_forecast(
        latitude=state["latitude"],
        longitude=state["longitude"],
        horizon_days=state["horizon_days"],
    )
    return {"forecast": forecast}


@lru_cache(maxsize=1)
def _ipcc_retriever() -> HybridRetriever:
    """Build (once) the measured hybrid retriever over the on-disk IPCC corpus."""
    return HybridRetriever.build(list(load_corpus_chunks()))


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
        answer = answer_with_guard(question, chunks)
    except (CorpusError, AnswerError, GeminiError) as exc:
        print(f"[research] IPCC grounding unavailable ({exc}) — report ships without citations")
        return {}
    return {"ipcc_answer": answer, "ipcc_chunks": chunks}


def synthesize(state: AgentState) -> dict:
    """Turn the forecast into a structured, provenanced RiskReport."""
    forecast = state["forecast"]
    hazard = state["hazard"]
    assert forecast is not None  # guaranteed by the graph path (plan -> call -> here)

    if hazard is Hazard.EXTREME_PRECIP:
        metric = max(forecast.precipitation_sum)
        level = _precip_level(metric)
        driver = RiskDriver(factor="precipitation", detail=f"max daily {metric} mm")
        summary = f"Peak daily rainfall of {metric} mm over {state['horizon_days']} days."
    elif hazard is Hazard.HEATWAVE:
        metric = max(forecast.temperature_2m_max)
        level = _heat_level(metric)
        driver = RiskDriver(factor="temperature", detail=f"max daily {metric} °C")
        summary = f"Peak daily max temperature of {metric} °C over {state['horizon_days']} days."
    else:  # Hazard.WIND — km/h from Open-Meteo, also shown in m/s for ERA5 comparability
        metric = max(forecast.wind_speed_10m_max)
        metric_ms = metric * _KMH_TO_MS
        level = _wind_level(metric)
        driver = RiskDriver(
            factor="wind", detail=f"max daily {metric} km/h ({metric_ms:.1f} m/s)"
        )
        summary = (
            f"Peak daily wind of {metric} km/h ({metric_ms:.1f} m/s) "
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
    confidence = _DAY1_CONFIDENCE
    stat = state.get("hazard_stat")
    if stat is not None:
        hazard_stats = [stat]
        confidence = _CLIMATOLOGY_CONFIDENCE
        levels_txt = ", ".join(
            f"{r.return_period_years}yr={round(r.level, 1)}" for r in stat.return_levels
        )
        drivers.append(
            RiskDriver(
                factor="climatology",
                detail=f"{stat.n_years}-yr ERA5 GEV ({stat.variable})",
            )
        )
        summary += f" ERA5 return levels ({levels_txt})."

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
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", _route_after_plan, {"call": "call", "end": END})
    builder.add_edge("call", "research")
    builder.add_edge("research", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


_GRAPH = _build_graph()


def run_agent(
    location: str,
    latitude: float,
    longitude: float,
    hazard: Hazard,
    horizon_days: int = 7,
    hazard_stat: Optional[HazardStat] = None,
) -> RiskReport:
    """Run the agent end-to-end and return the RiskReport.

    Pass `hazard_stat` (from tools.climatology.climatology_hazard_stat) to ground
    the report in ERA5 GEV climatology and raise its confidence.
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
    final = _GRAPH.invoke(state)
    return final["report"]
