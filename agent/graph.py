r"""The agent: a minimal 3-node LangGraph that fills a RiskReport for one query.

Flow:  START -> plan -> (call -> synthesize) -> END
                    \-------- refusal --------/

- plan       : is this hazard answerable with our data? if not, write a refusal.
- call       : run get_forecast, put the ForecastResult on the shared state.
- synthesize : turn the forecast into a RiskReport (deterministic Day-1 heuristic;
               real ERA5/IPCC grounding comes later).

Every node reads/writes one shared `AgentState` (the "clipboard"). Nodes depend on
the state shape, not on each other — which is what lets this grow into the
MASTER-PLAN's 4 parallel agents by *adding nodes*, not rewiring.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.contracts import (
    DataProvenance,
    Hazard,
    RiskDriver,
    RiskLevel,
    RiskReport,
)
from tools.forecast import ForecastResult, get_forecast, OPEN_METEO_URL

# Hazards we can actually answer today (have a data path in get_forecast).
_ANSWERABLE = {Hazard.HEATWAVE, Hazard.EXTREME_PRECIP}
_DAY1_CONFIDENCE = 0.3  # low on purpose: crude heuristic, no climatology yet.


class AgentState(TypedDict, total=False):
    """The shared clipboard passed between nodes."""

    location: str
    latitude: float
    longitude: float
    hazard: Hazard
    horizon_days: int
    forecast: Optional[ForecastResult]
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
    else:  # Hazard.HEATWAVE
        metric = max(forecast.temperature_2m_max)
        level = _heat_level(metric)
        driver = RiskDriver(factor="temperature", detail=f"max daily {metric} °C")
        summary = f"Peak daily max temperature of {metric} °C over {state['horizon_days']} days."

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
    report = RiskReport(
        location=state["location"],
        hazard=hazard,
        horizon_days=state["horizon_days"],
        risk_level=level,
        summary=summary,
        drivers=[driver],
        provenance=[provenance],
        confidence=_DAY1_CONFIDENCE,
    )
    return {"report": report}


def _route_after_plan(state: AgentState) -> str:
    """If plan already produced a (refusal) report, skip to the end."""
    return "end" if state.get("report") is not None else "call"


def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("plan", plan)
    builder.add_node("call", call)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", _route_after_plan, {"call": "call", "end": END})
    builder.add_edge("call", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


_GRAPH = _build_graph()


def run_agent(
    location: str,
    latitude: float,
    longitude: float,
    hazard: Hazard,
    horizon_days: int = 7,
) -> RiskReport:
    """Run the agent end-to-end and return the RiskReport."""
    final = _GRAPH.invoke(
        {
            "location": location,
            "latitude": latitude,
            "longitude": longitude,
            "hazard": hazard,
            "horizon_days": horizon_days,
        }
    )
    return final["report"]
