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
from tools.hazard_stats import HazardStat

# Hazards we can actually answer today (have a data path in get_forecast).
_ANSWERABLE = {Hazard.HEATWAVE, Hazard.EXTREME_PRECIP, Hazard.WIND}
_KMH_TO_MS = 1.0 / 3.6  # Open-Meteo reports km/h; ERA5 return levels are m/s.
_DAY1_CONFIDENCE = 0.3  # low on purpose: crude heuristic, no climatology yet.
_CLIMATOLOGY_CONFIDENCE = 0.6  # bumped when ERA5 GEV climatology grounds the report.


class AgentState(TypedDict, total=False):
    """The shared clipboard passed between nodes."""

    location: str
    latitude: float
    longitude: float
    hazard: Hazard
    horizon_days: int
    forecast: Optional[ForecastResult]
    hazard_stat: Optional[HazardStat]  # optional ERA5 climatology, injected by caller
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

    report = RiskReport(
        location=state["location"],
        hazard=hazard,
        horizon_days=state["horizon_days"],
        risk_level=level,
        summary=summary,
        drivers=drivers,
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
    hazard_stat: Optional[HazardStat] = None,
) -> RiskReport:
    """Run the agent end-to-end and return the RiskReport.

    Pass `hazard_stat` (from tools.era5.era5_hazard_stat) to ground the report in
    ERA5 GEV climatology and raise its confidence.
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
