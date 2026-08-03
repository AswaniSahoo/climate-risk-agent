"""weather-mcp: forecast + hazard-climatology tools over MCP (stdio only).

First of the two split servers (ipcc-rag-mcp is the other). Read-only, typed,
hosts hardcoded upstream; coordinates are range-validated at the tool boundary.

Run:  uv run mcp dev tools/weather_mcp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from agent.contracts import Hazard
from tools.climatology import climatology_hazard_stat
from tools.forecast import ForecastResult, get_forecast
from tools.hazard_stats import HazardStat

mcp = MCPServer("weather")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Daily weather forecast",
        read_only_hint=True,
        # Live Open-Meteo API, so the set of entities this can reach is open.
        # destructive_hint/idempotent_hint are only meaningful when
        # read_only_hint is false, so they are deliberately left unset.
        open_world_hint=True,
    )
)
def forecast(latitude: float, longitude: float, horizon_days: int = 7) -> ForecastResult:
    """Daily forecast (precipitation, max temperature, max wind) from Open-Meteo."""
    return get_forecast(latitude, longitude, horizon_days)


@mcp.tool(
    annotations=ToolAnnotations(
        title="ERA5 hazard return levels",
        read_only_hint=True,
        open_world_hint=True,  # fetches ERA5 archive data over the network
    )
)
def hazard_climatology(latitude: float, longitude: float, hazard: Hazard) -> HazardStat:
    """ERA5 return levels (10/50/100-yr) for a hazard at a location, with full provenance."""
    return climatology_hazard_stat(latitude, longitude, hazard)


if __name__ == "__main__":
    mcp.run(transport="stdio")
