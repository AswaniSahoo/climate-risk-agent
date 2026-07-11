"""weather-mcp: forecast + hazard-climatology tools over MCP (stdio only).

First of the two split servers (ipcc-rag-mcp is the other). Read-only, typed,
hosts hardcoded upstream; coordinates are range-validated at the tool boundary.

Run:  uv run mcp dev tools/weather_mcp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from agent.contracts import Hazard
from tools.climatology import climatology_hazard_stat
from tools.forecast import ForecastResult, get_forecast
from tools.hazard_stats import HazardStat

mcp = FastMCP("weather")


@mcp.tool()
def forecast(latitude: float, longitude: float, horizon_days: int = 7) -> ForecastResult:
    """Daily forecast (precipitation, max temperature, max wind) from Open-Meteo."""
    return get_forecast(latitude, longitude, horizon_days)


@mcp.tool()
def hazard_climatology(latitude: float, longitude: float, hazard: Hazard) -> HazardStat:
    """ERA5 return levels (10/50/100-yr) for a hazard at a location, with full provenance."""
    return climatology_hazard_stat(latitude, longitude, hazard)


if __name__ == "__main__":
    mcp.run()
