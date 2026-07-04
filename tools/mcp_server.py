"""MCP server exposing the climate tools over the Model Context Protocol.

A thin wrapper: it imports the existing get_forecast function and re-exposes it
as a standard MCP tool, so any MCP client (Claude Desktop, Cursor, ...) can call
it. The tool logic itself lives in tools/forecast.py and is unchanged.

Run as a stdio server:  uv run python -m tools.mcp_server
"""
from mcp.server.fastmcp import FastMCP

from tools.forecast import get_forecast

mcp = FastMCP("climate-tools")


@mcp.tool()
def forecast(latitude: float, longitude: float, horizon_days: int = 7) -> dict:
    """Daily precipitation (mm) and max temperature (°C) for the next N days at a location."""
    return get_forecast(latitude, longitude, horizon_days).model_dump(mode="json")


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
