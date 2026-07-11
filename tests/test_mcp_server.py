"""Tests for the MCP server (tools/mcp_server.py).

Uses FastMCP's in-memory calls (list_tools / call_tool) — no subprocess, no
network. HTTP is still mocked via pytest-httpx, so the whole thing is offline
and deterministic.
"""
import json

from tools.mcp_server import mcp

CANNED = {
    "latitude": 22.26,
    "longitude": 84.85,
    "timezone": "Asia/Kolkata",
    "daily_units": {
        "time": "iso8601",
        "precipitation_sum": "mm",
        "temperature_2m_max": "°C",
        "wind_speed_10m_max": "km/h",
    },
    "daily": {
        "time": ["2026-07-02", "2026-07-03", "2026-07-04"],
        "precipitation_sum": [14.2, 0.1, 55.0],
        "temperature_2m_max": [35.1, 36.4, 33.0],
        "wind_speed_10m_max": [70.0, 45.0, 30.0],
    },
}


async def test_forecast_tool_is_registered():
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert "forecast" in names


async def test_forecast_tool_returns_parsed_series(httpx_mock):
    httpx_mock.add_response(json=CANNED)

    result = await mcp.call_tool(
        "forecast", {"latitude": 22.26, "longitude": 84.85, "horizon_days": 3}
    )

    payload = json.loads(result[0].text)
    assert payload["precipitation_sum"] == [14.2, 0.1, 55.0]
    assert payload["timezone"] == "Asia/Kolkata"
