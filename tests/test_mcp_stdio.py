"""Boot both MCP servers as REAL subprocesses and speak the protocol to them.

Every other MCP test calls tools in-memory, so a server that cannot actually
start -- broken import, missing entrypoint, a stray print() polluting stdout --
still passes them. These spawn the process and do a real stdio handshake.

Deliberately stops at list_tools: calling a tool needs network + credentials.
The credential path that once hung this server (google-auth shelling out to the
`gcloud` CLI, whose child never returns once FastMCP owns stdio) is pinned by
tests/test_gemini_client.py instead.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOT_TIMEOUT_S = 120  # generous: cold import of numpy/bm25; a hang must fail, not wait

SERVERS = [
    ("tools.weather_mcp", {"forecast", "hazard_climatology"}),
    ("tools.ipcc_mcp", {"search_ipcc", "answer_ipcc"}),
]


async def _tool_names(module: str) -> set[str]:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", module],
        env=dict(os.environ),
        cwd=str(REPO_ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=BOOT_TIMEOUT_S)
            listed = await asyncio.wait_for(session.list_tools(), timeout=BOOT_TIMEOUT_S)
            return {tool.name for tool in listed.tools}


@pytest.mark.parametrize("module,expected", SERVERS)
async def test_server_boots_over_stdio_and_advertises_its_tools(module, expected):
    assert await _tool_names(module) == expected
