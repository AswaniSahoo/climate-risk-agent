"""Tests for the split MCP servers (weather-mcp + ipcc-rag-mcp), in-memory.

No network, no corpus PDFs: upstream functions are monkeypatched and the
corpus loader is stubbed with canned chunks — these tests pin the MCP wiring
(tool names, schemas, result shapes), not the underlying logic (tested elsewhere).
"""
import json

import pytest

import tools.ipcc_mcp as ipcc_mcp
import tools.weather_mcp as weather_mcp
from rag.answer import CitedAnswer
from rag.chunk import Chunk


@pytest.fixture(autouse=True)
def _fresh_ipcc_index(monkeypatch):
    canned = (
        Chunk(chunk_id="doc.pdf#p5#0", source="doc.pdf", page=5,
              text="heavy precipitation intensifies about 7% per degree of warming"),
        Chunk(chunk_id="doc.pdf#p9#0", source="doc.pdf", page=9,
              text="glaciers are committed to continue melting for centuries"),
    )
    monkeypatch.setattr(ipcc_mcp, "load_corpus_chunks", lambda: canned)
    # keep tests offline: corpus embedding degrades to the loud BM25-only path
    import rag.retrieve as retrieve_mod

    def no_network(texts, *, task_type, cache):
        raise retrieve_mod.EmbeddingError("offline test")

    monkeypatch.setattr(retrieve_mod, "cached_embed_texts", no_network)
    ipcc_mcp._retriever.cache_clear()
    yield
    ipcc_mcp._retriever.cache_clear()


async def test_weather_server_exposes_both_tools():
    tools = {t.name for t in await weather_mcp.mcp.list_tools()}
    assert tools == {"forecast", "hazard_climatology"}


def _parsed(result) -> list[dict]:
    """Normalize a CallToolResult to a list of dicts.

    SDK v2 hands back a fully constructed CallToolResult (no auto-wrapping):
    one TextContent block per returned item, plus `structured_content` carrying
    the same data typed.
    """
    assert result.is_error is False, result.content
    return [json.loads(block.text) for block in result.content]


async def test_ipcc_search_returns_page_cited_excerpts():
    result = await ipcc_mcp.mcp.call_tool("search_ipcc", {"question": "heavy precipitation", "top_k": 1})
    [excerpt] = _parsed(result)
    assert excerpt["chunk_id"] == "doc.pdf#p5#0"
    assert excerpt["page"] == 5


async def test_ipcc_search_caps_top_k():
    result = await ipcc_mcp.mcp.call_tool("search_ipcc", {"question": "warming melting", "top_k": 999})
    assert len(_parsed(result)) <= ipcc_mcp._MAX_TOP_K


async def test_ipcc_answer_delegates_to_guarded_answerer(monkeypatch):
    def fake_answer(question, chunks, **_):
        return CitedAnswer(
            answer="cited answer", citations=[chunks[0].chunk_id], abstain=False,
            allowed_ids=[c.chunk_id for c in chunks],
        )

    import rag.answer

    monkeypatch.setattr(rag.answer, "answer_with_guard", fake_answer)
    result = await ipcc_mcp.mcp.call_tool(
        "answer_ipcc", {"question": "how much does heavy precipitation intensify?"}
    )
    [payload] = _parsed(result)
    assert payload["abstain"] is False
    assert payload["citations"] == ["doc.pdf#p5#0"]
    assert "allowed_ids" not in payload


async def test_every_tool_is_annotated_as_read_only():
    """Clients build their safety guardrails from annotations.

    An unannotated tool has to be treated as potentially destructive, so a
    client may gate it behind a confirmation prompt. All four of ours only read.
    """
    for server in (weather_mcp.mcp, ipcc_mcp.mcp):
        for tool in await server.list_tools():
            assert tool.annotations is not None, f"{tool.name} has no annotations"
            assert tool.annotations.read_only_hint is True, f"{tool.name} not marked read-only"
            assert tool.annotations.title, f"{tool.name} has no human-readable title"


async def test_open_world_hint_matches_whether_the_tool_leaves_the_corpus():
    """forecast and hazard_climatology call live external APIs (open world).
    The IPCC tools answer from a fixed corpus shipped with the repo (closed)."""
    weather = {t.name: t for t in await weather_mcp.mcp.list_tools()}
    ipcc = {t.name: t for t in await ipcc_mcp.mcp.list_tools()}
    assert weather["forecast"].annotations.open_world_hint is True
    assert weather["hazard_climatology"].annotations.open_world_hint is True
    assert ipcc["search_ipcc"].annotations.open_world_hint is False
    assert ipcc["answer_ipcc"].annotations.open_world_hint is False


async def test_answer_ipcc_publishes_an_output_schema():
    """A bare dict return leaves clients with nothing to validate against."""
    [tool] = [t for t in await ipcc_mcp.mcp.list_tools() if t.name == "answer_ipcc"]
    assert tool.output_schema, "answer_ipcc must declare an output_schema"
    assert "allowed_ids" not in (tool.output_schema.get("properties") or {})


async def test_tools_are_listed_in_a_deterministic_order():
    """MCP 2026-07-28 says servers SHOULD return tools/list in a deterministic
    order, so clients can cache the list and LLM prompt caches stay warm."""
    for server in (weather_mcp.mcp, ipcc_mcp.mcp):
        first = [t.name for t in await server.list_tools()]
        second = [t.name for t in await server.list_tools()]
        assert first == second
