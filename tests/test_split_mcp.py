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
    """Normalize call_tool output to a list of dicts.

    FastMCP returns either list[TextContent] or (list[TextContent], structured)
    depending on the tool's return shape — accept both.
    """
    if isinstance(result, tuple):
        result = result[0]
    blocks = [json.loads(item.text) for item in result]
    if len(blocks) == 1 and isinstance(blocks[0], list):
        return blocks[0]
    return blocks


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
