"""ipcc-rag-mcp: the IPCC retrieval + cited-answer tools over MCP (stdio only).

Second of the two split servers (weather-mcp is the other). Tools are narrow,
typed, and read-only per the security model:
- search_ipcc: hybrid BM25 + dense retrieval -> page-cited excerpts. No LLM, but
  the dense half embeds the query, so this DOES need Gemini/Vertex credentials.
- answer_ipcc: scope guard + retrieval + cited LLM answer (generation on top)

Run:  uv run mcp dev tools/ipcc_mcp.py
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from rag.corpus import load_corpus_chunks
from rag.retrieve import HybridRetriever

# An MCP client hands the server a deliberately SHORT allow-list of environment
# variables (PATH, APPDATA, TEMP...) so that a server cannot harvest the user's
# secrets. Our Gemini/Vertex settings are not on that list, so a client-launched
# server starts with no credentials even when the user's own shell has them --
# measured in MCP Inspector: search_ipcc degraded to BM25-only and answer_ipcc
# failed outright. Reading .env here makes the server self-sufficient under any
# client. override=False, so anything the client DID pass always wins.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

mcp = MCPServer("ipcc-rag")

_MAX_TOP_K = 10  # denial-of-wallet + context-size cap on the tool surface
# A/B-measured on the frozen set (2026-07-12): k=8 admits table-header chunks
# (GWL column labels) -> fewer false refusals on regional-table rows, false_answer 0
_ANSWER_TOP_K = 8


class Excerpt(BaseModel):
    """One retrieved excerpt, traceable to a PDF page."""

    chunk_id: str
    source: str
    page: int = Field(ge=1)
    text: str


class Answer(BaseModel):
    """A cited answer, or a grounded refusal to give one."""

    answer: str
    citations: list[str] = Field(
        default_factory=list,
        description="chunk_ids backing the answer, validated against what retrieval returned",
    )
    abstain: bool
    abstain_reason: str = ""


@lru_cache(maxsize=1)
def _retriever() -> HybridRetriever:
    return HybridRetriever.build(list(load_corpus_chunks()))


@mcp.tool(
    annotations=ToolAnnotations(
        title="Search IPCC AR6 WG1",
        read_only_hint=True,
        # Closed world: a fixed corpus shipped with the repo, not the open web.
        open_world_hint=False,
    )
)
def search_ipcc(question: str, top_k: int = 5) -> list[Excerpt]:
    """Search the IPCC AR6 WG1 corpus (SPM, Ch.11, Ch.12); returns page-cited excerpts."""
    top_k = max(1, min(top_k, _MAX_TOP_K))
    return [
        Excerpt(chunk_id=c.chunk_id, source=c.source, page=c.page, text=c.text)
        for c in _retriever().retrieve(question, top_k=top_k)
    ]


@mcp.tool(
    annotations=ToolAnnotations(
        title="Answer from IPCC AR6 with citations",
        read_only_hint=True,
        open_world_hint=False,
    )
)
def answer_ipcc(question: str) -> Answer:
    """Answer a climate question with citations from the IPCC corpus.

    Refuses out-of-scope hazards (deterministic guard) and questions the corpus
    cannot support. Citations are validated against retrieved chunks.
    """
    from rag.answer import answer_with_guard
    from rag.answer_cache import AnswerCache

    result = answer_with_guard(
        question,
        _retriever().retrieve(question, top_k=_ANSWER_TOP_K),
        cache=AnswerCache(),  # process-wide backend: shared Redis tier when configured
    )
    # allowed_ids is retrieval bookkeeping, not part of the answer contract
    return Answer.model_validate(result.model_dump(exclude={"allowed_ids"}))


if __name__ == "__main__":
    mcp.run(transport="stdio")
