"""ipcc-rag-mcp: the IPCC retrieval + cited-answer tools over MCP (stdio only).

Second of the two split servers (weather-mcp is the other). Tools are narrow,
typed, and read-only per the security model:
- search_ipcc: BM25 over the corpus -> page-cited excerpts (no LLM, no key)
- answer_ipcc: scope guard + retrieval + cited LLM answer (needs GEMINI_API_KEY)

Run:  uv run mcp dev tools/ipcc_mcp.py
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from rag.corpus import load_corpus_chunks
from rag.retrieve import HybridRetriever

mcp = FastMCP("ipcc-rag")

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


@lru_cache(maxsize=1)
def _retriever() -> HybridRetriever:
    return HybridRetriever.build(list(load_corpus_chunks()))


@mcp.tool()
def search_ipcc(question: str, top_k: int = 5) -> list[Excerpt]:
    """Search the IPCC AR6 WG1 corpus (SPM, Ch.11, Ch.12); returns page-cited excerpts."""
    top_k = max(1, min(top_k, _MAX_TOP_K))
    return [
        Excerpt(chunk_id=c.chunk_id, source=c.source, page=c.page, text=c.text)
        for c in _retriever().retrieve(question, top_k=top_k)
    ]


@mcp.tool()
def answer_ipcc(question: str) -> dict:
    """Answer a climate question with citations from the IPCC corpus.

    Refuses out-of-scope hazards (deterministic guard) and questions the corpus
    cannot support. Citations are validated against retrieved chunks.
    """
    from rag.answer import answer_with_guard
    from rag.answer_cache import AnswerCache

    result = answer_with_guard(
        question,
        _retriever().retrieve(question, top_k=_ANSWER_TOP_K),
        cache=AnswerCache(Path("data/cache/answers")),
    )
    return result.model_dump(exclude={"allowed_ids"})


if __name__ == "__main__":
    mcp.run()
