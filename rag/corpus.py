"""The IPCC corpus: one place that knows which PDFs we index and how to load them.

Shared by the eval runners and the MCP server so "the corpus" can never drift
between consumers. Loading is memoized — parsing 439 pages costs a few seconds
and the result is immutable for a given set of PDFs.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from rag.chunk import Chunk, chunk_pages
from rag.parse import extract_pages

CORPUS_DIR = Path("data/ipcc")
CORPUS_FILES = (
    "IPCC_AR6_WGI_SPM.pdf",
    "IPCC_AR6_WGI_Chapter11.pdf",
    "IPCC_AR6_WGI_Chapter12.pdf",
)


class CorpusError(RuntimeError):
    """Raised when the corpus PDFs are not on disk (run scripts/download_ipcc.py)."""


@lru_cache(maxsize=1)
def load_corpus_chunks() -> tuple[Chunk, ...]:
    """Parse + chunk the full corpus (memoized; tuple = immutable)."""
    missing = [name for name in CORPUS_FILES if not (CORPUS_DIR / name).exists()]
    if missing:
        raise CorpusError(f"corpus PDFs missing: {missing} — run scripts/download_ipcc.py")
    pages = []
    for name in CORPUS_FILES:
        pages.extend(extract_pages(CORPUS_DIR / name))
    return tuple(chunk_pages(pages))
