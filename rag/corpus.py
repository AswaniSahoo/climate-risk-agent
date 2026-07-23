"""The IPCC corpus: one place that knows which PDFs we index and how to load them.

Shared by the eval runners and the MCP server so "the corpus" can never drift
between consumers. Loading is memoized — parsing 439 pages costs a few seconds
and the result is immutable for a given set of PDFs.
"""
from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path

from rag.chunk import Chunk, chunk_pages
from rag.parse import extract_pages

_log = logging.getLogger(__name__)

CORPUS_DIR = Path("data/ipcc")
CORPUS_FILES = (
    "IPCC_AR6_WGI_SPM.pdf",
    "IPCC_AR6_WGI_Chapter11.pdf",
    "IPCC_AR6_WGI_Chapter12.pdf",
)

# Parsing 439 PDF pages into 2730 chunks costs ~70 s (measured), and it is pure
# function of (the PDFs, the chunking code). On Cloud Run that ran on EVERY cold
# container and dominated the 219 s first response. So the result is cached to
# disk and baked into the image at build time.
CHUNK_CACHE = Path("data/cache/chunks.json")


def _fingerprint() -> str:
    """Identity of a chunking result: the PDFs plus the code that chunks them.

    Hashing rag/chunk.py's source means a re-chunk invalidates the cache
    automatically — no version constant anyone can forget to bump.
    """
    h = hashlib.sha256()
    for name in CORPUS_FILES:
        path = CORPUS_DIR / name
        h.update(name.encode())
        h.update(str(path.stat().st_size).encode() if path.exists() else b"missing")
    h.update((Path(__file__).parent / "chunk.py").read_bytes())
    return h.hexdigest()[:16]


class CorpusError(RuntimeError):
    """Raised when the corpus PDFs are not on disk (run scripts/download_ipcc.py)."""


def corpus_present() -> bool:
    """Cheap pre-check: are all corpus PDFs on disk? (No parsing, no exception —
    lets an entrypoint decide to fetch the corpus before first use, e.g. a fresh
    Cloud Run container where the Docker bake step never ran.)"""
    return all((CORPUS_DIR / name).exists() for name in CORPUS_FILES)


def build_chunk_cache() -> int:
    """Parse the corpus and write the chunk cache. Called at Docker build time
    (and safe to call locally); returns the number of chunks written."""
    chunks = _parse_corpus()
    CHUNK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CHUNK_CACHE.write_text(
        json.dumps(
            {
                "fingerprint": _fingerprint(),
                "chunks": [c.model_dump() for c in chunks],
            }
        ),
        encoding="utf-8",
    )
    _log.info("chunk cache written: %d chunks -> %s", len(chunks), CHUNK_CACHE)
    return len(chunks)


def _parse_corpus() -> tuple[Chunk, ...]:
    """The expensive path: extract every page, then chunk it (~70 s)."""
    missing = [name for name in CORPUS_FILES if not (CORPUS_DIR / name).exists()]
    if missing:
        raise CorpusError(f"corpus PDFs missing: {missing} — run scripts/download_ipcc.py")
    pages = []
    for name in CORPUS_FILES:
        pages.extend(extract_pages(CORPUS_DIR / name))
    return tuple(chunk_pages(pages))


@lru_cache(maxsize=1)
def load_corpus_chunks() -> tuple[Chunk, ...]:
    """The corpus as chunks (memoized; tuple = immutable).

    Reads the on-disk cache when its fingerprint matches the current PDFs and
    chunking code, otherwise parses from scratch. A stale or corrupt cache is a
    LOUD miss that falls back to parsing — never a silently wrong corpus.
    """
    if CHUNK_CACHE.exists():
        try:
            payload = json.loads(CHUNK_CACHE.read_text(encoding="utf-8"))
            if payload.get("fingerprint") == _fingerprint():
                chunks = tuple(Chunk.model_validate(c) for c in payload["chunks"])
                _log.info("corpus loaded from chunk cache (%d chunks)", len(chunks))
                return chunks
            _log.warning("chunk cache is stale (corpus or chunker changed) — reparsing")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            _log.warning("chunk cache unreadable (%s) — reparsing", exc)
    return _parse_corpus()
