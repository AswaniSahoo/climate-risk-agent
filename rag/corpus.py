"""The IPCC corpus: one place that knows which PDFs we index and how to load them.

Shared by the eval runners and the MCP server so "the corpus" can never drift
between consumers. Loading is memoized — parsing 439 pages costs a few seconds
and the result is immutable for a given set of PDFs.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from rag.chunk import Chunk, chunk_pages
from rag.parse import extract_pages
from tools.cache_backend import source_bytes

_log = logging.getLogger(__name__)

CORPUS_DIR = Path("data/ipcc")
CORPUS_FILES = (
    "IPCC_AR6_WGI_SPM.pdf",
    "IPCC_AR6_WGI_Chapter11.pdf",
    "IPCC_AR6_WGI_Chapter12.pdf",
)

# The SHA-256 of each PDF. These files are pypdf's entire input, so they are
# pinned: scripts/download_ipcc.py checks a download, or a copy it finds on
# disk, before keeping it, and _parse_corpus checks again right before pypdf
# runs, since the runtime parses data/ipcc on its own when the chunk cache is
# missing or stale. Pinned 2026-11-11: the copies this project has been parsing
# and a fresh download from ipcc.ch that day hashed the same. If the IPCC
# republishes a file, re-download it, re-run the eval, then change the hash.
CORPUS_SHA256 = {
    "IPCC_AR6_WGI_SPM.pdf": "37c9fa760757524c8d69630a89cecab9ebcacb66786970a3439da0619875e446",
    "IPCC_AR6_WGI_Chapter11.pdf": "d0dc4fae531a7b6a76a456cb12db9c7d68099265e377973d508484c3289ddce1",
    "IPCC_AR6_WGI_Chapter12.pdf": "4f1bb6b1ecce77b7a0f7f2322a2ba34e53c1a1e5bcb48d5db7f6a2d80b0fd4a0",
}

# Parsing 439 PDF pages into 2730 chunks costs ~70 s (measured), and it is pure
# function of (the PDFs, the chunking code). On Cloud Run that ran on EVERY cold
# container and dominated the 219 s first response. So the result is cached to
# disk and baked into the image at build time.
CHUNK_CACHE = Path("data/cache/chunks.json")


_CHUNKER_SOURCE = Path(__file__).parent / "chunk.py"  # module attr so tests can redirect it


def corpus_file_sizes() -> dict[str, int | None]:
    """Each corpus PDF's size in bytes, or None when it is not on disk."""
    sizes: dict[str, int | None] = {}
    for name in CORPUS_FILES:
        path = CORPUS_DIR / name
        sizes[name] = path.stat().st_size if path.exists() else None
    return sizes


def fingerprint_for(sizes: Mapping[str, int | None]) -> str:
    """The fingerprint `_fingerprint` returns for these PDF sizes.

    Pure in `sizes`, so a check that runs without the PDFs (CI) can rebuild
    the corpus version a shipped cache entry was keyed with. The chunker's
    source is read with LF line endings, so a CRLF checkout agrees.
    """
    h = hashlib.sha256()
    for name in CORPUS_FILES:
        size = sizes.get(name)
        h.update(name.encode())
        h.update(str(size).encode() if size is not None else b"missing")
    h.update(source_bytes(_CHUNKER_SOURCE))
    return h.hexdigest()[:16]


def _fingerprint() -> str:
    """Identity of a chunking result: the PDFs plus the code that chunks them.

    Hashing rag/chunk.py's source means a re-chunk invalidates the cache
    automatically, with no version constant anyone can forget to bump.
    """
    return fingerprint_for(corpus_file_sizes())


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


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_pins() -> None:
    """Refuse to parse any corpus PDF whose bytes are not the pinned ones."""
    for name in CORPUS_FILES:
        actual = file_sha256(CORPUS_DIR / name)
        if actual != CORPUS_SHA256[name]:
            raise CorpusError(
                f"{name} does not match its pinned sha256 (expected {CORPUS_SHA256[name]}, "
                f"got {actual}); refusing to parse it. Delete it and run scripts/download_ipcc.py"
            )


def _parse_corpus() -> tuple[Chunk, ...]:
    """The expensive path: extract every page, then chunk it (~70 s)."""
    missing = [name for name in CORPUS_FILES if not (CORPUS_DIR / name).exists()]
    if missing:
        raise CorpusError(f"corpus PDFs missing: {missing} — run scripts/download_ipcc.py")
    _check_pins()  # before pypdf reads a byte
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
