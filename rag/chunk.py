"""Split page text into retrievable chunks — without losing the page number.

Three invariants, each pinned by a test:
1. every chunk keeps its `source` + `page`  -> page-level citations stay possible
2. chunks never split a word in half        -> cuts land on whitespace
3. consecutive chunks overlap               -> a sentence across a boundary is
                                               still retrievable from one chunk
"""
from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from rag.parse import PageText

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 150


class Chunk(BaseModel):
    """One retrievable unit of an IPCC document, traceable to a page."""

    chunk_id: str
    source: str
    page: int = Field(ge=1)
    text: str


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Cut `text` into <=max_chars windows that start and end on word boundaries."""
    if len(text) <= max_chars:
        return [text] if text else []

    pieces: list[str] = []
    start, length = 0, len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            space = text.rfind(" ", start, end)
            if space > start:
                end = space  # cut on whitespace, never mid-word

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break

        candidate = end - overlap
        if candidate <= start:
            candidate = end
        space = text.rfind(" ", start, candidate)
        next_start = space + 1 if space != -1 else candidate
        start = next_start if next_start > start else end + 1

    return pieces


def chunk_pages(
    pages: Iterable[PageText],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Turn parsed pages into overlapping, page-tagged chunks."""
    chunks: list[Chunk] = []
    for page in pages:
        for index, piece in enumerate(_split_text(page.text, max_chars, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{page.source}#p{page.page}#{index}",
                    source=page.source,
                    page=page.page,
                    text=piece,
                )
            )
    return chunks
