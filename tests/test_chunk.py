"""Tests for chunking page text into retrievable units (rag/chunk.py).

Chunks must never lose the page number (page-level citations depend on it),
never split a word in half, and overlap slightly so a sentence straddling a
boundary is still retrievable.
"""
import re

from rag.chunk import Chunk, chunk_pages
from rag.parse import PageText

WORDS = [f"w{i}" for i in range(1000)]
LONG_TEXT = " ".join(WORDS)


def _page(text: str, page: int = 7) -> PageText:
    return PageText(source="IPCC_AR6_WGI_Chapter11.pdf", page=page, text=text)


def test_short_page_becomes_one_chunk_carrying_source_and_page():
    chunks = chunk_pages([_page("a short page")])

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == "a short page"
    assert chunks[0].page == 7
    assert chunks[0].source == "IPCC_AR6_WGI_Chapter11.pdf"


def test_long_page_splits_into_bounded_chunks_with_unique_ids():
    chunks = chunk_pages([_page(LONG_TEXT)], max_chars=1200, overlap=150)

    assert len(chunks) > 1
    assert all(len(c.text) <= 1200 for c in chunks)
    assert all(c.page == 7 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunks_never_split_a_word_and_lose_nothing():
    chunks = chunk_pages([_page(LONG_TEXT)], max_chars=1200, overlap=150)

    for chunk in chunks:
        assert all(re.fullmatch(r"w\d+", tok) for tok in chunk.text.split())

    seen = {tok for chunk in chunks for tok in chunk.text.split()}
    assert seen == set(WORDS)


def test_consecutive_chunks_overlap():
    chunks = chunk_pages([_page(LONG_TEXT)], max_chars=1200, overlap=150)

    last_word_of_first = chunks[0].text.split()[-1]
    assert last_word_of_first in chunks[1].text.split()
