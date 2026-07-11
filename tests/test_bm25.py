"""Tests for the zero-dependency BM25 index (rag/bm25.py).

Pure and offline. Pins the properties retrieval correctness rests on:
tokenization (NFKC fixes PDF ligatures), idf (rare terms dominate), and
deterministic top-k ranking.
"""
from rag.bm25 import BM25Index, tokenize
from rag.chunk import Chunk


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"doc#p1#{i}", source="doc.pdf", page=1, text=text)


def test_tokenize_normalizes_ligatures_and_case():
    # pypdf extracts "influence" with the ﬂ ligature glyph; NFKC folds it back.
    assert tokenize("Human inﬂuence") == ["human", "influence"]


def test_query_term_ranks_containing_chunk_first():
    index = BM25Index(
        [
            _chunk(0, "heavy precipitation intensifies with warming"),
            _chunk(1, "glaciers are committed to continue melting"),
            _chunk(2, "wind gusts over coastal regions"),
        ]
    )
    results = index.query("glaciers melting", top_k=3)
    assert results[0][0].chunk_id == "doc#p1#1"


def test_rare_term_outweighs_common_term():
    # "warming" is everywhere (low idf); "stilling" appears once (high idf).
    index = BM25Index(
        [
            _chunk(0, "warming trends and warming projections"),
            _chunk(1, "surface wind stilling under warming"),
            _chunk(2, "warming of oceans and warming of land"),
        ]
    )
    results = index.query("stilling warming", top_k=3)
    assert results[0][0].chunk_id == "doc#p1#1"


def test_top_k_limits_and_orders_by_score():
    index = BM25Index(
        [
            _chunk(0, "monsoon monsoon rainfall over india"),  # strongest: tf=2
            _chunk(1, "monsoon onset dates in kerala"),
            _chunk(2, "monsoon variability and drivers"),
            _chunk(3, "arctic sea ice decline"),  # no overlap: must not appear
        ]
    )
    results = index.query("monsoon", top_k=2)
    assert len(results) == 2  # top_k caps it (3 chunks match)
    assert results[0][0].chunk_id == "doc#p1#0"
    assert results[0][1] >= results[1][1]


def test_no_matching_terms_returns_empty():
    index = BM25Index([_chunk(0, "heavy precipitation over asia")])
    assert index.query("zzz qqq", top_k=5) == []
