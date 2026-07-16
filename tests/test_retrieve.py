"""Tests for the production retriever (rag/retrieve.py): hybrid with loud fallback."""
import numpy as np

import rag.retrieve as retrieve_mod
from rag.chunk import Chunk
from rag.retrieve import HybridRetriever


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"d.pdf#p{i+1}#0", source="d.pdf", page=i + 1, text=text)


CHUNKS = [
    _chunk(0, "heavy precipitation intensifies with warming rx1day"),
    _chunk(1, "glaciers melting committed centuries"),
    _chunk(2, "surface wind stilling tropics"),
]


def test_bm25_only_when_no_dense_matrix():
    retriever = HybridRetriever(CHUNKS, doc_matrix=None)
    top = retriever.retrieve("glaciers melting", top_k=2)
    assert top[0].chunk_id == CHUNKS[1].chunk_id


def test_hybrid_fuses_dense_and_bm25(monkeypatch):
    # dense matrix points question at chunk 2 even though bm25 favors chunk 0;
    # fusion must rank BOTH above the irrelevant glacier chunk.
    matrix = np.array([[1.0, 0.0], [0.0, 0.0], [0.9, 0.1]])
    retriever = HybridRetriever(CHUNKS, doc_matrix=matrix)
    monkeypatch.setattr(retriever, "_embed_query", lambda q: [0.9, 0.1])

    top = retriever.retrieve("rainfall intensification rx1day", top_k=2)

    ids = {c.chunk_id for c in top}
    assert CHUNKS[1].chunk_id not in ids
    assert len(ids) == 2


def test_query_embedding_failure_falls_back_to_bm25_loudly(monkeypatch, caplog):
    matrix = np.eye(3)
    retriever = HybridRetriever(CHUNKS, doc_matrix=matrix)

    def broken(question):
        raise retrieve_mod.EmbeddingError("no auth")

    monkeypatch.setattr(retriever, "_embed_query", broken)
    top = retriever.retrieve("stilling winds", top_k=1)
    assert top[0].chunk_id == CHUNKS[2].chunk_id  # bm25 still answers
    assert "falling back to BM25" in caplog.text  # loud (WARNING), not silent


def test_build_without_auth_degrades_to_bm25(monkeypatch, tmp_path, caplog):
    def no_auth(texts, *, task_type, cache):
        raise retrieve_mod.EmbeddingError("no Gemini auth configured")

    monkeypatch.setattr(retrieve_mod, "cached_embed_texts", no_auth)
    retriever = HybridRetriever.build(CHUNKS, cache_dir=tmp_path)
    assert retriever.dense_enabled is False
    assert "BM25-only" in caplog.text
    assert retriever.retrieve("glaciers", top_k=1)[0].chunk_id == CHUNKS[1].chunk_id
