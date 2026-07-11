"""Tests for embeddings + disk cache + dense index + RRF (all offline).

The SDK seam (rag.gemini_client.embed_batch) is monkeypatched; cosine and RRF
are pure math on tiny vectors.
"""
import numpy as np
import pytest

import rag.embed as embed_mod
from rag.chunk import Chunk
from rag.dense import DenseIndex
from rag.embed import DiskVectorCache, EmbeddingError, cache_key, cached_embed_texts, embed_texts
from rag.gemini_client import GeminiError
from rag.hybrid import rrf_fuse


def _chunk(i: int, text: str = "x") -> Chunk:
    return Chunk(chunk_id=f"d#p1#{i}", source="d.pdf", page=i + 1, text=text)


@pytest.fixture(autouse=True)
def _no_pacing(monkeypatch):
    monkeypatch.setattr(embed_mod, "_sleep", lambda s: None)


def _fake_batches(*results):
    """Seam stub returning canned batch results (or raising) in order."""
    queue = list(results)

    def fake(texts, *, task_type, dims):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return [[float(i)] * 4 for i in range(len(texts))]

    return fake


# --- embed_texts (seam mocked) ---


def test_embed_texts_batches_and_concatenates(monkeypatch):
    calls = []

    def fake(texts, *, task_type, dims):
        calls.append(len(texts))
        return [[0.0] * dims for _ in texts]

    monkeypatch.setattr(embed_mod, "embed_batch", fake)
    vectors = embed_texts(["t"] * (embed_mod._BATCH_SIZE + 1), task_type="RETRIEVAL_DOCUMENT")
    assert len(vectors) == embed_mod._BATCH_SIZE + 1
    assert calls == [embed_mod._BATCH_SIZE, 1]


def test_embed_texts_wraps_seam_error(monkeypatch):
    monkeypatch.setattr(embed_mod, "embed_batch", _fake_batches(GeminiError("quota")))
    with pytest.raises(EmbeddingError, match="quota"):
        embed_texts(["a"], task_type="RETRIEVAL_QUERY")


def test_disk_cache_roundtrip_and_miss(tmp_path):
    cache = DiskVectorCache(tmp_path)
    assert cache.get("some-key") is None
    cache.put("some-key", [0.1, 0.2])
    assert cache.get("some-key") == pytest.approx([0.1, 0.2])


def test_cached_embed_persists_completed_batches_on_midway_failure(tmp_path, monkeypatch):
    # Quota can bite mid-run: batch 1's vectors must already be on disk so the
    # next run resumes instead of re-paying.
    monkeypatch.setattr(
        embed_mod, "embed_batch", _fake_batches("ok-batch", GeminiError("quota"))
    )
    cache = DiskVectorCache(tmp_path)
    texts = [f"t{i}" for i in range(embed_mod._BATCH_SIZE + 1)]
    with pytest.raises(EmbeddingError):
        cached_embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", cache=cache)
    assert cache.get(cache_key("t0", "RETRIEVAL_DOCUMENT")) is not None  # batch 1 kept
    assert cache.get(cache_key(f"t{embed_mod._BATCH_SIZE}", "RETRIEVAL_DOCUMENT")) is None


def test_cached_embed_makes_no_call_on_full_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        embed_mod, "embed_batch",
        lambda *a, **k: pytest.fail("network must not be touched on cache hit"),
    )
    cache = DiskVectorCache(tmp_path)
    cache.put(cache_key("hello", "RETRIEVAL_QUERY"), [1.0, 2.0])
    [vector] = cached_embed_texts(["hello"], task_type="RETRIEVAL_QUERY", cache=cache)
    assert vector == pytest.approx([1.0, 2.0])


# --- dense index (pure math) ---


def test_dense_index_ranks_by_cosine():
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    index = DenseIndex(chunks, matrix)
    results = index.query([0.0, 2.0], top_k=2)  # magnitude must not matter
    assert results[0][0].chunk_id == "d#p1#1"
    assert results[1][0].chunk_id == "d#p1#2"


# --- RRF fusion (pure math) ---


def test_rrf_rewards_agreement_across_rankings():
    a, b, c = _chunk(0), _chunk(1), _chunk(2)
    fused = rrf_fuse([[a, b, c], [b, c, a]], top_k=3)
    # b: ranks 2+1 -> best combined; a: 1+3; c: 3+2
    assert [ch.chunk_id for ch in fused] == [b.chunk_id, a.chunk_id, c.chunk_id]


def test_rrf_handles_item_missing_from_one_ranking():
    a, b = _chunk(0), _chunk(1)
    fused = rrf_fuse([[a, b], [a]], top_k=2)
    assert fused[0].chunk_id == a.chunk_id
