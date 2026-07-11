"""Tests for Gemini embeddings + disk cache + dense index + RRF (all offline).

Network is mocked (pytest-httpx); cosine and RRF are pure math on tiny vectors.
"""
import numpy as np
import pytest

from rag.chunk import Chunk
from rag.dense import DenseIndex
from rag.embed import DiskVectorCache, EmbeddingError, embed_texts
from rag.hybrid import rrf_fuse


def _chunk(i: int, text: str = "x") -> Chunk:
    return Chunk(chunk_id=f"d#p1#{i}", source="d.pdf", page=i + 1, text=text)


# --- embed_texts (mocked HTTP) ---


def _batch_response(n: int, dims: int = 4) -> dict:
    return {"embeddings": [{"values": [float(i)] * dims} for i in range(n)]}


def test_embed_texts_parses_batch_response(httpx_mock):
    httpx_mock.add_response(json=_batch_response(2))
    vectors = embed_texts(["a", "b"], task_type="RETRIEVAL_DOCUMENT", api_key="k")
    assert len(vectors) == 2
    assert vectors[1] == [1.0, 1.0, 1.0, 1.0]


def test_embed_texts_splits_into_batches(httpx_mock, monkeypatch):
    from rag.embed import _BATCH_SIZE

    monkeypatch.setattr("rag.embed._sleep", lambda s: None)  # no real pacing in tests
    httpx_mock.add_response(json=_batch_response(_BATCH_SIZE))
    httpx_mock.add_response(json=_batch_response(1))
    vectors = embed_texts(["t"] * (_BATCH_SIZE + 1), task_type="RETRIEVAL_DOCUMENT", api_key="k")
    assert len(vectors) == _BATCH_SIZE + 1
    assert len(httpx_mock.get_requests()) == 2


def test_embed_texts_retries_429_with_server_delay(httpx_mock, monkeypatch):
    slept = []
    monkeypatch.setattr("rag.embed._sleep", slept.append)
    httpx_mock.add_response(status_code=429, text="Please retry in 7.5s.")
    httpx_mock.add_response(json=_batch_response(1))
    vectors = embed_texts(["a"], task_type="RETRIEVAL_DOCUMENT", api_key="k")
    assert len(vectors) == 1
    assert slept == [8.5]  # server-suggested 7.5s + 1


def test_embed_texts_raises_typed_error_on_http_failure(httpx_mock):
    httpx_mock.add_response(status_code=500)
    with pytest.raises(EmbeddingError):
        embed_texts(["a"], task_type="RETRIEVAL_QUERY", api_key="k")


def test_embed_texts_gives_up_after_max_429_retries(httpx_mock, monkeypatch):
    monkeypatch.setattr("rag.embed._sleep", lambda s: None)
    for _ in range(8):  # _RETRY_MAX
        httpx_mock.add_response(status_code=429, text="Please retry in 1s.")
    with pytest.raises(EmbeddingError, match="retries exhausted"):
        embed_texts(["a"], task_type="RETRIEVAL_QUERY", api_key="k")


def test_disk_cache_roundtrip_and_miss(tmp_path):
    cache = DiskVectorCache(tmp_path)
    assert cache.get("some-key") is None
    cache.put("some-key", [0.1, 0.2])
    assert cache.get("some-key") == pytest.approx([0.1, 0.2])


def test_cached_embed_persists_completed_batches_on_midway_failure(tmp_path, httpx_mock, monkeypatch):
    # Quota can bite mid-run: batch 1's vectors must already be on disk so the
    # next run resumes instead of re-paying.
    from rag.embed import _BATCH_SIZE, cache_key, cached_embed_texts

    monkeypatch.setattr("rag.embed._sleep", lambda s: None)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    httpx_mock.add_response(json=_batch_response(_BATCH_SIZE))
    httpx_mock.add_response(status_code=500)

    cache = DiskVectorCache(tmp_path)
    texts = [f"t{i}" for i in range(_BATCH_SIZE + 1)]
    with pytest.raises(EmbeddingError):
        cached_embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", cache=cache)
    assert cache.get(cache_key("t0", "RETRIEVAL_DOCUMENT")) is not None  # batch 1 kept
    assert cache.get(cache_key(f"t{_BATCH_SIZE}", "RETRIEVAL_DOCUMENT")) is None


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
