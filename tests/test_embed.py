"""Tests for embeddings + disk cache + dense index + RRF (all offline).

The SDK seam (rag.gemini_client.embed_batch) is monkeypatched; cosine and RRF
are pure math on tiny vectors.
"""
import logging
from pathlib import Path

import numpy as np
import pytest

import rag.embed as embed_mod
from rag.chunk import Chunk
from rag.dense import DenseIndex
from rag.embed import (
    DiskVectorCache,
    EmbeddingError,
    cache_key,
    cached_embed_texts,
    embed_texts,
    pack_path,
    vector_digest,
    write_pack,
)
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


# --- the vector pack: the same vectors in ONE file, read before the files ---
#
# A Cloud Run gen2 container pulls image files on first read, so 2,730 vector
# files cost 17.5 s on the live revision against 0.15 s locally. The pack must
# answer exactly what the files answer, bit for bit, and anything wrong with it
# must cost the slow path, never a wrong vector or a crash.


def _vector_files(directory, n=6, dims=8, seed=0):
    """`n` document vectors as files, with awkward float32 bits in each."""
    rng = np.random.default_rng(seed)
    store = DiskVectorCache(directory)
    keys = []
    for i in range(n):
        vector = rng.standard_normal(dims).astype(np.float32)
        vector[:4] = [-0.0, 1e-45, np.finfo(np.float32).max, np.finfo(np.float32).tiny]
        key = cache_key(f"chunk {i}", "RETRIEVAL_DOCUMENT")
        store.put(key, vector.tolist())
        keys.append(key)
    return store, keys


def _file_of(store, key):
    return store.directory / f"{vector_digest(key)}.npy"


def _pack(store, keys):
    vectors = {vector_digest(k): np.load(_file_of(store, k)) for k in keys}
    return write_pack(pack_path(store.directory), vectors)


def test_the_pack_sits_beside_the_folder_it_packs():
    assert pack_path("data/cache/embeddings") == Path("data/cache/embeddings.pack.npy")


def test_a_packed_vector_is_bit_for_bit_its_file_and_needs_no_file(tmp_path):
    store, keys = _vector_files(tmp_path / "embeddings")
    files = {k: np.load(_file_of(store, k)) for k in keys}
    assert _pack(store, keys) == len(keys)
    for key in keys:
        _file_of(store, key).unlink()  # only the pack can answer now

    cache = DiskVectorCache(tmp_path / "embeddings")
    for key in keys:
        vector = cache.get(key)
        assert vector == files[key].tolist()
        assert np.asarray(vector, dtype=np.float32).tobytes() == files[key].tobytes()


def test_the_corpus_is_read_from_the_pack_in_one_file_open(tmp_path, monkeypatch):
    store, keys = _vector_files(tmp_path / "embeddings", n=60)
    _pack(store, keys)
    texts = [f"chunk {i}" for i in range(60)]
    real_load = np.load
    opened = []

    def counting_load(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(embed_mod.np, "load", counting_load)
    monkeypatch.setattr(
        embed_mod, "embed_batch", lambda *a, **k: pytest.fail("every vector is in the pack")
    )
    vectors = cached_embed_texts(
        texts, task_type="RETRIEVAL_DOCUMENT", cache=DiskVectorCache(tmp_path / "embeddings")
    )

    assert len(vectors) == 60
    assert opened == ["embeddings.pack.npy"]


def test_a_vector_the_pack_lacks_is_read_from_its_file(tmp_path):
    store, keys = _vector_files(tmp_path / "embeddings")
    _pack(store, keys[:-1])

    cache = DiskVectorCache(tmp_path / "embeddings")

    assert cache.get(keys[-1]) == np.load(_file_of(store, keys[-1])).tolist()
    assert cache.get(cache_key("never embedded", "RETRIEVAL_QUERY")) is None


def test_a_new_query_vector_goes_to_a_file_and_leaves_the_pack_alone(tmp_path):
    store, keys = _vector_files(tmp_path / "embeddings")
    _pack(store, keys)
    pack = pack_path(store.directory)
    before = pack.read_bytes()
    question = cache_key("a new question", "RETRIEVAL_QUERY")

    DiskVectorCache(tmp_path / "embeddings").put(question, [0.5, 0.25])

    assert pack.read_bytes() == before
    assert _file_of(store, question).exists()
    assert DiskVectorCache(tmp_path / "embeddings").get(question) == [0.5, 0.25]


@pytest.mark.parametrize("damage", ["truncated", "not numpy", "wrong layout"])
def test_a_damaged_pack_is_a_loud_miss_and_the_files_still_answer(tmp_path, caplog, damage):
    store, keys = _vector_files(tmp_path / "embeddings")
    _pack(store, keys)
    pack = pack_path(store.directory)
    if damage == "truncated":
        pack.write_bytes(pack.read_bytes()[:-10])
    elif damage == "not numpy":
        pack.write_bytes(b"not a numpy file at all")
    else:
        np.save(pack, np.zeros((len(keys), 8), dtype=np.float32))

    with caplog.at_level(logging.WARNING):
        vector = DiskVectorCache(tmp_path / "embeddings").get(keys[0])

    assert vector == np.load(_file_of(store, keys[0])).tolist()
    assert "vector pack" in caplog.text


def test_a_rewritten_pack_is_read_again(tmp_path):
    store, keys = _vector_files(tmp_path / "embeddings")
    _pack(store, keys)
    assert DiskVectorCache(tmp_path / "embeddings").get(keys[0]) is not None

    replacement = np.arange(8, dtype=np.float32)
    write_pack(pack_path(store.directory), {vector_digest(keys[0]): replacement})

    assert DiskVectorCache(tmp_path / "embeddings").get(keys[0]) == replacement.tolist()


def test_write_pack_refuses_what_it_would_have_to_cast(tmp_path):
    target = tmp_path / "embeddings.pack.npy"
    with pytest.raises(ValueError, match="float32"):
        write_pack(target, {"a" * 64: np.zeros(4, dtype=np.float64)})
    with pytest.raises(ValueError, match="width"):
        write_pack(target, {"a" * 64: np.zeros(4, dtype=np.float32),
                            "b" * 64: np.zeros(5, dtype=np.float32)})
    with pytest.raises(ValueError, match="sha256"):
        write_pack(target, {"not a digest": np.zeros(4, dtype=np.float32)})
    with pytest.raises(ValueError, match="nothing"):
        write_pack(target, {})

    assert list(tmp_path.iterdir()) == []
