"""Tests for the two build steps that bake the corpus vectors into the image.

A deploy whose build context lacks data/cache/embeddings used to ship an image
without vectors: every cold container's first report then embedded all 2,730
chunks through the API, at 62 s per batch of 50. The image now takes them from
the eval-cache-v1 release asset, pinned by sha256 (scripts/fetch_eval_cache.py),
and refuses to build unless every chunk it will serve has its vector on disk
(scripts/check_baked_index.py).
"""
import hashlib
import json

import httpx
import pytest

from rag.embed import DiskVectorCache, cache_key
from scripts import check_baked_index as check
from scripts import fetch_eval_cache as fetch

URL = "https://github.com/AswaniSahoo/climate-risk-agent/releases/download/eval-cache-v1/eval-cache-v1.tar.gz"
PAYLOAD = b"not really a tarball, just bytes to hash"


def test_the_asset_is_written_when_its_sha256_matches(httpx_mock, tmp_path):
    httpx_mock.add_response(url=URL, content=PAYLOAD)
    out = tmp_path / "eval-cache.tar.gz"

    fetch.fetch(URL, hashlib.sha256(PAYLOAD).hexdigest(), out)

    assert out.read_bytes() == PAYLOAD


def test_a_sha256_mismatch_fails_and_leaves_no_file(httpx_mock, tmp_path):
    httpx_mock.add_response(url=URL, content=PAYLOAD + b"tampered")
    out = tmp_path / "eval-cache.tar.gz"

    with pytest.raises(fetch.AssetError) as failed:
        fetch.fetch(URL, hashlib.sha256(PAYLOAD).hexdigest(), out)

    assert "sha256 mismatch" in str(failed.value)
    assert hashlib.sha256(PAYLOAD).hexdigest() in str(failed.value)
    assert list(tmp_path.iterdir()) == []  # nothing half-trusted left behind


def test_a_failed_download_fails(httpx_mock, tmp_path):
    httpx_mock.add_response(url=URL, status_code=404)

    with pytest.raises(fetch.AssetError):
        fetch.fetch(URL, "0" * 64, tmp_path / "eval-cache.tar.gz")


def test_the_pin_must_look_like_a_sha256():
    with pytest.raises(fetch.AssetError):
        fetch.fetch(URL, "7f91a5ce", "unused.tar.gz")


def test_the_command_line_exits_non_zero_on_a_mismatch(httpx_mock, tmp_path, capsys):
    httpx_mock.add_response(url=URL, content=PAYLOAD)

    code = fetch.main(["--url", URL, "--sha256", "0" * 64, "--out", str(tmp_path / "x.tgz")])

    assert code == 1
    assert "sha256 mismatch" in capsys.readouterr().err


# --- every chunk the image serves has its document vector on disk ------------

def _baked(tmp_path, texts, vectors_for):
    chunks = tmp_path / "chunks.json"
    chunks.write_text(json.dumps({"fingerprint": "f", "chunks": [
        {"chunk_id": f"d.pdf#p{i}#0", "source": "d.pdf", "page": i, "text": text}
        for i, text in enumerate(texts, 1)]}))
    store = DiskVectorCache(tmp_path / "embeddings")
    for text in vectors_for:
        store.put(cache_key(text, "RETRIEVAL_DOCUMENT"), [0.1, 0.2])
    return chunks, tmp_path / "embeddings"


def test_a_complete_index_passes_and_counts_what_it_saw(tmp_path):
    chunks, vectors = _baked(tmp_path, ["a", "b", "c"], ["a", "b", "c"])
    DiskVectorCache(vectors).put(cache_key("an eval question", "RETRIEVAL_QUERY"), [0.3])

    result = check.coverage(chunks, vectors)

    assert (result.chunks, result.document_vectors, result.vector_files) == (3, 3, 4)
    assert result.missing == []


def test_a_chunk_without_its_vector_fails_the_build_with_the_reason(tmp_path, capsys):
    chunks, vectors = _baked(tmp_path, ["a", "b", "c"], ["a", "b"])

    code = check.main(["--chunks", str(chunks), "--vectors", str(vectors)])

    err = capsys.readouterr().err
    assert code == 1
    assert "3 chunks" in err and "2" in err and "would embed" in err


def test_an_empty_index_fails(tmp_path):
    chunks, vectors = _baked(tmp_path, [], [])

    assert check.main(["--chunks", str(chunks), "--vectors", str(vectors)]) == 1


def test_a_query_vector_does_not_stand_in_for_a_document_vector(tmp_path):
    chunks, vectors = _baked(tmp_path, ["a"], [])
    DiskVectorCache(vectors).put(cache_key("a", "RETRIEVAL_QUERY"), [0.1])

    assert check.coverage(chunks, vectors).missing == ["d.pdf#p1#0"]


def test_the_command_line_passes_on_a_complete_index(tmp_path, capsys):
    chunks, vectors = _baked(tmp_path, ["a", "b"], ["a", "b"])

    assert check.main(["--chunks", str(chunks), "--vectors", str(vectors)]) == 0
    assert "2 chunks, 2 document vectors" in capsys.readouterr().out


def test_an_http_error_type_is_what_the_fetch_wraps():
    assert issubclass(fetch.AssetError, RuntimeError)
    assert httpx.HTTPError  # the fetch is plain httpx, mocked above with pytest-httpx
