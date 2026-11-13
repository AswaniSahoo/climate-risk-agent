"""Tests for the two build steps that bake the corpus vectors into the image.

A deploy whose build context lacks data/cache/embeddings used to ship an image
without vectors: every cold container's first report then embedded all 2,730
chunks through the API, at 62 s per batch of 50. The image now takes them from
the eval-cache-v1 release asset, pinned by sha256 (scripts/fetch_eval_cache.py),
and refuses to build unless every chunk it will serve has its vector on disk
(scripts/check_baked_index.py).

The vectors are also packed into one file (scripts/pack_vectors.py), because a
Cloud Run gen2 container spent 17.5 s reading them as 2,730 files. The check
then holds the pack to the files, bit for bit.
"""
import fnmatch
import hashlib
import json
import re
from pathlib import Path

import httpx
import numpy as np
import pytest

from rag.embed import DiskVectorCache, cache_key, pack_path, read_pack, vector_digest, write_pack
from scripts import check_baked_index as check
from scripts import fetch_eval_cache as fetch
from scripts import pack_vectors

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
    for i, text in enumerate(vectors_for):  # distinct vectors, so a swapped row would show
        store.put(cache_key(text, "RETRIEVAL_DOCUMENT"), [0.1 + i, 0.2, -0.0])
    return chunks, tmp_path / "embeddings"


def _args(chunks, vectors):
    return ["--chunks", str(chunks), "--vectors", str(vectors)]


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


def test_the_command_line_passes_on_a_complete_index_and_its_pack(tmp_path, capsys):
    chunks, vectors = _baked(tmp_path, ["a", "b"], ["a", "b"])

    assert pack_vectors.main(_args(chunks, vectors)) == 0
    assert check.main(_args(chunks, vectors)) == 0
    out = capsys.readouterr().out
    assert "2 chunks, 2 document vectors" in out
    assert "2 in the pack" in out and "bit for bit" in out


# --- the pack: every chunk's document vector in one file, exactly ------------

def test_the_pack_holds_each_chunk_vector_bit_for_bit_and_no_query_vector(tmp_path, capsys):
    chunks, vectors = _baked(tmp_path, ["a", "b", "c"], ["a", "b", "c"])
    DiskVectorCache(vectors).put(cache_key("an eval question", "RETRIEVAL_QUERY"), [0.3, 0.4, 0.5])

    assert pack_vectors.main(_args(chunks, vectors)) == 0

    packed = read_pack(pack_path(vectors))
    files = DiskVectorCache(vectors)
    keys = [cache_key(text, "RETRIEVAL_DOCUMENT") for text in "abc"]
    assert set(packed) == {vector_digest(key) for key in keys}
    for key in keys:
        assert packed[vector_digest(key)].tobytes() == files.file_vector(key).tobytes()
    assert "3 chunks" in capsys.readouterr().out


def test_two_chunks_with_one_text_share_one_row(tmp_path, capsys):
    chunks, vectors = _baked(tmp_path, ["same", "same", "other"], ["same", "other"])

    assert pack_vectors.main(_args(chunks, vectors)) == 0
    assert len(read_pack(pack_path(vectors))) == 2
    assert check.main(_args(chunks, vectors)) == 0
    assert "3 in the pack" in capsys.readouterr().out


def test_the_pack_step_refuses_an_incomplete_index_and_writes_no_pack(tmp_path, capsys):
    chunks, vectors = _baked(tmp_path, ["a", "b"], ["a"])

    assert pack_vectors.main(_args(chunks, vectors)) == 1
    assert not pack_path(vectors).exists()
    assert "would embed" in capsys.readouterr().err


def test_the_check_fails_without_the_pack(tmp_path, capsys):
    chunks, vectors = _baked(tmp_path, ["a", "b"], ["a", "b"])

    assert check.main(_args(chunks, vectors)) == 1
    err = capsys.readouterr().err
    assert "VECTOR PACK MISSING" in err and "scripts/pack_vectors.py" in err


def test_the_check_fails_when_the_pack_lacks_a_chunk(tmp_path, capsys):
    chunks, vectors = _baked(tmp_path, ["a", "b"], ["a", "b"])
    key = cache_key("a", "RETRIEVAL_DOCUMENT")
    write_pack(pack_path(vectors), {vector_digest(key): DiskVectorCache(vectors).file_vector(key)})

    assert check.main(_args(chunks, vectors)) == 1
    err = capsys.readouterr().err
    assert "VECTOR PACK WRONG" in err and "d.pdf#p2#0" in err


def test_the_check_fails_when_a_packed_vector_is_one_bit_off_its_file(tmp_path, capsys):
    chunks, vectors = _baked(tmp_path, ["a", "b"], ["a", "b"])
    assert pack_vectors.main(_args(chunks, vectors)) == 0
    key = cache_key("b", "RETRIEVAL_DOCUMENT")
    vector = DiskVectorCache(vectors).file_vector(key)
    vector[0] = np.nextafter(vector[0], np.float32(2.0))  # the next float32 up: one ulp
    np.save(vectors / f"{vector_digest(key)}.npy", vector)

    assert check.main(_args(chunks, vectors)) == 1
    err = capsys.readouterr().err
    assert "VECTOR PACK WRONG" in err and "d.pdf#p2#0" in err and "1 differ" in err


def test_the_check_fails_on_an_unreadable_pack(tmp_path, capsys):
    chunks, vectors = _baked(tmp_path, ["a"], ["a"])
    pack_path(vectors).write_bytes(b"not a numpy file")

    assert check.main(_args(chunks, vectors)) == 1
    assert "VECTOR PACK WRONG" in capsys.readouterr().err


def test_an_http_error_type_is_what_the_fetch_wraps():
    assert issubclass(fetch.AssetError, RuntimeError)
    assert httpx.HTTPError  # the fetch is plain httpx, mocked above with pytest-httpx


# --- the Dockerfile wires the steps in order ---------------------------------

def _builder_runs() -> list[str]:
    """The RUN commands of the builder stage, continuation lines joined."""
    text = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")
    builder = re.split(r"^FROM ", text.replace("\\\n", " "), flags=re.MULTILINE)[1]
    return [line.strip()[4:] for line in builder.splitlines() if line.strip().startswith("RUN ")]


def test_the_image_packs_the_vectors_after_the_chunk_cache_and_checks_them_last():
    runs = _builder_runs()
    chunk_cache = next(i for i, run in enumerate(runs) if "build_chunk_cache" in run)
    packing = next(i for i, run in enumerate(runs) if "scripts.pack_vectors" in run)
    checking = next(i for i, run in enumerate(runs) if "scripts.check_baked_index" in run)

    assert chunk_cache < packing < checking == len(runs) - 1


def test_a_local_copy_of_the_asset_stays_out_of_the_build_context():
    """pack_eval_cache.py writes, and unpack_eval_cache.py reads, the asset at the
    repo root by default. `COPY . .` would carry that 9 MB copy into the image,
    beside the vectors the build fetches by sha256 for itself."""
    from scripts.pack_eval_cache import ARCHIVE_NAME

    root = Path(__file__).resolve().parents[1]
    for ignore_file in (".dockerignore", ".gcloudignore"):
        patterns = [line.strip() for line in (root / ignore_file).read_text("utf-8").splitlines()
                    if line.strip() and not line.startswith("#")]
        assert any(fnmatch.fnmatch(ARCHIVE_NAME, p) for p in patterns), ignore_file
