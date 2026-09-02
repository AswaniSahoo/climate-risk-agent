"""Pack/unpack must round-trip, and must REFUSE a cache that no longer matches.

The dangerous failure is not a crash, it is acceptance: installing vectors built
by a different model or a different chunker would produce eval numbers that look
fine and are wrong. So most of these tests assert on the refusal.

Everything runs against a fake corpus in tmp_path: the real PDFs are 50 MB and
git-ignored, and the checks are about identity, not content.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from scripts import pack_eval_cache, unpack_eval_cache
from scripts.pack_eval_cache import MANIFEST_NAME, EvalCacheError, pack

PDFS = ("IPCC_AR6_WGI_SPM.pdf", "IPCC_AR6_WGI_Chapter11.pdf", "IPCC_AR6_WGI_Chapter12.pdf")


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A miniature data/ tree, wired into both scripts and into rag.corpus."""
    from rag import corpus

    corpus_dir = tmp_path / "data" / "ipcc"
    cache_root = tmp_path / "data" / "cache"
    embed_dir = cache_root / "embeddings"
    chunk_cache = cache_root / "chunks.json"

    corpus_dir.mkdir(parents=True)
    embed_dir.mkdir(parents=True)
    for i, name in enumerate(PDFS):
        (corpus_dir / name).write_bytes(b"%PDF-fake" + bytes([i]) * (100 + i))
    for i in range(3):
        np.save(embed_dir / f"{i:064x}.npy", np.full(768, i, dtype=np.float32))
    chunk_cache.write_text(
        json.dumps({"fingerprint": "abc", "chunks": [{"chunk_id": "c1"}, {"chunk_id": "c2"}]}),
        encoding="utf-8",
    )

    # rag.corpus owns _fingerprint(), which reads ITS module globals.
    monkeypatch.setattr(corpus, "CORPUS_DIR", corpus_dir)
    monkeypatch.setattr(corpus, "CORPUS_FILES", PDFS)
    for module in (pack_eval_cache, unpack_eval_cache):
        for attr, value in (
            ("CORPUS_DIR", corpus_dir),
            ("CORPUS_FILES", PDFS),
            ("CACHE_ROOT", cache_root),
            ("EMBED_CACHE_DIR", embed_dir),
            ("CHUNK_CACHE", chunk_cache),
        ):
            if hasattr(module, attr):
                monkeypatch.setattr(module, attr, value)

    return {
        "corpus_dir": corpus_dir,
        "cache_root": cache_root,
        "embed_dir": embed_dir,
        "chunk_cache": chunk_cache,
        "archive": tmp_path / "eval-cache-v1.tar.gz",
    }


def _wipe_cache(repo: dict) -> None:
    for path in repo["embed_dir"].glob("*.npy"):
        path.unlink()
    repo["chunk_cache"].unlink()


def _rebuild(src: Path, dst: Path, *, mutate=None, extra: tuple[str, bytes] | None = None) -> Path:
    """Copy an archive, optionally mutating the manifest or adding a member."""
    with tarfile.open(src, "r:gz") as old, tarfile.open(dst, "w:gz") as new:
        for member in old.getmembers():
            handle = old.extractfile(member)
            data = handle.read() if handle else b""
            if member.name == MANIFEST_NAME and mutate is not None:
                payload = json.loads(data.decode("utf-8"))
                mutate(payload)
                data = json.dumps(payload).encode("utf-8")
                member.size = len(data)
            new.addfile(member, io.BytesIO(data))
        if extra is not None:
            name, blob = extra
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            new.addfile(info, io.BytesIO(blob))
    return dst


# --------------------------------------------------------------------------
# pack
# --------------------------------------------------------------------------

def test_manifest_pins_model_chunker_and_corpus(fake_repo):
    from rag.corpus import _fingerprint
    from rag.embed import DIMS, MODEL

    manifest = pack(fake_repo["archive"])

    assert manifest["schema"] == pack_eval_cache.SCHEMA
    assert manifest["embed_model"] == MODEL
    assert manifest["embed_dims"] == DIMS
    assert manifest["chunker_fingerprint"] == _fingerprint()
    assert manifest["pdf_sizes"] == {
        name: (fake_repo["corpus_dir"] / name).stat().st_size for name in PDFS
    }
    assert manifest["n_vectors"] == 3
    assert manifest["n_chunks"] == 2
    assert len(manifest["contents_sha256"]) == 64


def test_pack_refuses_when_the_embedding_cache_is_empty(fake_repo):
    for path in fake_repo["embed_dir"].glob("*.npy"):
        path.unlink()
    with pytest.raises(EvalCacheError, match="no .npy vectors"):
        pack(fake_repo["archive"])


def test_pack_refuses_when_a_corpus_pdf_is_missing(fake_repo):
    (fake_repo["corpus_dir"] / PDFS[0]).unlink()
    with pytest.raises(EvalCacheError, match="download_ipcc"):
        pack(fake_repo["archive"])


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------

def test_round_trip_restores_every_vector_and_the_chunk_cache(fake_repo):
    before = {p.name: p.read_bytes() for p in fake_repo["embed_dir"].glob("*.npy")}
    chunks_before = fake_repo["chunk_cache"].read_bytes()
    pack(fake_repo["archive"])
    _wipe_cache(fake_repo)

    manifest = unpack_eval_cache.unpack(fake_repo["archive"])

    after = {p.name: p.read_bytes() for p in fake_repo["embed_dir"].glob("*.npy")}
    assert after == before
    assert fake_repo["chunk_cache"].read_bytes() == chunks_before
    assert manifest["n_vectors"] == 3


# --------------------------------------------------------------------------
# refusals: every one of these would otherwise publish a wrong number
# --------------------------------------------------------------------------

def test_missing_archive_names_the_pack_and_upload_commands(fake_repo):
    with pytest.raises(EvalCacheError) as exc:
        unpack_eval_cache.unpack(fake_repo["archive"])
    message = str(exc.value)
    assert "scripts.pack_eval_cache" in message
    assert "gh release create eval-cache-v1" in message


def test_embedding_model_swap_is_refused(fake_repo, monkeypatch):
    pack(fake_repo["archive"])
    _wipe_cache(fake_repo)
    monkeypatch.setattr(unpack_eval_cache, "MODEL", "gemini-embedding-99")

    with pytest.raises(EvalCacheError, match="MODEL mismatch"):
        unpack_eval_cache.unpack(fake_repo["archive"])
    assert list(fake_repo["embed_dir"].glob("*.npy")) == []  # nothing was installed


def test_changed_corpus_is_refused(fake_repo):
    pack(fake_repo["archive"])
    _wipe_cache(fake_repo)
    (fake_repo["corpus_dir"] / PDFS[1]).write_bytes(b"%PDF-a-different-edition")

    with pytest.raises(EvalCacheError, match="corpus mismatch"):
        unpack_eval_cache.unpack(fake_repo["archive"])


def test_changed_chunker_is_refused(fake_repo, monkeypatch):
    """rag/chunk.py edited since the cache was packed: same PDFs, different chunks."""
    pack(fake_repo["archive"])
    _wipe_cache(fake_repo)
    monkeypatch.setattr(unpack_eval_cache, "_fingerprint", lambda: "0123456789abcdef")

    with pytest.raises(EvalCacheError, match="chunker mismatch"):
        unpack_eval_cache.unpack(fake_repo["archive"])


def test_corrupt_payload_is_refused(fake_repo, tmp_path):
    pack(fake_repo["archive"])
    tampered = _rebuild(
        fake_repo["archive"],
        tmp_path / "tampered.tar.gz",
        mutate=lambda m: m.update(contents_sha256="0" * 64),
    )
    _wipe_cache(fake_repo)

    with pytest.raises(EvalCacheError, match="checksum mismatch"):
        unpack_eval_cache.unpack(tampered)
    assert list(fake_repo["embed_dir"].glob("*.npy")) == []


def test_path_traversal_member_is_refused(fake_repo, tmp_path):
    pack(fake_repo["archive"])
    hostile = _rebuild(
        fake_repo["archive"], tmp_path / "hostile.tar.gz", extra=("../evil.npy", b"x")
    )
    _wipe_cache(fake_repo)

    with pytest.raises(EvalCacheError, match="unsafe path"):
        unpack_eval_cache.unpack(hostile)


def test_archive_without_a_manifest_is_refused(fake_repo, tmp_path):
    naked = tmp_path / "naked.tar.gz"
    with tarfile.open(naked, "w:gz") as archive:
        info = tarfile.TarInfo("chunks.json")
        info.size = 2
        archive.addfile(info, io.BytesIO(b"{}"))

    with pytest.raises(EvalCacheError, match="manifest.json missing"):
        unpack_eval_cache.unpack(naked)


def test_cli_exits_one_on_a_mismatch(fake_repo, monkeypatch):
    pack(fake_repo["archive"])
    monkeypatch.setattr(unpack_eval_cache, "MODEL", "gemini-embedding-99")
    monkeypatch.setattr("sys.argv", ["unpack", "--archive", str(fake_repo["archive"])])

    with pytest.raises(SystemExit) as exc:
        unpack_eval_cache.main()
    assert exc.value.code == 1
