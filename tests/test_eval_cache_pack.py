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


def _without_chunk_cache(repo: dict, dst: Path) -> Path:
    """The packed archive minus its chunk cache, with the manifest checksum fixed
    so the payload check passes and the INSTALL step is the one under test."""
    chunk_name = repo["chunk_cache"].name
    payload = [(v, f"embeddings/{v.name}") for v in repo["embed_dir"].glob("*.npy")]
    expected = pack_eval_cache.contents_sha256(payload)

    with tarfile.open(repo["archive"], "r:gz") as old, tarfile.open(dst, "w:gz") as new:
        for member in old.getmembers():
            if member.name == chunk_name:
                continue
            handle = old.extractfile(member)
            data = handle.read() if handle else b""
            if member.name == MANIFEST_NAME:
                manifest = json.loads(data.decode("utf-8"))
                manifest["contents_sha256"] = expected
                data = json.dumps(manifest).encode("utf-8")
                member.size = len(data)
            new.addfile(member, io.BytesIO(data))
    return dst


def test_an_archive_without_a_chunk_cache_fails_with_the_remediation(fake_repo, tmp_path):
    """shutil.move raised a raw FileNotFoundError here, which escaped main() as a
    traceback instead of the REMEDIATION message the script exists to print."""
    pack(fake_repo["archive"])
    reduced = _without_chunk_cache(fake_repo, tmp_path / "no-chunks.tar.gz")
    _wipe_cache(fake_repo)

    with pytest.raises(EvalCacheError) as exc:
        unpack_eval_cache.unpack(reduced)

    message = str(exc.value)
    assert "chunk cache" in message
    assert "scripts.pack_eval_cache" in message  # the remediation, not a traceback


def test_a_half_installed_cache_is_never_left_behind(fake_repo, tmp_path):
    """The refusal above must happen BEFORE any vector moves: new vectors beside
    an old chunk cache is exactly the mismatched pair this script refuses."""
    pack(fake_repo["archive"])
    reduced = _without_chunk_cache(fake_repo, tmp_path / "no-chunks.tar.gz")
    _wipe_cache(fake_repo)

    with pytest.raises(EvalCacheError):
        unpack_eval_cache.unpack(reduced)

    assert list(fake_repo["embed_dir"].glob("*.npy")) == []


def test_stale_vectors_are_cleared_before_the_new_ones_land(fake_repo):
    """Vectors are named by content hash, so a leftover from an earlier corpus is
    not overwritten by the move — it lingers in the cache and keeps answering."""
    pack(fake_repo["archive"])
    stale = fake_repo["embed_dir"] / ("f" * 64 + ".npy")
    np.save(stale, np.zeros(768, dtype=np.float32))
    assert stale.exists()

    unpack_eval_cache.unpack(fake_repo["archive"])

    assert not stale.exists()
    assert len(list(fake_repo["embed_dir"].glob("*.npy"))) == 3  # only what was packed

