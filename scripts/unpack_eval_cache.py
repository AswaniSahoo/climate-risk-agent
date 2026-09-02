"""Install a packed eval cache into data/cache, or fail loudly, never quietly.

The whole point of the archive is that the evals must NOT call the embedding API
in CI. That guarantee only holds if the vectors in the archive were produced by
the same model, the same dimensionality and the same chunker as the repo running
right now. A silently-accepted stale cache would still produce numbers, just
wrong ones, and CI would publish them.

So this refuses to install unless every field in the manifest still matches the
live repo, and it verifies the payload bytes before a single file lands in
data/cache (staged in a temp dir, moved into place only on success).

Run:  uv run python -m scripts.unpack_eval_cache
      uv run python -m scripts.unpack_eval_cache --archive /tmp/eval-cache-v1.tar.gz
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import NoReturn

from rag.corpus import CHUNK_CACHE, _fingerprint
from rag.embed import DIMS, MODEL
from scripts.pack_eval_cache import (
    ARCHIVE_NAME,
    ASSET_NAME,
    CACHE_ROOT,
    EMBED_CACHE_DIR,
    MANIFEST_NAME,
    PACK_COMMAND,
    REUPLOAD_COMMAND,
    SCHEMA,
    UPLOAD_COMMAND,
    EvalCacheError,
    contents_sha256,
    pdf_sizes,
)

REMEDIATION = f"""
The evals cannot run without a matching embedding cache, and re-embedding the
corpus in CI is deliberately not an option (2,730 chunks; the Gemini free tier
cannot do it inside a job). Fix it by re-packing and re-uploading, locally:

  {PACK_COMMAND}
  {REUPLOAD_COMMAND}

If the {ASSET_NAME} release does not exist yet, create it instead:

  {UPLOAD_COMMAND}
"""


def _fail(reason: str) -> NoReturn:
    raise EvalCacheError(f"{reason}\n{REMEDIATION}")


def _safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Reject anything that is not a plain file inside the archive root."""
    members = []
    for member in archive.getmembers():
        name = member.name
        if member.isdir():
            continue
        if not member.isfile():
            _fail(f"archive holds a non-regular member ({name!r}). Refusing to extract")
        if name.startswith(("/", "\\")) or ".." in Path(name).parts:
            _fail(f"archive holds an unsafe path ({name!r}). Refusing to extract")
        members.append(member)
    return members


def read_manifest(archive: tarfile.TarFile) -> dict:
    try:
        handle = archive.extractfile(MANIFEST_NAME)
    except KeyError:
        handle = None
    if handle is None:
        _fail(f"{MANIFEST_NAME} missing from the archive: it was not built by {PACK_COMMAND}")
    return json.loads(handle.read().decode("utf-8"))


def verify_manifest(manifest: dict) -> None:
    """Every mismatch is a hard stop, reported with both values."""
    if manifest.get("schema") != SCHEMA:
        _fail(f"manifest schema {manifest.get('schema')!r}, expected {SCHEMA!r}")

    if manifest.get("embed_model") != MODEL:
        _fail(
            f"embedding MODEL mismatch: cache was built with {manifest.get('embed_model')!r}, "
            f"this repo embeds with {MODEL!r}. Vectors from a different model are not comparable."
        )
    if manifest.get("embed_dims") != DIMS:
        _fail(
            f"embedding DIMS mismatch: cache has {manifest.get('embed_dims')!r}, repo uses {DIMS!r}"
        )

    current_pdfs = pdf_sizes()
    if manifest.get("pdf_sizes") != current_pdfs:
        _fail(
            f"corpus mismatch: cache was built over {manifest.get('pdf_sizes')}, "
            f"the PDFs on disk are {current_pdfs}"
        )

    current_fingerprint = _fingerprint()
    if manifest.get("chunker_fingerprint") != current_fingerprint:
        _fail(
            f"chunker mismatch: cache fingerprint {manifest.get('chunker_fingerprint')!r} != "
            f"repo fingerprint {current_fingerprint!r} (rag/chunk.py changed since the cache "
            "was packed, so the cached vectors describe different chunks)"
        )


def unpack(archive_path: Path) -> dict:
    """Verify then install. Returns the manifest on success."""
    if not archive_path.exists():
        _fail(f"archive not found: {archive_path}")

    with tarfile.open(archive_path, "r:gz") as archive:
        manifest = read_manifest(archive)
        print(f"manifest: {json.dumps({k: v for k, v in manifest.items() if k != 'pdf_sizes'})}")
        verify_manifest(manifest)
        print("manifest matches the repo (model, dims, corpus, chunker) - extracting")

        members = _safe_members(archive)
        CACHE_ROOT.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=CACHE_ROOT.parent) as staging_name:
            staging = Path(staging_name)
            # filter="data" (PEP 706) strips absolute paths, links and odd modes.
            if hasattr(tarfile, "data_filter"):
                archive.extractall(staging, members=members, filter="data")
            else:  # pragma: no cover - Python < 3.11.4
                archive.extractall(staging, members=members)

            staged = [
                (staging / m.name, m.name) for m in members if m.name != MANIFEST_NAME
            ]
            actual = contents_sha256(staged)
            if actual != manifest.get("contents_sha256"):
                _fail(
                    f"payload checksum mismatch: archive contents hash to {actual}, manifest "
                    f"claims {manifest.get('contents_sha256')} - the download is corrupt"
                )
            print(f"payload sha256 verified ({actual[:16]}...)")

            # Everything the install needs must be present BEFORE anything is
            # moved: half an install (new vectors, old chunk cache) is the
            # mismatched pair this whole script exists to refuse. A missing
            # member used to surface as a raw FileNotFoundError out of
            # shutil.move, escaping main() as a traceback with no remediation.
            staged_chunk_cache = staging / CHUNK_CACHE.name
            if not staged_chunk_cache.exists():
                _fail(
                    f"archive has no {CHUNK_CACHE.name}: the chunk cache is missing, so the "
                    "vectors cannot be matched to chunks"
                )

            EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            CHUNK_CACHE.parent.mkdir(parents=True, exist_ok=True)
            # Clear the old vectors first (the manifest and the payload checksum
            # have both passed by now, so the replacement is known good). Vectors
            # are named by content hash, so leftovers from an earlier corpus are
            # not overwritten by the move — they linger and keep answering.
            for stale in EMBED_CACHE_DIR.glob("*.npy"):
                stale.unlink()

            moved = 0
            try:
                for source in (staging / "embeddings").glob("*.npy"):
                    shutil.move(str(source), str(EMBED_CACHE_DIR / source.name))
                    moved += 1
                shutil.move(str(staged_chunk_cache), str(CHUNK_CACHE))
            except OSError as exc:
                _fail(f"could not install the cache into {CACHE_ROOT}: {exc}")

    print(f"installed {moved} vectors -> {EMBED_CACHE_DIR}")
    print(f"installed chunk cache ({manifest['n_chunks']} chunks) -> {CHUNK_CACHE}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path(ARCHIVE_NAME), help=f"the {ASSET_NAME} tarball"
    )
    args = parser.parse_args()
    try:
        unpack(args.archive)
    except EvalCacheError as exc:
        print(f"\nEVAL CACHE UNUSABLE\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
