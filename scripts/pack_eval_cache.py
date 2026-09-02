"""Pack the eval caches (embeddings + chunks) into one verifiable archive.

WHY this exists: the retrieval/e2e evals are only affordable because they never
re-embed. Embedding 2,730 chunks costs a free-tier key ~50 minutes of paced
batches, so a CI run that rebuilt the cache would be both slow and expensive.
The cache is git-ignored (`data/` is), which leaves exactly one portable place
to keep it: a GitHub Release asset, fetched by the eval workflow.

An out-of-date cache is worse than no cache: it would score TODAY's chunker
against YESTERDAY's vectors and report a plausible, wrong number. So the archive
carries a manifest that pins the four things the numbers depend on:

    embed_model + embed_dims  : a model swap changes every vector
    chunker_fingerprint       : hash of (PDF sizes + rag/chunk.py source)
    pdf_sizes                 : which corpus was embedded
    contents_sha256           : the bytes themselves, so corruption is caught

`unpack_eval_cache.py` re-checks all four against the live repo and refuses to
install a mismatched cache. See DEPLOY.md ("Eval cache release asset").

Run:  uv run python -m scripts.pack_eval_cache
      uv run python -m scripts.pack_eval_cache --out /tmp/eval-cache-v1.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from rag.corpus import CHUNK_CACHE, CORPUS_DIR, CORPUS_FILES, _fingerprint
from rag.embed import DIMS, MODEL

# The release asset name. Bump the "v1" (here, in the workflow, and in DEPLOY.md)
# only if the ARCHIVE LAYOUT changes. A new corpus or a new embedding model is
# already caught by the manifest, and re-uploading the same asset name is fine.
ASSET_NAME = "eval-cache-v1"
ARCHIVE_NAME = f"{ASSET_NAME}.tar.gz"
MANIFEST_NAME = "manifest.json"
SCHEMA = "climate-risk-agent/eval-cache/1"

CACHE_ROOT = Path("data/cache")
EMBED_CACHE_DIR = CACHE_ROOT / "embeddings"

PACK_COMMAND = "uv run python -m scripts.pack_eval_cache"
# First time: create the release that carries the asset. Afterwards: re-upload.
UPLOAD_COMMAND = (
    f'gh release create {ASSET_NAME} {ARCHIVE_NAME} '
    f'--title "Eval cache v1" --notes "Embedding + chunk cache for the eval workflow."'
)
REUPLOAD_COMMAND = f"gh release upload {ASSET_NAME} {ARCHIVE_NAME} --clobber"


class EvalCacheError(RuntimeError):
    """Raised when the cache cannot be packed or does not match the repo."""


def _payload_files() -> list[tuple[Path, str]]:
    """(source path, name inside the archive) for every file we ship."""
    if not EMBED_CACHE_DIR.is_dir():
        raise EvalCacheError(
            f"embedding cache not found at {EMBED_CACHE_DIR}. Run the retrieval eval "
            f"once with GEMINI_API_KEY set to populate it, then re-run {PACK_COMMAND}"
        )
    vectors = sorted(EMBED_CACHE_DIR.glob("*.npy"))
    if not vectors:
        raise EvalCacheError(f"{EMBED_CACHE_DIR} holds no .npy vectors, nothing to pack")
    if not CHUNK_CACHE.exists():
        raise EvalCacheError(
            f"chunk cache not found at {CHUNK_CACHE}. Build it with "
            "`uv run python -c 'from rag.corpus import build_chunk_cache; build_chunk_cache()'`"
        )
    files = [(path, f"embeddings/{path.name}") for path in vectors]
    files.append((CHUNK_CACHE, CHUNK_CACHE.name))
    return files


def contents_sha256(files: list[tuple[Path, str]], *, progress: bool = False) -> str:
    """Hash of the payload: every (archive name, size, bytes), in sorted order.

    Length-prefixed and name-prefixed so two different file sets can never hash
    the same. Streamed, because the embedding cache is thousands of small files.
    """
    digest = hashlib.sha256()
    bar = tqdm(sorted(files, key=lambda f: f[1]), desc="hashing", unit="file", disable=not progress)
    for path, arcname in bar:
        digest.update(arcname.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def pdf_sizes() -> dict[str, int]:
    """Byte size of each corpus PDF: the cheap half of the corpus identity."""
    sizes: dict[str, int] = {}
    for name in CORPUS_FILES:
        path = CORPUS_DIR / name
        if not path.exists():
            raise EvalCacheError(
                f"corpus PDF missing: {path}. Run `uv run python -m scripts.download_ipcc`"
            )
        sizes[name] = path.stat().st_size
    return sizes


def build_manifest(files: list[tuple[Path, str]], *, progress: bool = False) -> dict:
    """The contract the unpacker verifies against the live repo."""
    n_chunks = len(json.loads(CHUNK_CACHE.read_text(encoding="utf-8"))["chunks"])
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embed_model": MODEL,
        "embed_dims": DIMS,
        "chunker_fingerprint": _fingerprint(),
        "pdf_sizes": pdf_sizes(),
        "n_vectors": sum(1 for _, arc in files if arc.startswith("embeddings/")),
        "n_chunks": n_chunks,
        "contents_sha256": contents_sha256(files, progress=progress),
    }


def pack(out_path: Path) -> dict:
    """Write the archive; return the manifest that went into it."""
    files = _payload_files()
    manifest = build_manifest(files, progress=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blob = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    with tarfile.open(out_path, "w:gz") as archive:
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(blob)
        archive.addfile(info, io.BytesIO(blob))
        for path, arcname in tqdm(sorted(files, key=lambda f: f[1]), desc="packing", unit="file"):
            archive.add(path, arcname=arcname)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(ARCHIVE_NAME), help="archive to write")
    args = parser.parse_args()

    manifest = pack(args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out} ({size_mb:.1f} MB)")
    print(f"  embed model   : {manifest['embed_model']} @ {manifest['embed_dims']}d")
    print(f"  chunker       : {manifest['chunker_fingerprint']}")
    print(f"  vectors/chunks: {manifest['n_vectors']} / {manifest['n_chunks']}")
    print(f"  contents sha  : {manifest['contents_sha256']}")
    print(f"\nupload it once as a release asset:\n  {UPLOAD_COMMAND}")
    print(f"replace an existing asset:\n  {REUPLOAD_COMMAND}")


if __name__ == "__main__":
    main()
