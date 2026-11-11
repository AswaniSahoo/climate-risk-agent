"""Fail the image build unless every chunk it will serve has its document vector on disk.

WHY. HybridRetriever.build (rag/retrieve.py) embeds, through the API, every
chunk whose RETRIEVAL_DOCUMENT vector is missing from data/cache/embeddings,
at 62 s per batch of 50 under the default pacing. An image that shipped
without the vectors would pay that on the first report of every cold
container. The count of chunks in the baked chunks.json and the count of
their document vectors must therefore be equal (2,730 and 2,730 for the
corpus pinned today); the asset also carries query vectors for the eval
questions, which are counted apart and never stand in for a document vector.

The same vectors also sit in one packed file (scripts/pack_vectors.py), which
the runtime reads before the files: a Cloud Run gen2 container spent 17.5 s
reading them as 2,730 files. So the check also fails unless the pack holds
every chunk's vector and each packed row equals its vector file, bit for bit.

Run (the Dockerfile's last builder step):  python -m scripts.check_baked_index
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rag.corpus import CHUNK_CACHE
from rag.embed import DiskVectorCache, cache_key, pack_path, read_pack, vector_digest
from rag.retrieve import DEFAULT_CACHE_DIR


@dataclass
class Coverage:
    chunks: int
    document_vectors: int
    vector_files: int
    missing: list[str] = field(default_factory=list)
    pack_bytes: int | None = None  # None: there is no pack file
    packed: int = 0  # chunks whose packed row equals their vector file
    pack_missing: list[str] = field(default_factory=list)
    pack_mismatched: list[str] = field(default_factory=list)
    pack_error: str | None = None


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def coverage(chunks_path: Path | str = CHUNK_CACHE,
             vectors_dir: Path | str = DEFAULT_CACHE_DIR,
             pack: Path | str | None = None) -> Coverage:
    """How many of the baked chunks have a document vector file, and the pack's row for it."""
    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))["chunks"]
    store = DiskVectorCache(vectors_dir)
    pack_file = Path(pack) if pack is not None else pack_path(vectors_dir)
    packed: Mapping[str, np.ndarray] = {}
    pack_error = None
    if pack_file.exists():
        try:
            packed = read_pack(pack_file)
        except Exception as exc:  # noqa: BLE001 - reported, so the build fails with the reason
            pack_error = str(exc)
    missing: list[str] = []
    pack_missing: list[str] = []
    mismatched: list[str] = []
    for chunk in chunks:
        key = cache_key(chunk["text"], "RETRIEVAL_DOCUMENT")
        on_disk = store.file_vector(key)  # the file itself, never the pack
        if on_disk is None:
            missing.append(chunk["chunk_id"])
        row = packed.get(vector_digest(key))
        if row is None:
            pack_missing.append(chunk["chunk_id"])
        elif on_disk is not None and not _same_bits(row, on_disk):
            mismatched.append(chunk["chunk_id"])
    files = len(list(Path(vectors_dir).glob("*.npy")))
    return Coverage(
        len(chunks), len(chunks) - len(missing), files, missing,
        pack_bytes=pack_file.stat().st_size if pack_file.exists() else None,
        packed=len(chunks) - len(pack_missing) - len(mismatched),
        pack_missing=pack_missing, pack_mismatched=mismatched, pack_error=pack_error,
    )


def incomplete_message(result: Coverage) -> str:
    """Why an index with chunks lacking their vector file cannot ship."""
    return (
        f"BAKED INDEX INCOMPLETE: the image would serve {result.chunks} chunks but only "
        f"{result.document_vectors} have a document vector on disk, so every cold start "
        f"would embed the other {result.chunks - result.document_vectors} through the "
        "embedding API. The vectors come from the eval-cache-v1 asset: after a re-chunk "
        "or an embedding-model change, rebuild it (scripts/pack_eval_cache.py), upload it "
        "and update the pinned sha256 in the Dockerfile. First missing: "
        f"{', '.join(result.missing[:3]) or 'none (no chunks at all)'}."
    )


def pack_problem(result: Coverage, pack: Path) -> str | None:
    """Why the pack cannot ship, or None when every row is there and equals its file."""
    if result.pack_bytes is None:
        return (
            f"VECTOR PACK MISSING: {pack} does not exist, so a cold container would read the "
            f"{result.chunks} chunks' vectors file by file (17.5 s on Cloud Run gen2 on "
            "2026-11-11). Run scripts/pack_vectors.py after the chunk cache is built."
        )
    if result.pack_error or result.pack_missing or result.pack_mismatched:
        unreadable = f" It does not read as a pack: {result.pack_error}." if result.pack_error else ""
        first = (result.pack_mismatched + result.pack_missing)[:3]
        return (
            f"VECTOR PACK WRONG: in {pack}, {len(result.pack_missing)} of {result.chunks} chunks "
            f"have no row and {len(result.pack_mismatched)} differ from their vector file.{unreadable}"
            f" Re-run scripts/pack_vectors.py. First affected: {', '.join(first)}."
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=Path, default=CHUNK_CACHE)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--pack", type=Path, default=None,
                        help="the packed vectors (default: beside --vectors, <folder>.pack.npy)")
    args = parser.parse_args(argv)
    pack = args.pack if args.pack is not None else pack_path(args.vectors)
    result = coverage(args.chunks, args.vectors, pack)
    if result.chunks == 0 or result.missing:
        print(incomplete_message(result), file=sys.stderr)
        return 1
    problem = pack_problem(result, pack)
    if problem is not None:
        print(problem, file=sys.stderr)
        return 1
    assert result.pack_bytes is not None
    print(f"baked index: {result.chunks} chunks, {result.document_vectors} document vectors "
          f"on disk ({result.vector_files} vector files in all, the rest are query vectors), "
          f"{result.packed} in the pack {pack} ({result.pack_bytes / 1e6:.1f} MB), "
          "equal to their files bit for bit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
