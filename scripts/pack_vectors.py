"""Pack the baked corpus vectors into one file, so a cold container reads one file, not 2,730.

WHY. Cloud Run gen2 pulls image files on first read. On revision 00031
(2026-11-11, 1 vCPU) the first live IPCC answer of a process logged "corpus
loaded from chunk cache (2730 chunks)" at 19:29:14.66 and "all 2730 embeddings
loaded from cache" at 19:29:32.19: 17.5 s to read the vector files, 3,200 bytes
each, that a local disk reads in 0.15 s.

WHAT. The RETRIEVAL_DOCUMENT vector of every baked chunk, copied from its vector
file into data/cache/embeddings.pack.npy, one row per distinct chunk text. No
cast: rag.embed.write_pack refuses anything that is not already float32, so the
rows are the files' own bits. rag/embed.py reads the pack before the files; the
files stay, as the fallback and as the write path for newly embedded questions.
It refuses to pack an index that lacks a vector file, then reads the pack back
and checks every row against its file. scripts/check_baked_index.py does so
again as the build's last step.

Not scripts/pack_eval_cache.py, which packs the eval-cache release asset (a
tarball of the vector files).

Run (a Dockerfile builder step, after the chunk cache is built):
    python -m scripts.pack_vectors
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rag.corpus import CHUNK_CACHE
from rag.embed import DiskVectorCache, cache_key, pack_path, vector_digest, write_pack
from rag.retrieve import DEFAULT_CACHE_DIR
from scripts.check_baked_index import coverage, incomplete_message, pack_problem


def pack(chunks_path: Path | str = CHUNK_CACHE,
         vectors_dir: Path | str = DEFAULT_CACHE_DIR) -> int:
    """Write the pack of the baked chunks' document vectors; returns its rows."""
    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))["chunks"]
    store = DiskVectorCache(vectors_dir)
    vectors: dict[str, np.ndarray] = {}
    for chunk in chunks:
        key = cache_key(chunk["text"], "RETRIEVAL_DOCUMENT")
        vector = store.file_vector(key)
        if vector is None:
            raise FileNotFoundError(f"no vector file for {chunk['chunk_id']}")
        vectors[vector_digest(key)] = vector
    return write_pack(pack_path(vectors_dir), vectors)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=Path, default=CHUNK_CACHE)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)
    before = coverage(args.chunks, args.vectors)
    if before.chunks == 0 or before.missing:
        print(incomplete_message(before), file=sys.stderr)
        return 1
    rows = pack(args.chunks, args.vectors)
    target = pack_path(args.vectors)
    after = coverage(args.chunks, args.vectors, target)
    problem = pack_problem(after, target)
    if problem is not None:
        target.unlink(missing_ok=True)
        print(problem, file=sys.stderr)
        return 1
    assert after.pack_bytes is not None
    print(f"vector pack: {after.packed} chunks' document vectors in {rows} rows -> {target} "
          f"({after.pack_bytes / 1e6:.1f} MB), equal to their files bit for bit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
