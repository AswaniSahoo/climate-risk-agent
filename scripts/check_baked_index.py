"""Fail the image build unless every chunk it will serve has its document vector on disk.

WHY. HybridRetriever.build (rag/retrieve.py) embeds, through the API, every
chunk whose RETRIEVAL_DOCUMENT vector is missing from data/cache/embeddings,
at 62 s per batch of 50 under the default pacing. An image that shipped
without the vectors would pay that on the first report of every cold
container. The count of chunks in the baked chunks.json and the count of
their document vectors must therefore be equal (2,730 and 2,730 for the
corpus pinned today); the asset also carries query vectors for the eval
questions, which are counted apart and never stand in for a document vector.

Run (the Dockerfile's last builder step):  python -m scripts.check_baked_index
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rag.corpus import CHUNK_CACHE
from rag.embed import DiskVectorCache, cache_key
from rag.retrieve import DEFAULT_CACHE_DIR


@dataclass
class Coverage:
    chunks: int
    document_vectors: int
    vector_files: int
    missing: list[str] = field(default_factory=list)


def coverage(chunks_path: Path | str = CHUNK_CACHE,
             vectors_dir: Path | str = DEFAULT_CACHE_DIR) -> Coverage:
    """How many of the baked chunks have a readable document vector on disk."""
    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))["chunks"]
    store = DiskVectorCache(vectors_dir)
    missing = [c["chunk_id"] for c in chunks
               if store.get(cache_key(c["text"], "RETRIEVAL_DOCUMENT")) is None]
    files = len(list(Path(vectors_dir).glob("*.npy")))
    return Coverage(len(chunks), len(chunks) - len(missing), files, missing)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=Path, default=CHUNK_CACHE)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)
    result = coverage(args.chunks, args.vectors)
    if result.chunks == 0 or result.missing:
        print(
            f"BAKED INDEX INCOMPLETE: the image would serve {result.chunks} chunks but only "
            f"{result.document_vectors} have a document vector on disk, so every cold start "
            f"would embed the other {result.chunks - result.document_vectors} through the "
            "embedding API. The vectors come from the eval-cache-v1 asset: after a re-chunk "
            "or an embedding-model change, rebuild it (scripts/pack_eval_cache.py), upload it "
            "and update the pinned sha256 in the Dockerfile. First missing: "
            f"{', '.join(result.missing[:3]) or 'none (no chunks at all)'}.",
            file=sys.stderr,
        )
        return 1
    print(f"baked index: {result.chunks} chunks, {result.document_vectors} document vectors "
          f"on disk ({result.vector_files} vector files in all, the rest are query vectors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
