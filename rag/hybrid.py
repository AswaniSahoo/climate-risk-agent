"""Reciprocal Rank Fusion (RRF) of multiple retrieval rankings.

score(chunk) = Σ over rankings 1 / (k + rank). Rank-based, so BM25 scores and
cosine similarities never need to share a scale — the standard trick that makes
lexical+dense hybrids work without per-collection weight tuning. k=60 is the
literature default; we do NOT tune it against our own eval set (that would be
fitting the benchmark).
"""
from __future__ import annotations

from rag.chunk import Chunk

RRF_K = 60


def rrf_fuse(rankings: list[list[Chunk]], *, k: int = RRF_K, top_k: int = 10) -> list[Chunk]:
    scores: dict[str, float] = {}
    first_seen: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(chunk.chunk_id, chunk)
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return [first_seen[chunk_id] for chunk_id in ordered]
