"""Retrieval metrics: recall@k, MRR, Wilson 95% CI.

Retrieval returns ranked CHUNKS; gold labels are PAGES. So chunk hits are first
deduped to unique (source, page) pairs preserving rank order, and every metric
is computed over that page ranking. recall@k is any_of: one gold page in the
top-k pages = hit. Wilson CIs are published beside every rate because at n≈45 a
point estimate is a band, not a number.
"""
from __future__ import annotations

import math

from evals.schema import PageRef
from rag.chunk import Chunk

Page = tuple[str, int]  # (source, page)


def unique_pages(ranked_chunks: list[Chunk]) -> list[Page]:
    """Collapse a chunk ranking to a page ranking (first hit keeps the rank)."""
    seen: set[Page] = set()
    pages: list[Page] = []
    for chunk in ranked_chunks:
        page = (chunk.source, chunk.page)
        if page not in seen:
            seen.add(page)
            pages.append(page)
    return pages


def recall_at_k(retrieved: list[Page], gold: list[PageRef], k: int) -> bool:
    """True if any gold page appears in the top-k retrieved pages (any_of)."""
    gold_pages = {(ref.source, ref.page) for ref in gold}
    return any(page in gold_pages for page in retrieved[:k])


def mrr(retrieved: list[Page], gold: list[PageRef]) -> float:
    """Reciprocal rank of the first gold page (0.0 when none retrieved)."""
    gold_pages = {(ref.source, ref.page) for ref in gold}
    for rank, page in enumerate(retrieved, start=1):
        if page in gold_pages:
            return 1.0 / rank
    return 0.0


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion (well-behaved at small n)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)
