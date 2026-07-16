"""The production retriever: BM25 + dense fused by RRF, with a loud BM25 fallback.

This is the one retrieval entry point the answering path, the MCP server, and
the end-to-end eval all share — so the measured hybrid numbers (headline R@3
91% vs 82% BM25-only) describe the system users actually get.

Fallback is graceful AND loud (observable-CLI rule): without Gemini auth or a
warm cache the retriever degrades to BM25-only and says so — a key-less clone
still works, and nobody mistakes degraded mode for the measured hybrid.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from rag.bm25 import BM25Index
from rag.chunk import Chunk
from rag.dense import DenseIndex
from rag.embed import DiskVectorCache, EmbeddingError, cached_embed_texts
from rag.hybrid import rrf_fuse

_log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache/embeddings")
_CANDIDATES = 50  # per-retriever candidate depth before fusion


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], *, doc_matrix: np.ndarray | None,
                 cache: DiskVectorCache | None = None):
        self.chunks = list(chunks)
        self._bm25 = BM25Index(self.chunks)
        self._dense = DenseIndex(self.chunks, doc_matrix) if doc_matrix is not None else None
        self._cache = cache

    @property
    def dense_enabled(self) -> bool:
        return self._dense is not None

    @classmethod
    def build(cls, chunks: list[Chunk], *, cache_dir: Path | str = DEFAULT_CACHE_DIR
              ) -> "HybridRetriever":
        """Embed the corpus through the disk cache; degrade to BM25-only loudly."""
        cache = DiskVectorCache(cache_dir)
        try:
            matrix = np.asarray(
                cached_embed_texts([c.text for c in chunks],
                                   task_type="RETRIEVAL_DOCUMENT", cache=cache)
            )
        except EmbeddingError as exc:
            _log.warning(
                "dense unavailable (%s) — running BM25-only "
                "(measured hybrid quality requires embeddings)", exc,
            )
            return cls(chunks, doc_matrix=None, cache=cache)
        return cls(chunks, doc_matrix=matrix, cache=cache)

    def _embed_query(self, question: str) -> list[float]:
        cache = self._cache or DiskVectorCache(DEFAULT_CACHE_DIR)
        [vector] = cached_embed_texts([question], task_type="RETRIEVAL_QUERY", cache=cache)
        return vector

    def retrieve(self, question: str, top_k: int = 5) -> list[Chunk]:
        """Ranked chunks for a question: RRF(bm25, dense), or BM25 on fallback."""
        lexical = [c for c, _ in self._bm25.query(question, top_k=_CANDIDATES)]
        if self._dense is None:
            return lexical[:top_k]
        try:
            query_vector = self._embed_query(question)
        except EmbeddingError as exc:
            _log.warning(
                "query embedding failed (%s) — falling back to BM25 for this question", exc
            )
            return lexical[:top_k]
        semantic = [c for c, _ in self._dense.query(query_vector, top_k=_CANDIDATES)]
        return rrf_fuse([lexical, semantic], top_k=top_k)
