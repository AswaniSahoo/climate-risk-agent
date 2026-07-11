"""Zero-dependency BM25 (Okapi) index over chunks.

Deliberately hand-rolled (~50 lines): no heavyweight retrieval dep, full control
of tokenization, and fully deterministic — the same index and query always give
the same ranking, so published recall@k numbers are reproducible.

Tokenization applies NFKC first: pypdf extracts ligature glyphs (ﬂ, ﬁ) so the
corpus literally contains "inﬂuence"; NFKC folds those back to ASCII at INDEX
time. We do not normalize in rag.parse — the frozen gold-set quotes are verbatim
against the raw extraction and must stay byte-stable.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from rag.chunk import Chunk

_WORD = re.compile(r"[a-z0-9]+")

# Standard Okapi constants: k1 = term-frequency saturation, b = length penalty.
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """NFKC-fold (ligatures → letters), lowercase, split to alphanumeric words."""
    return _WORD.findall(unicodedata.normalize("NFKC", text).lower())


class BM25Index:
    """In-memory BM25 over a fixed chunk list (corpus is static per build)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._doc_terms = [Counter(tokenize(c.text)) for c in chunks]
        self._doc_lens = [sum(t.values()) for t in self._doc_terms]
        self._avg_len = (sum(self._doc_lens) / len(chunks)) if chunks else 0.0

        doc_freq: Counter[str] = Counter()
        for terms in self._doc_terms:
            doc_freq.update(terms.keys())
        n = len(chunks)
        self._idf = {
            term: math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in doc_freq.items()
        }

    def query(self, text: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Rank chunks by BM25 score; drops zero-score chunks (no term overlap)."""
        scores = [0.0] * len(self.chunks)
        for term in tokenize(text):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, doc in enumerate(self._doc_terms):
                tf = doc.get(term, 0)
                if not tf:
                    continue
                norm = K1 * (1 - B + B * self._doc_lens[i] / self._avg_len)
                scores[i] += idf * tf * (K1 + 1) / (tf + norm)

        ranked = sorted(
            ((self.chunks[i], s) for i, s in enumerate(scores) if s > 0),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]
