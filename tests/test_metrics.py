"""Tests for retrieval metrics (evals/metrics.py). Pure math, offline."""
import pytest

from evals.metrics import mrr, recall_at_k, unique_pages, wilson_ci
from evals.schema import PageRef
from rag.chunk import Chunk


def _chunk(source: str, page: int, i: int = 0) -> Chunk:
    return Chunk(chunk_id=f"{source}#p{page}#{i}", source=source, page=page, text="x")


GOLD = [PageRef(source="a.pdf", page=7)]


def test_unique_pages_dedupes_preserving_rank_order():
    ranked = [_chunk("a.pdf", 3, 0), _chunk("a.pdf", 3, 1), _chunk("b.pdf", 1), _chunk("a.pdf", 3, 2)]
    assert unique_pages(ranked) == [("a.pdf", 3), ("b.pdf", 1)]


def test_recall_at_k_hits_within_k_only():
    retrieved = [("a.pdf", 1), ("a.pdf", 2), ("a.pdf", 7)]
    assert recall_at_k(retrieved, GOLD, k=3) is True
    assert recall_at_k(retrieved, GOLD, k=2) is False


def test_recall_any_of_gold_pages():
    gold = [PageRef(source="a.pdf", page=7), PageRef(source="b.pdf", page=2)]
    assert recall_at_k([("b.pdf", 2)], gold, k=3) is True


def test_mrr_is_reciprocal_rank_of_first_gold():
    assert mrr([("a.pdf", 1), ("a.pdf", 7)], GOLD) == pytest.approx(0.5)
    assert mrr([("a.pdf", 7)], GOLD) == pytest.approx(1.0)
    assert mrr([("a.pdf", 1)], GOLD) == 0.0


def test_wilson_ci_known_value():
    # 36/45 = 80%: Wilson 95% interval ≈ (0.662, 0.891) — the reason we publish
    # CIs: at n=45 a "point estimate" of 80% is really "somewhere in the 66-89 band".
    lo, hi = wilson_ci(36, 45)
    assert lo == pytest.approx(0.6618, abs=1e-3)
    assert hi == pytest.approx(0.8910, abs=1e-3)


def test_wilson_ci_degenerate_n_zero():
    assert wilson_ci(0, 0) == (0.0, 1.0)
