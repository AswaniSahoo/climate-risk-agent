"""Tests for the frozen gold eval set (evals/gold_set.json).

Structure tests are offline. The verbatim-quote test parses the real PDFs, so it
skips when the corpus isn't on disk (data/ is git-ignored; fetch via
scripts/download_ipcc.py) — everywhere else it mechanically enforces
"hand-verified": every supporting quote must appear verbatim on a gold page.
"""
from pathlib import Path

import pytest

from evals.gold_set import FROZEN_HASH_PATH, load_gold_set
from evals.schema import Slice

CORPUS_DIR = Path("data/ipcc")
CORPUS_FILES = {
    "IPCC_AR6_WGI_SPM.pdf",
    "IPCC_AR6_WGI_Chapter11.pdf",
    "IPCC_AR6_WGI_Chapter12.pdf",
}

EXPECTED_QUOTAS = {
    Slice.SINGLE_PAGE: 12,
    Slice.MULTI_PAGE: 10,
    Slice.REGIONAL_TABLE: 10,
    Slice.OUT_OF_SCOPE_HAZARD: 4,
    Slice.OUT_OF_CORPUS: 3,
    Slice.PREMISE_INJECTION: 4,
    Slice.DUPLICATE_REGION: 2,
}


@pytest.fixture(scope="module")
def gold_set():
    return load_gold_set()


def test_gold_set_has_45_valid_questions(gold_set):
    assert len(gold_set.questions) == 45


def test_slice_quotas_match_spec(gold_set):
    counts: dict[Slice, int] = {}
    for q in gold_set.questions:
        counts[q.slice] = counts.get(q.slice, 0) + 1
    assert counts == EXPECTED_QUOTAS


def test_question_ids_are_unique(gold_set):
    ids = [q.id for q in gold_set.questions]
    assert len(ids) == len(set(ids))


def test_gold_sources_are_known_corpus_files(gold_set):
    for q in gold_set.questions:
        for ref in q.gold_pages:
            assert ref.source in CORPUS_FILES, f"{q.id}: unknown source {ref.source}"


def test_frozen_hash_matches():
    """The freeze: editing any question breaks this until deliberately re-frozen."""
    frozen = FROZEN_HASH_PATH.read_text(encoding="utf-8").strip()
    assert load_gold_set().content_hash() == frozen


@pytest.mark.skipif(
    not all((CORPUS_DIR / f).exists() for f in CORPUS_FILES),
    reason="IPCC PDFs not on disk (run scripts/download_ipcc.py)",
)
def test_supporting_quotes_are_verbatim_on_a_gold_page(gold_set):
    """Machine-enforced 'hand-verified': each quote is an exact substring of the
    extracted text of at least one of its gold pages."""
    from rag.parse import extract_pages

    page_text: dict[tuple[str, int], str] = {}
    for name in CORPUS_FILES:
        for page in extract_pages(CORPUS_DIR / name):
            page_text[(page.source, page.page)] = page.text

    failures = []
    for q in gold_set.questions:
        if not q.supporting_quote:
            continue
        if not any(
            q.supporting_quote in page_text.get((r.source, r.page), "")
            for r in q.gold_pages
        ):
            failures.append(q.id)
    assert not failures, f"quotes not found verbatim on any gold page: {failures}"
