"""Tests for the HELD-OUT v2 test set (evals/gold_set_v2.json).

Same enforcement as the dev set (tests/test_gold_set.py) plus the split
guarantees: v2 questions must not collide with dev-set ids OR duplicate dev
questions (paraphrase-level contamination is caught in Fable review; exact
text duplication is machine-caught here).

All tests skip until the v2 set is assembled and frozen.
"""
from pathlib import Path

import pytest

from evals.gold_set import (
    TEST_SET_FROZEN_HASH_PATH,
    TEST_SET_PATH,
    load_gold_set,
    load_test_set,
)
from evals.schema import Slice

pytestmark = pytest.mark.skipif(
    not TEST_SET_PATH.exists(), reason="v2 test set not assembled yet"
)

CORPUS_DIR = Path("data/ipcc")
CORPUS_FILES = {
    "IPCC_AR6_WGI_SPM.pdf",
    "IPCC_AR6_WGI_Chapter11.pdf",
    "IPCC_AR6_WGI_Chapter12.pdf",
}

# Weakness-weighted quotas (design 2026-07-17), with the 2026-07-18
# reallocation MP 20->15, PI 12->17 (corpus supplies ~15 distinct
# supported-hazard multi-page facts; PI is author-generated and higher value).
# Total 105.
EXPECTED_QUOTAS = {
    Slice.SINGLE_PAGE: 24,
    Slice.MULTI_PAGE: 15,
    Slice.REGIONAL_TABLE: 22,
    Slice.OUT_OF_SCOPE_HAZARD: 10,
    Slice.OUT_OF_CORPUS: 8,
    Slice.PREMISE_INJECTION: 17,
    Slice.DUPLICATE_REGION: 9,
}


@pytest.fixture(scope="module")
def test_set():
    return load_test_set()


def test_has_105_valid_questions(test_set):
    assert len(test_set.questions) == 105


def test_slice_quotas_match_design(test_set):
    counts: dict[Slice, int] = {}
    for q in test_set.questions:
        counts[q.slice] = counts.get(q.slice, 0) + 1
    assert counts == EXPECTED_QUOTAS


def test_ids_unique_within_v2_and_disjoint_from_dev(test_set):
    v2_ids = [q.id for q in test_set.questions]
    assert len(v2_ids) == len(set(v2_ids))
    dev_ids = {q.id for q in load_gold_set().questions}
    assert not dev_ids & set(v2_ids)


def test_no_exact_question_duplication_against_dev(test_set):
    """Held-out means held out: identical question text vs dev = contamination."""
    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    dev_questions = {norm(q.question) for q in load_gold_set().questions}
    dupes = [q.id for q in test_set.questions if norm(q.question) in dev_questions]
    assert not dupes, f"v2 questions duplicate dev questions: {dupes}"


def test_gold_sources_are_known_corpus_files(test_set):
    for q in test_set.questions:
        for ref in q.gold_pages:
            assert ref.source in CORPUS_FILES, f"{q.id}: unknown source {ref.source}"


def test_frozen_hash_matches(test_set):
    frozen = TEST_SET_FROZEN_HASH_PATH.read_text(encoding="utf-8").strip()
    assert test_set.content_hash() == frozen


@pytest.mark.skipif(
    not all((CORPUS_DIR / f).exists() for f in CORPUS_FILES),
    reason="IPCC PDFs not on disk (run scripts/download_ipcc.py)",
)
def test_supporting_quotes_are_verbatim_on_a_gold_page(test_set):
    from rag.parse import extract_pages

    page_text: dict[tuple[str, int], str] = {}
    for name in CORPUS_FILES:
        for page in extract_pages(CORPUS_DIR / name):
            page_text[(page.source, page.page)] = page.text

    failures = []
    for q in test_set.questions:
        if not q.supporting_quote:
            continue
        if not any(
            q.supporting_quote in page_text.get((r.source, r.page), "")
            for r in q.gold_pages
        ):
            failures.append(q.id)
    assert not failures, f"quotes not found verbatim on any gold page: {failures}"
