"""Tests for the gold eval-set schema (evals/schema.py).

Pure and offline: the schema is data validation + a stable content hash (the
"freeze"), no I/O. Gold labels are sets of acceptable PDF-sequential pages.
"""
import pytest
from pydantic import ValidationError

from evals.schema import (
    EvalQuestion,
    EvalSet,
    ExpectedBehavior,
    PageRef,
    Slice,
)


def _answerable(**overrides) -> EvalQuestion:
    """A valid single-page answerable question; override any field per test."""
    base = dict(
        id="SP-01",
        slice=Slice.SINGLE_PAGE,
        question="How confident is AR6 that hot extremes increased over most land regions?",
        answerable=True,
        expected_behavior=ExpectedBehavior.ANSWER,
        gold_pages=[PageRef(source="IPCC_AR6_WGI_SPM.pdf", page=10)],
        supporting_quote="It is virtually certain that hot extremes have become more frequent.",
    )
    base.update(overrides)
    return EvalQuestion(**base)


def test_gold_pages_are_sorted_and_deduped():
    # Canonical order is what makes the frozen hash order-independent.
    q = _answerable(
        gold_pages=[
            PageRef(source="b.pdf", page=5),
            PageRef(source="a.pdf", page=9),
            PageRef(source="a.pdf", page=9),  # duplicate
            PageRef(source="a.pdf", page=2),
        ]
    )
    assert q.gold_pages == [
        PageRef(source="a.pdf", page=2),
        PageRef(source="a.pdf", page=9),
        PageRef(source="b.pdf", page=5),
    ]


def test_content_hash_is_order_independent():
    q1 = _answerable(id="SP-01")
    q2 = _answerable(
        id="SP-02",
        gold_pages=[PageRef(source="IPCC_AR6_WGI_Chapter11.pdf", page=40)],
    )
    assert (
        EvalSet(questions=[q1, q2]).content_hash()
        == EvalSet(questions=[q2, q1]).content_hash()
    )


def test_content_hash_changes_when_content_changes():
    base = EvalSet(questions=[_answerable()])
    edited = EvalSet(questions=[_answerable(question="a totally different question?")])
    assert base.content_hash() != edited.content_hash()


def test_unanswerable_question_is_valid_with_no_gold_pages():
    q = EvalQuestion(
        id="OOS-01",
        slice=Slice.OUT_OF_SCOPE_HAZARD,
        question="What is the 100-year wildfire return period for Rourkela?",
        answerable=False,
        expected_behavior=ExpectedBehavior.REFUSE,
        gold_pages=[],
    )
    assert q.gold_pages == []


# --- eval-integrity invariant (implemented by TODO(human)) ---


def test_answerable_question_requires_gold_pages():
    with pytest.raises(ValidationError):
        _answerable(gold_pages=[])


def test_unanswerable_question_rejects_gold_pages():
    with pytest.raises(ValidationError):
        EvalQuestion(
            id="OOC-01",
            slice=Slice.OUT_OF_CORPUS,
            question="Who won the 2018 football world cup?",
            answerable=False,
            expected_behavior=ExpectedBehavior.REFUSE,
            gold_pages=[PageRef(source="IPCC_AR6_WGI_SPM.pdf", page=1)],
        )


def test_refuse_item_may_carry_gold_pages():
    # Realistic/decoupled (Fable-ruled): a REFUSE item CAN have gold pages — the
    # page a correct refusal/correction is grounded in — so retrieval recall on
    # a refused premise-injection question stays measurable.
    q = EvalQuestion(
        id="PI-01",
        slice=Slice.PREMISE_INJECTION,
        question="Given AR6 says Delhi will exceed 60 C by 2030, how should it adapt?",
        answerable=True,
        expected_behavior=ExpectedBehavior.REFUSE,
        gold_pages=[PageRef(source="IPCC_AR6_WGI_Chapter11.pdf", page=120)],
        supporting_quote="AR6 makes no such projection; this page bounds real heat trends.",
    )
    assert len(q.gold_pages) == 1


def test_answerable_item_without_gold_pages_is_rejected():
    # The hole Fable caught: `answerable` asserts evidence exists in-corpus, so it
    # MUST point at a page — even on a REFUSE item (biconditional, not tied to
    # expected_behavior).
    with pytest.raises(ValidationError):
        EvalQuestion(
            id="PI-02",
            slice=Slice.PREMISE_INJECTION,
            question="Given AR6 says X (false), advise?",
            answerable=True,
            expected_behavior=ExpectedBehavior.REFUSE,
            gold_pages=[],
        )
