"""Typed home for the gold eval set.

An `EvalQuestion` is one benchmark item: the question, its slice, the gold label
(the set of acceptable pages), whether it is answerable, and the hand-verified
supporting quote. An `EvalSet` wraps the list and exposes a stable
`content_hash` — the "freeze": reordering questions leaves the hash unchanged,
but editing any content changes it, so a published recall number is auditable.

Gold pages are **PDF-sequential** page numbers (what `rag.chunk.Chunk.page`
carries), NOT the IPCC printed page number — retrieval emits the former, so the
eval must score against the same index.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class Slice(str, Enum):
    """The seven eval slices (Fable-locked spec). Slice-wise metrics expose which
    kind of question a retrieval change actually helped or hurt."""

    SINGLE_PAGE = "single_page"           # answer sits on one page
    MULTI_PAGE = "multi_page"             # any_of: several acceptable pages
    REGIONAL_TABLE = "regional_table"     # a region's row in an assessment table
    OUT_OF_SCOPE_HAZARD = "out_of_scope_hazard"  # hazard we don't support
    OUT_OF_CORPUS = "out_of_corpus"       # answerable by no document we hold
    PREMISE_INJECTION = "premise_injection"      # embeds a false premise
    DUPLICATE_REGION = "duplicate_region"        # traps the orphaned-cell bug


class ExpectedBehavior(str, Enum):
    """What a correct agent does with the question — drives the refusal matrix."""

    ANSWER = "answer"
    REFUSE = "refuse"


class PageRef(BaseModel, frozen=True):
    """One acceptable gold page: which PDF, which PDF-sequential page.

    Frozen so it is hashable and can never drift after authoring.
    """

    source: str
    page: int = Field(ge=1)


class EvalQuestion(BaseModel):
    """One benchmark item with a hand-verified gold label.

    `answerable` and `expected_behavior` are orthogonal on purpose (Fable-ruled):
    `answerable` is a *corpus fact* (the evidence exists in our three PDFs,
    scope-agnostic); `expected_behavior` is the *policy* call (answer or refuse).
    Decoupling them keeps retrieval recall measurable on a refused item — e.g. a
    premise-injection question whose gold page *refutes* the false premise, or an
    out-of-scope hazard (drought) whose evidence is in Ch.11 but that the agent
    must decline. Invariant: answerable <=> gold_pages non-empty; ANSWER => answerable.
    """

    id: str
    slice: Slice
    question: str
    answerable: bool  # corpus fact: evidence exists in our 3 docs (scope-agnostic)
    expected_behavior: ExpectedBehavior  # policy: answer or refuse
    # Acceptable gold pages (any_of). On a REFUSE item this is the page a correct
    # refusal/correction must be grounded in — not "the page that answers".
    gold_pages: list[PageRef] = Field(default_factory=list)
    supporting_quote: str = ""
    hazard: str | None = None  # free text: out-of-scope hazards aren't in our enum
    notes: str = ""

    @field_validator("gold_pages")
    @classmethod
    def _canonicalise(cls, refs: list[PageRef]) -> list[PageRef]:
        """Sort + dedupe gold pages so the frozen hash is order-independent."""
        unique = {(r.source, r.page) for r in refs}
        return [PageRef(source=s, page=p) for s, p in sorted(unique)]

    @model_validator(mode="after")
    def _check_answerability(self) -> "EvalQuestion":
        """Eval-integrity gate (Fable-ruled). `answerable` is a corpus fact,
        `expected_behavior` is orthogonal policy:
        - answerable <=> gold_pages non-empty  (evidence exists iff we point at it)
        - ANSWER      => answerable            (can't answer with no evidence)
        """
        if self.answerable and not self.gold_pages:
            raise ValueError("an answerable question must cite at least one gold page")
        if not self.answerable and self.gold_pages:
            raise ValueError("an unanswerable question must carry no gold pages")
        if self.expected_behavior is ExpectedBehavior.ANSWER and not self.answerable:
            raise ValueError("an ANSWER question must be answerable")
        return self


class EvalSet(BaseModel):
    """The full gold set. `content_hash` is the freeze."""

    questions: list[EvalQuestion]

    def content_hash(self) -> str:
        """Stable SHA-256 over the questions in canonical (id-sorted) order."""
        canonical = json.dumps(
            [q.model_dump(mode="json") for q in sorted(self.questions, key=lambda q: q.id)],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
