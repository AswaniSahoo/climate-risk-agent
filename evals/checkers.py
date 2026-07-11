"""End-to-end answer checkers: citation validity, numeric provenance, refusal cells.

These are the automated trust gates from the design (reliability ranked by
trust-per-hour): a citation must land on a gold page, every number in an answer
must exist in a cited excerpt, and refusal behavior is scored as a 4-cell
confusion matrix (false_answer = confabulation, the cell that must stay at 0).
"""
from __future__ import annotations

import re

from evals.schema import PageRef

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _page_of(chunk_id: str) -> tuple[str, int]:
    """chunk_id format: '<source>#p<page>#<index>' (see rag.chunk)."""
    source, page_part, _ = chunk_id.split("#")
    return source, int(page_part[1:])


def citation_hits_gold(citations: list[str], gold: list[PageRef]) -> bool:
    """True when any cited chunk lies on one of the gold pages."""
    gold_pages = {(ref.source, ref.page) for ref in gold}
    return any(_page_of(c) in gold_pages for c in citations)


def numbers_in_text(text: str) -> set[str]:
    """Numeric tokens in the text ('7', '0.20', '1901'); trailing dots trimmed."""
    return set(_NUMBER.findall(text))


def numeric_provenance_ok(answer: str, cited_texts: list[str]) -> bool:
    """Every number in the answer must appear in at least one cited excerpt.

    Vacuously true for number-free answers. String-level matching on purpose:
    '7' backs '7%' and '7 percent', but an invented '12' has no source and fails.
    """
    cited_numbers: set[str] = set()
    for text in cited_texts:
        cited_numbers |= numbers_in_text(text)
    return numbers_in_text(answer) <= cited_numbers


def refusal_cell(*, expected_refuse: bool, did_abstain: bool) -> str:
    """Which cell of the refusal confusion matrix this outcome lands in."""
    if expected_refuse:
        return "correct_refuse" if did_abstain else "false_answer"
    return "false_refuse" if did_abstain else "correct_answer"
