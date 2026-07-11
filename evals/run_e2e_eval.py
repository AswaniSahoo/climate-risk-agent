"""End-to-end eval: frozen gold set → guard + BM25 + LLM → trust metrics.

Reports:
- refusal 4-cell confusion matrix (false_answer = confabulation; target 0)
- citation validity: of non-abstaining answers on gold-bearing questions, how
  many cite at least one gold page
- numeric provenance: of non-abstaining answers, how many contain only numbers
  that exist in their cited excerpts
- grounded refusals: of correct refusals on gold-bearing questions, how many
  cite the refuting page (scope-guard refusals are pre-LLM and cite nothing;
  counted separately)

LLM calls are paced ~7 s apart (free-tier RPM). Run:
  uv run python -m evals.run_e2e_eval
"""
from __future__ import annotations

import time

from evals.checkers import citation_hits_gold, numeric_provenance_ok, refusal_cell
from evals.gold_set import load_gold_set
from evals.run_retrieval_eval import build_chunks
from evals.schema import ExpectedBehavior
from rag.answer import AnswerError, answer_with_guard
from rag.bm25 import BM25Index
from rag.scope import out_of_scope_hazard

import os as _os

# Default pacing fits the FREE tier (~5 RPM measured); paid tier: EVAL_LLM_PAUSE_S=0.5
_LLM_PAUSE_S = float(_os.environ.get("EVAL_LLM_PAUSE_S", "13.0"))
_TOP_K = 5


def main() -> None:
    gold = load_gold_set()
    chunks = build_chunks()
    index = BM25Index(chunks)
    by_id = {c.chunk_id: c for c in chunks}

    cells: dict[str, list[str]] = {}
    citation_valid: list[bool] = []
    numeric_ok: list[bool] = []
    grounded: list[bool] = []
    scope_refusals = 0
    errors: list[str] = []

    for q in gold.questions:
        top = [c for c, _ in index.query(q.question, top_k=_TOP_K)]
        guard_fired = out_of_scope_hazard(q.question) is not None
        try:
            result = answer_with_guard(q.question, top)
        except AnswerError as exc:
            errors.append(f"{q.id}: {exc}")
            continue
        if not guard_fired:
            time.sleep(_LLM_PAUSE_S)  # free-tier pacing

        expected_refuse = q.expected_behavior is ExpectedBehavior.REFUSE
        cell = refusal_cell(expected_refuse=expected_refuse, did_abstain=result.abstain)
        cells.setdefault(cell, []).append(q.id)

        if not result.abstain:
            cited_texts = [by_id[c].text for c in result.citations]
            numeric_ok.append(numeric_provenance_ok(result.answer, cited_texts))
            if q.gold_pages:
                citation_valid.append(citation_hits_gold(result.citations, q.gold_pages))
        elif expected_refuse and q.gold_pages:
            if guard_fired:
                scope_refusals += 1  # pre-LLM refusal: grounded-citation N/A
            else:
                grounded.append(citation_hits_gold(result.citations, q.gold_pages))

    print("\n-- refusal confusion matrix --")
    for cell in ("correct_answer", "correct_refuse", "false_refuse", "false_answer"):
        ids = cells.get(cell, [])
        tail = f"  <- {', '.join(ids)}" if cell in ("false_refuse", "false_answer") and ids else ""
        print(f"{cell:15s} {len(ids):2d}{tail}")

    def rate(name: str, flags: list[bool]) -> None:
        if flags:
            print(f"{name}: {sum(flags)}/{len(flags)} = {sum(flags)/len(flags):.0%}")

    print()
    rate("citation validity (non-abstain, gold-bearing)", citation_valid)
    rate("numeric provenance (non-abstain)", numeric_ok)
    rate("grounded refusals (LLM refusals w/ gold)", grounded)
    print(f"scope-guard refusals (pre-LLM, ungrounded by design): {scope_refusals}")
    if errors:
        print("\nERRORS:", *errors, sep="\n  ")


if __name__ == "__main__":
    main()
