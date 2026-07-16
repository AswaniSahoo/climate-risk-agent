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

import os as _os
import time

from evals.checkers import citation_hits_gold, numeric_provenance_ok, refusal_cell
from evals.gold_set import load_gold_set
from evals.run_retrieval_eval import build_chunks
from evals.schema import ExpectedBehavior
from rag.answer import AnswerError, answer_with_guard
from rag.retrieve import HybridRetriever
from rag.scope import out_of_scope_hazard

# Default pacing fits the FREE tier (~5 RPM measured); paid tier: EVAL_LLM_PAUSE_S=0.5
_LLM_PAUSE_S = float(_os.environ.get("EVAL_LLM_PAUSE_S", "13.0"))
# top_k=8 (A/B-measured 2026-07-12): lets table-HEADER chunks (GWL column labels)
# into context, fixing column-ambiguity false refusals on regional-table rows
# (RT-07/RT-10); matrix 33/11/1/0 vs 31/11/3/0 at k=5, false_answer 0 at both
_TOP_K = int(_os.environ.get("EVAL_TOP_K", "8"))
# EVAL_CLAIM_JUDGE=1 adds the claim-level LLM judge (one extra call per
# non-abstaining answer) — reported NEXT TO the deterministic checkers, never
# instead of them.
_CLAIM_JUDGE = _os.environ.get("EVAL_CLAIM_JUDGE", "") == "1"


def main() -> None:
    from obs.log import configure

    configure()  # runner owns logging config
    gold = load_gold_set()
    chunks = build_chunks()
    retriever = HybridRetriever.build(chunks)  # the measured 91% path; loud BM25 fallback
    print(f"retriever: dense_enabled={retriever.dense_enabled}")
    by_id = {c.chunk_id: c for c in chunks}

    cells: dict[str, list[str]] = {}
    citation_valid: list[bool] = []
    numeric_ok: list[bool] = []
    grounded: list[bool] = []
    claims_supported: list[bool] = []
    claim_totals = [0, 0]  # [supported, total] across all judged answers
    scope_refusals = 0
    errors: list[str] = []

    from tqdm import tqdm

    progress = tqdm(gold.questions, desc="e2e", unit="q")
    for q in progress:
        top = retriever.retrieve(q.question, top_k=_TOP_K)
        guard_fired = out_of_scope_hazard(q.question) is not None
        try:
            result = answer_with_guard(q.question, top)
        except AnswerError as exc:
            errors.append(f"{q.id}: {exc}")
            progress.write(f"  {q.id:7s} ERROR: {exc}")
            continue
        if not guard_fired:
            time.sleep(_LLM_PAUSE_S)  # free-tier pacing

        expected_refuse = q.expected_behavior is ExpectedBehavior.REFUSE
        cell = refusal_cell(expected_refuse=expected_refuse, did_abstain=result.abstain)
        cells.setdefault(cell, []).append(q.id)
        how = "guard" if guard_fired else "llm"
        progress.write(f"  {q.id:7s} {cell:15s} ({how}, {len(result.citations)} citations)")

        if not result.abstain:
            cited_texts = [by_id[c].text for c in result.citations]
            numeric_ok.append(numeric_provenance_ok(result.answer, cited_texts))
            if q.gold_pages:
                citation_valid.append(citation_hits_gold(result.citations, q.gold_pages))
            if _CLAIM_JUDGE:
                from evals.claim_judge import judge_claims

                judgment = judge_claims(result.answer, cited_texts)
                claims_supported.append(judgment.all_supported)
                claim_totals[0] += judgment.n_supported
                claim_totals[1] += judgment.n_claims
                if not judgment.all_supported:
                    bad = [c.claim for c in judgment.claims if not c.supported]
                    progress.write(f"          unsupported claim(s): {bad}")
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
    rate("answers with ALL claims judge-supported", claims_supported)
    if claim_totals[1]:
        print(f"claim support (LLM judge): {claim_totals[0]}/{claim_totals[1]} "
              f"= {claim_totals[0]/claim_totals[1]:.0%}")
    print(f"scope-guard refusals (pre-LLM, ungrounded by design): {scope_refusals}")
    if errors:
        print("\nERRORS:", *errors, sep="\n  ")

    # Committed artifact: the numbers as a verifiable file, not README prose.
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    def _rate(flags: list[bool]) -> dict:
        return {"passed": sum(flags), "total": len(flags)}

    artifact = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gold_set_sha256_file": "evals/gold_set.sha256",
        "top_k": _TOP_K,
        "matrix": {cell: sorted(ids) for cell, ids in sorted(cells.items())},
        "citation_validity": _rate(citation_valid),
        "numeric_provenance": _rate(numeric_ok),
        "grounded_refusals": _rate(grounded),
        "claim_judge": (
            {"answers_fully_supported": _rate(claims_supported),
             "claims_supported": claim_totals[0], "claims_total": claim_totals[1]}
            if claim_totals[1] else None
        ),
        "scope_guard_refusals": scope_refusals,
        "errors": errors,
    }
    out_dir = Path("evals/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"e2e-{datetime.now(timezone.utc):%Y-%m-%d}.json"
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nartifact written: {out_path} (commit it — release-gate evidence)")


if __name__ == "__main__":
    main()
