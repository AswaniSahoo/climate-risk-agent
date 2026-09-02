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
  uv run python -m evals.run_e2e_eval --scope-stage2 embed   # semantic guard arm

`--scope-stage2 off|embed|llm` mirrors the CRG_SCOPE_STAGE2 env var, so the
three guard arms are measured by the same runner on the same frozen set and
land in three separate artifacts (the arm is in the filename).
"""
from __future__ import annotations

import argparse
import os as _os
import time

from evals.checkers import citation_hits_gold, numeric_provenance_ok, refusal_cell
from evals.gold_set import load_gold_set, load_test_set
from evals.run_retrieval_eval import build_chunks
from evals.schema import ExpectedBehavior
from rag.answer import AnswerError, answer_with_guard
from rag.retrieve import HybridRetriever
from rag.scope import STAGE2_MODES, scope_verdict, stage2_mode

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
# EVAL_SET=test runs the HELD-OUT v2 set — release gates ONLY (exposure
# protocol in DEPLOY.md). Default dev: an accidental run must never burn
# the test set.
_EVAL_SET = _os.environ.get("EVAL_SET", "dev")


def _parse_args() -> argparse.Namespace:
    # Plain-ASCII description on purpose: __doc__ carries arrows, and a Windows
    # cp1252 console raises UnicodeEncodeError printing them (measured on --help).
    parser = argparse.ArgumentParser(
        description="End-to-end eval: frozen gold set through the guard, retrieval and the LLM."
    )
    parser.add_argument(
        "--scope-stage2",
        choices=STAGE2_MODES,
        default=None,
        help="semantic scope-guard arm (mirrors CRG_SCOPE_STAGE2; default: the env var, else off)",
    )
    return parser.parse_args()


def main() -> None:
    from obs.log import configure

    args = _parse_args()
    if args.scope_stage2 is not None:
        _os.environ["CRG_SCOPE_STAGE2"] = args.scope_stage2

    configure()  # runner owns logging config
    from rag.scope_semantic import reset_stats, stats

    reset_stats()
    arm = stage2_mode()
    gold = load_test_set() if _EVAL_SET == "test" else load_gold_set()
    print(f"eval set: {_EVAL_SET} ({len(gold.questions)} questions) | scope stage 2: {arm}")
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

    by_slice: dict[str, dict[str, list[str]]] = {}
    stage2_notes: list[str] = []

    progress = tqdm(gold.questions, desc="e2e", unit="q")
    for q in progress:
        top = retriever.retrieve(q.question, top_k=_TOP_K)
        # Same call the answer path makes; memoised per (question, arm) so the
        # LLM arm is charged once per question, not twice.
        decision = scope_verdict(q.question)
        guard_fired = decision.out_of_scope is not None
        if decision.stage != "lexical":
            stage2_notes.append(
                f"{q.id}: stage2={decision.stage} "
                f"verdict={decision.out_of_scope or decision.hazard_hint or 'defer'} "
                f"[{decision.detail}]"
            )
        try:
            result = answer_with_guard(q.question, top)
        except AnswerError as exc:
            errors.append(f"{q.id}: {exc}")
            progress.write(f"  {q.id:7s} ERROR: {exc}")
            continue
        # Pace whenever this question cost a generation call: the answer call,
        # and/or the stage-2 classifier in the llm arm.
        if not guard_fired or (arm == "llm" and decision.stage == "llm"):
            time.sleep(_LLM_PAUSE_S)  # free-tier pacing

        expected_refuse = q.expected_behavior is ExpectedBehavior.REFUSE
        cell = refusal_cell(expected_refuse=expected_refuse, did_abstain=result.abstain)
        cells.setdefault(cell, []).append(q.id)
        by_slice.setdefault(q.slice.value, {}).setdefault(cell, []).append(q.id)
        how = "guard" if guard_fired else "llm"
        if guard_fired and decision.stage != "lexical":
            how = f"guard:{decision.stage}"
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

    print("\n-- per-slice cells --")
    for slice_name in sorted(by_slice):
        cell_counts = {c: len(ids) for c, ids in sorted(by_slice[slice_name].items())}
        print(f"{slice_name:20s} {cell_counts}")

    s2 = stats()
    n_q = len(gold.questions)
    print(f"\n-- scope stage 2 ({arm}) --")
    print(f"routed to stage 2: {s2['calls']}/{n_q} | refusals: {s2['verdicts']} | "
          f"hazard hints: {s2['hints']} | failures: {s2['failures']}")
    print(f"added latency: {s2['wall_ms']:.0f} ms total = "
          f"{s2['wall_ms']/n_q:.1f} ms/question | live API calls: {s2['api_calls']} | "
          f"added est cost: ${s2['est_cost_usd']:.6f} = "
          f"${s2['est_cost_usd']/n_q:.8f}/question")
    for note in stage2_notes:
        print(f"  {note}")
    if errors:
        print("\nERRORS:", *errors, sep="\n  ")

    # Committed artifact: the numbers as a verifiable file, not README prose.
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    def _rate(flags: list[bool]) -> dict:
        return {"passed": sum(flags), "total": len(flags)}

    # Model identity + measured cost/latency belong IN the artifact: a number
    # without the model that produced it is not evidence (a silent model swap
    # is invisible to tests and CI, and only the eval can catch the change).
    from obs.telemetry import snapshot, summarize
    from rag.gemini_client import EMBED_MODEL, GENERATE_MODEL

    artifact = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eval_set": _EVAL_SET,
        "generate_model": GENERATE_MODEL,
        "embed_model": EMBED_MODEL,
        "telemetry": summarize(snapshot()),
        "gold_set_sha256_file": (
            "evals/gold_set_v2.sha256" if _EVAL_SET == "test" else "evals/gold_set.sha256"
        ),
        "top_k": _TOP_K,
        "scope_stage2": {
            "arm": arm,
            "questions": len(gold.questions),
            "routed": s2["calls"],
            "refusals": s2["verdicts"],
            "hazard_hints": s2["hints"],
            "failures": s2["failures"],
            "live_api_calls": s2["api_calls"],
            "added_wall_ms_total": round(s2["wall_ms"], 1),
            "added_wall_ms_per_question": round(s2["wall_ms"] / len(gold.questions), 2),
            "added_est_cost_usd": round(s2["est_cost_usd"], 8),
            "added_est_cost_usd_per_question": round(
                s2["est_cost_usd"] / len(gold.questions), 10
            ),
            "notes": stage2_notes,
        },
        "matrix": {cell: sorted(ids) for cell, ids in sorted(cells.items())},
        "matrix_by_slice": {
            name: {cell: sorted(ids) for cell, ids in sorted(cell_map.items())}
            for name, cell_map in sorted(by_slice.items())
        },
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
    # Model goes in the FILENAME so a bake-off across models cannot overwrite
    # its own evidence (same set + same day + different model = different file).
    suffix = "-test" if _EVAL_SET == "test" else ""
    # The guard arm goes in the filename for the same reason the model does: a
    # bake-off across arms must not overwrite its own evidence.
    arm_slug = "" if arm == "off" else f"-stage2-{arm}"
    slug = GENERATE_MODEL.replace("/", "-")
    out_path = (
        out_dir / f"e2e{suffix}-{slug}{arm_slug}-{datetime.now(timezone.utc):%Y-%m-%d}.json"
    )
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nartifact written: {out_path} (commit it — release-gate evidence)")


if __name__ == "__main__":
    main()
