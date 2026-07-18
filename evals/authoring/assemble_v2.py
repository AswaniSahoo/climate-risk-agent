"""Assemble the held-out v2 test set from the authored batch files.

Run AFTER all four batches are banked and reviewed:
    uv run python evals/authoring/assemble_v2.py

Validates every question through the EvalQuestion invariant, enforces the
design quotas, checks id/text collisions against the dev set, then writes
evals/gold_set_v2.json + evals/gold_set_v2.sha256 (the freeze) together —
the sidecar is generated in the same step so they can never drift.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.gold_set import (  # noqa: E402
    TEST_SET_FROZEN_HASH_PATH,
    TEST_SET_PATH,
    load_gold_set,
)
from evals.schema import EvalSet  # noqa: E402

BATCHES = ["v2_rt_dr.json", "v2_sp_ooc.json", "v2_mp_osh.json", "v2_pi.json"]
# Reallocation 2026-07-18 (Fable design call, flagged for Aswani's review):
# multi_page 20->15 — the 3-document corpus honestly supplies ~15 DISTINCT
# supported-hazard multi-page facts; the rest are drought/compound (forbidden
# subjects). Padding MP with weak items violates the no-over-engineering rule.
# The 5 moved to premise_injection (12->17): author-generated, no corpus-supply
# ceiling, and the highest-value adversarial-robustness slice. Total stays 105
# and the reallocation follows the same "don't over-invest already-strong
# slices" logic that set the original weights (MP was 90% R@3 in v1).
QUOTAS = {
    "single_page": 24, "multi_page": 15, "regional_table": 22,
    "out_of_scope_hazard": 10, "out_of_corpus": 8,
    "premise_injection": 17, "duplicate_region": 9,
}


def main() -> None:
    here = Path(__file__).parent
    questions = []
    for name in BATCHES:
        path = here / name
        if not path.exists():
            sys.exit(f"MISSING batch: {path} — assemble only after all batches are banked")
        data = json.loads(path.read_text(encoding="utf-8"))
        batch = data["questions"] if isinstance(data, dict) else data
        questions.extend(batch)
        print(f"{name}: {len(batch)} questions")

    test_set = EvalSet.model_validate({"questions": questions})

    counts = Counter(q.slice.value for q in test_set.questions)
    assert dict(counts) == QUOTAS, f"quota mismatch: {dict(counts)} != {QUOTAS}"
    assert len(test_set.questions) == 105

    ids = [q.id for q in test_set.questions]
    assert len(ids) == len(set(ids)), "duplicate ids within v2"
    dev = load_gold_set()
    assert not {q.id for q in dev.questions} & set(ids), "id collision with dev set"
    dev_texts = {" ".join(q.question.lower().split()) for q in dev.questions}
    dupes = [q.id for q in test_set.questions
             if " ".join(q.question.lower().split()) in dev_texts]
    assert not dupes, f"question text duplicates dev set: {dupes}"

    TEST_SET_PATH.write_text(
        json.dumps(test_set.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    TEST_SET_FROZEN_HASH_PATH.write_text(test_set.content_hash() + "\n", encoding="utf-8")
    print(f"\nFROZEN: {TEST_SET_PATH.name} ({len(test_set.questions)} q)")
    print(f"hash: {test_set.content_hash()}")


if __name__ == "__main__":
    main()
