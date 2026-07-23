"""Is the answerer deterministic at temperature 0, per model?

Why this exists: two load-bearing claims in this project assume determinism.
  1. `rag.answer_cache` keys on (question, chunk texts, model) and serves the
     stored answer forever — only sound if the model is deterministic.
  2. Published eval numbers are reproducible — only true if a rerun of the same
     frozen questions yields the same answers.

Gemini 3.x models sample "thinking" tokens before emitting structured output,
so determinism is an empirical question, not an assumption. This probe answers
it with repeats of ONE fixed (question, chunks) pair per model.

Run:  uv run python -m evals.determinism_probe            # default models
      CRG_PROBE_MODELS=gemini-2.5-flash,... uv run python -m evals.determinism_probe
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODELS = ("gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.6-flash")
REPEATS = int(os.environ.get("CRG_PROBE_REPEATS", "5"))


def _outcome_key(ans) -> str:
    """Identity of an answer for determinism purposes: the parts a caller acts on."""
    payload = {
        "abstain": ans.abstain,
        "citations": sorted(ans.citations),
        "answer_sha": hashlib.sha256(ans.answer.encode("utf-8")).hexdigest()[:12],
    }
    return json.dumps(payload, sort_keys=True)


def main() -> None:
    from obs.log import configure

    configure()
    import rag.gemini_client as gc
    from agent.contracts import Hazard
    from agent.graph import _IPCC_TOP_K, _ipcc_question, _ipcc_retriever
    from rag.answer import AnswerError, answer_question

    # One fixed, representative in-scope question + its retrieved chunks.
    question = _ipcc_question(Hazard.EXTREME_PRECIP, "Rourkela, India")
    chunks = _ipcc_retriever().retrieve(question, top_k=_IPCC_TOP_K)
    print(f"question: {question}")
    print(f"chunks:   {[(c.source[-13:-4], c.page) for c in chunks]}")
    print(f"repeats:  {REPEATS} per model\n")

    models = tuple(
        m.strip() for m in os.environ.get("CRG_PROBE_MODELS", ",".join(DEFAULT_MODELS)).split(",") if m.strip()
    )
    results: dict[str, dict] = {}
    for model in models:
        gc.GENERATE_MODEL = model
        gc._reset_clients()
        outcomes: list[str] = []
        errors: list[str] = []
        for i in range(REPEATS):
            try:
                outcomes.append(_outcome_key(answer_question(question, chunks)))
            except (AnswerError, Exception) as exc:  # noqa: BLE001 — probe classifies, never crashes
                errors.append(f"{type(exc).__name__}: {exc}")
            print(f"  {model} [{i + 1}/{REPEATS}]", flush=True)
        distinct = sorted(set(outcomes))
        deterministic = len(distinct) <= 1 and not errors
        results[model] = {
            "repeats": REPEATS,
            "distinct_outcomes": len(distinct),
            "deterministic": deterministic,
            "outcomes": [json.loads(d) for d in distinct],
            "errors": errors,
        }
        verdict = "DETERMINISTIC" if deterministic else "NON-DETERMINISTIC"
        print(f"{model}: {verdict} — {len(distinct)} distinct outcome(s) over {REPEATS} runs")
        for d in distinct:
            o = json.loads(d)
            print(f"    abstain={o['abstain']} citations={len(o['citations'])} answer={o['answer_sha']}")
        if errors:
            print(f"    errors: {errors}")
        print()

    out_dir = Path("evals/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"determinism-{datetime.now(timezone.utc):%Y-%m-%d}.json"
    out.write_text(
        json.dumps(
            {
                "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "question": question,
                "chunk_ids": [c.chunk_id for c in chunks],
                "models": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"artifact written: {out}")


if __name__ == "__main__":
    main()
