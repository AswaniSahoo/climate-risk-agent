"""Run the frozen gold set against retrievers → per-slice recall@k, MRR, Wilson CIs.

Ablation across three modes — bm25 (lexical), dense (gemini-embedding-2 cosine),
hybrid (RRF fusion) — so every layer must justify itself with a delta on the
same frozen questions.

`--reranker` / `--rewrite` add a FOURTH column to the same run: a second-stage
reranker over a wider RRF pool, and/or a neutral query rewrite. The baseline
three are always recomputed alongside it, so "added latency" and "added cost"
are within-run deltas against the hybrid column of the same process, not
against a number from a different day.

The question TEXT is the query — never the supporting quote (that would leak the
answer's wording into retrieval and inflate every number). Headline recall is
computed over ANSWER-behavior items; REFUSE-with-gold slices are diagnostics.

Embeddings go through a disk cache (data/cache/embeddings, git-ignored): the
first run needs GEMINI_API_KEY + network, after that the eval is offline and
deterministic. Without a key or cache, the bm25 column still runs (a reviewer
can reproduce the lexical numbers with zero credentials).

Run:  uv run python -m evals.run_retrieval_eval
      uv run python -m evals.run_retrieval_eval --reranker minilm
      uv run python -m evals.run_retrieval_eval --reranker gemini --rewrite on
"""
from __future__ import annotations

import argparse
import os as _os
import time
from collections.abc import Callable
from pathlib import Path

from evals.gold_set import load_gold_set, load_test_set
from evals.metrics import mrr, recall_at_k, unique_pages, wilson_ci
from evals.schema import EvalQuestion, ExpectedBehavior, Slice
from rag.bm25 import BM25Index
from rag.chunk import Chunk, chunk_pages
from rag.dense import DenseIndex
from rag.embed import DiskVectorCache, EmbeddingError, cached_embed_texts
from rag.hybrid import rrf_fuse
from rag.parse import extract_pages

CORPUS_DIR = Path("data/ipcc")
CORPUS_FILES = [
    "IPCC_AR6_WGI_SPM.pdf",
    "IPCC_AR6_WGI_Chapter11.pdf",
    "IPCC_AR6_WGI_Chapter12.pdf",
]
CACHE_DIR = Path("data/cache/embeddings")
K_VALUES = (3, 5, 10)
TOP_CHUNKS = 50  # enough chunk hits to yield >=10 unique pages after dedupe
# The reranked arm returns its whole reordered pool (rag.retrieve._RERANK_POOL),
# so R@3/R@5 are measured over the RERANKED order rather than over a truncated
# top-5 that could not be scored at k=10 at all.
RERANK_DEPTH = 30
# Same free-tier pacing knob the e2e eval uses (~5 RPM measured); a paid Vertex
# key can drop it: EVAL_LLM_PAUSE_S=0.5
_LLM_PAUSE_S = float(_os.environ.get("EVAL_LLM_PAUSE_S", "13.0"))

Retriever = Callable[[str], list[Chunk]]


def build_chunks() -> list[Chunk]:
    pages = []
    for name in CORPUS_FILES:
        pages.extend(extract_pages(CORPUS_DIR / name))
    chunks = chunk_pages(pages)
    print(f"corpus: {len(pages)} pages -> {len(chunks)} chunks")
    return chunks


def arm_label(reranker: str, rewrite: str) -> str:
    """Name of the extra column, or "" when the run is the plain ablation."""
    if reranker == "none" and rewrite == "off":
        return ""
    parts = ["hybrid"]
    if reranker != "none":
        parts.append(reranker)
    if rewrite == "on":
        parts.append("rewrite")
    return "+".join(parts)


def build_retrievers(chunks: list[Chunk], *, reranker: str = "none", rewrite: str = "off",
                     ) -> dict[str, Retriever]:
    bm25 = BM25Index(chunks)
    retrievers: dict[str, Retriever] = {
        "bm25": lambda q: [c for c, _ in bm25.query(q, top_k=TOP_CHUNKS)]
    }

    try:
        import numpy as np

        cache = DiskVectorCache(CACHE_DIR)
        matrix = np.asarray(
            cached_embed_texts([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT", cache=cache)
        )
        dense = DenseIndex(chunks, matrix)

        def dense_retrieve(question: str) -> list[Chunk]:
            [vector] = cached_embed_texts([question], task_type="RETRIEVAL_QUERY", cache=cache)
            return [c for c, _ in dense.query(vector, top_k=TOP_CHUNKS)]

        retrievers["dense"] = dense_retrieve
        retrievers["hybrid"] = lambda q: rrf_fuse(
            [retrievers["bm25"](q), dense_retrieve(q)], top_k=TOP_CHUNKS
        )

        label = arm_label(reranker, rewrite)
        if label:
            # The arm runs the PRODUCTION retriever (rag/retrieve.py) with the
            # optional stages switched on — measuring the wiring users would
            # get, not a parallel implementation that could drift from it.
            from rag.rerank import build_reranker
            from rag.retrieve import HybridRetriever
            from rag.rewrite import neutral_query

            arm = HybridRetriever(
                chunks, doc_matrix=matrix, cache=cache,
                reranker=build_reranker(reranker),
                rewriter=neutral_query if rewrite == "on" else None,
            )
            retrievers[label] = lambda q: arm.retrieve(q, top_k=RERANK_DEPTH)
    except EmbeddingError as exc:
        print(f"(dense/hybrid skipped: {exc} — bm25 column is still fully reproducible)")
    return retrievers


def report(label: str, items: list[tuple[EvalQuestion, list[tuple[str, int]]]]) -> dict | None:
    n = len(items)
    if not n:
        return None
    parts = []
    row: dict = {"label": label, "n": n, "recall": {}}
    for k in K_VALUES:
        hits = sum(recall_at_k(pages, q.gold_pages, k) for q, pages in items)
        lo, hi = wilson_ci(hits, n)
        parts.append(f"R@{k} {hits/n:5.0%} [{lo:.0%}-{hi:.0%}]")
        row["recall"][f"@{k}"] = {"rate": round(hits / n, 4),
                                  "wilson95": [round(lo, 4), round(hi, 4)]}
    mean_mrr = sum(mrr(pages, q.gold_pages) for q, pages in items) / n
    row["mrr"] = round(mean_mrr, 4)
    print(f"{label:22s} n={n:2d}  " + "  ".join(parts) + f"  MRR {mean_mrr:.2f}")
    return row


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


def score_questions(retrieve: Retriever, questions: list[EvalQuestion], *, paced: bool,
                    ) -> tuple[list[tuple[EvalQuestion, list]], dict]:
    """Retrieve for every question, measuring wall time and model spend per question.

    Cost comes from the telemetry span around the call, so it is whatever the
    single Gemini seam actually recorded — an arm cannot under-report by
    forgetting to add up its own calls.
    """
    from obs.telemetry import Span, estimate_cost_usd

    scored, latencies, costs = [], [], []
    for i, q in enumerate(questions):
        with Span("retrieve") as span:
            pages = unique_pages(retrieve(q.question))
        scored.append((q, pages))
        latencies.append(span.wall_ms)
        costs.append(estimate_cost_usd(span.events))
        if paced and i < len(questions) - 1:
            time.sleep(_LLM_PAUSE_S)
    stats = {
        "p50_ms": round(_percentile(latencies, 0.50), 1),
        "p95_ms": round(_percentile(latencies, 0.95), 1),
        "est_cost_usd_per_question": round(sum(costs) / len(costs), 6) if costs else 0.0,
        "est_cost_usd_total": round(sum(costs), 6),
    }
    return scored, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reranker", choices=("none", "gemini", "minilm"), default="none",
                        help="second-stage reranker over a wider RRF pool (default: none)")
    parser.add_argument("--rewrite", choices=("on", "off"), default="off",
                        help="neutral query rewrite before retrieval (default: off)")
    return parser.parse_args()


def main() -> None:
    from obs.log import configure

    configure()  # runner owns logging config
    args = parse_args()
    # EVAL_SET=test = the held-out v2 set: release gates ONLY (see DEPLOY.md).
    eval_set = _os.environ.get("EVAL_SET", "dev")
    gold = load_test_set() if eval_set == "test" else load_gold_set()
    label = arm_label(args.reranker, args.rewrite)
    print(f"eval set: {eval_set} ({len(gold.questions)} questions)")
    print(f"arm: reranker={args.reranker} rewrite={args.rewrite}"
          + (f" -> extra column '{label}'" if label else " (baseline ablation only)"))
    chunks = build_chunks()
    questions = [q for q in gold.questions if q.gold_pages]

    # Only the arm can make per-question model calls, so only it needs pacing.
    llm_arm = args.reranker == "gemini" or args.rewrite == "on"

    artifact_rows: dict[str, list[dict]] = {}
    cost_rows: dict[str, dict] = {}
    for name, retrieve in build_retrievers(
        chunks, reranker=args.reranker, rewrite=args.rewrite
    ).items():
        scored, stats = score_questions(
            retrieve, questions, paced=llm_arm and name == label
        )
        cost_rows[name] = stats
        print(f"\n===== {name} =====")
        rows = []
        for s in Slice:
            rows.append(report(s.value, [(q, p) for q, p in scored if q.slice is s]))
        rows.append(report(
            "HEADLINE (answer)",
            [(q, p) for q, p in scored if q.expected_behavior is ExpectedBehavior.ANSWER],
        ))
        rows.append(report(
            "diagnostic (refuse)",
            [(q, p) for q, p in scored if q.expected_behavior is ExpectedBehavior.REFUSE],
        ))
        artifact_rows[name] = [r for r in rows if r]
        print(f"{'cost/latency':22s}      p50 {stats['p50_ms']:8.1f} ms  "
              f"p95 {stats['p95_ms']:8.1f} ms  "
              f"est ${stats['est_cost_usd_per_question']:.6f}/question")

    # "Added" cost is always measured against the hybrid column of THIS run:
    # same process, same machine, same warm caches.
    base = cost_rows.get("hybrid")
    if label and base and label in cost_rows:
        arm = cost_rows[label]
        cost_rows[label]["added_p50_ms_vs_hybrid"] = round(arm["p50_ms"] - base["p50_ms"], 1)
        cost_rows[label]["added_cost_usd_per_question_vs_hybrid"] = round(
            arm["est_cost_usd_per_question"] - base["est_cost_usd_per_question"], 6
        )
        print(f"\nADDED by {label}: p50 +{cost_rows[label]['added_p50_ms_vs_hybrid']:.1f} ms/question"
              f"  est +${cost_rows[label]['added_cost_usd_per_question_vs_hybrid']:.6f}/question")

    # Committed artifact: the ablation as a verifiable file, not README prose.
    import json
    from datetime import datetime, timezone

    from rag.gemini_client import EMBED_MODEL, GENERATE_MODEL

    out_dir = Path("evals/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-test" if eval_set == "test" else ""
    # Default run keeps the historical filename so existing artifacts stay
    # comparable; an arm gets its own file so a bake-off can't overwrite itself.
    arm_slug = f"-{args.reranker}-rewrite{args.rewrite}" if label else ""
    out_path = out_dir / f"retrieval{suffix}{arm_slug}-{datetime.now(timezone.utc):%Y-%m-%d}.json"
    out_path.write_text(json.dumps({
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eval_set": eval_set,
        "arm": {"reranker": args.reranker, "rewrite": args.rewrite, "column": label or None,
                "rerank_depth": RERANK_DEPTH if label else None},
        "generate_model": GENERATE_MODEL,
        "embed_model": EMBED_MODEL,
        "gold_set_sha256_file": (
            "evals/gold_set_v2.sha256" if eval_set == "test" else "evals/gold_set.sha256"
        ),
        "retrievers": artifact_rows,
        "cost_latency": cost_rows,
    }, indent=2), encoding="utf-8")
    print(f"\nartifact written: {out_path} (commit it — release-gate evidence)")


if __name__ == "__main__":
    main()
