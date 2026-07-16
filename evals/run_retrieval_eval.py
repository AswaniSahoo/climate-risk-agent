"""Run the frozen gold set against retrievers → per-slice recall@k, MRR, Wilson CIs.

Ablation across three modes — bm25 (lexical), dense (gemini-embedding-2 cosine),
hybrid (RRF fusion) — so every layer must justify itself with a delta on the
same frozen questions.

The question TEXT is the query — never the supporting quote (that would leak the
answer's wording into retrieval and inflate every number). Headline recall is
computed over ANSWER-behavior items; REFUSE-with-gold slices are diagnostics.

Embeddings go through a disk cache (data/cache/embeddings, git-ignored): the
first run needs GEMINI_API_KEY + network, after that the eval is offline and
deterministic. Without a key or cache, the bm25 column still runs (a reviewer
can reproduce the lexical numbers with zero credentials).

Run:  uv run python -m evals.run_retrieval_eval
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from evals.gold_set import load_gold_set
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

Retriever = Callable[[str], list[Chunk]]


def build_chunks() -> list[Chunk]:
    pages = []
    for name in CORPUS_FILES:
        pages.extend(extract_pages(CORPUS_DIR / name))
    chunks = chunk_pages(pages)
    print(f"corpus: {len(pages)} pages -> {len(chunks)} chunks")
    return chunks


def build_retrievers(chunks: list[Chunk]) -> dict[str, Retriever]:
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


def main() -> None:
    from obs.log import configure

    configure()  # runner owns logging config
    gold = load_gold_set()
    chunks = build_chunks()
    questions = [q for q in gold.questions if q.gold_pages]

    artifact_rows: dict[str, list[dict]] = {}
    for name, retrieve in build_retrievers(chunks).items():
        scored = [(q, unique_pages(retrieve(q.question))) for q in questions]
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

    # Committed artifact: the ablation as a verifiable file, not README prose.
    import json
    from datetime import datetime, timezone

    out_dir = Path("evals/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"retrieval-{datetime.now(timezone.utc):%Y-%m-%d}.json"
    out_path.write_text(json.dumps({
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gold_set_sha256_file": "evals/gold_set.sha256",
        "retrievers": artifact_rows,
    }, indent=2), encoding="utf-8")
    print(f"\nartifact written: {out_path} (commit it — release-gate evidence)")


if __name__ == "__main__":
    main()
