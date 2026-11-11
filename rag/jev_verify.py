"""Citation support check with Jev: does each cited IPCC page back the statement?

What a report already guarantees, structurally: every citation is a chunk that
retrieval actually returned (`CitedAnswer` in rag/answer.py), so a fabricated
reference cannot be emitted. What that cannot see is a real, retrieved page
that does not say what the sentence claims. LIMITATIONS.md names the gap
("claim-level entailment ... is not yet automated"), and the dev-set citation
validity of 41/48 measures it at page level.

For every (sentence, cited passage) pair this asks Jev TypeSafe's own
citation-check question (docs "cookbooks/citation_check", jev-1.14.0 numbers
dated 2026-11-03): does the passage support the claim, contradict it, or say
nothing about it. One request per pair, because answers move with everything
else in the state (TypeSafe's docs on large state, and an independent
code-review calibration study), so each pair is judged on its own.

Signal, not authority. The result only annotates the report: a strength per
citation and the sentences no cited page supports. It never removes or adds a
citation, never touches the risk level or the confidence, and on any failure
the report ships exactly as it would with Jev off, marked `unavailable`.

The passages are public IPCC text. The statement is Gemini's answer, which can
name the queried location, so in verify mode the location reaches TypeSafe too
(LIMITATIONS.md, "Jev").
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Literal

from agent.contracts import CitationEvidence, EvidenceCheck
from rag.chunk import Chunk
from rag.jev_client import ChoiceAnswer, JevClient, JevError, get_client, verify_policy

_log = logging.getLogger(__name__)

# Fixed before any measurement: the cookbook's 0.8 auto-accept line for
# "strong" and "contradicted", and the docs' coin-flip line for "weak".
SUPPORT_STRONG = 0.8
SUPPORT_MODERATE = 0.5
CONTRADICTS = 0.8

# A report's IPCC statement is a few sentences citing a few of the 8 chunks
# retrieval returns (agent/graph.py _IPCC_TOP_K), so every cited chunk can be
# checked. The caps bound one report at 32 requests, all sent at once, so the
# check costs about one round trip; whatever a cap leaves out is reported as
# unchecked and the check as partial, never as unsupported.
MAX_STATEMENTS = 4
MAX_PASSAGES = 8
STATEMENT_CHARS = 600
PASSAGE_CHARS = 6000
MAX_WORKERS = MAX_STATEMENTS * MAX_PASSAGES
# How long past one call's budget the whole check waits before giving up on
# the pairs still running: each call stops itself at its own budget.
DEADLINE_GRACE_S = 0.5

# Verbatim from TypeSafe's citation-check cookbook, with its "section" called a
# passage here because that is what this corpus is cut into.
RELATION_QUESTION = {
    "relation": {
        "type": "choice",
        "instructions": "How does `passage` relate to `claim`?",
        "criteria": {
            "supports": "The passage states the claim or directly implies that it is true",
            "contradicts": "The passage states the opposite of the claim or implies it is false",
            "says_nothing": "The passage does not address what the claim asserts, either way",
        },
    }
}

# A sentence ends at . ! or ? followed by a space and a capital, a quote or a
# bracket. Digits are excluded on purpose: "1.5 °C" has no space after its
# point, and "Fig. 11.3" must not split.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(\[])")


def _all_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text or "").strip()
    sentences = [s.strip() for s in _SENTENCE_END.split(flat) if s.strip()]
    return [s[:STATEMENT_CHARS] for s in sentences if len(s) >= 12]


def split_statements(text: str) -> list[str]:
    """The statement as separately checkable sentences, capped in count and length."""
    return _all_sentences(text)[:MAX_STATEMENTS]


Strength = Literal["strong", "moderate", "weak", "contradicted"]


def strength(supports: float, contradicts: float) -> Strength:
    """The label a reader sees for one citation."""
    if contradicts >= CONTRADICTS and contradicts > supports:
        return "contradicted"
    if supports >= SUPPORT_STRONG:
        return "strong"
    if supports >= SUPPORT_MODERATE:
        return "moderate"
    return "weak"


def verify_answer(
    statement: str, cited: Sequence[Chunk], *, client: JevClient | None = None
) -> EvidenceCheck:
    """Evidence strength for each cited page of `statement`. Never raises."""
    try:
        return _verify(statement, cited, client)
    except Exception as exc:  # noqa: BLE001 (a signal must degrade, never take a report down)
        _log.exception("citation support check failed unexpectedly; the report ships without it")
        return EvidenceCheck(status="unavailable", detail=f"unavailable: {type(exc).__name__}")


def _verify(statement: str, cited: Sequence[Chunk], client: JevClient | None) -> EvidenceCheck:
    every_sentence = _all_sentences(statement)
    sentences = every_sentence[:MAX_STATEMENTS]
    every_passage: list[Chunk] = []
    for chunk in cited:
        if chunk.chunk_id not in {p.chunk_id for p in every_passage}:
            every_passage.append(chunk)
    passages = every_passage[:MAX_PASSAGES]
    if not sentences or not passages:
        return EvidenceCheck(status="unavailable", detail="nothing to check")
    try:
        jev = client or get_client()
    except JevError as exc:
        return EvidenceCheck(status="unavailable", detail=f"unavailable: {exc.reason}")

    pairs = [(i, j) for i in range(len(sentences)) for j in range(len(passages))]
    answers: dict[tuple[int, int], ChoiceAnswer | None] = {}
    models: set[str] = set()
    failures: list[str] = []
    policy = verify_policy()

    def ask(i: int, j: int):
        state = {"claim": sentences[i], "passage": passages[j].text[:PASSAGE_CHARS]}
        return jev.ask(state, RELATION_QUESTION, op="verify", policy=policy)

    # Every pair is sent at once, and the whole check gets the same budget as
    # one call: a slow Jev delays the research step by that much at most. A pair
    # still running at the end is abandoned (it stops at its own budget) and
    # counted as unchecked; nothing here waits for it.
    pool = ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(pairs)))
    futures = {pool.submit(ask, i, j): (i, j) for i, j in pairs}
    try:
        for future in as_completed(futures, timeout=policy.budget_s + DEADLINE_GRACE_S):
            pair = futures[future]
            try:
                result = future.result()
                answers[pair] = result.choice("relation")
                models.add(result.model)
            except Exception as exc:  # noqa: BLE001 (one pair failing is that pair, unchecked)
                answers[pair] = None
                failures.append(exc.reason if isinstance(exc, JevError) else type(exc).__name__)
    except FuturesTimeout:
        for pair in pairs:
            if pair not in answers:
                answers[pair] = None
                failures.append("the check's time budget ran out")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    # One entry per cited page, in the order the statement cited them, which is
    # the order agent/graph.py builds its page-level Citation list in. A page
    # past the passage cap is still listed, as unchecked.
    pages: dict[tuple[str, int], list[int]] = {}
    for j, chunk in enumerate(every_passage):
        pages.setdefault((chunk.source, chunk.page), []).append(j)

    evidence: list[CitationEvidence] = []
    for (source, page), indexes in pages.items():
        got = [
            (j, answer)
            for j in indexes
            for i in range(len(sentences))
            if (answer := answers.get((i, j))) is not None
        ]
        if not got:
            evidence.append(CitationEvidence(
                source=source, locator=f"p{page}", chunk_id=every_passage[indexes[0]].chunk_id,
                supports=0.0, contradicts=0.0, strength="unchecked",
            ))
            continue
        best_j, best = max(got, key=lambda pair: pair[1].p("supports"))
        supports = best.p("supports")
        contradicts = max(answer.p("contradicts") for _, answer in got)
        evidence.append(CitationEvidence(
            source=source, locator=f"p{page}", chunk_id=every_passage[best_j].chunk_id,
            supports=round(supports, 4), contradicts=round(contradicts, 4),
            strength=strength(supports, contradicts),
        ))

    # "No cited page supports this sentence" is only said when every cited
    # page was actually read against it.
    unsupported: list[str] = []
    if len(passages) == len(every_passage):
        for i, sentence in enumerate(sentences):
            row = [answers.get((i, j)) for j in range(len(passages))]
            if all(a is not None for a in row) and max(
                a.p("supports") for a in row if a is not None
            ) < SUPPORT_MODERATE:
                unsupported.append(sentence)

    capped = len(sentences) < len(every_sentence) or len(passages) < len(every_passage)
    status: Literal["ok", "partial", "unavailable"]
    if failures and len(failures) == len(pairs):
        status = "unavailable"
    elif failures or capped:
        status = "partial"
    else:
        status = "ok"
    detail = f"{len(pairs) - len(failures)}/{len(pairs)} (sentence, passage) pairs checked"
    if capped:
        detail += (f"; checked {len(sentences)} of {len(every_sentence)} sentences and "
                   f"{len(passages)} of {len(every_passage)} passages")
    if failures:
        detail += f"; failures: {sorted(set(failures))[0]}"
        _log.warning("citation support check %s (%s)", status, detail)
    return EvidenceCheck(
        status=status,
        model=", ".join(sorted(models)) or None,
        detail=detail,
        citations=evidence,
        unsupported_statements=unsupported,
    )
