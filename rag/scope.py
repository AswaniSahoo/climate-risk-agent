"""Deterministic scope guard: refuse-before-retrieve for unsupported hazards.

Measured motive: in the live smoke the LLM answered a tropical-cyclone question
despite an explicit prompt rule to refuse — a prompt-level guard is a
suggestion. This guard is code: it runs BEFORE any LLM call, cannot be
prompt-injected, and is unit-tested.

Policy rule (not tuned to the eval set): a question is out of scope when it
mentions an unsupported hazard AND no supported hazard — compound questions
that involve a supported hazard (e.g. concurrent heatwaves and droughts) stay
in scope. v1 limitation, stated honestly: matching is lexical.

v2 (2026-09) adds a SECOND stage behind the CRG_SCOPE_STAGE2 flag for the one
bucket this lexical stage is blind to: questions that name no hazard vocabulary
at all. `scope_verdict()` is the combined entry point; stage 1 still runs first
and its verdict is final, so the deterministic, injection-proof guarantee above
is unchanged. See rag/scope_semantic.py.
"""
from __future__ import annotations

import os
import re
from collections import OrderedDict
from dataclasses import dataclass

# A marine heatwave is an oceanic extreme-heat EVENT — a hazard, not background
# earth-system science — but the phrase contains "heatwave" (a supported term),
# so it must be tested BEFORE the supported-hazard check or it slips through
# (held-out eval v2 caught exactly this false answer). Cryosphere topics
# (glaciers, sea ice, snowlines) are deliberately NOT here: they are answerable
# background science, consistent with the dev set's glacier-commitment item.
_MARINE_HEATWAVE = re.compile(r"\bmarine heat ?waves?\b|\bocean heat ?waves?\b", re.IGNORECASE)

# term-pattern -> canonical hazard name reported in the refusal
_UNSUPPORTED: dict[str, str] = {
    r"\bdroughts?\b|\baridity\b": "drought",
    r"\btropical[ -]cyclones?\b|\bhurricanes?\b|\btyphoons?\b|\bcyclones?\b": "tropical cyclone",
    r"\bsea[ -]levels?\b|\bcoastal flood\w*\b|\bstorm surges?\b": "coastal flooding / sea level",
    r"\bwild ?fires?\b|\bfire weather\b|\bfire seasons?\b|\bburn\w* area\b": "wildfire / fire weather",
    r"\bhail\b|\btornado\w*\b": "severe convective storm",
    r"\blandslides?\b|\bavalanches?\b": "landslide / avalanche",
}

_SUPPORTED = re.compile(
    r"\bheat ?waves?\b|\bhot extremes?\b|\bheat extremes?\b|\btemperature extremes?\b"
    r"|\bprecipitation\b|\brainfall\b|\brain\b|\bmonsoon\b"
    r"|\bwinds?\b|\bgusts?\b",
    re.IGNORECASE,
)


def out_of_scope_hazard(question: str) -> str | None:
    """Name the unsupported hazard a question is about, or None if in scope.

    A supported hazard anywhere in the question keeps it in scope (compound
    events involving our hazards are our business) — except a marine heatwave,
    which is an oceanic hazard we do not assess, checked first so its embedded
    "heatwave" cannot wave it through.
    """
    if _MARINE_HEATWAVE.search(question):
        return "marine heatwave"
    if _SUPPORTED.search(question):
        return None
    for pattern, name in _UNSUPPORTED.items():
        if re.search(pattern, question, re.IGNORECASE):
            return name
    return None


def has_supported_signal(question: str) -> bool:
    """Did the lexical stage SEE a supported hazard? (Not the same as "in scope".)

    Split out because it is the routing question for stage 2: a question with no
    supported signal and no unsupported signal is the bucket stage 1 is blind to.
    """
    return bool(_SUPPORTED.search(question))


# --- combined verdict (stage 1 + optional stage 2) --------------------------

STAGE2_MODES = ("off", "embed", "llm")


@dataclass(frozen=True)
class ScopeDecision:
    """What the guard concluded, and which stage concluded it.

    `out_of_scope` is the lexical stage's contract, unchanged: the named
    unsupported hazard, or None. `hazard_hint` is new and additive — a supported
    hazard recovered from a paraphrase the regex misses, which callers may use
    to route instead of refusing.
    """

    out_of_scope: str | None = None
    hazard_hint: str | None = None
    stage: str = "lexical"
    detail: str = ""


def stage2_mode() -> str:
    """Resolve CRG_SCOPE_STAGE2 (off | embed | llm). Unknown value -> off, loudly."""
    mode = os.environ.get("CRG_SCOPE_STAGE2", "off").strip().lower() or "off"
    if mode not in STAGE2_MODES:
        import logging

        logging.getLogger(__name__).warning(
            "CRG_SCOPE_STAGE2=%r is not one of %s — stage 2 stays OFF", mode, STAGE2_MODES
        )
        return "off"
    return mode


# One stage-2 result per (question, mode) per process. The eval runner and
# answer_with_guard both ask for the same verdict on the same question; without
# this the LLM arm would pay twice for one decision.
#
# BOUNDED, and keyed by user input: an unbounded dict here is a slow memory leak
# on any long-lived server (every distinct question ever asked, kept forever)
# and an easy one to trigger on purpose. 512 entries is far more than one eval
# run or one user session repeats, and the eviction cost of a wrong guess is one
# extra stage-2 call, not a wrong answer.
STAGE2_CACHE_MAXSIZE = 512
_stage2_cache: "OrderedDict[tuple[str, str], ScopeDecision]" = OrderedDict()


def clear_stage2_cache() -> None:
    _stage2_cache.clear()


def _remember_stage2(key: tuple[str, str], decision: "ScopeDecision") -> None:
    """Store one verdict, evicting the least recently used entry past the cap."""
    _stage2_cache[key] = decision
    _stage2_cache.move_to_end(key)
    while len(_stage2_cache) > STAGE2_CACHE_MAXSIZE:
        _stage2_cache.popitem(last=False)


def scope_verdict(question: str) -> ScopeDecision:
    """Stage 1, then stage 2 only where stage 1 is blind.

    Order is the safety argument, not an implementation detail:
    1. a lexical out-of-scope verdict STANDS — deterministic and unforgeable;
    2. a lexical supported-hazard signal means in scope — no model call needed;
    3. only the silent bucket (neither signal) reaches stage 2, and stage 2 may
       return no verdict, in which case behaviour is exactly what it is today.

    With CRG_SCOPE_STAGE2=off (the default) this is `out_of_scope_hazard` plus a
    wrapper — no extra call, no behaviour change.
    """
    lexical = out_of_scope_hazard(question)
    if lexical is not None:
        return ScopeDecision(out_of_scope=lexical, stage="lexical")
    if has_supported_signal(question):
        return ScopeDecision(stage="lexical")

    mode = stage2_mode()
    if mode == "off":
        return ScopeDecision(stage="lexical")

    key = (question, mode)
    cached = _stage2_cache.get(key)
    if cached is not None:
        _stage2_cache.move_to_end(key)  # LRU: a repeat keeps it alive
        return cached

    from rag import scope_semantic

    verdict = (
        scope_semantic.semantic_scope(question)
        if mode == "embed"
        else scope_semantic.llm_scope(question)
    )
    decision = ScopeDecision(
        out_of_scope=verdict.out_of_scope,
        hazard_hint=verdict.hazard_hint,
        stage=mode,
        detail=verdict.detail,
    )
    _remember_stage2(key, decision)
    return decision
