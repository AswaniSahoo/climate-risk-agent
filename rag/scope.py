"""Deterministic scope guard: refuse-before-retrieve for unsupported hazards.

Measured motive: in the live smoke the LLM answered a tropical-cyclone question
despite an explicit prompt rule to refuse — a prompt-level guard is a
suggestion. This guard is code: it runs BEFORE any LLM call, cannot be
prompt-injected, and is unit-tested.

Policy rule (not tuned to the eval set): a question is out of scope when it
mentions an unsupported hazard AND no supported hazard — compound questions
that involve a supported hazard (e.g. concurrent heatwaves and droughts) stay
in scope. v1 limitation, stated honestly: matching is lexical.
"""
from __future__ import annotations

import re

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
    events involving our hazards are our business).
    """
    if _SUPPORTED.search(question):
        return None
    for pattern, name in _UNSUPPORTED.items():
        if re.search(pattern, question, re.IGNORECASE):
            return name
    return None
