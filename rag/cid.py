"""Cited AR6 Ch.12 climatic impact-driver (CID) projections, parsed without an LLM.

The forecast tells you what the next week looks like. This tells you which way
the *baseline* is moving for the AR6 reference region the location sits in, in
the IPCC's own words and with its own calibrated confidence phrase.

Deliberately deterministic: retrieval + regex, no generation. Every field is
either copied out of a retrieved chunk or derived from it by a rule you can
read here, so `ProjectedChange`'s structural citation rule is trivially met and
there is nothing for a model to hallucinate.

WHY THIS FILTERS ON PROSE, NOT ON TABLE CELLS (measured, 2026-09-02)
--------------------------------------------------------------------
Ch.12's CID summary tables (12.3-12.10) are SYMBOL tables: each region/CID cell
is a coloured glyph, and the legend at the foot of the page maps glyph -> "High
confidence of increase" and friends. PDF text extraction keeps the glyphs out,
so a row chunk extracts as, verbatim:

    "Sahara (SAH)     4   [Table 12.3 | Summary of confidence in direction ...]"

Counted over data/cache/chunks.json: 1064 Ch.12 chunks, 99 row-atomic ones, 61
of those carrying a `Table 12.N` caption -- and ZERO carrying a per-CID
direction or confidence for their region. The 12 chunks that do contain a
confidence phrase alongside a region label got it from the shared legend or
footnote block, which belongs to no single region or CID; parsing those would
attribute "High confidence of increase" to whichever region happened to end the
page. That is exactly the fabrication this module exists to prevent.

The readable form of the same assessment is Ch.12's own regional CID prose
(Sections 12.4.x, headed "Extreme heat:", "Heavy precipitation and pluvial
flood:", "Mean wind speed:") -- which is where the tables' contents are stated
in words, region by region, with the calibrated phrase attached. So the filter
is: Chapter 12 + this region + this hazard's CID vocabulary + a direction verb
+ a calibrated confidence phrase, all inside ONE sentence, with the table
legend and footnote boilerplate excluded by name.

This module never edits rag/retrieve.py: it calls the shared retriever with a
wider top_k and post-filters, so retrieval quality stays one measured thing.
"""
from __future__ import annotations

import logging
import re
from typing import Protocol

from agent.contracts import ChangeDirection, Citation, Hazard, ProjectedChange
from rag.chunk import Chunk
from tools.ar6_regions import AR6Region

_log = logging.getLogger(__name__)

CH12_SOURCE = "IPCC_AR6_WGI_Chapter12.pdf"
DEFAULT_TOP_K = 30  # wide: the CID sentence is often outside the answering top-8


class _Retriever(Protocol):
    def retrieve(self, question: str, top_k: int) -> list[Chunk]: ...


# --- the CID vocabulary, taken from Ch.12's own section headings -------------
# 12.4.x.1 "Extreme heat", 12.4.x.2 "Heavy precipitation and pluvial flood" /
# "River flood", 12.4.x.3 "Mean wind speed" / "Severe wind storm".
_CID_QUERY_TERMS = {
    Hazard.HEATWAVE: "extreme heat and heat stress",
    Hazard.EXTREME_PRECIP: "heavy precipitation and pluvial flood",
    Hazard.WIND: "mean wind speed and severe wind storm",
}
_CID_TERM = {
    Hazard.HEATWAVE: re.compile(
        r"\b(extreme heat|heat ?waves?|heat stress|hot extremes|heat extremes"
        r"|extreme temperatures)\b",
        re.I,
    ),
    Hazard.EXTREME_PRECIP: re.compile(
        r"\b(heavy precipitation|extreme precipitation|pluvial flood\w*|river flood\w*"
        r"|heavy rainfall|extreme rainfall)\b",
        re.I,
    ),
    Hazard.WIND: re.compile(
        r"\b(mean wind speeds?|wind speeds?|severe wind storms?|wind ?storms?"
        r"|extreme winds?)\b",
        re.I,
    ),
}

_INCREASE = re.compile(r"\b(increas\w+|intensif\w+|more frequent|rise|rising)\b", re.I)
_DECREASE = re.compile(r"\b(decreas\w+|declin\w+|reduc\w+|slowdown|less frequent)\b", re.I)
_NO_CHANGE = re.compile(r"\b(no (?:significant )?change|little change|no broad signal)\b", re.I)
# AR6's own "we cannot say which way" marker — an answer, not a parse failure.
_LOW_CONF_DIRECTION = re.compile(r"low confidence in (?:the )?direction of change", re.I)

# The calibrated-uncertainty phrases (AR6 WGI Box 1.1). Matched with tolerant
# spacing because PDF extraction emits "( high confidence )".
_CONFIDENCE = re.compile(r"\b(?:very high|high|medium|low)\s+confidence\b", re.I)

_WARMING_OR_PERIOD = re.compile(
    r"(?:\bGWLs?\s+of\s+\d(?:\.\d)?\s*°C"
    r"|\bfor all GWLs\b"
    r"|\b\d(?:\.\d)?\s*°C\s*(?:GWL|global warming|of global warming|warming level)"
    r"|\bglobal warming (?:level )?of \d(?:\.\d)?\s*°C"
    r"|\bSSP\d-\d\.\d|\bRCP\d\.\d"
    r"|\bmid-?century\b|\bend of the century\b|\bby 20\d\d\b|\b21st century\b)",
    re.I,
)

# Table 12.x legend + footnote furniture. These strings state every direction and
# every confidence level at once, for no region in particular — the single
# largest mis-attribution risk in Ch.12, so they are excluded by name.
_BOILERPLATE = (
    "Medium confidence of decrease Medium confidence of increase",
    "Already emerged in the historical period",
    "Not broadly relevant",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+")
_WHITESPACE = re.compile(r"\s+")

# Ch.12 routinely packs two assessments into one sentence, each with its own
# calibrated phrase and its own region list:
#   "high confidence that mean wind speeds will decrease in Mediterranean areas
#    AND medium confidence of such decreases in Northern Europe"
# A positional "nearest phrase wins" rule reads that backwards (measured: it
# gave MED "medium confidence"), because English puts the phrase before the
# claim in one construction and after it in the other. So the sentence is cut
# into clauses and only the clause the region sits in is read.
_CLAUSE_SPLIT = re.compile(
    r",\s+and\s+|;\s+|,\s+except\s+|,\s+but\s+"
    r"|\s+and\s+(?=(?:very high|high|medium|low)\s+confidence\b)",
    re.I,
)


def _region_pattern(region: AR6Region) -> re.Pattern[str]:
    """Match this region by acronym (standalone token) or by its full AR6 name.

    Hyphen/space tolerance mirrors rag.chunk._region_row_pattern, because the
    same PDF renders "North-Western North America" both ways.
    """
    words = [re.escape(w) for w in re.split(r"[\s\-/]+", region.name)]
    name = r"[\s\-/]+".join(words)
    acronym = rf"(?<![A-Za-z]){re.escape(region.acronym)}(?![A-Za-z])"
    return re.compile(rf"{acronym}|{name}", re.I)


def _direction(sentence: str) -> ChangeDirection:
    """Direction as the sentence states it — never inferred beyond the words."""
    if _LOW_CONF_DIRECTION.search(sentence):
        return ChangeDirection.UNKNOWN
    if _NO_CHANGE.search(sentence):
        return ChangeDirection.NO_CHANGE
    up, down = bool(_INCREASE.search(sentence)), bool(_DECREASE.search(sentence))
    if up and down:
        return ChangeDirection.MIXED
    if up:
        return ChangeDirection.INCREASE
    if down:
        return ChangeDirection.DECREASE
    return ChangeDirection.UNKNOWN


def _focus(sentence: str, region_pattern: re.Pattern[str]) -> str:
    """The clause of `sentence` that speaks about this region.

    Falls back to the whole sentence unless exactly one clause names the region
    and that clause carries a calibrated phrase — an ambiguous split must not
    silently narrow the evidence.
    """
    clauses = _CLAUSE_SPLIT.split(sentence)
    if len(clauses) > 1:
        naming = [c for c in clauses if region_pattern.search(c)]
        if len(naming) == 1 and _CONFIDENCE.search(naming[0]):
            return naming[0]
    return sentence


def _confidence_phrase(focus: str) -> str | None:
    """The calibrated phrase governing this clause, verbatim — or None if two
    different ones apply and no rule can say which. Silence beats a wrong
    confidence level on a climate-risk report."""
    matches = list(_CONFIDENCE.finditer(focus))
    distinct = {_WHITESPACE.sub(" ", m.group(0)).strip().lower() for m in matches}
    if len(distinct) != 1:
        return None
    return _WHITESPACE.sub(" ", matches[0].group(0)).strip()


def _warming_level_or_period(focus: str) -> str | None:
    """The first warming level / scenario / period named in this clause."""
    match = _WARMING_OR_PERIOD.search(focus)
    return _WHITESPACE.sub(" ", match.group(0)).strip() if match else None


def _qualifying_sentences(
    chunk: Chunk, hazard: Hazard, region_pattern: re.Pattern[str]
) -> list[str]:
    """Sentences in this chunk that state a CID projection for this region.

    All four signals must sit in the SAME sentence — the region, the hazard's
    CID term, a direction verb and a calibrated confidence phrase — so a
    confidence phrase can never be borrowed from a neighbouring claim.
    """
    if chunk.source != CH12_SOURCE:
        return []
    found = []
    for sentence in _SENTENCE_SPLIT.split(chunk.text):
        if any(marker in sentence for marker in _BOILERPLATE):
            continue
        if not region_pattern.search(sentence):
            continue
        if not _CID_TERM[hazard].search(sentence):
            continue
        if not _CONFIDENCE.search(sentence):
            continue
        if not (_INCREASE.search(sentence) or _DECREASE.search(sentence)
                or _NO_CHANGE.search(sentence) or _LOW_CONF_DIRECTION.search(sentence)):
            continue
        found.append(_WHITESPACE.sub(" ", sentence).strip())
    return found


def cid_question(region: AR6Region, hazard: Hazard) -> str:
    """The deterministic retrieval query for one region/hazard CID projection."""
    return (
        f"How is {_CID_QUERY_TERMS[hazard]} projected to change in "
        f"{region.name} ({region.acronym}) in the AR6 regional assessment?"
    )


def projected_change_for(
    region: AR6Region,
    hazard: Hazard,
    *,
    retriever: _Retriever,
    top_k: int = DEFAULT_TOP_K,
) -> ProjectedChange | None:
    """The cited AR6 Ch.12 CID projection for this region/hazard, or None.

    None means "the corpus did not yield a qualifying statement" — the caller
    must then say regional projections were not found, never invent one.
    """
    chunks = retriever.retrieve(cid_question(region, hazard), top_k=top_k)
    pattern = _region_pattern(region)

    # Retrieval order is the ranking; the only preference on top of it is for a
    # sentence the chunker did not cut in half (chunk windows can end mid-clause),
    # because a truncated quote reads as a claim the IPCC did not finish making.
    candidates = [
        sentence
        for chunk in chunks
        for sentence in _qualifying_sentences(chunk, hazard, pattern)
    ]
    statement = next(
        (s for s in candidates if s.endswith(".")),
        candidates[0] if candidates else None,
    )
    if statement is None:
        _log.info(
            "no AR6 Ch.12 CID statement for %s/%s in the top-%d chunks",
            region.acronym, hazard.value, top_k,
        )
        return None

    # Cite every retrieved chunk that carries this exact sentence (overlapping
    # chunks repeat it), deduped to one citation per page.
    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        if statement not in _WHITESPACE.sub(" ", chunk.text):
            continue
        key = (chunk.source, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(source=chunk.source, locator=f"p{chunk.page}", chunk_id=chunk.chunk_id)
        )

    # Direction, confidence and timing all come from the region's own clause;
    # `statement` stays the whole sentence so a reader can check the narrowing.
    focus = _focus(statement, pattern)
    return ProjectedChange(
        region_acronym=region.acronym,
        region_name=region.name,
        hazard=hazard,
        statement=statement,
        direction=_direction(focus),
        confidence_language=_confidence_phrase(focus),
        warming_level_or_period=_warming_level_or_period(focus),
        citations=citations,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
    )
