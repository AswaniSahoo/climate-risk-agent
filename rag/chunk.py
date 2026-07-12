"""Split page text into retrievable chunks — without losing the page number.

Four invariants, each pinned by a test:
1. every chunk keeps its `source` + `page`  -> page-level citations stay possible
2. chunks never split a word in half        -> cuts land on whitespace
3. consecutive chunks overlap               -> a sentence across a boundary is
                                               still retrievable from one chunk
4. regional assessment tables are ROW-ATOMIC -> one region per chunk, header
   re-prepended on overflow. Measured motive: naive windows mix several regions'
   cells, and the duplicate_region eval slice scored 0% recall because of it.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

from rag.parse import PageText

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 150

# Official AR6 WG1 reference land regions (Iturbide et al., 2020). Anchoring row
# detection to this fixed vocabulary means "(CMIP6)" / "(RCM)" can never trigger
# a split — only a real region row can.
_AR6_REGIONS: dict[str, str] = {
    "GIC": "Greenland/Iceland", "NWN": "North-Western North America",
    "NEN": "North-Eastern North America", "WNA": "Western North America",
    "CNA": "Central North America", "ENA": "Eastern North America",
    "NCA": "Northern Central America", "SCA": "Southern Central America",
    "CAR": "Caribbean", "NWS": "North-Western South America",
    "NSA": "Northern South America", "NES": "North-Eastern South America",
    "SAM": "South American Monsoon", "SWS": "South-Western South America",
    "SES": "South-Eastern South America", "SSA": "Southern South America",
    "NEU": "Northern Europe", "WCE": "Western and Central Europe",
    "EEU": "Eastern Europe", "MED": "Mediterranean", "SAH": "Sahara",
    "WAF": "Western Africa", "CAF": "Central Africa",
    "NEAF": "North Eastern Africa", "SEAF": "South Eastern Africa",
    "WSAF": "West Southern Africa", "ESAF": "East Southern Africa",
    "MDG": "Madagascar", "RAR": "Russian Arctic", "WSB": "West Siberia",
    "ESB": "East Siberia", "RFE": "Russian Far East",
    "WCA": "West Central Asia", "ECA": "East Central Asia",
    "TIB": "Tibetan Plateau", "EAS": "East Asia", "ARP": "Arabian Peninsula",
    "SAS": "South Asia", "SEA": "South East Asia",
    "NAU": "Northern Australia", "CAU": "Central Australia",
    "EAU": "Eastern Australia", "SAU": "Southern Australia",
    "NZ": "New Zealand",
}


def _region_row_pattern() -> re.Pattern[str]:
    """Compile 'Region Name (ACR)' alternation, tolerant of hyphen/space variants."""
    alternatives = []
    for acronym, name in _AR6_REGIONS.items():
        words = [re.escape(w) for w in re.split(r"[\s\-/]+", name)]
        alternatives.append(r"[\s\-/]+".join(words) + r"\s*\(" + acronym + r"\)")
    return re.compile("|".join(alternatives))


_REGION_ROW = _region_row_pattern()

# Table caption, e.g. "Table 11.7 | Observed trends ... at 1.5°C, 2°C and 4°C ...".
# The caption is the ONLY place the GWL column labels appear — row cells all say
# "compared to the 1°C warming level" — so every row chunk must carry it.
# Measured twice: RT-01 false refusal + claim-judge "at 4°C" flags (RT-10/DR-02).
_TABLE_CAPTION = re.compile(r"Table \d+\.\d+\s*\|\s*[^|]{0,160}")


def _find_caption(text: str) -> str | None:
    """The page's table caption, word-safe trimmed, or None."""
    match = _TABLE_CAPTION.search(text)
    if not match:
        return None
    caption = match.group(0).strip()
    if match.end() < len(text) and " " in caption:  # trim a cut-off last word
        caption = caption.rsplit(" ", 1)[0]
    return caption


class Chunk(BaseModel):
    """One retrievable unit of an IPCC document, traceable to a page."""

    chunk_id: str
    source: str
    page: int = Field(ge=1)
    text: str


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Cut `text` into <=max_chars windows that start and end on word boundaries."""
    if len(text) <= max_chars:
        return [text] if text else []

    pieces: list[str] = []
    start, length = 0, len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            space = text.rfind(" ", start, end)
            if space > start:
                end = space  # cut on whitespace, never mid-word

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break

        candidate = end - overlap
        if candidate <= start:
            candidate = end
        space = text.rfind(" ", start, candidate)
        next_start = space + 1 if space != -1 else candidate
        start = next_start if next_start > start else end + 1

    return pieces


def _split_region_rows(text: str) -> list[tuple[str | None, str]] | None:
    """Split a table page into (row_label, row_text) segments.

    Returns None unless the page has >=2 region-row markers (a lone mention is
    prose, not a table). The preamble before the first row keeps label None.
    """
    matches = list(_REGION_ROW.finditer(text))
    if len(matches) < 2:
        return None

    segments: list[tuple[str | None, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        segments.append((None, preamble))
    for current, following in zip(matches, [*matches[1:], None]):
        end = following.start() if following else len(text)
        segments.append((current.group(0), text[current.start() : end].strip()))
    return segments


def chunk_pages(
    pages: Iterable[PageText],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Turn parsed pages into overlapping, page-tagged chunks.

    Regional-table pages are split row-atomically first; a row that overflows
    max_chars is windowed with its region label re-prepended to every window,
    so no window is ever orphaned from its region.
    """
    chunks: list[Chunk] = []
    carried_caption: dict[str, str] = {}  # source -> caption while its table continues
    for page in pages:
        segments = _split_region_rows(page.text)
        if segments is None:
            # prose page: any running table has ended — drop the carried caption
            carried_caption.pop(page.source, None)
            segments = [(None, page.text)]
            caption = None
        else:
            found = _find_caption(page.text)
            if found:
                carried_caption[page.source] = found
            caption = carried_caption.get(page.source)
        index = 0
        for label, segment in segments:
            for piece in _split_text(segment, max_chars, overlap):
                if label and not piece.startswith(label):
                    piece = f"{label} {piece}"
                if label and caption and "Table" not in piece:
                    # suffix (not prefix): keeps the label-first invariant; BM25
                    # and embeddings are position-blind, the LLM only needs the
                    # column labels PRESENT in the cited text
                    piece = f"{piece} [{caption}]"
                chunks.append(
                    Chunk(
                        chunk_id=f"{page.source}#p{page.page}#{index}",
                        source=page.source,
                        page=page.page,
                        text=piece,
                    )
                )
                index += 1
    return chunks
