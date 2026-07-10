"""Page-aware parsing of the IPCC AR6 PDFs.

The page number is captured at the very first step and rides along with every
downstream chunk. That is what makes page-level citations — and a measurable
citation-validity score — possible later.

Split like our other I/O:
- `clean_text`    : pure, unit-tested (de-hyphenation + whitespace normalisation).
- `extract_pages` : the file/PDF edge (pypdf), covered by one integration test.
"""
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field
from pypdf import PdfReader

# "tempera-\nture" -> "temperature" (PDF line-break hyphenation, not real hyphens).
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_WHITESPACE = re.compile(r"\s+")

# pypdf emits U+FFFD for glyphs it cannot map to a character. A stray one is
# harmless; a page made mostly of them is the Table of Contents, whose dot
# leaders ("......") use a symbol font. Those pages are navigation, not content,
# and would pollute retrieval with lists of section titles — so we drop them.
REPLACEMENT_CHAR = "�"
UNMAPPABLE_PAGE_RATIO = 0.2


class PageText(BaseModel):
    """The cleaned text of one page, tagged with where it came from."""

    source: str
    page: int = Field(ge=1)  # 1-based, as a human would cite it
    text: str


def clean_text(raw: str) -> str:
    """Re-join hyphenated words, blank out unmappable glyphs, collapse whitespace."""
    joined = _HYPHEN_LINEBREAK.sub(r"\1\2", raw)
    joined = joined.replace(REPLACEMENT_CHAR, " ")
    return _WHITESPACE.sub(" ", joined).strip()


def is_mostly_unmappable(raw: str, threshold: float = UNMAPPABLE_PAGE_RATIO) -> bool:
    """True when so many glyphs failed to map that the page isn't real content."""
    if not raw:
        return False
    return raw.count(REPLACEMENT_CHAR) / len(raw) > threshold


def extract_pages(pdf_path: Path | str) -> list[PageText]:
    """Read a PDF into one cleaned `PageText` per non-empty page (1-based)."""
    path = Path(pdf_path)
    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        if is_mostly_unmappable(raw):  # Table of Contents / symbol-font pages
            continue
        text = clean_text(raw)
        if text:  # skip blank/image-only pages
            pages.append(PageText(source=path.name, page=page_number, text=text))
    return pages
