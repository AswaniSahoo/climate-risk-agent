"""Tests for page-aware IPCC PDF parsing (rag/parse.py).

`clean_text` is pure and fully unit-tested. `extract_pages` is the file/PDF edge:
it gets one integration test against the real SPM, skipped when the PDFs haven't
been downloaded (so CI stays green without the data).
"""
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.parse import PageText, clean_text, extract_pages, is_mostly_unmappable

SPM = Path("data/ipcc/IPCC_AR6_WGI_SPM.pdf")
CH11 = Path("data/ipcc/IPCC_AR6_WGI_Chapter11.pdf")


def test_clean_text_dehyphenates_across_line_breaks():
    assert clean_text("tempera-\nture rises") == "temperature rises"


def test_clean_text_preserves_real_hyphens():
    assert clean_text("well-known effect") == "well-known effect"


def test_clean_text_collapses_whitespace_and_newlines():
    assert clean_text("Heat   waves\nare\n\n  more frequent  ") == "Heat waves are more frequent"


def test_clean_text_replaces_unmappable_glyphs_with_space():
    assert clean_text("Executive��Summary") == "Executive Summary"


def test_is_mostly_unmappable_flags_toc_dot_leaders():
    assert is_mostly_unmappable("Contents" + "�" * 100)


def test_is_mostly_unmappable_allows_a_stray_glyph():
    assert not is_mostly_unmappable("a mostly clean sentence with one � glyph in it")


def test_page_text_rejects_non_positive_page():
    with pytest.raises(ValidationError):
        PageText(source="x.pdf", page=0, text="hi")


@pytest.mark.skipif(not SPM.exists(), reason="IPCC PDFs not downloaded")
def test_extract_pages_reads_real_spm_with_page_numbers():
    pages = extract_pages(SPM)

    assert len(pages) > 10
    assert pages[0].page == 1
    assert pages[0].source == SPM.name
    assert all(page.text for page in pages)  # empty pages are dropped
    assert [p.page for p in pages] == sorted(p.page for p in pages)


@pytest.mark.skipif(not CH11.exists(), reason="IPCC PDFs not downloaded")
def test_extract_pages_drops_unmappable_toc_pages():
    pages = extract_pages(CH11)
    kept = {page.page for page in pages}

    assert 3 not in kept and 4 not in kept  # Table of Contents (dot-leader glyphs)
    assert 5 in kept  # Executive Summary survives
    assert all("�" not in page.text for page in pages)
