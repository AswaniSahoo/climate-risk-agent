"""Tests for row-atomic chunking of IPCC regional assessment tables.

The failure being fixed (measured: duplicate_region slice = 0% recall on naive
windows): a 1200-char window mixes several regions' table cells, so a query for
South Asia retrieves a chunk dominated by the Tibetan Plateau's row.
"""
from rag.chunk import chunk_pages
from rag.parse import PageText


def _page(text: str) -> PageText:
    return PageText(source="IPCC_AR6_WGI_Chapter11.pdf", page=131, text=text)


TABLE_TEXT = (
    "1643 Chapter 11 11 Region Observed Trends Projections "
    "Tibetan Plateau (TIB) Intensification of heavy precipitation limited evidence "
    "median increase of more than 2% in the 50-year Rx1day events "
    "South Asia (SAS) High confidence in the intensification of heavy precipitation "
    "median increase of more than 25% in the 50-year Rx1day events"
)


def test_table_page_splits_one_chunk_per_region_row():
    chunks = chunk_pages([_page(TABLE_TEXT)])
    row_starts = [c.text for c in chunks if c.text.startswith(("Tibetan Plateau (TIB)", "South Asia (SAS)"))]
    assert len(row_starts) == 2


def test_region_rows_do_not_bleed_into_each_other():
    chunks = chunk_pages([_page(TABLE_TEXT)])
    sas = next(c for c in chunks if c.text.startswith("South Asia (SAS)"))
    assert "Tibetan Plateau" not in sas.text
    assert "High confidence" in sas.text
    assert sas.page == 131  # page-level citation preserved


def test_long_row_windows_reprepend_region_header():
    long_row = "South Asia (SAS) " + "heavy precipitation intensifies strongly " * 60
    chunks = chunk_pages([_page("Region Observed Trends " + long_row + " East Asia (EAS) short row")])
    sas_chunks = [c for c in chunks if "(SAS)" in c.text]
    assert len(sas_chunks) >= 2  # row overflowed max_chars
    assert all(c.text.startswith("South Asia (SAS)") for c in sas_chunks)


def test_non_region_parentheses_do_not_trigger_table_mode():
    prose = (
        "Climate models (RCM) and ensembles (CMIP6) project heavy precipitation "
        "increases over land (IPCC). " * 10
    )
    chunks = chunk_pages([_page(prose)])
    # one region marker at most -> plain window chunking, no row splitting
    assert all(not c.text.startswith("(") for c in chunks)
    assert "".join(c.text for c in chunks)  # still produced normal chunks


def test_prose_page_with_single_region_mention_stays_windowed():
    prose = "In South Asia (SAS) the monsoon intensifies. " * 40
    chunks = chunk_pages([_page(prose)])
    assert len(chunks) > 1  # windowed as before, not one giant row
