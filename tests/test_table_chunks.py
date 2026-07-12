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


# --- table-caption carry (measured motive, twice): row cells say "compared to
# the 1°C warming level" for EVERY column — the 1.5/2/4°C labels live only in
# the table caption. Without the caption in the chunk, the model correctly
# refused to attribute figures (RT-01 false refusal) and the claim judge
# flagged "at 4°C" as unsupported on RT-10/DR-02. ---

_CAPTION = (
    "Table 11.7 | Observed trends, human contribution, and projected changes "
    "at 1.5°C, 2°C and 4°C of global warming"
)


def _pageN(text: str, page: int) -> PageText:
    return PageText(source="IPCC_AR6_WGI_Chapter11.pdf", page=page, text=text)


def test_caption_is_suffixed_to_row_chunks_on_the_caption_page():
    chunks = chunk_pages([_page(_CAPTION + " " + TABLE_TEXT)])
    sas = next(c for c in chunks if c.text.startswith("South Asia (SAS)"))
    assert "Table 11.7" in sas.text and "4°C" in sas.text
    assert sas.text.startswith("South Asia (SAS)")  # label-first invariant kept


def test_caption_carries_to_continuation_pages_of_the_same_table():
    first = _pageN(_CAPTION + " " + TABLE_TEXT, page=119)
    continuation = _pageN(TABLE_TEXT, page=120)  # no caption on the page itself
    chunks = chunk_pages([first, continuation])
    cont_sas = next(c for c in chunks if c.page == 120 and c.text.startswith("South Asia (SAS)"))
    assert "Table 11.7" in cont_sas.text


def test_prose_page_ends_the_carry():
    first = _pageN(_CAPTION + " " + TABLE_TEXT, page=119)
    prose = _pageN("Plain discussion of extremes, no table rows here. " * 30, page=120)
    orphan_table = _pageN(TABLE_TEXT, page=121)  # table w/o caption after prose
    chunks = chunk_pages([first, prose, orphan_table])
    orphan_sas = next(c for c in chunks if c.page == 121 and "South Asia (SAS)" in c.text)
    assert "Table 11.7" not in orphan_sas.text  # stale caption must not leak


def test_preamble_segments_do_not_get_the_caption_suffix():
    chunks = chunk_pages([_page(_CAPTION + " Region Observed Trends " + TABLE_TEXT)])
    preamble = [c for c in chunks if not c.text.startswith(("Tibetan Plateau", "South Asia"))]
    assert all(not c.text.endswith("]") for c in preamble)
