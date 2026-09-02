"""Tests for the AR6 Ch.12 CID projection layer (rag/cid.py).

No HTTP and no LLM: the retriever is stubbed and every parse is a pure function,
because the whole point of this module is that the "projected change" section is
derived by rule from retrieved text rather than generated.

Every sentence used below is verbatim from data/cache/chunks.json (Ch.12 of the
AR6 WG1 corpus); its chunk_id and page are named in the fixture so a reviewer
can check the quote against the PDF.
"""
import pytest
from pydantic import ValidationError

import agent.graph as graph_mod
from agent.contracts import ChangeDirection, Citation, Hazard, ProjectedChange
from rag.chunk import Chunk
from rag.cid import CH12_SOURCE, _qualifying_sentences, _region_pattern, projected_change_for
from tools.ar6_regions import AR6Region

MED = AR6Region(acronym="MED", name="Mediterranean")
NEU = AR6Region(acronym="NEU", name="Northern Europe")
SAM = AR6Region(acronym="SAM", name="South American Monsoon")
SWS = AR6Region(acronym="SWS", name="South-Western South America")
NAU = AR6Region(acronym="NAU", name="Northern Australia")
WNA = AR6Region(acronym="WNA", name="Western North America")
ESAF = AR6Region(acronym="ESAF", name="East Southern Africa")
TIB = AR6Region(acronym="TIB", name="Tibetan Plateau")

# --- five real Ch.12 sentences, quoted with their source chunk ---------------
WIND_EUROPE = (  # IPCC_AR6_WGI_Chapter12.pdf#p58#6
    "There is high confidence that mean wind speeds will decrease in Mediterranean "
    "areas and medium confidence of such decreases in Northern Europe for global "
    "warming levels of 2°C or more and beyond the middle of the century."
)
PRECIP_SOUTH_AMERICA = (  # IPCC_AR6_WGI_Chapter12.pdf#p50#1
    "Chapter 11 projections indicate low confidence of increase, compared to the "
    "modern period, in the intensity and frequency of heavy precipitation in SCA and "
    "SWS for all GWLs, and medium confidence of increase in NSA, NES, SSA, SAM and "
    "SES for GWL of 4°C."
)
PRECIP_AUSTRALIA = (  # IPCC_AR6_WGI_Chapter12.pdf#p44#0
    "Heavy precipitation and pluvial flooding are projected to increase with medium "
    "confidence in Northern Australia and Central Australia."
)
WIND_NORTH_AMERICA = (  # IPCC_AR6_WGI_Chapter12.pdf#p66#4
    "model types is a reduction in wind speed in Western North America (high confidence)."
)
WIND_AFRICA = (  # IPCC_AR6_WGI_Chapter12.pdf#p29#5
    "equal or superior to 2°C, high confidence of a decrease in frequency of cyclones "
    "landing in SEAF, ESAF and MDG, and low confidence of a general increase in wind "
    "storms in most African regions located south of the Sahel."
)
# The Table 12.x legend: every direction and every confidence level at once, for
# no region in particular. The single biggest mis-attribution risk in Ch.12.
TABLE_LEGEND = (
    "New Zealand (NZ) 2 4 6 7 Already emerged in the historical period (medium to high "
    "confidence) Medium confidence of decrease Medium confidence of increase High "
    "confidence of decrease High confidence of increase Low confidence in direction of "
    "change Not broadly relevant [Table 12.5 | Summary of confidence in direction of "
    "projected change in climatic impact-drivers in Australasia]"
)


def _chunk(text: str, *, source: str = CH12_SOURCE, page: int = 58, index: int = 0) -> Chunk:
    return Chunk(chunk_id=f"{source}#p{page}#{index}", source=source, page=page, text=text)


class _StubRetriever:
    """Stands in for HybridRetriever; records the top_k it was asked for."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.asked_top_k: int | None = None

    def retrieve(self, question: str, top_k: int) -> list[Chunk]:
        self.asked_top_k = top_k
        return self.chunks[:top_k]


# --- the filter: Chapter 12 only, and only for the region asked about --------


def test_filter_keeps_only_chapter_12_chunks_naming_this_region():
    ch11 = _chunk(WIND_EUROPE, source="IPCC_AR6_WGI_Chapter11.pdf")
    ch12 = _chunk(WIND_EUROPE)
    pattern = _region_pattern(MED)

    assert _qualifying_sentences(ch11, Hazard.WIND, pattern) == []
    assert _qualifying_sentences(ch12, Hazard.WIND, pattern) == [WIND_EUROPE]


def test_filter_rejects_a_chapter_12_sentence_about_another_region():
    ch12 = _chunk(PRECIP_AUSTRALIA, page=44)
    assert _qualifying_sentences(ch12, Hazard.EXTREME_PRECIP, _region_pattern(NAU))
    assert _qualifying_sentences(ch12, Hazard.EXTREME_PRECIP, _region_pattern(SWS)) == []


def test_filter_rejects_the_table_legend_and_the_wrong_hazard():
    legend = _chunk(TABLE_LEGEND, page=47)
    nz = AR6Region(acronym="NZ", name="New Zealand")
    # The legend names the region and states four directions with four confidence
    # levels — parsing it would attribute all of them to whichever region ends
    # the table page. It must never survive the filter.
    assert _qualifying_sentences(legend, Hazard.EXTREME_PRECIP, _region_pattern(nz)) == []
    # right region and chapter, wrong CID vocabulary
    assert _qualifying_sentences(
        _chunk(WIND_EUROPE), Hazard.EXTREME_PRECIP, _region_pattern(MED)
    ) == []


# --- direction / confidence parsing on real Ch.12 sentences ------------------


@pytest.mark.parametrize(
    "region, hazard, text, page, direction, confidence, when",
    [
        # Two assessments in one sentence: MED gets "high", NEU gets "medium".
        (MED, Hazard.WIND, WIND_EUROPE, 58, ChangeDirection.DECREASE, "high confidence", None),
        (NEU, Hazard.WIND, WIND_EUROPE, 58, ChangeDirection.DECREASE, "medium confidence", None),
        # Two region lists, two confidence levels, two warming levels.
        (SAM, Hazard.EXTREME_PRECIP, PRECIP_SOUTH_AMERICA, 50,
         ChangeDirection.INCREASE, "medium confidence", "GWL of 4°C"),
        (SWS, Hazard.EXTREME_PRECIP, PRECIP_SOUTH_AMERICA, 50,
         ChangeDirection.INCREASE, "low confidence", "for all GWLs"),
        (NAU, Hazard.EXTREME_PRECIP, PRECIP_AUSTRALIA, 44,
         ChangeDirection.INCREASE, "medium confidence", None),
        (WNA, Hazard.WIND, WIND_NORTH_AMERICA, 66,
         ChangeDirection.DECREASE, "high confidence", None),
        (ESAF, Hazard.WIND, WIND_AFRICA, 29, ChangeDirection.DECREASE, "high confidence", None),
    ],
)
def test_direction_and_confidence_parse_from_real_rows(
    region, hazard, text, page, direction, confidence, when
):
    retriever = _StubRetriever([_chunk(text, page=page)])

    change = projected_change_for(region, hazard, retriever=retriever)

    assert change is not None
    assert change.statement == text  # verbatim, never paraphrased
    assert change.direction is direction
    assert change.confidence_language == confidence
    assert change.warming_level_or_period == when
    assert change.region_acronym == region.acronym


def test_ambiguous_confidence_is_reported_as_none_not_guessed():
    # Two calibrated phrases, no clause boundary that separates them -> the
    # module says nothing rather than picking one.
    text = (
        "Mean wind speed and wind power potential are projected to decrease in Western "
        "North America (medium confidence) with differences between global and regional "
        "models lending low confidence elsewhere."
    )
    change = projected_change_for(
        WNA, Hazard.WIND, retriever=_StubRetriever([_chunk(text, page=67)])
    )

    assert change is not None
    assert change.direction is ChangeDirection.DECREASE
    assert change.confidence_language is None


def test_retriever_is_queried_wide_and_returns_none_when_nothing_qualifies():
    retriever = _StubRetriever([
        _chunk(WIND_EUROPE),                                     # wrong region
        _chunk(PRECIP_AUSTRALIA, page=44),                       # wrong region
        _chunk(WIND_EUROPE, source="IPCC_AR6_WGI_Chapter11.pdf"),  # wrong chapter
    ])

    assert projected_change_for(TIB, Hazard.WIND, retriever=retriever) is None
    assert retriever.asked_top_k == 30  # wider than the answering top_k of 8


def test_citations_carry_the_chunk_id_and_dedupe_by_page():
    chunks = [
        _chunk(WIND_EUROPE, page=58, index=6),
        _chunk(WIND_EUROPE, page=58, index=7),  # overlap repeats the sentence
        _chunk(WIND_EUROPE, page=59, index=0),
    ]
    change = projected_change_for(MED, Hazard.WIND, retriever=_StubRetriever(chunks))

    assert change is not None
    assert [(c.locator, c.chunk_id) for c in change.citations] == [
        ("p58", "IPCC_AR6_WGI_Chapter12.pdf#p58#6"),
        ("p59", "IPCC_AR6_WGI_Chapter12.pdf#p59#0"),
    ]
    assert all(c.chunk_id in change.retrieved_chunk_ids for c in change.citations)


# --- the structural citation rule (mirrors CitedAnswer in rag/answer.py) -----


def _projected_change(**overrides) -> ProjectedChange:
    payload = dict(
        region_acronym="MED",
        region_name="Mediterranean",
        hazard=Hazard.WIND,
        statement=WIND_EUROPE,
        direction=ChangeDirection.DECREASE,
        confidence_language="high confidence",
        citations=[Citation(source=CH12_SOURCE, locator="p58",
                            chunk_id=f"{CH12_SOURCE}#p58#6")],
        retrieved_chunk_ids=[f"{CH12_SOURCE}#p58#6", f"{CH12_SOURCE}#p58#7"],
    )
    payload.update(overrides)
    return ProjectedChange(**payload)


def test_citation_to_a_non_retrieved_chunk_id_is_rejected():
    with pytest.raises(ValidationError, match="not among the retrieved chunks"):
        _projected_change(
            citations=[Citation(source=CH12_SOURCE, locator="p999",
                                chunk_id=f"{CH12_SOURCE}#p999#0")]
        )


def test_citation_without_a_chunk_id_is_rejected():
    with pytest.raises(ValidationError, match="must carry a chunk_id"):
        _projected_change(citations=[Citation(source=CH12_SOURCE, locator="p58")])


def test_uncited_projected_change_is_rejected():
    with pytest.raises(ValidationError, match="must cite at least one"):
        _projected_change(citations=[])


# --- the graph node ----------------------------------------------------------

CANNED_FORECAST = {
    "latitude": 41.9,
    "longitude": 12.5,
    "timezone": "Europe/Rome",
    "daily_units": {"time": "iso8601", "precipitation_sum": "mm",
                    "temperature_2m_max": "°C", "wind_speed_10m_max": "km/h",
                    "wind_gusts_10m_max": "km/h"},
    "daily": {
        "time": ["2026-07-02", "2026-07-03"],
        "precipitation_sum": [2.0, 1.0],
        "temperature_2m_max": [33.0, 31.0],
        "wind_speed_10m_max": [40.0, 22.0],
        "wind_gusts_10m_max": [61.0, 35.0],
    },
}


def test_graph_run_attaches_a_cited_projected_change(httpx_mock, monkeypatch):
    httpx_mock.add_response(json=CANNED_FORECAST)
    retriever = _StubRetriever([_chunk(WIND_EUROPE, page=58, index=6)])
    monkeypatch.setattr(graph_mod, "_ipcc_retriever", lambda: retriever)
    # research() is not under test here; keep it out of the way (no LLM call).
    monkeypatch.setattr(
        graph_mod, "answer_with_guard",
        lambda q, chunks, **_: (_ for _ in ()).throw(graph_mod.AnswerError("no llm")),
    )

    report = graph_mod.run_agent(  # Rome -> MED
        location="Rome", latitude=41.9, longitude=12.5,
        hazard=Hazard.WIND, horizon_days=2,
    )

    change = report.projected_change
    assert change is not None
    assert (change.region_acronym, change.direction) == ("MED", ChangeDirection.DECREASE)
    assert change.confidence_language == "high confidence"
    assert [c.chunk_id for c in change.citations] == [f"{CH12_SOURCE}#p58#6"]
    assert all(c.chunk_id in change.retrieved_chunk_ids for c in change.citations)
    assert change.statement in report.summary  # verbatim in the prose too


def test_graph_run_says_so_when_no_projection_is_found(httpx_mock, monkeypatch):
    httpx_mock.add_response(json=CANNED_FORECAST)
    # Ch.11 only: nothing survives the Ch.12 filter.
    monkeypatch.setattr(
        graph_mod, "_ipcc_retriever",
        lambda: _StubRetriever([_chunk(WIND_EUROPE, source="IPCC_AR6_WGI_Chapter11.pdf")]),
    )
    monkeypatch.setattr(
        graph_mod, "answer_with_guard",
        lambda q, chunks, **_: (_ for _ in ()).throw(graph_mod.AnswerError("no llm")),
    )

    report = graph_mod.run_agent(
        location="Rome", latitude=41.9, longitude=12.5,
        hazard=Hazard.WIND, horizon_days=2,
    )

    assert report.projected_change is None
    assert "No AR6 Ch.12 regional projection" in report.summary
    assert report.risk_level is not None  # the forecast/hazard path still works


def test_ocean_point_skips_the_projection_without_breaking_the_report(httpx_mock, monkeypatch):
    httpx_mock.add_response(json=CANNED_FORECAST)
    # The very chunk that resolves for Rome is available — the only reason
    # nothing is attached is that an ocean point has no AR6 land region.
    monkeypatch.setattr(
        graph_mod, "_ipcc_retriever",
        lambda: _StubRetriever([_chunk(WIND_EUROPE, page=58, index=6)]),
    )
    monkeypatch.setattr(
        graph_mod, "answer_with_guard",
        lambda q, chunks, **_: (_ for _ in ()).throw(graph_mod.AnswerError("no llm")),
    )

    report = graph_mod.run_agent(  # mid North Atlantic
        location="North Atlantic", latitude=35.0, longitude=-40.0,
        hazard=Hazard.WIND, horizon_days=2,
    )

    assert report.projected_change is None
    assert "outside the AR6 land reference regions" in report.summary
    assert report.risk_level is not None
