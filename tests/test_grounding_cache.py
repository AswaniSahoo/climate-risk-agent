"""The IPCC grounding cache: one cited answer per (hazard, AR6 area), shared by every city in it.

research() asks `_ipcc_question(hazard, _ipcc_area(state))`, and the area is the
AR6 region label, never the city. So Berlin and Paris (both WCE) ask the same
question, retrieve the same chunks and need the same answer. The cache stores
that answer once, under a key that also carries the corpus, retrieval, prompt
and model versions, so any of those changing asks again instead of replaying.

Everything here runs on fakes: no corpus, no embeddings, no model call.
"""
import pytest
from pydantic import ValidationError

import agent.graph as graph_mod
import rag.answer_cache as answer_cache_mod
from agent.contracts import ChangeDirection, Citation, Hazard, ProjectedChange
from rag.answer import CitedAnswer
from rag.answer_cache import Grounding, GroundingCache, ProjectionCache, ProjectionResult
from rag.chunk import Chunk

_CHUNKS = (
    Chunk(chunk_id="IPCC_AR6_WGI_Chapter11.pdf#p124#1", source="IPCC_AR6_WGI_Chapter11.pdf",
          page=124, text="hot extremes will intensify in Western and Central Europe"),
    Chunk(chunk_id="IPCC_AR6_WGI_Chapter12.pdf#p40#0", source="IPCC_AR6_WGI_Chapter12.pdf",
          page=40, text="extreme heat is projected to increase in WCE"),
    Chunk(chunk_id="IPCC_AR6_WGI_SPM.pdf#p16#0", source="IPCC_AR6_WGI_SPM.pdf",
          page=16, text="hot extremes have become more frequent"),
)

BERLIN = {"location": "Berlin, Western and Central Europe (WCE)", "latitude": 52.52,
          "longitude": 13.40, "hazard": Hazard.HEATWAVE, "horizon_days": 7}
PARIS = {"location": "Paris, Western and Central Europe (WCE)", "latitude": 48.85,
         "longitude": 2.35, "hazard": Hazard.HEATWAVE, "horizon_days": 7}
DELHI = {"location": "Delhi, South Asia (SAS)", "latitude": 28.61,
         "longitude": 77.21, "hazard": Hazard.HEATWAVE, "horizon_days": 7}


class _Retriever:
    """Stands in for HybridRetriever. `dense` is whether the dense half 'ran'."""

    def __init__(self, dense: bool = True) -> None:
        self.dense = dense
        self.questions: list[str] = []

    def retrieve_traced(self, question, top_k):
        self.questions.append(question)
        return list(_CHUNKS)[:top_k], self.dense

    def retrieve(self, question, top_k):
        return self.retrieve_traced(question, top_k)[0]


def _cited(question, chunks, **_):
    return CitedAnswer(answer="Hot extremes are projected to intensify.",
                       citations=[chunks[1].chunk_id], abstain=False,
                       allowed_ids=[c.chunk_id for c in chunks])


@pytest.fixture
def grounding(tmp_path, monkeypatch):
    """A disk-backed grounding cache in tmp_path, plus counters on the expensive parts."""
    cache = GroundingCache(tmp_path / "grounding")
    monkeypatch.setattr(graph_mod, "_grounding_cache", lambda: cache)
    retriever = _Retriever()
    built: list[int] = []

    def build():
        built.append(1)
        return retriever

    asked: list[str] = []

    def answer(question, chunks, **kwargs):
        asked.append(question)
        return _cited(question, chunks, **kwargs)

    monkeypatch.setattr(graph_mod, "_ipcc_retriever", build)
    monkeypatch.setattr(graph_mod, "answer_with_guard", answer)
    return {"cache": cache, "retriever": retriever, "built": built, "asked": asked}


def test_two_cities_in_one_area_share_one_grounding(grounding):
    berlin = graph_mod.research(BERLIN)
    paris = graph_mod.research(PARIS)

    assert len(grounding["asked"]) == 1  # Paris cost no model call
    assert len(grounding["retriever"].questions) == 1  # and no retrieval
    assert paris["ipcc_answer"] == berlin["ipcc_answer"]
    # The cited chunk travels with the answer, so citations still map to pages.
    assert [c.chunk_id for c in paris["ipcc_chunks"]] == [_CHUNKS[1].chunk_id]


def test_a_hit_never_builds_the_retriever(grounding):
    graph_mod.research(BERLIN)
    grounding["built"].clear()

    graph_mod.research(PARIS)

    assert grounding["built"] == []  # no corpus load, no embedding matrix, no BM25 index


def test_a_different_area_is_a_different_entry(grounding):
    graph_mod.research(BERLIN)
    graph_mod.research(DELHI)

    assert len(grounding["asked"]) == 2
    assert "South Asia (SAS)" in grounding["asked"][1]


def test_a_bm25_only_grounding_is_never_cached(grounding):
    grounding["retriever"].dense = False  # the query embedding failed: BM25 ranking

    graph_mod.research(BERLIN)
    graph_mod.research(PARIS)

    assert len(grounding["asked"]) == 2


def test_an_abstention_is_never_cached(grounding, monkeypatch):
    def abstain(question, chunks, **_):
        grounding["asked"].append(question)
        return CitedAnswer(answer="", citations=[], abstain=True,
                           allowed_ids=[c.chunk_id for c in chunks])

    monkeypatch.setattr(graph_mod, "answer_with_guard", abstain)

    graph_mod.research(BERLIN)
    graph_mod.research(PARIS)

    assert len(grounding["asked"]) == 2


def test_a_stand_in_retriever_without_the_traced_call_is_never_cached(grounding, monkeypatch):
    class _PlainRetriever:
        def retrieve(self, question, top_k):
            return list(_CHUNKS)[:top_k]

    monkeypatch.setattr(graph_mod, "_ipcc_retriever", lambda: _PlainRetriever())

    graph_mod.research(BERLIN)
    graph_mod.research(PARIS)

    assert len(grounding["asked"]) == 2


def _key(**overrides):
    base = {"hazard": "heatwave", "area": "Western and Central Europe (WCE)",
            "question": "How are hot extremes ... over WCE?", "top_k": 8}
    base.update(overrides)
    return GroundingCache(backend=None).key(**base)


def test_the_key_moves_with_every_version_it_claims_to_carry(monkeypatch):
    base = _key()

    assert _key(hazard="wind") != base
    assert _key(area="South Asia (SAS)") != base
    assert _key(question="another wording") != base
    assert _key(top_k=5) != base
    monkeypatch.setattr(answer_cache_mod, "GENERATE_MODEL", "gemini-9-flash")
    assert _key() != base  # model
    monkeypatch.undo()
    monkeypatch.setattr(answer_cache_mod, "corpus_version", lambda: "a-new-corpus")
    assert _key() != base  # corpus
    monkeypatch.undo()
    monkeypatch.setattr(answer_cache_mod, "prompt_version", lambda: "a-new-prompt")
    assert _key() != base  # prompt
    monkeypatch.undo()
    monkeypatch.setattr(answer_cache_mod, "retrieval_version", lambda: "a-new-ranker")
    assert _key() != base  # retrieval
    monkeypatch.undo()
    assert _key() == base


def test_the_prompt_version_follows_the_regional_rule_switch(monkeypatch):
    monkeypatch.delenv("CRG_REGIONAL_GROUNDING", raising=False)
    off = answer_cache_mod.prompt_version()
    monkeypatch.setenv("CRG_REGIONAL_GROUNDING", "1")

    assert answer_cache_mod.prompt_version() != off  # a different rule set is a different prompt


def test_a_grounding_must_carry_every_chunk_it_cites():
    answer = _cited("q", list(_CHUNKS))

    with pytest.raises(ValidationError):
        Grounding(hazard="heatwave", area="WCE", question="q", answer=answer,
                  cited_chunks=[_CHUNKS[0]])  # cites _CHUNKS[1], which is missing


def test_an_abstention_cannot_be_stored_as_a_grounding():
    abstained = CitedAnswer(answer="", citations=[], abstain=True,
                            allowed_ids=[c.chunk_id for c in _CHUNKS])

    with pytest.raises(ValidationError):
        Grounding(hazard="heatwave", area="WCE", question="q", answer=abstained, cited_chunks=[])


def test_a_grounding_round_trips_through_the_cache(tmp_path):
    cache = GroundingCache(tmp_path)
    grounded = Grounding.from_research(hazard="heatwave", area="WCE", question="q",
                                       answer=_cited("q", list(_CHUNKS)), chunks=list(_CHUNKS))

    cache.put("k", grounded)

    assert cache.get("k") == grounded
    assert [c.chunk_id for c in grounded.cited_chunks] == [_CHUNKS[1].chunk_id]


# --- the Ch.12 projection: also a function of (region, hazard) only ---------

def _projection(region, hazard, chunks):
    return ProjectedChange.from_retrieval(
        chunks, region_acronym=region.acronym, region_name=region.name, hazard=hazard,
        statement="Extreme heat will increase in WCE (high confidence).",
        direction=ChangeDirection.INCREASE, confidence_language="high confidence",
        citations=[Citation(source=chunks[0].source, locator=f"p{chunks[0].page}",
                            chunk_id=chunks[0].chunk_id)],
    )


@pytest.fixture
def projection(tmp_path, monkeypatch):
    cache = ProjectionCache(tmp_path / "projection")
    monkeypatch.setattr(graph_mod, "_projection_cache", lambda: cache)
    retriever = _Retriever()
    monkeypatch.setattr(graph_mod, "_ipcc_retriever", lambda: retriever)
    computed: list[str] = []
    found = {"value": True}

    def fake_projected_change_for(region, hazard, *, retriever):
        computed.append(region.acronym)
        chunks = retriever.retrieve(f"How is extreme heat projected in {region.acronym}?", top_k=3)
        return _projection(region, hazard, chunks) if found["value"] else None

    monkeypatch.setattr(graph_mod, "projected_change_for", fake_projected_change_for)
    return {"cache": cache, "retriever": retriever, "computed": computed, "found": found}


def test_two_cities_in_one_region_share_one_projection(projection):
    berlin = graph_mod.project(BERLIN)
    paris = graph_mod.project(PARIS)

    assert projection["computed"] == ["WCE"]
    assert paris["projected_change"] == berlin["projected_change"]


def test_a_region_with_no_projection_is_remembered_as_not_found(projection):
    projection["found"]["value"] = False

    first = graph_mod.project(BERLIN)
    second = graph_mod.project(PARIS)

    assert projection["computed"] == ["WCE"]
    assert first == second == {"projection_note": graph_mod._NOT_FOUND_NOTE}


def test_a_bm25_only_projection_is_never_cached(projection):
    projection["retriever"].dense = False

    graph_mod.project(BERLIN)
    graph_mod.project(PARIS)

    assert projection["computed"] == ["WCE", "WCE"]


def test_a_projection_result_round_trips_through_the_cache(tmp_path):
    from tools.ar6_regions import region_for

    cache = ProjectionCache(tmp_path)
    region = region_for(52.52, 13.40)
    found = ProjectionResult(change=_projection(region, Hazard.HEATWAVE, list(_CHUNKS)))
    missing = ProjectionResult(change=None)

    cache.put("found", found)
    cache.put("missing", missing)

    assert cache.get("found") == found
    assert cache.get("missing") == missing
    assert cache.key(region, Hazard.HEATWAVE) != cache.key(region, Hazard.WIND)


# --- end to end through the graph -------------------------------------------

def test_the_second_city_gets_the_same_citations_and_a_cache_badge(grounding, httpx_mock):
    from agent.progress import StepStatus

    canned = {
        "latitude": 52.52, "longitude": 13.40, "timezone": "Europe/Berlin",
        "daily_units": {}, "daily": {
            "time": ["2026-07-02", "2026-07-03"],
            "precipitation_sum": [1.0, 0.0], "temperature_2m_max": [30.0, 31.0],
            "wind_speed_10m_max": [10.0, 12.0], "wind_gusts_10m_max": [20.0, 25.0],
        },
    }
    httpx_mock.add_response(json=canned)
    httpx_mock.add_response(json=canned)
    events = []

    def on_step(event):
        if event.status is StepStatus.FINISHED:
            events.append(event)

    first = graph_mod.run_agent(BERLIN["location"], 52.52, 13.40, Hazard.HEATWAVE, horizon_days=2)
    second = graph_mod.run_agent(PARIS["location"], 48.85, 2.35, Hazard.HEATWAVE,
                                 horizon_days=2, on_step=on_step)

    assert second.citations == first.citations != []
    assert len(grounding["asked"]) == 1
    research = next(e for e in events if e.node == "research")
    assert research.detail == "cached (disk)"
