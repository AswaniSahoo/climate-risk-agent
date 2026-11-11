"""Tests for scripts/precompute_grounding.py: the IPCC grounding per (hazard, AR6 area).

The precompute runs the graph's own research() and project() nodes, so what it
ships is by construction what a live run would have cached. These tests check
the plumbing around that: which pairs exist, that the written entries sit under
the live keys and answer a later live run, and that the run stops before its
model-call budget. No corpus, embedding or model call: the nodes' expensive
parts are faked, as in tests/test_grounding_cache.py.
"""
import json

import pytest

import agent.graph as graph_mod
from agent.contracts import Hazard
from rag.answer import CitedAnswer
from rag.answer_cache import GroundingCache, ProjectionCache
from rag.chunk import Chunk
from scripts import precompute_grounding as pre
from tools.ar6_regions import region_for
from tools.cache_backend import DiskCache, _with_seed

_CHUNKS = (
    Chunk(chunk_id="IPCC_AR6_WGI_Chapter11.pdf#p124#1", source="IPCC_AR6_WGI_Chapter11.pdf",
          page=124, text="hot extremes will intensify"),
    Chunk(chunk_id="IPCC_AR6_WGI_Chapter12.pdf#p40#0", source="IPCC_AR6_WGI_Chapter12.pdf",
          page=40, text="extreme heat is projected to increase"),
)


def _record(op: str) -> None:
    """What the real SDK seam records for each model call (rag/gemini_client.py)."""
    from obs.telemetry import record

    record(op=op, model="fake", latency_ms=1.0, tokens_in=1, tokens_out=0, retries=0, ok=True)


class _Retriever:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def retrieve_traced(self, question, top_k):
        self.questions.append(question)
        _record("embed")  # the real retriever embeds the query: one model call
        return list(_CHUNKS)[:top_k], True

    def retrieve(self, question, top_k):
        return self.retrieve_traced(question, top_k)[0]


@pytest.fixture
def nodes(tmp_path, monkeypatch):
    """Disk-backed caches the nodes write to, and counters on the faked expensive parts."""
    grounding = GroundingCache(tmp_path / "live" / "grounding")
    projection = ProjectionCache(tmp_path / "live" / "projection")
    monkeypatch.setattr(graph_mod, "_grounding_cache", lambda: grounding)
    monkeypatch.setattr(graph_mod, "_projection_cache", lambda: projection)
    retriever = _Retriever()
    monkeypatch.setattr(graph_mod, "_ipcc_retriever", lambda: retriever)
    asked: list[str] = []
    abstain_on: set[str] = set()

    def answer(question, chunks, **_):
        asked.append(question)
        _record("generate")
        if any(area in question for area in abstain_on):
            return CitedAnswer(answer="", citations=[], abstain=True,
                               allowed_ids=[c.chunk_id for c in chunks])
        return CitedAnswer(answer="Hot extremes intensify.", citations=[chunks[0].chunk_id],
                           abstain=False, allowed_ids=[c.chunk_id for c in chunks])

    monkeypatch.setattr(graph_mod, "answer_with_guard", answer)
    monkeypatch.setattr(graph_mod, "projected_change_for",
                        lambda region, hazard, *, retriever: retriever.retrieve("cid", 3) and None)
    return {"asked": asked, "abstain_on": abstain_on, "retriever": retriever,
            "grounding": grounding, "projection": projection}


def _pair(acronym: str, hazard: Hazard = Hazard.HEATWAVE) -> pre.Pair:
    return next(p for p in pre.all_pairs() if p.region.acronym == acronym and p.hazard is hazard)


def test_every_land_region_gets_a_point_that_maps_back_to_it():
    points = pre.region_points()

    assert len(points) == 46  # the AR6 land set, tools/ar6_regions.py
    assert len({region.acronym for region, _, _ in points}) == 46
    for region, lat, lon in points:
        assert region_for(lat, lon) == region


def test_the_full_set_is_every_answerable_hazard_in_every_land_region():
    pairs = pre.all_pairs()

    assert len(pairs) == 3 * 46
    assert {p.hazard for p in pairs} == set(graph_mod._ANSWERABLE)


def test_the_six_presets_need_six_pairs_in_two_areas():
    pairs = pre.preset_pairs()

    assert len(pairs) == 6
    assert {p.region.acronym for p in pairs} == {"SAS", "WCE"}
    assert {p.hazard for p in pairs} == set(graph_mod._ANSWERABLE)


def test_a_run_writes_both_entries_under_the_keys_a_live_run_reads(nodes, tmp_path):
    seed = tmp_path / "seed" / "ipcc.json"

    summary = pre.run([_pair("WCE")], seed_path=seed, max_calls=30)

    assert summary.written == ["heatwave/WCE"]
    rows = json.loads(seed.read_text())["entries"]
    assert {row["namespace"] for row in rows} == {"grounding", "projection"}
    region = region_for(52.52, 13.40)
    question = graph_mod._ipcc_question(Hazard.HEATWAVE, region.label)
    g_key = nodes["grounding"].key(hazard="heatwave", area=region.label, question=question,
                                   top_k=graph_mod._IPCC_TOP_K)
    assert nodes["grounding"].json_cache.key(g_key) in {row["key"] for row in rows}
    p_key = nodes["projection"].key(region, Hazard.HEATWAVE)
    assert nodes["projection"].json_cache.key(p_key) in {row["key"] for row in rows}


def test_a_later_live_run_reads_the_shipped_entries_without_any_work(nodes, tmp_path, monkeypatch):
    seed_dir = tmp_path / "seed"
    pre.run([_pair("WCE")], seed_path=seed_dir / "ipcc.json", max_calls=30)
    asked_before, retrieved_before = len(nodes["asked"]), len(nodes["retriever"].questions)

    seeded = _with_seed(DiskCache(tmp_path / "cold"), {"CRG_CACHE_SEED_DIR": str(seed_dir)})
    monkeypatch.setattr(graph_mod, "_grounding_cache", lambda: GroundingCache(backend=seeded))
    monkeypatch.setattr(graph_mod, "_projection_cache", lambda: ProjectionCache(backend=seeded))
    berlin = {"location": "Berlin, Western and Central Europe (WCE)", "latitude": 52.52,
              "longitude": 13.40, "hazard": Hazard.HEATWAVE, "horizon_days": 7}

    research = graph_mod.research(berlin)
    project = graph_mod.project(berlin)

    assert research["ipcc_answer"].answer == "Hot extremes intensify."
    assert project == {"projection_note": graph_mod._NOT_FOUND_NOTE}
    assert len(nodes["asked"]) == asked_before  # no model call
    assert len(nodes["retriever"].questions) == retrieved_before  # no retrieval at all


def test_a_pair_already_shipped_is_skipped(nodes, tmp_path):
    seed = tmp_path / "ipcc.json"
    pre.run([_pair("WCE")], seed_path=seed, max_calls=30)

    summary = pre.run([_pair("WCE")], seed_path=seed, max_calls=30)

    assert summary.skipped == ["heatwave/WCE"]
    assert len(nodes["asked"]) == 1


def test_an_abstention_is_reported_and_never_shipped(nodes, tmp_path):
    nodes["abstain_on"].add("(WCE)")
    seed = tmp_path / "ipcc.json"

    summary = pre.run([_pair("WCE")], seed_path=seed, max_calls=30)

    assert summary.written == []
    assert summary.not_cached == ["heatwave/WCE: no grounding entry (abstained, BM25-only or failed)"]
    rows = json.loads(seed.read_text())["entries"]
    assert [row["namespace"] for row in rows] == ["projection"]  # the projection still ships


def test_the_run_stops_before_a_pair_could_pass_the_call_budget(nodes, tmp_path):
    summary = pre.run([_pair("WCE"), _pair("SAS")], seed_path=tmp_path / "ipcc.json",
                      max_calls=pre.MAX_CALLS_PER_PAIR + 1)

    assert summary.written == ["heatwave/WCE"]
    assert summary.deferred == ["heatwave/SAS"]


def test_check_counts_current_and_stale_entries(nodes, tmp_path):
    seed = tmp_path / "ipcc.json"
    pre.run([_pair("WCE")], seed_path=seed, max_calls=30)
    payload = json.loads(seed.read_text())
    payload["entries"].append({"key": "crg:v1:grounding:retired", "namespace": "grounding",
                               "value": {}})
    seed.write_text(json.dumps(payload))

    result = pre.check([_pair("WCE"), _pair("SAS")], seed_path=seed)

    assert result == {"pairs": 2, "current": 1, "missing": ["heatwave/SAS"], "stale": 1}
