"""Every shipped precomputed entry must sit under the key the current code builds.

An entry in tools/cache_seed/ answers a request only while its key matches the
one the live code builds. An edit to the fitting code, the ranking code,
rag/answer.py, rag/cid.py or the model moves that key, and the entry then
never matches again: production quietly goes back to the slow path with
nothing red anywhere. This test is the red. It needs no corpus, no network and
no model: each key is rebuilt from the entry's own fields, with the corpus
version taken from the PDF sizes the seed file records.

It assumes the default model and bootstrap settings, the ones production runs
with (CRG_GENERATE_MODEL, CRG_EMBED_MODEL and CRG_BOOTSTRAP_N unset or at their
defaults). A shell that overrides them sees other keys, and this test says so.
"""
import json

import rag.answer_cache as answer_cache_mod
import tools.climatology as clim
from rag.corpus import CORPUS_FILES, fingerprint_for
from scripts import precompute_climatology as clim_pre
from scripts import precompute_grounding as ipcc_pre

_RERUN = (
    "\nRerun the matching precompute and commit the refreshed seed file:\n"
    "  uv run python -m scripts.precompute_grounding --presets --prune-stale"
    "   (IPCC answers and Ch.12 projections; --only projection after a rag/cid.py change)\n"
    "  uv run python -m scripts.precompute_climatology --refit   (ERA5 fits, offline when the "
    "entry stores its series; otherwise a paced run)\n"
    "If you only set CRG_GENERATE_MODEL, CRG_EMBED_MODEL or CRG_BOOTSTRAP_N in this shell, "
    "unset them: the shipped keys use the production defaults."
)


def test_every_shipped_seed_entry_matches_the_key_the_current_code_builds():
    stale = clim_pre.stale_rows() + ipcc_pre.stale_rows()

    assert not stale, "shipped seed entries the current code no longer builds:\n  " + (
        "\n  ".join(stale)) + _RERUN


def test_the_ipcc_seed_records_the_corpus_it_was_built_on():
    header = json.loads(ipcc_pre.SEED_FILE.read_text(encoding="utf-8"))

    assert set(header["corpus_files"]) == set(CORPUS_FILES)
    assert all(isinstance(size, int) for size in header["corpus_files"].values())
    assert fingerprint_for(header["corpus_files"]) == header["version"]["corpus"]


def _copy(source, tmp_path):
    target = tmp_path / source.name
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_an_edited_prompt_retires_every_grounding_entry(tmp_path, monkeypatch):
    seed = _copy(ipcc_pre.SEED_FILE, tmp_path)
    monkeypatch.setattr(answer_cache_mod, "prompt_version", lambda: "an-edited-prompt")

    stale = ipcc_pre.stale_rows(seed)

    groundings = sum(1 for row in json.loads(seed.read_text())["entries"]
                     if row["namespace"] == "grounding")
    assert groundings and len([s for s in stale if s.startswith("grounding ")]) == groundings
    assert not [s for s in stale if s.startswith("projection ")]  # the projection has no prompt


def test_an_edited_ranker_retires_the_projections_too(tmp_path, monkeypatch):
    seed = _copy(ipcc_pre.SEED_FILE, tmp_path)
    monkeypatch.setattr(answer_cache_mod, "retrieval_version", lambda: "an-edited-ranker")

    stale = ipcc_pre.stale_rows(seed)

    assert len(stale) == len(json.loads(seed.read_text())["entries"])


def test_a_local_corpus_unlike_the_recorded_one_is_flagged(tmp_path, monkeypatch):
    seed = _copy(ipcc_pre.SEED_FILE, tmp_path)
    monkeypatch.setattr(ipcc_pre, "corpus_file_sizes", lambda: {name: 1 for name in CORPUS_FILES})

    stale = ipcc_pre.stale_rows(seed)

    assert any("corpus on disk" in line for line in stale)


def test_another_bootstrap_count_retires_every_fit(tmp_path, monkeypatch):
    seed = _copy(clim_pre.SEED_FILE, tmp_path)
    monkeypatch.setattr(clim, "_N_BOOT", 150)
    monkeypatch.setattr(clim, "_TREND_N_BOOT", 150)

    stale = clim_pre.stale_rows(seed)

    assert len(stale) == len(json.loads(seed.read_text())["entries"]) > 0


def test_a_seed_without_recorded_corpus_sizes_cannot_pass(tmp_path):
    payload = json.loads(ipcc_pre.SEED_FILE.read_text(encoding="utf-8"))
    payload.pop("corpus_files", None)
    seed = tmp_path / "ipcc.json"
    seed.write_text(json.dumps(payload), encoding="utf-8")

    assert ipcc_pre.stale_rows(seed) == ["ipcc.json: no corpus_files in the header, so its "
                                         "keys cannot be rebuilt"]
