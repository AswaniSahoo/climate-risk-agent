"""Cache keys hash source code, so line endings must not move them.

A Windows checkout with core.autocrlf holds the same code with CRLF endings.
Hashed as raw bytes it would build different keys from a Linux checkout, and
every shipped precomputed entry (tools/cache_seed/) would miss there.
"""
from pathlib import Path

import rag.answer_cache as answer_cache_mod
import rag.corpus as corpus_mod
from tools.cache_backend import source_bytes

RAG = Path(__file__).resolve().parents[1] / "rag"


def _crlf_copy(source: Path, target: Path) -> Path:
    text = source.read_bytes().replace(b"\r\n", b"\n")
    target.write_bytes(text.replace(b"\n", b"\r\n"))
    return target


def test_source_bytes_reads_crlf_as_lf(tmp_path):
    path = tmp_path / "x.py"
    path.write_bytes(b"a = 1\r\nb = 2\r\n")

    assert source_bytes(path) == b"a = 1\nb = 2\n"


def test_the_corpus_fingerprint_ignores_line_endings_in_the_chunker(tmp_path, monkeypatch):
    sizes = {"IPCC_AR6_WGI_SPM.pdf": 1, "IPCC_AR6_WGI_Chapter11.pdf": 2,
             "IPCC_AR6_WGI_Chapter12.pdf": None}
    lf = corpus_mod.fingerprint_for(sizes)

    monkeypatch.setattr(corpus_mod, "_CHUNKER_SOURCE", _crlf_copy(RAG / "chunk.py", tmp_path / "chunk.py"))

    assert corpus_mod.fingerprint_for(sizes) == lf


def test_the_grounding_version_digests_ignore_line_endings(tmp_path, monkeypatch):
    names = (*answer_cache_mod._RETRIEVAL_SOURCES, *answer_cache_mod._PROMPT_SOURCES,
             *answer_cache_mod._PROJECTION_SOURCES)
    answer_cache_mod._source_digest.cache_clear()
    lf = {group: answer_cache_mod._source_digest(group) for group in (
        answer_cache_mod._RETRIEVAL_SOURCES, answer_cache_mod._PROMPT_SOURCES,
        answer_cache_mod._PROJECTION_SOURCES)}
    for name in names:
        _crlf_copy(RAG / name, tmp_path / name)
    monkeypatch.setattr(answer_cache_mod, "_SOURCE_DIR", tmp_path)
    answer_cache_mod._source_digest.cache_clear()
    try:
        crlf = {group: answer_cache_mod._source_digest(group) for group in lf}
    finally:
        answer_cache_mod._source_digest.cache_clear()

    assert crlf == lf


def test_a_real_edit_still_moves_the_digest(tmp_path, monkeypatch):
    """Normalising line endings must not blind the fingerprint to code changes."""
    answer_cache_mod._source_digest.cache_clear()
    before = answer_cache_mod._source_digest(("answer.py",))
    edited = tmp_path / "answer.py"
    edited.write_bytes(source_bytes(RAG / "answer.py") + b"\n# an edit\n")
    monkeypatch.setattr(answer_cache_mod, "_SOURCE_DIR", tmp_path)
    answer_cache_mod._source_digest.cache_clear()
    try:
        assert answer_cache_mod._source_digest(("answer.py",)) != before
    finally:
        answer_cache_mod._source_digest.cache_clear()


def test_python_sources_check_out_with_lf_everywhere():
    """tools/climatology.py hashes its own raw bytes into every ERA5 fit key, and
    changing that function would retire the shipped fits, so this file guards it
    at the source: git writes .py files with LF even under core.autocrlf."""
    rules = (Path(__file__).resolve().parents[1] / ".gitattributes").read_text(encoding="utf-8")

    assert "*.py text eol=lf" in rules.splitlines()
