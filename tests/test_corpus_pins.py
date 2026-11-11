"""pypdf only ever parses the pinned IPCC PDFs (rag/corpus.py).

The download script checks the pins when it fetches or finds a file, but the
runtime parses data/ipcc on its own whenever the chunk cache is missing or
stale. So the parse path checks the same pins, right before pypdf runs.
"""
from __future__ import annotations

import hashlib

import pytest

import rag.corpus as corpus


@pytest.fixture
def fake_corpus(tmp_path, monkeypatch):
    """Two tiny 'PDFs' pinned to their own bytes; parsing is recorded, not run."""
    files = {"A.pdf": b"%PDF-1.7 a", "B.pdf": b"%PDF-1.7 b"}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    monkeypatch.setattr(corpus, "CORPUS_DIR", tmp_path)
    monkeypatch.setattr(corpus, "CORPUS_FILES", tuple(files))
    monkeypatch.setattr(corpus, "CORPUS_SHA256",
                        {n: hashlib.sha256(d).hexdigest() for n, d in files.items()})
    parsed: list[str] = []
    monkeypatch.setattr(corpus, "extract_pages", lambda path: parsed.append(path.name) or [])
    monkeypatch.setattr(corpus, "chunk_pages", lambda pages: [])
    return tmp_path, parsed


def test_pinned_files_are_parsed(fake_corpus):
    _, parsed = fake_corpus

    corpus._parse_corpus()

    assert parsed == ["A.pdf", "B.pdf"]


def test_a_file_that_fails_its_pin_is_never_parsed(fake_corpus):
    directory, parsed = fake_corpus
    (directory / "B.pdf").write_bytes(b"%PDF-1.7 swapped on disk")

    with pytest.raises(corpus.CorpusError, match="B.pdf.*sha256"):
        corpus._parse_corpus()

    assert parsed == []  # checked before pypdf sees any file


def test_every_corpus_file_has_a_pin():
    assert set(corpus.CORPUS_SHA256) == set(corpus.CORPUS_FILES)
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in corpus.CORPUS_SHA256.values())
