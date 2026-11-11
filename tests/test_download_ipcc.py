"""The corpus download (scripts/download_ipcc.py) accepts only the pinned bytes.

pypdf parses whatever lands in data/ipcc, at image build and at runtime when the
chunk cache is missing, so the PDFs are its whole input. Each one is pinned by
SHA-256 in the manifest, checked after download and before reuse.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

import scripts.download_ipcc as dl
from rag.corpus import CORPUS_DIR, CORPUS_FILES

GOOD = b"%PDF-1.7 the pinned bytes"
URL = "https://www.ipcc.ch/report/ar6/wg1/downloads/report/TEST.pdf"


@pytest.fixture
def manifest(tmp_path, monkeypatch) -> Path:
    """A one-entry manifest pinned to GOOD, downloading into tmp_path."""
    monkeypatch.setattr(dl, "DEST", tmp_path)
    monkeypatch.setattr(
        dl, "DOCS", {"TEST.pdf": dl.CorpusPdf(URL, hashlib.sha256(GOOD).hexdigest())}
    )
    return tmp_path


def test_a_download_with_the_pinned_hash_lands(manifest, httpx_mock):
    httpx_mock.add_response(url=URL, content=GOOD)

    dl.main()

    assert (manifest / "TEST.pdf").read_bytes() == GOOD


def test_a_download_with_other_bytes_fails_loudly_and_leaves_nothing(manifest, httpx_mock):
    httpx_mock.add_response(url=URL, content=b"%PDF-1.7 something else")

    with pytest.raises(dl.CorpusDownloadError, match="does not match the pinned sha256"):
        dl.main()

    assert list(manifest.iterdir()) == []  # no target, no temp file left behind


def test_an_existing_copy_is_verified_not_trusted(manifest, httpx_mock):
    (manifest / "TEST.pdf").write_bytes(b"%PDF-1.7 tampered on disk")

    with pytest.raises(dl.CorpusDownloadError, match="TEST.pdf"):
        dl.main()

    # not replaced behind the reader's back either: the file stays for inspection
    assert (manifest / "TEST.pdf").read_bytes() == b"%PDF-1.7 tampered on disk"
    assert httpx_mock.get_requests() == []


def test_an_existing_good_copy_is_not_downloaded_again(manifest, httpx_mock):
    (manifest / "TEST.pdf").write_bytes(GOOD)

    dl.main()

    assert httpx_mock.get_requests() == []


def test_the_manifest_pins_every_corpus_file_from_ipcc_ch():
    assert set(dl.DOCS) == set(CORPUS_FILES)  # one list of PDFs, not two that drift
    for name, pdf in dl.DOCS.items():
        assert re.fullmatch(r"[0-9a-f]{64}", pdf.sha256), name
        assert pdf.url == f"https://www.ipcc.ch/report/ar6/wg1/downloads/report/{name}"


@pytest.mark.skipif(not all((CORPUS_DIR / n).exists() for n in CORPUS_FILES),
                    reason="IPCC PDFs not on disk (run scripts/download_ipcc.py)")
def test_the_pdfs_on_disk_match_the_pins():
    for name, pdf in dl.DOCS.items():
        digest = hashlib.sha256((CORPUS_DIR / name).read_bytes()).hexdigest()
        assert digest == pdf.sha256, name
