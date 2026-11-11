"""Fetch the IPCC AR6 WG1 documents we index: SPM + Chapter 11 + Chapter 12.

Chapter 11 ("Weather and Climate Extreme Events in a Changing Climate") covers
heat, precipitation and wind extremes — exactly our three hazards. Chapter 12
("Climatic Impact-Drivers and their Changes") assesses those same hazards *by
region*, which is what a location-based agent needs for the regional/table eval
questions. The SPM supplies the headline assessed statements.

PDFs land in data/ (git-ignored: large, and best fetched from the source).

Integrity: every PDF is pinned by SHA-256 (CORPUS_SHA256 in rag/corpus.py).
pypdf parses whatever lands in data/ipcc, at image build and at runtime
whenever the chunk cache is missing, so these files are its entire input and
nothing else may reach it. A download is written to a temporary file, hashed,
and moved into place only if it matches; a copy already on disk is hashed too
rather than trusted. Any mismatch stops with CorpusDownloadError, naming the
file and both hashes. rag/corpus.py checks the same pins again before parsing.

Run:  uv run python -m scripts.download_ipcc
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from rag.corpus import CORPUS_DIR, CORPUS_FILES, CORPUS_SHA256, file_sha256

BASE = "https://www.ipcc.ch/report/ar6/wg1/downloads/report"


@dataclass(frozen=True)
class CorpusPdf:
    url: str
    sha256: str


# Fetched with HTTP 200 and no redirect on 2026-11-11, when the pins were set.
DOCS = {name: CorpusPdf(f"{BASE}/{name}", CORPUS_SHA256[name]) for name in CORPUS_FILES}
DEST = CORPUS_DIR


class CorpusDownloadError(RuntimeError):
    """A corpus PDF, downloaded or already on disk, failed its SHA-256 check."""


def _mismatch(name: str, where: str, expected: str, actual: str) -> CorpusDownloadError:
    return CorpusDownloadError(
        f"{name} {where} does not match the pinned sha256 in scripts/download_ipcc.py "
        f"(expected {expected}, got {actual})"
    )


def _download(name: str, pdf: CorpusPdf, target: Path) -> None:
    """Stream to a temporary file beside `target`; move it in only if it matches."""
    handle, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(handle, "wb") as out:
            with httpx.stream("GET", pdf.url, follow_redirects=True, timeout=180) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    digest.update(chunk)
                    out.write(chunk)
        if digest.hexdigest() != pdf.sha256:
            raise _mismatch(name, f"downloaded from {pdf.url}", pdf.sha256, digest.hexdigest())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, pdf in DOCS.items():
        target = DEST / name
        if target.exists():
            actual = file_sha256(target)
            if actual != pdf.sha256:
                raise _mismatch(name, f"on disk at {target}", pdf.sha256, actual)
            print(f"ok {name} (exists, sha256 verified)")
            continue
        print(f"downloading {name} ...")
        _download(name, pdf, target)
        print(f"  -> {target} ({target.stat().st_size / 1e6:.1f} MB, sha256 verified)")


if __name__ == "__main__":
    try:
        main()
    except CorpusDownloadError as exc:
        sys.exit(f"corpus check failed: {exc}")
