"""Fetch the IPCC AR6 WG1 documents we index: SPM + Chapter 11 + Chapter 12.

Chapter 11 ("Weather and Climate Extreme Events in a Changing Climate") covers
heat, precipitation and wind extremes — exactly our three hazards. Chapter 12
("Climatic Impact-Drivers and their Changes") assesses those same hazards *by
region*, which is what a location-based agent needs for the regional/table eval
questions. The SPM supplies the headline assessed statements.

PDFs land in data/ (git-ignored: large, and best fetched from the source).

Integrity: every PDF is pinned by SHA-256 below. pypdf parses whatever lands in
data/ipcc, at image build and at runtime whenever the chunk cache is missing,
so these files are its entire input and nothing else may reach it. A download
is written to a temporary file, hashed, and moved into place only if it matches;
a copy already on disk is hashed too rather than trusted. Any mismatch stops
with CorpusDownloadError, naming the file and both hashes.

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

BASE = "https://www.ipcc.ch/report/ar6/wg1/downloads/report"


@dataclass(frozen=True)
class CorpusPdf:
    url: str
    sha256: str


# Pinned 2026-11-11: the copies this project has been parsing and a fresh
# download from these URLs that day hashed the same (HTTP 200, no redirect).
# If the IPCC ever republishes a file, re-download it, re-run the eval, then
# update the hash here on purpose.
DOCS = {
    "IPCC_AR6_WGI_SPM.pdf": CorpusPdf(
        f"{BASE}/IPCC_AR6_WGI_SPM.pdf",
        "37c9fa760757524c8d69630a89cecab9ebcacb66786970a3439da0619875e446",
    ),
    "IPCC_AR6_WGI_Chapter11.pdf": CorpusPdf(
        f"{BASE}/IPCC_AR6_WGI_Chapter11.pdf",
        "d0dc4fae531a7b6a76a456cb12db9c7d68099265e377973d508484c3289ddce1",
    ),
    "IPCC_AR6_WGI_Chapter12.pdf": CorpusPdf(
        f"{BASE}/IPCC_AR6_WGI_Chapter12.pdf",
        "4f1bb6b1ecce77b7a0f7f2322a2ba34e53c1a1e5bcb48d5db7f6a2d80b0fd4a0",
    ),
}
DEST = Path("data/ipcc")


class CorpusDownloadError(RuntimeError):
    """A corpus PDF, downloaded or already on disk, failed its SHA-256 check."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
            actual = _sha256(target)
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
