"""Download the eval-cache-v1 release asset for the image build; refuse it unless its sha256 matches.

The Dockerfile pins the digest GitHub reports for the asset (read with `gh
release view eval-cache-v1 --json assets` on 2026-11-11: name
eval-cache-v1.tar.gz, 9,338,994 bytes, digest sha256:7f91a5ce...425423). The pin
guards against a replaced asset; scripts/unpack_eval_cache.py then checks the
archive's own manifest against this repo (embedding model, dims, chunker)
before a single vector lands, which guards against a mismatched one.

Run:  python -m scripts.fetch_eval_cache --sha256 <64 hex> --out /tmp/eval-cache.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

DEFAULT_URL = ("https://github.com/AswaniSahoo/climate-risk-agent/releases/download/"
               "eval-cache-v1/eval-cache-v1.tar.gz")


class AssetError(RuntimeError):
    """The asset could not be fetched, or it is not the pinned one."""


def fetch(url: str, sha256: str, out: Path | str) -> Path:
    """Stream `url` to `out`, hashing as it goes; keep the file only if it matches."""
    expected = sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise AssetError("the pinned sha256 must be 64 hex characters")
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for block in response.iter_bytes():
                    digest.update(block)
                    handle.write(block)
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise AssetError(f"could not download {url}: {exc}") from exc
    actual = digest.hexdigest()
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise AssetError(
            f"sha256 mismatch for {url}: expected {expected}, got {actual}. The asset was "
            "replaced or the download is corrupt, so it is not baked into the image.")
    os.replace(partial, target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--sha256", required=True, help="the pinned digest, 64 hex characters")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        path = fetch(args.url, args.sha256, args.out)
    except AssetError as exc:
        print(f"EVAL CACHE ASSET REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"eval cache asset verified (sha256 {args.sha256[:16]}...): {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
