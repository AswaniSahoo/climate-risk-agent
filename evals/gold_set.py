"""Loader for the frozen gold eval set.

The questions live in gold_set.json (data as data — no Python-literal escaping
of quotes/ligatures). Loading validates every item against the EvalQuestion
invariant, so a mislabel cannot even be constructed. gold_set.sha256 holds the
frozen content hash; tests/test_gold_set.py asserts they match, so any edit is
loud and deliberate (re-freeze = update the sidecar in the same commit).

Authored BEFORE retrieval existed (2026-07-11); gold pages are PDF-sequential.
"""
from __future__ import annotations

from pathlib import Path

from evals.schema import EvalSet

GOLD_SET_PATH = Path(__file__).parent / "gold_set.json"
FROZEN_HASH_PATH = Path(__file__).parent / "gold_set.sha256"


def load_gold_set() -> EvalSet:
    """Load + validate the frozen gold set from JSON."""
    return EvalSet.model_validate_json(GOLD_SET_PATH.read_text(encoding="utf-8"))
