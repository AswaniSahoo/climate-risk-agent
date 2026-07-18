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

# v2 split (2026-07-17): the 45 above became the DEV set (they steered every
# design decision, so they cannot claim held-out). gold_set_v2.json is the
# HELD-OUT TEST set — run at release gates only, diagnosed never.
TEST_SET_PATH = Path(__file__).parent / "gold_set_v2.json"
TEST_SET_FROZEN_HASH_PATH = Path(__file__).parent / "gold_set_v2.sha256"


def load_gold_set() -> EvalSet:
    """Load + validate the frozen gold set from JSON (the DEV set since v2)."""
    return EvalSet.model_validate_json(GOLD_SET_PATH.read_text(encoding="utf-8"))


def load_test_set() -> EvalSet:
    """Load + validate the frozen HELD-OUT test set (release gates only)."""
    return EvalSet.model_validate_json(TEST_SET_PATH.read_text(encoding="utf-8"))
