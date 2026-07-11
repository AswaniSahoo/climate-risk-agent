"""Tests for the end-to-end answer checkers (evals/checkers.py). Pure, offline."""
from evals.checkers import (
    citation_hits_gold,
    numbers_in_text,
    numeric_provenance_ok,
    refusal_cell,
)
from evals.schema import PageRef

GOLD = [PageRef(source="a.pdf", page=16)]


def test_citation_hits_gold_on_page_match():
    assert citation_hits_gold(["a.pdf#p16#0"], GOLD) is True
    assert citation_hits_gold(["a.pdf#p2#0", "b.pdf#p16#1"], GOLD) is False
    assert citation_hits_gold([], GOLD) is False


def test_numbers_in_text_extracts_and_normalizes():
    text = "about 7% per 1°C; sea level rose 0.20 [0.15 to 0.25] m since 1901"
    assert numbers_in_text(text) == {"7", "1", "0.20", "0.15", "0.25", "1901"}


def test_numeric_provenance_passes_when_all_numbers_are_cited():
    answer = "Extreme rain intensifies about 7% per 1°C of warming."
    cited = ["events are projected to intensify by about 7% for each 1°C of global warming"]
    assert numeric_provenance_ok(answer, cited) is True


def test_numeric_provenance_fails_on_invented_number():
    answer = "Extreme rain intensifies about 12% per 1°C."
    cited = ["events are projected to intensify by about 7% for each 1°C of global warming"]
    assert numeric_provenance_ok(answer, cited) is False


def test_numeric_provenance_trivially_true_without_numbers():
    assert numeric_provenance_ok("Heavy rain will worsen in most regions.", ["any text"]) is True


def test_refusal_cells():
    assert refusal_cell(expected_refuse=True, did_abstain=True) == "correct_refuse"
    assert refusal_cell(expected_refuse=False, did_abstain=False) == "correct_answer"
    assert refusal_cell(expected_refuse=False, did_abstain=True) == "false_refuse"
    assert refusal_cell(expected_refuse=True, did_abstain=False) == "false_answer"
