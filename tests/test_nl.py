"""Tests for agent/nl.py — deterministic natural-language query parsing.

Lexical on purpose (same philosophy as the scope guard): classification that
runs before any LLM cannot be prompt-injected and costs nothing. Misses are
typed and loud, never guessed.
"""
import pytest

from agent.contracts import Hazard
from agent.nl import ParsedQuery, parse_query


def test_heatwave_query_parses_fully():
    q = parse_query("How risky are heatwaves in Rourkela over the next 10 days?")
    assert q.hazard is Hazard.HEATWAVE
    assert q.place == "Rourkela"
    assert q.horizon_days == 10
    assert q.out_of_scope is None


@pytest.mark.parametrize(
    ("text", "hazard"),
    [
        ("extreme rainfall risk for Mumbai next week", Hazard.EXTREME_PRECIP),
        ("will the monsoon bring heavy rain in Kolkata", Hazard.EXTREME_PRECIP),
        ("wind gust risk near Chennai", Hazard.WIND),
        ("is Delhi facing extreme heat this week", Hazard.HEATWAVE),
    ],
)
def test_hazard_classification(text, hazard):
    assert parse_query(text).hazard is hazard


def test_unsupported_hazard_is_flagged_not_guessed():
    q = parse_query("what is the wildfire risk in Sydney next week")
    assert q.hazard is None
    assert q.out_of_scope is not None
    assert "wildfire" in q.out_of_scope


def test_supported_hazard_keeps_compound_query_in_scope():
    q = parse_query("heatwave and drought risk in Delhi")  # compound: heat is ours
    assert q.hazard is Hazard.HEATWAVE
    assert q.out_of_scope is None


def test_no_hazard_found_is_none_not_a_guess():
    q = parse_query("what's the weather like in Paris")
    assert q.hazard is None and q.out_of_scope is None


@pytest.mark.parametrize(
    ("text", "days"),
    [
        ("heat risk in Delhi next week", 7),
        ("heat risk in Delhi for the next 3 days", 3),
        ("heat risk in Delhi tomorrow", 2),
        ("heat risk in Delhi over the next two weeks", 14),
        ("heat risk in Delhi", 7),  # default
        ("heat risk in Delhi next 99 days", 16),  # capped at the tool boundary max
    ],
)
def test_horizon_extraction(text, days):
    assert parse_query(text).horizon_days == days


@pytest.mark.parametrize(
    ("text", "place"),
    [
        ("heatwave risk in New Delhi next week", "New Delhi"),
        ("extreme rain near Port Blair", "Port Blair"),
        ("wind risk for Berlin", "Berlin"),
        ("heatwaves in Rourkela?", "Rourkela"),
    ],
)
def test_place_extraction(text, place):
    assert parse_query(text).place == place


def test_missing_place_is_none():
    q = parse_query("how bad will heatwaves get next week")
    assert q.place is None
    assert isinstance(q, ParsedQuery)
