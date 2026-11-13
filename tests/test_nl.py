"""Tests for agent/nl.py — deterministic natural-language query parsing.

Lexical on purpose (same philosophy as the scope guard): classification that
runs before any LLM cannot be prompt-injected and costs nothing. Misses are
typed and loud, never guessed.
"""
import pytest

from agent.contracts import Hazard
from agent.nl import ParsedQuery, parse_query, refusal_reason


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


# --- everyday words -------------------------------------------------------------------
# The live app (2026-11-11, CRG_JEV off) refused "How windy will it get in Berlin over
# the next 7 days?" with "could not identify a supported hazard" while "wind gusts"
# worked. With Jev off these patterns are the whole front door.


@pytest.mark.parametrize(
    ("text", "hazard", "place", "days"),
    [
        ("How windy will it get in Berlin over the next 7 days?", Hazard.WIND, "Berlin", 7),
        ("Will it be windy in Mumbai this week?", Hazard.WIND, "Mumbai", 7),
        ("How hot will it get in Delhi next week?", Hazard.HEATWAVE, "Delhi", 7),
    ],
)
def test_the_questions_the_live_app_refused_now_parse_fully(text, hazard, place, days):
    q = parse_query(text)
    assert (q.hazard, q.place, q.horizon_days) == (hazard, place, days)
    assert refusal_reason(q) is None


@pytest.mark.parametrize(
    ("text", "hazard"),
    [
        ("Are the hottest days getting hotter in Jaipur?", Hazard.HEATWAVE),
        ("Will Seville see scorching afternoons this week?", Hazard.HEATWAVE),
        ("How bad will the summer scorchers get in Phoenix?", Hazard.HEATWAVE),
        ("Will Rome swelter next week?", Hazard.HEATWAVE),
        ("Is London in for a sweltering week?", Hazard.HEATWAVE),
        ("Will it be rainy in Kochi this week?", Hazard.EXTREME_PRECIP),
        ("Is a downpour likely in Kolkata tomorrow?", Hazard.EXTREME_PRECIP),
        ("Could a deluge hit Chennai this week?", Hazard.EXTREME_PRECIP),
        ("Will Mumbai be deluged next week?", Hazard.EXTREME_PRECIP),
        ("Will cloudbursts hit Shimla more often?", Hazard.EXTREME_PRECIP),
        ("Is a cloud burst likely over Dehradun?", Hazard.EXTREME_PRECIP),
        ("Is it getting windier on the coast near Perth?", Hazard.WIND),
        ("Which is the windiest week ahead in Wellington?", Hazard.WIND),
        ("Will it be gusty in Hamburg tomorrow?", Hazard.WIND),
        ("Will it be gusting hard in Oslo tonight?", Hazard.WIND),
        ("Will gales hit Aberdeen next week?", Hazard.WIND),
        ("Is a gale-force blow coming to Galway?", Hazard.WIND),
        ("Will a windstorm hit Dublin this week?", Hazard.WIND),
    ],
)
def test_everyday_words_name_our_hazards(text, hazard):
    q = parse_query(text)
    assert q.hazard is hazard and q.out_of_scope is None


@pytest.mark.parametrize(
    ("text", "refused_as"),
    [
        # "storm" names none of ours: storm surge, tropical storms, hail and tornado
        # storms, snowstorms and dust storms are all hazards we do not assess.
        ("Are storms getting stronger in Berlin?", None),
        ("Will a thunderstorm hit Berlin tomorrow?", None),
        ("Will a cyclone hit Odisha next week?", "tropical cyclone"),
        ("Will the drought in Nairobi get worse this week?", "drought"),
        ("Will coastal flooding hit Mumbai next week?", "coastal flooding / sea level"),
        ("Will the river flood in Patna next week?", None),
        ("Will hail hit Denver tomorrow?", "severe convective storm"),
        ("Will it be cold in Moscow next week?", None),
        ("Will it snow in Moscow next week?", None),
        ("Is a blizzard coming to Chicago?", None),
    ],
)
def test_hazards_we_do_not_assess_are_still_refused(text, refused_as):
    q = parse_query(text)
    assert q.hazard is None and q.out_of_scope == refused_as
    assert refusal_reason(q) is not None


@pytest.mark.parametrize(
    ("text", "refused_as"),
    [
        ("Will hot, dry and windy weather raise the wildfire risk near Athens?",
         "wildfire / fire weather"),
        ("Will it be windy when the typhoon reaches Manila?", "tropical cyclone"),
        ("Will gales drive a storm surge into Chennai?", "coastal flooding / sea level"),
        ("Will the sea get hotter in a marine heatwave off Perth?", "marine heatwave"),
    ],
)
def test_an_everyday_word_never_lets_in_a_hazard_the_guard_names(text, refused_as):
    """"Hot, dry and windy" is how fire weather is described: an everyday word names
    our hazard only when the scope guard names no unsupported one."""
    q = parse_query(text)
    assert q.hazard is None and q.out_of_scope == refused_as


def test_an_everyday_word_never_moves_a_reading_the_older_words_make():
    # "rain" named this question before "hot" did anything; enum order must not flip it
    q = parse_query("Will the hot weather end in heavy rain in Mumbai?")
    assert q.hazard is Hazard.EXTREME_PRECIP


@pytest.mark.parametrize(
    "text",
    [
        "Is Delhi a climate change hot spot?",  # a hotspot of change, not heat
        "How much warming will Delhi see by 2050?",  # the background trend, no hazard
        "Will temperatures fall below zero in Moscow next week?",  # hot or cold: unsaid
        "Is there a hotel near the beach in Goa?",
        "Will the windscreen survive the drive to Windsor?",
        "They ate with gusto in Rome",
    ],
)
def test_words_that_do_not_say_which_hazard_name_none(text):
    assert parse_query(text).hazard is None
