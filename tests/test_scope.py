"""Tests for the deterministic scope guard (rag/scope.py).

Live smoke showed the prompt-level scope rule being ignored by the LLM — so the
guard is CODE in front of the model (refuse-before-retrieve), not a prompt hope.
"""
from rag.scope import out_of_scope_hazard


def test_pure_unsupported_hazard_is_flagged():
    assert out_of_scope_hazard("Has meteorological drought increased in South Asia?") == "drought"
    assert out_of_scope_hazard("Are Category 3-5 tropical cyclones becoming more common?") == "tropical cyclone"
    assert out_of_scope_hazard("Will coastal flooding from sea level rise get worse?") == "coastal flooding / sea level"
    assert out_of_scope_hazard("Is the fire weather season lengthening in India?") == "wildfire / fire weather"


def test_marine_heatwave_is_flagged_despite_the_word_heatwave():
    """A marine heatwave is an oceanic hazard we do not assess. The phrase
    contains 'heatwave' (supported), so it must be caught before the supported
    check — held-out eval v2 caught it slipping through as a false answer."""
    assert out_of_scope_hazard("How have marine heatwaves changed since the 1980s?") == "marine heatwave"
    assert out_of_scope_hazard("Is ocean heatwave frequency rising in the Bay of Bengal?") == "marine heatwave"


def test_cryosphere_background_science_stays_answerable():
    """Glaciers / sea ice / snowlines are background earth-system science
    (like Arctic warming), NOT hazards this agent refuses — consistent with the
    dev set's glacier-commitment ANSWER item. The guard must NOT flag them."""
    assert out_of_scope_hazard("Are mountain and polar glaciers committed to shrink?") is None
    assert out_of_scope_hazard("How does 2011-2020 Arctic sea ice area compare historically?") is None


def test_supported_hazards_pass():
    assert out_of_scope_hazard("Will heatwaves intensify over India?") is None
    assert out_of_scope_hazard("How much will extreme daily precipitation intensify?") is None
    assert out_of_scope_hazard("Are wind gusts increasing over Chennai?") is None


def test_compound_question_with_supported_hazard_passes():
    # Drought appears, but the question is about (in-scope) heatwave compounds.
    assert out_of_scope_hazard("Have concurrent heatwaves and droughts become more frequent?") is None


def test_non_hazard_science_question_passes():
    assert out_of_scope_hazard("What were CO2 concentrations in 2019?") is None
