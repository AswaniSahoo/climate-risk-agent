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


def test_supported_hazards_pass():
    assert out_of_scope_hazard("Will heatwaves intensify over India?") is None
    assert out_of_scope_hazard("How much will extreme daily precipitation intensify?") is None
    assert out_of_scope_hazard("Are wind gusts increasing over Chennai?") is None


def test_compound_question_with_supported_hazard_passes():
    # Drought appears, but the question is about (in-scope) heatwave compounds.
    assert out_of_scope_hazard("Have concurrent heatwaves and droughts become more frequent?") is None


def test_non_hazard_science_question_passes():
    assert out_of_scope_hazard("What were CO2 concentrations in 2019?") is None
