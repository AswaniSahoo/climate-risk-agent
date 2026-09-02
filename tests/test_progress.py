"""Tests for agent/progress.py: the words and the arithmetic behind the panel.

Everything here is pure (no Streamlit, no langgraph, no network), which is the
whole reason the module exists: the label mapping, the cache badge, the elapsed
formatting and the error-to-sentence map are pinned by tests instead of being
discovered by a person watching the UI go wrong.
"""
import pytest

from agent.progress import (
    CLIMATOLOGY,
    GRAPH_NODES,
    LOCATE,
    STEP_ORDER,
    StepEvent,
    StepStatus,
    cache_badge,
    emit_step,
    error_sentence,
    finished_detail,
    format_elapsed,
    step_label,
    step_meaning,
    track,
)


def _cache_event(namespace: str, backend: str, *, hit: bool) -> dict:
    """A telemetry row in the exact shape tools/cache_backend.py:_record emits."""
    return {
        "op": f"cache:{namespace}",
        "model": backend,
        "latency_ms": 0.4,
        "tokens_in": 0,
        "tokens_out": 0,
        "retries": 0,
        "ok": True,
        "cached": hit,
    }


# --- labels -----------------------------------------------------------------

def test_every_displayed_step_has_a_label_and_a_meaning():
    """A step with no words is a spinner with extra steps."""
    for node in STEP_ORDER:
        assert step_label(node) != node, f"{node} has no label"
        assert step_meaning(node), f"{node} has no meaning line"


def test_the_forecast_label_names_the_horizon_that_was_actually_asked_for():
    assert step_label("call", horizon_days=16) == "Fetching the 16-day forecast (Open-Meteo)"
    assert step_label("call", horizon_days=7) == "Fetching the 7-day forecast (Open-Meteo)"


def test_the_forecast_label_invents_no_horizon_when_it_does_not_know_one():
    assert step_label("call") == "Fetching the forecast (Open-Meteo)"


def test_the_climatology_meaning_carries_the_expected_wait():
    """The one step that can take minutes has to say so, and say why it won't next time."""
    meaning = step_meaning(CLIMATOLOGY)
    assert "1 to 2 minutes" in meaning
    assert "cache" in meaning


def test_the_graph_node_names_match_the_graph():
    from agent.graph import _GRAPH

    assert set(GRAPH_NODES) <= set(_GRAPH.nodes)


# --- cache badges -----------------------------------------------------------

def test_a_cache_hit_is_badged_with_the_tier_that_served_it():
    records = [_cache_event("forecast", "redis", hit=True)]

    assert cache_badge(records, "call") == "cached (redis)"


def test_a_disk_hit_says_disk():
    records = [_cache_event("hazard_fit", "disk", hit=True)]

    assert cache_badge(records, CLIMATOLOGY) == "cached (disk)"


def test_a_cache_miss_gets_no_badge():
    """A miss is recorded too, so the flag decides, not the presence of a row."""
    records = [_cache_event("forecast", "disk", hit=False)]

    assert cache_badge(records, "call") == ""


def test_another_namespace_does_not_badge_this_step():
    records = [_cache_event("answers", "redis", hit=True)]

    assert cache_badge(records, "call") == ""


def test_a_step_with_no_cache_never_claims_one():
    assert cache_badge([_cache_event("forecast", "redis", hit=True)], "synthesize") == ""


# --- finished detail --------------------------------------------------------

def test_a_missing_regional_projection_is_stated_on_the_step_that_found_none():
    detail = finished_detail("project", {"projection_note": "ocean point"}, [])

    assert detail == "no regional projection for this point"


def test_a_refusal_is_stated_on_the_planning_step():
    detail = finished_detail("plan", {"report": object()}, [])

    assert "refused" in detail


def test_a_cache_hit_and_an_absence_are_both_shown():
    detail = finished_detail(
        "call", {}, [_cache_event("forecast", "disk", hit=True)]
    )

    assert detail == "cached (disk)"


# --- elapsed formatting -----------------------------------------------------

@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0.0, "0.00 s"),
        (0.02, "0.02 s"),       # a warm cache read must not read as "0 s"
        (1.44, "1.4 s"),
        (42.3, "42 s"),
        (82.0, "1 m 22 s"),
        (-1.0, "0.00 s"),        # clock skew never prints a negative wait
    ],
)
def test_format_elapsed(seconds, expected):
    assert format_elapsed(seconds) == expected


# --- error sentences --------------------------------------------------------

class ForecastError(RuntimeError):
    pass


class GeocodeError(RuntimeError):
    pass


class ClimatologyError(RuntimeError):
    pass


class GeminiError(RuntimeError):
    pass


class CorpusError(RuntimeError):
    pass


class HTTPStatusError(Exception):
    pass


def test_an_upstream_forecast_outage_reads_as_an_outage_not_a_user_error():
    sentence = error_sentence(ForecastError("Open-Meteo request failed: 503"))

    assert "Open-Meteo" in sentence
    assert "try again" in sentence.lower()
    assert "503" not in sentence  # the raw failure stays in the log


def test_an_unresolvable_place_tells_the_reader_what_to_do_instead():
    sentence = error_sentence(GeocodeError("no match"))

    assert "latitude and longitude" in sentence


def test_a_climatology_failure_offers_the_ungrounded_run():
    assert "untick" in error_sentence(ClimatologyError("archive down"))


def test_missing_gemini_credentials_and_exhausted_quota_read_differently():
    auth = error_sentence(GeminiError("no Gemini auth configured: set ..."))
    quota = error_sentence(GeminiError("rate-limit retries exhausted"))

    assert "credentials" in auth
    assert "quota" in quota
    assert auth != quota


def test_a_429_from_the_model_is_a_quota_sentence():
    assert "quota" in error_sentence(GeminiError("Gemini call failed: 429 RESOURCE_EXHAUSTED"))


def test_a_network_error_gets_the_network_sentence():
    assert "network call" in error_sentence(HTTPStatusError("500 Server Error"))


def test_bad_coordinates_point_at_the_sidebar():
    assert "latitude" in error_sentence(ValueError("latitude 999 out of range"))


def test_a_missing_corpus_says_so():
    assert "IPCC AR6 corpus" in error_sentence(CorpusError("no PDFs"))


def test_an_unknown_failure_still_gets_one_sentence_and_never_a_traceback():
    sentence = error_sentence(ZeroDivisionError("division by zero"))

    assert len(sentence) < 200  # a sentence, not a dump
    assert "server log" in sentence  # where the traceback actually went
    assert "Traceback" not in sentence
    assert "division by zero" not in sentence


# --- emitters ---------------------------------------------------------------

def test_emit_step_is_a_no_op_when_nobody_is_listening():
    emit_step(None, "call", StepStatus.STARTED)  # must not raise


def test_track_reports_started_then_finished_with_measured_time():
    seen: list[StepEvent] = []

    with track(seen.append, LOCATE):
        pass

    assert [e.status for e in seen] == [StepStatus.STARTED, StepStatus.FINISHED]
    assert all(e.label == "Resolving location" for e in seen)
    assert seen[-1].seconds >= 0.0


def test_track_reports_a_failure_and_re_raises_it():
    """The panel must not leave a step spinning, and the caller still decides."""
    seen: list[StepEvent] = []

    with pytest.raises(ClimatologyError):
        with track(seen.append, CLIMATOLOGY):
            raise ClimatologyError("archive down")

    assert [e.status for e in seen] == [StepStatus.STARTED, StepStatus.FAILED]
    assert seen[-1].detail == "ClimatologyError"
