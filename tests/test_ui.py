"""Streamlit UI smoke tests (streamlit.testing.AppTest: headless, offline).

The agent + climatology are monkeypatched: these pin that the UI renders every
RiskReport path honestly (risk, citations, refusal), not the agent logic.
"""
from datetime import datetime, timezone

import pytest
from streamlit.testing.v1 import AppTest

import agent.graph as graph_mod
import agent.location as location_mod
import agent.nl as nl_mod
import tools.climatology as climatology_mod
from agent.contracts import (
    Citation,
    DataProvenance,
    Hazard,
    RiskDriver,
    RiskLevel,
    RiskReport,
)

_APP = "ui/app.py"


def _report(**overrides) -> RiskReport:
    base = dict(
        location="Rourkela, India",
        hazard=Hazard.HEATWAVE,
        horizon_days=7,
        confidence=0.6,
        risk_level=RiskLevel.HIGH,
        summary="Peak daily max temperature of 42.0 °C over 7 days.",
        drivers=[RiskDriver(factor="temperature", detail="max daily 42.0 °C")],
        citations=[Citation(source="IPCC_AR6_WGI_Chapter11.pdf", locator="p124")],
        provenance=[DataProvenance(
            source="Open-Meteo", url="https://api.open-meteo.com/v1/forecast",
            retrieved_at=datetime.now(timezone.utc), params={},
        )],
    )
    base.update(overrides)
    return RiskReport(**base)


@pytest.fixture
def stubbed(monkeypatch):
    import streamlit as st

    def fake_run_agent(**kwargs):
        return _report(hazard=kwargs["hazard"], horizon_days=kwargs["horizon_days"])

    monkeypatch.setattr(graph_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(nl_mod, "run_agent_nl", lambda *a, **k: _report())
    monkeypatch.setattr(
        climatology_mod, "climatology_hazard_stat",
        lambda *a, **k: (_ for _ in ()).throw(climatology_mod.ClimatologyError("offline test")),
    )
    monkeypatch.setattr(st, "map", lambda *a, **k: None)


def _run_clicked(at: AppTest) -> AppTest:
    at.run()
    at.sidebar.button(key="assess").set_value(True)  # by key: the sidebar has example buttons too
    return at.run()


# AppTest classifies EVERY expandable block that carries an icon as a `Status`
# (element_tree.py: `if block.expandable.icon`), so the app's icon'd expanders
# land in `at.status` alongside the real one. The run's own panel is the block
# wearing an icon that only st.status sets.
_STATUS_ICONS = {"spinner", ":material/check:", ":material/error:"}


def _panel(at: AppTest):
    """The st.status container the run rendered its steps into."""
    panels = [s for s in at.status if s.icon in _STATUS_ICONS]
    assert panels, "the run rendered no status panel"
    return panels[0]


def test_report_path_renders_risk_and_citations(stubbed):
    at = _run_clicked(AppTest.from_file(_APP, default_timeout=30))

    assert not at.exception
    rendered = " ".join(el.value for el in at.markdown) + " ".join(s.value for s in at.subheader)
    assert "HIGH" in rendered
    assert "IPCC_AR6_WGI_Chapter11.pdf" in rendered
    assert "p124" in rendered


def test_manual_coordinates_reach_the_agent_unchanged(stubbed, monkeypatch):
    """Any point on Earth: a southern + western pair must arrive with its signs."""
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return _report(hazard=kwargs["hazard"], horizon_days=kwargs["horizon_days"])

    monkeypatch.setattr(graph_mod, "run_agent", capture)
    monkeypatch.setattr(location_mod, "region_for", lambda lat, lon: None)

    at = AppTest.from_file(_APP, default_timeout=30)
    at.session_state["lat_input"] = -33.90  # Sydney
    at.session_state["lon_input"] = -70.67  # deliberately both negative
    _run_clicked(at)  # the seeded name belongs to other coordinates -> dropped

    assert captured["latitude"] == pytest.approx(-33.90)
    assert captured["longitude"] == pytest.approx(-70.67)
    assert "33.9000°S" in captured["location"] and "70.6700°W" in captured["location"]


def test_point_with_no_ar6_region_shows_the_notice_and_still_runs(stubbed, monkeypatch):
    monkeypatch.setattr(location_mod, "region_for", lambda lat, lon: None)

    at = AppTest.from_file(_APP, default_timeout=30)
    at.session_state["lat_input"] = 0.0  # mid-Pacific
    at.session_state["lon_input"] = -140.0
    _run_clicked(at)  # the seeded name belongs to other coordinates -> dropped

    assert not at.exception
    assert any("IPCC AR6 land region" in i.value for i in at.info)
    assert at.subheader  # the report still rendered: no region is not a refusal


def test_progress_panel_shows_step_labels_timings_and_cache_badges(stubbed, monkeypatch):
    """What the agent emits is what the reader sees, including a warm-cache hit."""
    from agent.progress import StepEvent, StepStatus, step_label

    def emitting_run_agent(**kwargs):
        on_step = kwargs["on_step"]
        label = step_label("call", horizon_days=kwargs["horizon_days"])
        on_step(StepEvent(node="call", label=label, status=StepStatus.STARTED))
        on_step(StepEvent(node="call", label=label, status=StepStatus.FINISHED,
                          seconds=1.44, detail="cached (redis)"))
        return _report(hazard=kwargs["hazard"], horizon_days=kwargs["horizon_days"])

    monkeypatch.setattr(graph_mod, "run_agent", emitting_run_agent)
    at = _run_clicked(AppTest.from_file(_APP, default_timeout=30))

    assert not at.exception
    panel = _panel(at)
    assert panel.state == "complete"
    assert panel.label.startswith("Report ready in")  # title becomes the total

    rendered = " ".join(m.value for m in at.markdown)
    assert "Fetching the 7-day forecast (Open-Meteo)" in rendered
    assert "1.4 s" in rendered
    assert "cached (redis)" in rendered
    assert "Resolving location" in rendered  # the UI's own step, same panel


def test_the_first_visit_wait_is_stated_next_to_the_assess_button(stubbed):
    at = AppTest.from_file(_APP, default_timeout=30)
    at.run()

    hints = " ".join(c.value for c in at.sidebar.caption)
    assert "1 to 2 minutes" in hints
    assert "cached" in hints


def test_results_carry_a_collapsed_how_to_read_this_report_explainer(stubbed):
    at = _run_clicked(AppTest.from_file(_APP, default_timeout=30))

    assert "How to read this report" in [e.label for e in at.status]
    rendered = " ".join(m.value for m in at.markdown)
    assert "return-level curve" in rendered  # what the risk band means
    assert "LIMITATIONS.md" in rendered  # where the limits live


def test_a_failed_run_renders_one_sentence_and_keeps_the_traceback_off_the_page(
    stubbed, monkeypatch
):
    from tools.forecast import ForecastError

    def boom(**kwargs):
        raise ForecastError("Open-Meteo request failed: 503 Server Error")

    monkeypatch.setattr(graph_mod, "run_agent", boom)
    at = _run_clicked(AppTest.from_file(_APP, default_timeout=30))

    assert not at.exception  # a crash must never reach the page
    panel = _panel(at)
    assert panel.state == "error"

    messages = " ".join(e.value for e in at.error) + panel.label
    assert "Open-Meteo" in messages
    assert "try again" in messages.lower()
    assert "Traceback" not in messages and "503" not in messages
    assert not at.subheader  # nothing pretends a report was produced


def test_refusal_path_renders_as_refusal_not_risk(stubbed, monkeypatch):
    monkeypatch.setattr(
        graph_mod, "run_agent",
        lambda **kw: _report(risk_level=None, citations=[], drivers=[],
                             summary="", confidence=0.0,
                             refusal="wind risk is not supported yet"),
    )
    at = _run_clicked(AppTest.from_file(_APP, default_timeout=30))

    assert not at.exception
    assert any("Refused" in e.value for e in at.error)
    assert not at.subheader  # no risk badge on a refusal


def test_suggestion_chip_synchronizes_location_state(stubbed):
    """Clicking a suggestion chip synchronizes coordinates and query text."""
    at = AppTest.from_file(_APP, default_timeout=30)
    at.run()

    berlin_btn = next(b for b in at.button if "Berlin" in b.label)
    berlin_btn.click().run()

    assert not at.exception
    assert at.session_state["lat_input"] == pytest.approx(52.52)
    assert at.session_state["lon_input"] == pytest.approx(13.40)
    assert at.session_state["place_name"] == "Berlin"
    assert at.session_state["place_country"] == "Germany"
    assert "Berlin" in at.session_state["nl_query"]


def test_sidebar_preset_location_synchronizes_coordinates(stubbed):
    """Clicking an example city button in the sidebar moves the coordinates."""
    at = AppTest.from_file(_APP, default_timeout=30)
    at.run()

    at.sidebar.button(key="ex_Berlin, Germany").click().run()

    assert not at.exception
    assert at.session_state["lat_input"] == pytest.approx(52.52)
    assert at.session_state["lon_input"] == pytest.approx(13.40)
    assert at.session_state["place_name"] == "Berlin"


def test_header_renders_brand_logo_and_eyebrow(stubbed):
    """Header integrates brand logo and decision-grade badge."""
    at = AppTest.from_file(_APP, default_timeout=30)
    at.run()

    assert not at.exception
    markdown_content = " ".join(m.value for m in at.markdown)
    assert "cra-brand-logo" in markdown_content
    assert "Climate-Risk Analyst Agent" in markdown_content
    assert "DECISION-GRADE CLIMATE INTELLIGENCE" in markdown_content


def test_place_search_synchronizes_coordinates(stubbed, monkeypatch):
    """Searching for a place via the sidebar updates coordinates and place state."""
    from agent.location import ResolvedPlace

    monkeypatch.setattr(
        location_mod,
        "resolve_place",
        lambda query: ResolvedPlace(
            name="Berlin", country="Germany", latitude=52.52, longitude=13.40
        ),
    )

    at = AppTest.from_file(_APP, default_timeout=30)
    at.run()

    at.sidebar.text_input(key="place_query").input("Berlin").run()
    find_btn = next(b for b in at.sidebar.button if "Find place" in b.label)
    find_btn.click().run()

    assert not at.exception
    assert at.session_state["lat_input"] == pytest.approx(52.52)
    assert at.session_state["lon_input"] == pytest.approx(13.40)
    assert at.session_state["place_name"] == "Berlin"


def test_plain_language_ask_synchronizes_coordinates(stubbed, monkeypatch):
    """Running a plain-language query synchronizes coordinates before generating report."""
    from agent.location import ResolvedPlace

    monkeypatch.setattr(
        location_mod,
        "resolve_place",
        lambda query: ResolvedPlace(
            name="Berlin", country="Germany", latitude=52.52, longitude=13.40
        ),
    )

    at = AppTest.from_file(_APP, default_timeout=30)
    at.run()

    at.text_input(key="nl_query").input("How risky are heatwaves in Berlin over the next 7 days?").run()
    ask_btn = next(b for b in at.button if "Ask" in b.label)
    ask_btn.click().run()

    assert not at.exception
    assert at.session_state["lat_input"] == pytest.approx(52.52)
    assert at.session_state["lon_input"] == pytest.approx(13.40)
    assert at.session_state["place_name"] == "Berlin"


