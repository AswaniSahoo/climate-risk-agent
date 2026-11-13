"""What a reader can take away from a report: its JSON file, and a way to flag a wrong number."""
from urllib.parse import parse_qs, urlsplit

import pytest

from agent.contracts import Hazard, RiskLevel, RiskReport
from ui.report_export import data_correctness_url, report_file_name


def _report(**overrides) -> RiskReport:
    base = dict(location="Rourkela, India", hazard=Hazard.HEATWAVE, horizon_days=7,
                confidence=0.5, risk_level=RiskLevel.HIGH, summary="s")
    base.update(overrides)
    return RiskReport(**base)


def test_the_file_is_named_after_the_place_hazard_and_horizon():
    assert report_file_name(_report()) == "risk-report-rourkela-india-heatwave-7d.json"


def test_accents_and_symbols_in_a_place_name_become_plain_ascii():
    name = report_file_name(_report(location="São Paulo, Brasil (23.55°S, 46.63°W)"))

    assert name == "risk-report-sao-paulo-brasil-23-55s-46-63w-heatwave-7d.json"


def test_a_refusal_is_named_as_one_and_never_after_the_question():
    """On an instant refusal the location is the reader's own question (ui/app.py)."""
    refusal = _report(hazard=None, risk_level=None, refusal="out of scope",
                      location="How bad will floods be at 12 Park Street next week?")

    assert report_file_name(refusal) == "risk-report-refusal.json"


def test_a_refusal_has_no_flag_link():
    refusal = _report(risk_level=None, refusal="out of scope")

    with pytest.raises(ValueError):
        data_correctness_url(refusal)


def test_a_long_place_is_cut_short_in_the_issue_title():
    title = parse_qs(urlsplit(data_correctness_url(_report(location="x" * 500))).query)["title"]

    assert title == [f"Data correctness: heatwave, {'x' * 80}, 7 days"]


def test_the_flag_link_opens_the_data_correctness_template_titled_with_the_report():
    url = data_correctness_url(_report())
    parts = urlsplit(url)
    query = parse_qs(parts.query)

    assert parts.netloc == "github.com"
    assert parts.path == "/AswaniSahoo/climate-risk-agent/issues/new"
    assert query["template"] == ["data_correctness.md"]
    assert query["title"] == ["Data correctness: heatwave, Rourkela, India, 7 days"]
