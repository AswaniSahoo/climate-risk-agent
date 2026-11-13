"""What a reader can take away from a report: its JSON file, and a way to flag a wrong number.

Both are plain values for ui/app.py to hand to Streamlit. Neither one calls a
model or reruns the report: the file is the RiskReport the page already holds,
and the flag is a link to the repo's data-correctness issue template, titled
with the report it is about so the issue names what was wrong where.
"""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlencode

from agent.contracts import RiskReport

ISSUES_NEW_URL = "https://github.com/AswaniSahoo/climate-risk-agent/issues/new"
TEMPLATE = "data_correctness.md"  # .github/ISSUE_TEMPLATE/data_correctness.md
_TITLE_PLACE_CHARS = 80  # a resolved place name; never a pasted paragraph


def _slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")[:60].strip("-")


def report_file_name(report: RiskReport) -> str:
    """`risk-report-rourkela-india-heatwave-7d.json`: the place, hazard and horizon.

    A refusal is named `risk-report-refusal.json`: on an instant refusal the
    report's location is the reader's own question, which stays out of the name.
    """
    if report.refusal is not None or report.hazard is None:
        return "risk-report-refusal.json"
    return (f"risk-report-{_slug(report.location) or 'place'}-{report.hazard.value}"
            f"-{report.horizon_days}d.json")


def data_correctness_url(report: RiskReport) -> str:
    """A new issue on the data-correctness template, titled with this report.

    For an assessed report only: a refusal has no number or citation to flag,
    and its location can be the reader's whole question (ui/app.py shows no
    flag on a refusal).
    """
    if report.refusal is not None or report.hazard is None:
        raise ValueError("a refusal has no number or citation to flag")
    place = report.location[:_TITLE_PLACE_CHARS]
    title = f"Data correctness: {report.hazard.value}, {place}, {report.horizon_days} days"
    return f"{ISSUES_NEW_URL}?{urlencode({'template': TEMPLATE, 'title': title})}"
