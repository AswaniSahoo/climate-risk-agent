"""The UI's per-session cap on reports that need new model calls
(ui/session_budget.py over rag/spend_guard.py). A plain dict stands in for
st.session_state; tests/test_ui.py checks the app wraps each run in it."""
from __future__ import annotations

import pytest

from rag import spend_guard
from ui.session_budget import USED_KEY, fresh_report_scope

FLASH = "gemini-2.5-flash"


@pytest.fixture(autouse=True)
def _budget_off(monkeypatch):
    # The session cap is under test here, not the daily budget.
    monkeypatch.setenv("CRG_DAILY_MODEL_BUDGET_USD", "off")


def _fresh_report(state) -> str:
    """One report that needs a model call; what came of that call."""
    with fresh_report_scope(state):
        try:
            spend_guard.reserve_generate(FLASH, "p")
        except spend_guard.BudgetRefused as exc:
            return str(exc)
    return "sent"


def test_the_cap_pauses_model_calls_after_n_fresh_reports(monkeypatch):
    monkeypatch.setenv("CRG_SESSION_FRESH_REPORTS", "2")
    state: dict = {}

    assert _fresh_report(state) == "sent"
    assert _fresh_report(state) == "sent"
    paused = _fresh_report(state)

    assert "limit of 2 reports" in paused and "cached" in paused
    assert state[USED_KEY] == 2  # a paused run made no call, so it is not counted


def test_a_report_served_from_cache_is_not_counted(monkeypatch):
    monkeypatch.setenv("CRG_SESSION_FRESH_REPORTS", "1")
    state: dict = {}

    for _ in range(3):
        with fresh_report_scope(state):
            pass  # every answer came from a cache: no model call was reserved

    assert state.get(USED_KEY, 0) == 0
    assert _fresh_report(state) == "sent"


def test_sessions_are_counted_separately(monkeypatch):
    monkeypatch.setenv("CRG_SESSION_FRESH_REPORTS", "1")
    first, second = {}, {}

    assert _fresh_report(first) == "sent"
    assert _fresh_report(second) == "sent"
    assert _fresh_report(first) != "sent"


@pytest.mark.parametrize("value,expected", [("off", "sent"), ("0", "paused")])
def test_off_never_pauses_and_zero_always_does(monkeypatch, value, expected):
    monkeypatch.setenv("CRG_SESSION_FRESH_REPORTS", value)

    outcome = _fresh_report({})

    assert (outcome == "sent") is (expected == "sent")
