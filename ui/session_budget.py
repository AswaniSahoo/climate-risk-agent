"""Per-session cap on reports that need new model calls (the public UI).

The daily budget in rag/spend_guard.py is the hard stop for the whole
deployment. This cap is the fair share: once one browser session has run
CRG_SESSION_FRESH_REPORTS reports that needed new model calls, its later
reports still run, with model calls paused. An answer already cached keeps its
citations; anything that needs a new call takes the degraded path, and the
report says why (agent/graph.py). A reload starts a new session, so this stops
one tab from spending the day's budget by accident, not a determined client.

Kept out of ui/app.py so the page needs only one `with` per run, and imported
at boot, so it must stay light: rag/spend_guard.py pulls in no SDK.
"""
from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

from rag import spend_guard

# st.session_state key: how many of this session's reports made a model call.
USED_KEY = "_crg_fresh_reports_used"


@contextmanager
def fresh_report_scope(session_state: MutableMapping[str, Any]) -> Iterator[spend_guard.CallScope]:
    """Run one report under this session's cap.

    A report counts once it has reserved at least one model call; one served
    entirely from caches does not count, and neither does one whose calls were
    all refused.
    """
    limit = spend_guard.session_fresh_report_limit()
    used = int(session_state.get(USED_KEY, 0) or 0)
    blocked = None if limit is None or used < limit else spend_guard.session_limit_reason(limit)
    with spend_guard.call_scope(blocked=blocked) as scope:
        try:
            yield scope
        finally:
            if scope.calls:
                session_state[USED_KEY] = used + 1
