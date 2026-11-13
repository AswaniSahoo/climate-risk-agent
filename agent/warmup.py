"""Import the report path's heavy modules on one background thread, once per process.

WHY. The first report a process serves pays for imports the first page never
needed: langgraph and the rest of agent.graph's import tree on the sidebar's
path, and scipy.stats for a place with no shipped ERA5 fit. They are imported
inside the functions that use them on purpose, so the page itself never waits
for them (tests/test_import_hygiene.py). On a Cloud Run gen2 container each file
is pulled from image storage on its first read, which turned 2,730 small vector
files into 17.5 s (DEPLOY.md), so these imports are expected to cost more there
than on a local disk; how much is not measured yet. This moves them into the
time a person spends reading the first page. ui/app.py logs the warm-up's state
at the start and at the end of every run, so the Cloud Run log shows whether a
first report ran beside it.

WHEN. Once per process, not per session: the thread is module state, and every
call after the first returns the same thread. ui/app.py calls `start` at the end
of its script run, once the page is drawn. The API calls it at startup, where
api/app.py has already imported agent.graph and agent.nl at module level, so
there it only adds scipy.stats. Not earlier in the UI: Streamlit runs no app
code before the first session connects, and a thread importing beside the first
script run's own imports would compete with drawing the first page for the one
vCPU.

SAFETY. A session that needs one of these modules while the warm-up is still
importing it waits on Python's per-module import lock, then gets the finished
module: the same module, at no extra cost over importing it itself
(tests/test_warmup.py runs that race). A failed import is logged and left to
the first report, which imports it itself and fails there as it would have.
CRG_WARMUP=off turns the warm-up off.
"""
from __future__ import annotations

import importlib
import logging
import os
import threading
import time
from collections.abc import Sequence

_log = logging.getLogger(__name__)

# agent.graph brings in langgraph; agent.nl is the plain-language front door on
# top of it; scipy.stats is the GEV fit's one heavy import.
MODULES = ("agent.graph", "agent.nl", "scipy.stats")

_lock = threading.Lock()
_thread: threading.Thread | None = None
_finished: float | None = None  # seconds the warm-up took, once it is done


def _enabled() -> bool:
    return os.environ.get("CRG_WARMUP", "on").strip().lower() != "off"


def _run(modules: Sequence[str]) -> None:
    global _finished
    started = time.perf_counter()
    loaded = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - a warm-up must never take the process down
            _log.warning("warm-up: importing %s failed (%s); the first report imports it itself",
                         name, exc)
        else:
            loaded.append(name)
    _finished = time.perf_counter() - started
    _log.info("warm-up: imported %s in %.2f s", ", ".join(loaded) or "nothing", _finished)


def start(modules: Sequence[str] = MODULES) -> threading.Thread | None:
    """Start the warm-up, once per process; later calls return the same thread."""
    global _thread
    if not _enabled():
        return None
    with _lock:
        if _thread is None:
            _thread = threading.Thread(target=_run, args=(tuple(modules),), name="warm-up",
                                       daemon=True)
            _thread.start()
        return _thread


def status() -> str:
    """For the UI's per-run log line (ui/app.py `_log_run`): off, not started, running,
    or done in N s."""
    if not _enabled():
        return "off"
    if _thread is None:
        return "not started"
    if _finished is None:
        return "running"
    return f"done in {_finished:.2f} s"
