"""The startup warm-up: the report path's heavy imports, once per process, in the background.

The race is the part that matters. A session that asks for a report while the
warm-up is still importing must wait for that import and then work exactly as
it would have without the warm-up. That race cannot be staged in this process
(every module is long imported by the rest of the suite), so it runs in a fresh
interpreter, with the warm-up's own imports slowed down on purpose.
"""
import ast
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import agent.warmup as warmup

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fresh(monkeypatch):
    """A process that has not started its warm-up yet."""
    monkeypatch.setattr(warmup, "_thread", None)
    monkeypatch.setattr(warmup, "_finished", None)
    monkeypatch.delenv("CRG_WARMUP", raising=False)


def test_the_report_path_is_what_it_imports():
    assert warmup.MODULES == ("agent.graph", "agent.nl", "scipy.stats")


def test_one_daemon_thread_per_process_however_often_it_is_asked(fresh):
    first = warmup.start(("json", "csv"))
    again = warmup.start(("json", "csv"))
    first.join(timeout=30)

    assert first is again and first.daemon
    assert warmup.status().startswith("done in ")


def test_a_failed_import_is_logged_and_the_others_still_load(fresh, caplog):
    with caplog.at_level(logging.INFO, logger="agent.warmup"):
        warmup.start(("no_such_module_for_the_warm_up", "json")).join(timeout=30)

    assert "no_such_module_for_the_warm_up" in caplog.text and "imports it itself" in caplog.text
    assert "imported json" in caplog.text


def test_it_can_be_switched_off(fresh, monkeypatch):
    monkeypatch.setenv("CRG_WARMUP", "off")

    assert warmup.start(("json",)) is None
    assert warmup.status() == "off"


_RACE = r"""
import importlib.abc, json, sys, threading, time

class SlowWarmUp(importlib.abc.MetaPathFinder):
    # Only the warm-up thread is slowed, and only inside the two imports a report needs.
    def find_spec(self, name, path=None, target=None):
        if threading.current_thread().name == "warm-up" and name in (
                "langgraph.graph", "scipy.stats._continuous_distns"):
            time.sleep(0.8)
        return None

sys.meta_path.insert(0, SlowWarmUp())
import agent.warmup as warmup
thread = warmup.start()
time.sleep(0.1)  # let the warm-up get inside its first import
alive_at_report = thread.is_alive()
t0 = time.perf_counter()
from agent import graph  # what the sidebar's report does first
graph_ready = callable(graph.run_agent)
from tools.hazard_stats import _fit  # what a first fit does first
fitted = _fit(json.loads(sys.argv[1]))
waited = time.perf_counter() - t0
thread.join(timeout=60)
print(json.dumps({"alive_at_report": alive_at_report, "graph_ready": graph_ready,
                  "fitted": fitted, "waited": waited, "status": warmup.status()}))
"""


def test_a_report_that_arrives_mid_warm_up_waits_for_it_and_gets_the_same_numbers():
    from tools.hazard_stats import _fit

    seed = json.loads((REPO_ROOT / "tools" / "cache_seed" / "climatology.json").read_text("utf-8"))
    maxima = next(r for r in seed["entries"] if r["hazard"] == "heatwave")["series"]["maxima"]
    env = {k: v for k, v in os.environ.items() if k not in ("CRG_WARMUP", "PYTEST_CURRENT_TEST")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", _RACE, json.dumps(maxima)], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    assert result["alive_at_report"], "the report did not arrive while the warm-up was importing"
    assert result["graph_ready"]
    assert tuple(result["fitted"]) == _fit(maxima)
    assert result["waited"] >= 0.5  # it waited for the slowed imports rather than racing them
    assert result["status"].startswith("done in ")


def _top_level(path: Path) -> list[ast.stmt]:
    return ast.parse(path.read_text(encoding="utf-8")).body


def _calls_warm_up(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_start_warmup"
               for n in ast.walk(node))


def _under_the_pytest_guard(node: ast.AST) -> bool:
    return isinstance(node, ast.If) and "PYTEST_CURRENT_TEST" in ast.unparse(node.test)


def test_the_ui_warms_up_once_its_page_is_drawn_and_never_under_pytest():
    body = _top_level(REPO_ROOT / "ui" / "app.py")
    warm = [i for i, node in enumerate(body) if _calls_warm_up(node)]
    drawn = max(i for i, node in enumerate(body)
                if isinstance(node, ast.If) and ast.unparse(node.test) == "report is not None")

    assert len(warm) == 1 and warm[0] > drawn
    assert _under_the_pytest_guard(body[warm[0]])


def test_the_api_warms_up_at_startup_and_never_under_pytest():
    lifespan = next(node for node in _top_level(REPO_ROOT / "api" / "app.py")
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == "_lifespan")
    guards = [n for n in ast.walk(lifespan) if _under_the_pytest_guard(n) and _calls_warm_up(n)]

    assert len(guards) == 1
