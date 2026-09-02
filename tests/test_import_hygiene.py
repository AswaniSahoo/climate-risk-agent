"""Booting the UI must not pay for the report path.

Streamlit's cold start on Cloud Run is bounded by the startup probe (240 s
deadline, hit twice in production). scipy.stats, langgraph and google.genai
together cost ~50 s of cold import and none of them is needed to render the
first page — they are imported inside the functions that actually use them.
This test is the guard: a stray top-level import anywhere in the chain
`ui.app -> agent.* -> tools.* -> rag.*` puts them back and fails here.

Runs in a subprocess because sys.modules in THIS process is already polluted
by the rest of the suite.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = ("scipy", "google.genai", "langgraph")

PROBE = """
import sys
import ui.app  # noqa: F401
roots = {m.split(".")[0] for m in sys.modules}
print("scipy" if "scipy" in roots else "-")
print("google.genai" if "google.genai" in sys.modules else "-")
print("langgraph" if "langgraph" in roots else "-")
print("TOTAL", len(sys.modules))
"""


def test_importing_ui_app_pulls_no_scipy_genai_or_langgraph():
    env = dict(os.environ)
    # ui/app.py's own hermetic switch: skips the corpus download and the
    # embedding self-test, so this test needs no network and no credentials.
    env["PYTEST_CURRENT_TEST"] = "import-hygiene"
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=REPO_ROOT, env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=300,
    )
    assert proc.returncode == 0, f"import ui.app failed:\n{proc.stderr[-2000:]}"

    loaded = [line for line in proc.stdout.split() if line in FORBIDDEN]
    assert loaded == [], (
        f"import ui.app eagerly loaded {loaded} — move the import inside the "
        f"function that uses it (see tools/hazard_stats.py for the pattern)."
    )
