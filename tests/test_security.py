"""Security Tier 1 invariants, pinned as tests.

Each test is a control from the threat model: if a refactor weakens it, the
suite goes red — security by regression pin, not by policy document.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest

from agent.contracts import Hazard

# --- SSRF: outbound targets are hardcoded constants, never caller-supplied ---

def test_outbound_targets_are_pinned():
    from rag.gemini_client import EMBED_MODEL, GENERATE_MODEL
    from tools.climatology import ARCHIVE_URL
    from tools.forecast import OPEN_METEO_URL

    assert OPEN_METEO_URL.startswith("https://api.open-meteo.com/")
    assert ARCHIVE_URL.startswith("https://archive-api.open-meteo.com/")
    # Gemini goes through the official SDK (Google-controlled endpoints); what we
    # pin is the MODEL choice — stable GA versions, upgraded deliberately.
    assert EMBED_MODEL == "gemini-embedding-2"
    assert GENERATE_MODEL == "gemini-2.5-flash"


def test_jev_targets_and_models_are_pinned():
    """A Jev provider receives user questions when CRG_JEV is on, so where they
    can go is a reviewed list of two HTTPS hosts, not a config value. TypeSafe
    direct is the default and pins a version; the gateway grants only the
    alias (HTTP 403 model_not_granted for jev-1.14.0, probed 2026-11-11)."""
    from rag.jev_client import DEFAULT_MODEL, JEV_BASE_URL, PROVIDERS, JevClient

    assert JEV_BASE_URL == "https://api.typesafe.ai"
    assert set(PROVIDERS) == {"https://api.typesafe.ai", "https://api.experientiallabs.ai"}
    assert all(url.startswith("https://") and p.base_url == url for url, p in PROVIDERS.items())
    assert DEFAULT_MODEL == "jev-1.14.0"
    assert PROVIDERS["https://api.typesafe.ai"].default_model == "jev-1.14.0"
    assert PROVIDERS["https://api.experientiallabs.ai"].default_model == "jev-latest"
    client = JevClient("sentinel-key")
    try:
        assert str(client._http.base_url).rstrip("/") == JEV_BASE_URL
    finally:
        client.close()


def test_jev_is_off_unless_asked_for(monkeypatch):
    from rag.jev_client import jev_mode

    monkeypatch.delenv("CRG_JEV", raising=False)
    assert jev_mode() == "off"


def test_gemini_auth_is_explicit_never_a_silent_fallback(monkeypatch):
    import rag.gemini_client as gc

    gc._reset_clients()
    for var in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(gc.GeminiError):
        gc._client()
    gc._reset_clients()


# --- input validation: coordinates are range-checked at the tool boundary ---

@pytest.mark.parametrize("lat,lon", [(91.0, 0.0), (-91.0, 0.0), (0.0, 181.0), (0.0, -181.0)])
def test_forecast_rejects_out_of_range_coordinates(lat, lon):
    from tools.forecast import get_forecast

    with pytest.raises(ValueError, match="range"):
        get_forecast(lat, lon, horizon_days=3)


@pytest.mark.parametrize("lat,lon", [(90.5, 0.0), (0.0, -180.5)])
def test_climatology_rejects_out_of_range_coordinates(lat, lon):
    from tools.climatology import climatology_hazard_stat

    with pytest.raises(ValueError, match="range"):
        climatology_hazard_stat(lat, lon, Hazard.HEATWAVE)


def test_forecast_rejects_absurd_horizon():
    from tools.forecast import get_forecast

    with pytest.raises(ValueError, match="horizon"):
        get_forecast(0.0, 0.0, horizon_days=10_000)  # denial-of-wallet guard


# --- secrets: the API key can never appear in serialized output ---

def test_api_key_never_serializes_into_answers(monkeypatch):
    from rag.answer import CitedAnswer

    sentinel = "TEST-SENTINEL-KEY-a1b2c3"
    monkeypatch.setenv("GEMINI_API_KEY", sentinel)
    answer = CitedAnswer(
        answer="x", citations=["c#p1#0"], abstain=False, allowed_ids=["c#p1#0"]
    )
    assert sentinel not in answer.model_dump_json()
    assert sentinel not in json.dumps(CitedAnswer.model_json_schema())


# --- repo hygiene: no Google API key pattern in any tracked file ---

def test_no_api_key_committed_to_repo():
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    key_pattern = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
    offenders = []
    for path in tracked:
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if key_pattern.search(text):
            offenders.append(path)
    assert not offenders, f"Google API key pattern found in: {offenders}"


# --- supply chain: workflows run reviewed code with the least token ---------

_WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

# A tag such as `v4` is a pointer the action's owner can move, so every action
# is pinned to a full commit SHA, with the release it came from in a comment.
# Resolved with `gh api repos/<repo>/git/ref/tags/<tag>` on 2026-11-11 (all
# three are lightweight tags on commits). Each SHA is the commit the floating
# major tag (`v4`, `v5`) pointed at that day, so the pin changed no behaviour.
# Newer majors exist (checkout v7.0.1, setup-uv v10.3.3, upload-artifact
# v7.0.2); moving to one is a deliberate upgrade, not part of a pin.
PINNED_ACTIONS = {
    "actions/checkout": ("11d5960a326750d5838078e36cf38b85af677262", "v4.4.0"),
    "astral-sh/setup-uv": ("d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86", "v5.4.2"),
    "actions/upload-artifact": ("ea165f8d65b6e75b540449e92b4886f43607fa02", "v4.6.2"),
}

# setup-uv otherwise downloads whatever uv is newest, unverified: v5.4.2 checks
# a checksum it is given, or one from its own list, which predates this uv.
# 0.13.7 is what the last green CI run installed; the checksum is the release's
# own uv-x86_64-unknown-linux-gnu.tar.gz.sha256, which matches the digest
# GitHub reports for that asset.
UV_VERSION = "0.13.7"
UV_LINUX_X86_64_SHA256 = "c5221ff41b7892bfc223543883600125f6a47fe845093e0f5bdff4f359196d83"


def _workflows() -> list[tuple[str, str]]:
    found = [(p.name, p.read_text(encoding="utf-8")) for p in sorted(_WORKFLOWS.glob("*.yml"))]
    assert found, "no workflows found: did .github/workflows move?"
    return found


def _steps(text: str) -> list[str]:
    """Each step of a workflow as its own block of text (steps sit at six spaces)."""
    return re.split(r"\n(?=      - )", text)[1:]


def _step_name(step: str) -> str:
    match = re.match(r"      - name: ([^\n]+)", step)
    return match.group(1) if match else step.splitlines()[0].strip()


def test_every_action_is_pinned_to_a_full_commit_sha():
    uses = [
        (name, ref, tail)
        for name, text in _workflows()
        for ref, tail in re.findall(r"uses:\s*(\S+)([^\n]*)", text)
    ]
    assert uses, "no actions found"
    for name, ref, tail in uses:
        action, _, sha = ref.partition("@")
        assert action in PINNED_ACTIONS, f"{name}: {action} is not a reviewed action"
        want_sha, want_tag = PINNED_ACTIONS[action]
        assert sha == want_sha, f"{name}: {ref} is not pinned to {want_sha}"
        assert tail.strip() == f"# {want_tag}", f"{name}: {ref} lacks the '# {want_tag}' comment"


def test_every_workflow_defaults_the_token_to_read_only():
    """A job that needs more asks at job level, with its reason, or it reads only."""
    for name, text in _workflows():
        assert re.search(r"^permissions:\n  contents: read\n", text, re.MULTILINE), (
            f"{name} has no top-level `permissions: contents: read`"
        )


def test_checkout_never_leaves_the_token_in_git_config():
    """No job pushes, so none needs the token persisted into .git/config."""
    for name, text in _workflows():
        checkouts = [s for s in _steps(text) if "uses: actions/checkout@" in s]
        assert checkouts, f"{name} has no checkout step"
        for step in checkouts:
            assert "persist-credentials: false" in step, f"{name}: checkout keeps the token"


def test_setup_uv_installs_a_pinned_and_verified_uv():
    for name, text in _workflows():
        for step in (s for s in _steps(text) if "uses: astral-sh/setup-uv@" in s):
            assert f'version: "{UV_VERSION}"' in step, f"{name}: uv version is not pinned"
            assert f'checksum: "{UV_LINUX_X86_64_SHA256}"' in step, f"{name}: uv is not verified"


def test_ci_installs_exactly_the_lock():
    """`--locked` fails the job when uv.lock no longer matches pyproject.toml."""
    syncs = re.findall(r"uv sync[^\n]*", dict(_workflows())["ci.yml"])
    assert syncs and all("--locked" in s for s in syncs), syncs


def test_the_gemini_key_reaches_only_the_steps_that_call_gemini():
    """Job-level env hands a secret to every step, third-party actions included.
    Only the presence check and the four eval steps that call the model get it."""
    workflows = dict(_workflows())
    text = workflows["evals.yml"]
    head, _, _ = text.partition("    steps:")
    assert "secrets.GEMINI_API_KEY" not in head, "GEMINI_API_KEY is set above step level"
    holders = {_step_name(s) for s in _steps(text) if "secrets.GEMINI_API_KEY" in s}
    assert holders == {
        "Require the GEMINI_API_KEY secret",
        "Retrieval eval (dev)",
        "End-to-end eval (dev)",
        "Retrieval eval (held-out test set)",
        "End-to-end eval (held-out test set)",
    }
    for name, other in workflows.items():
        if name != "evals.yml":
            assert "secrets." not in other, f"{name} reads a secret"


def test_the_eval_job_runs_without_the_daily_spend_limit():
    """rag/spend_guard.py would refuse calls partway through 165 questions, and
    the refusals would score as answers the model got wrong."""
    text = dict(_workflows())["evals.yml"]
    head, _, _ = text.partition("    steps:")
    assert 'CRG_DAILY_MODEL_BUDGET_USD: "off"' in head


def test_dependency_audit_reads_the_lock_not_its_own_environment():
    """`uvx pip-audit` with no input audits the throwaway environment uvx built
    for pip-audit itself, so the job passed whatever the project locked. The
    audit has to read uv.lock: exported with every group and its hashes, then
    checked as pinned, hashed requirements with no resolver in the loop."""
    text = dict(_workflows())["security.yml"]
    export = re.search(r"uv export [^\n]*", text)
    assert export, "the lock is never exported"
    for flag in ("--frozen", "--all-groups", "--no-emit-project",
                 "--format requirements.txt", "-o requirements-audit.txt"):
        assert flag in export.group(0), f"uv export lacks {flag}"
    audit = re.search(r"uvx --from pip-audit==\d+\.\d+\.\d+ pip-audit [^\n]*", text)
    assert audit, "pip-audit does not run from a pinned version"
    for flag in ("--strict", "--require-hashes", "--disable-pip", "-r requirements-audit.txt"):
        assert flag in audit.group(0), f"pip-audit lacks {flag}"


def test_every_ignored_advisory_says_why_it_cannot_reach_us():
    """An ignore is a claim about this codebase, so each one carries it. Only
    pypdf is ignored, on one ground that the code itself enforces: pypdf reads
    nothing but the IPCC PDFs pinned in rag/corpus.py, checked before parsing,
    and each of these advisories needs an attacker-crafted PDF."""
    from rag.corpus import CORPUS_FILES, CORPUS_SHA256

    text = dict(_workflows())["security.yml"]
    block = re.search(r"\n\s+ignored=\(\n(.*?)\n\s+\)\n", text, re.S)
    assert block, "no ignore list in security.yml"
    entries = [line.strip() for line in block.group(1).splitlines()
               if line.strip() and not line.strip().startswith("#")]
    assert entries, "an empty ignore list should be deleted, not kept"
    for entry in entries:
        advisory, _, reason = entry.partition("#")
        assert re.fullmatch(r"(PYSEC|CVE|GHSA)-[\w-]+", advisory.strip()), entry
        assert "pypdf" in reason and "pinned IPCC PDFs" in reason, f"no reason: {entry}"
    assert len({e.partition('#')[0].strip() for e in entries}) == len(entries), "duplicate ID"
    assert '"${ignored[@]/#/--ignore-vuln=}"' in text, "the list never reaches pip-audit"
    assert set(CORPUS_SHA256) == set(CORPUS_FILES)  # the ground the reason stands on


def test_no_workflow_can_reach_vertex():
    """rag/gemini_client.py goes to Vertex only when GOOGLE_GENAI_USE_VERTEXAI is
    true and Google credentials exist. No workflow sets either."""
    for name, text in _workflows():
        for marker in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT",
                       "GOOGLE_APPLICATION_CREDENTIALS", "google-github-actions/auth",
                       "workload_identity_provider"):
            assert marker not in text, f"{name} mentions {marker}"


# --- containers: non-root at runtime, nothing secret in the build context ---

_ROOT = Path(__file__).resolve().parents[1]
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)


def _instructions(dockerfile: str) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) per line, continuation lines joined."""
    text = (_ROOT / dockerfile).read_text(encoding="utf-8").replace("\\\n", " ")
    return [
        (m.group(1).upper(), m.group(2).strip())
        for m in re.finditer(r"^\s*([A-Za-z]+)\s+(.*)$", text, re.MULTILINE)
        if not m.group(0).lstrip().startswith("#")
    ]


@pytest.mark.parametrize("dockerfile", ["Dockerfile", "Dockerfile.mcp"])
def test_the_runtime_stage_runs_as_a_named_non_root_user(dockerfile):
    """The last USER after the last FROM is what the container runs as. The MCP
    image says it again rather than trusting whatever image BASE points at."""
    steps = _instructions(dockerfile)
    last_from = max(i for i, (op, _) in enumerate(steps) if op == "FROM")
    users = [arg for op, arg in steps[last_from:] if op == "USER"]
    assert users, f"{dockerfile}: the runtime stage never sets USER"
    assert users[-1] == "appuser", f"{dockerfile}: runs as {users[-1]!r}"


@pytest.mark.parametrize("dockerfile", ["Dockerfile", "Dockerfile.mcp"])
def test_no_secret_is_baked_in_through_env_or_build_args(dockerfile):
    """Credentials arrive at run time (env or Secret Manager), never at build
    time: an ENV or ARG value is stored in the image config and its history."""
    for op, arg in _instructions(dockerfile):
        if op in {"ENV", "ARG"}:
            names = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=", arg) or [arg.split()[0]]
            leaked = [n for n in names if _SECRET_NAME.search(n)]
            assert not leaked, f"{dockerfile}: {op} carries {leaked}"
        if op in {"COPY", "ADD"}:
            assert ".env" not in arg.split(), f"{dockerfile}: {op} copies an env file"


def test_dockerignore_keeps_secrets_local_state_and_git_out_of_the_image():
    """`COPY . .` ships whatever the context holds, so the ignore file is the
    control. Patterns match from the context root; `**/` reaches subdirectories."""
    patterns = {
        line.strip()
        for line in (_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    required = {
        # version control and credentials
        ".git", ".env", ".env.*", "**/.env", "**/.env.*", "**/secrets.toml",
        "**/*.pem", "**/*.key", "**/*credentials*.json", "**/.mcp_publisher_token",
        "**/.mcpregistry_*",
        # local caches and state
        ".venv", "venv", "**/__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
        "data/cache/kv", "data/telemetry", ".firecrawl",
        # not needed at runtime (nothing under agent, rag, tools, ui, api or obs
        # imports or reads them), and .gcloudignore already leaves them out
        "tests", "evals",
    }
    assert not required - patterns, f".dockerignore lacks {sorted(required - patterns)}"
    assert "data" not in patterns, "data/ must stay in: it carries the embedding cache"


def test_gcloudignore_keeps_secrets_out_of_the_cloud_build_upload():
    """`gcloud run deploy --source .` uploads the context to a Cloud Build bucket
    before .dockerignore is ever read. gitignore syntax: no slash, any depth."""
    patterns = {
        line.strip()
        for line in (_ROOT / ".gcloudignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    required = {".git", ".env", ".env.*", "secrets.toml", "*.pem", "*.key",
                "*credentials*.json", ".mcp_publisher_token", ".mcpregistry_*",
                "data/cache/kv", "data/telemetry", ".venv", "venv"}
    assert not required - patterns, f".gcloudignore lacks {sorted(required - patterns)}"

