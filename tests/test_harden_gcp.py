"""scripts/harden_gcp.sh is prepared, not run: it must not run by accident."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "harden_gcp.sh"


def test_the_gcp_script_refuses_to_run_without_an_explicit_yes(tmp_path):
    """It writes to the project and its billing account, so its first act is to
    refuse unless HARDEN_GCP_CONFIRMED=yes. PATH is empty here, so even a broken
    guard could not reach gcloud."""
    assert "NEEDS ASWANI'S EXPLICIT YES BEFORE IT RUNS" in SCRIPT.read_text(encoding="utf-8")

    bash = shutil.which("bash")
    assert bash, "bash is needed to run this check"
    proc = subprocess.run(
        [bash, str(SCRIPT)], cwd=tmp_path, env={"PATH": str(tmp_path)},
        capture_output=True, text=True, timeout=30,
    )

    assert proc.returncode == 1
    assert "Refusing to run" in proc.stderr and "gcloud" not in proc.stdout


def test_the_script_matches_the_commands_deploy_md_shows():
    """DEPLOY.md is what gets read before the yes; the script must do what it says."""
    deploy = (SCRIPT.parents[1] / "DEPLOY.md").read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")
    for fragment in (
        "gcloud services enable secretmanager.googleapis.com",
        "--permissions=aiplatform.endpoints.predict --stage=GA",
        "--role=\"roles/secretmanager.secretAccessor\"",
        "--replication-policy=automatic --data-file=-",
        "--min-instances=0 --max-instances=2",
        "--budget-amount=900INR --credit-types-treatment=exclude-all-credits",
        "--threshold-rule=percent=0.5 --threshold-rule=percent=0.9",
        "--threshold-rule=percent=1.0",
        "HARDEN_GCP_CONFIRMED=yes",
    ):
        assert fragment in script, f"script lacks {fragment}"
        assert fragment in deploy, f"DEPLOY.md lacks {fragment}"
