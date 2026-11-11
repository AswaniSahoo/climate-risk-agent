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
    """DEPLOY.md is what gets read before the yes; the script must do what it says.
    The resources are the ones created on 2026-11-11 (Cloud Storage, not Upstash)."""
    deploy = (SCRIPT.parents[1] / "DEPLOY.md").read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")
    for fragment in (
        "HARDEN_GCP_CONFIRMED=yes",
        "--permissions=aiplatform.endpoints.predict --stage=GA",
        "--permissions=storage.objects.get,storage.objects.create,storage.objects.delete",
        "--uniform-bucket-level-access --public-access-prevention",
        "--soft-delete-duration=0",
        '{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 400}}]}',
        "gcloud storage buckets add-iam-policy-binding",
        "--budget-amount=900INR",
        "--credit-types-treatment=exclude-all-credits",
        "--filter-projects=",
        "--threshold-rule=percent=0.5 --threshold-rule=percent=0.9",
        "--threshold-rule=percent=1.0",
        "--min-instances=0 --max-instances=1",
        "CRG_CACHE_GCS_BUCKET=",
        "--no-traffic --tag=candidate",
        "--to-tags=candidate=100",
    ):
        assert fragment in script, f"script lacks {fragment}"
        assert fragment in deploy, f"DEPLOY.md lacks {fragment}"


def test_every_create_is_preceded_by_a_describe():
    """Running it again must change nothing that already exists."""
    script = SCRIPT.read_text(encoding="utf-8")
    for describe, create in (
        ("gcloud iam service-accounts describe", "gcloud iam service-accounts create"),
        ("gcloud iam roles describe \"$MODEL_ROLE\"", "gcloud iam roles create \"$MODEL_ROLE\""),
        ("gcloud iam roles describe \"$CACHE_ROLE\"", "gcloud iam roles create \"$CACHE_ROLE\""),
        ("gcloud projects get-iam-policy", "gcloud projects add-iam-policy-binding"),
        ("gcloud storage buckets describe", "gcloud storage buckets create"),
        ("gcloud storage buckets get-iam-policy", "gcloud storage buckets add-iam-policy-binding"),
        ("gcloud billing budgets list", "gcloud billing budgets create"),
    ):
        assert describe in script and create in script, (describe, create)
        assert script.index(describe) < script.index(create), f"{create} before {describe}"


def test_the_bucket_binding_retries_while_the_new_role_propagates():
    """On 2026-11-11 the first binding failed with "does not exist in the
    resource's hierarchy" because the new custom role had not propagated."""
    script = SCRIPT.read_text(encoding="utf-8")
    binding = script.index("gcloud storage buckets add-iam-policy-binding")
    loop = script.rfind("for attempt in", 0, binding)
    assert loop != -1 and "sleep" in script[binding: binding + 900]


def test_no_upstash_and_no_secret_manager_step():
    """Aswani chose Cloud Storage over Upstash; there is no secret to mount."""
    script = SCRIPT.read_text(encoding="utf-8").lower()
    for gone in ("upstash", "secretmanager", "gcloud secrets", "--update-secrets"):
        assert gone not in script, gone
