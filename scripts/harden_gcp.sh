#!/usr/bin/env bash
# The GCP side of the climate-risk-agent demo, kept as a script that can be run
# again (DEPLOY.md, "Hardening the GCP side"). Aswani chose Cloud Storage for
# the shared cache and approved these changes on 2026-11-11; the identity, both
# roles, the bucket and the budget were created that day. Every step describes
# before it creates, so running it again changes nothing that already exists.
#
# NEEDS ASWANI'S EXPLICIT YES BEFORE IT RUNS. It writes to project
# climate-risk-agent and to its billing account, and `candidate` and `promote`
# change what the public service runs. The guard below stays even though the
# setup is approved, so the script never runs by accident.
#
# Usage, from the repository root:
#   HARDEN_GCP_CONFIRMED=yes scripts/harden_gcp.sh setup      # identity, roles, bucket, budget
#   HARDEN_GCP_CONFIRMED=yes scripts/harden_gcp.sh candidate  # a new revision, no traffic
#   HARDEN_GCP_CONFIRMED=yes scripts/harden_gcp.sh promote    # after the candidate checks out
#
# Flags checked against the help text of gcloud 593.0.0 on 2026-11-11.
set -euo pipefail

if [[ "${HARDEN_GCP_CONFIRMED:-}" != "yes" ]]; then
  echo "Refusing to run: this changes project climate-risk-agent and its billing account." >&2
  echo "It needs Aswani's explicit yes. Read DEPLOY.md first, then run it as" >&2
  echo "  HARDEN_GCP_CONFIRMED=yes $0 setup|candidate|promote" >&2
  exit 1
fi

PROJECT=climate-risk-agent
REGION=us-central1
SERVICE=climate-risk-agent
SA_NAME=climate-risk-runtime
SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
MODEL_ROLE=crgModelInvoker
CACHE_ROLE=crgCacheReadWrite
BUCKET=climate-risk-agent-cache
BUDGET_NAME="climate-risk-agent INR 900"

step() { printf '\n== %s\n' "$*"; }
quietly() { "$@" >/dev/null 2>&1; }

# Reads an IAM policy as JSON on stdin; succeeds when ROLE is bound to MEMBER.
has_binding() {
  python3 -c '
import json, sys
policy, role, member = json.load(sys.stdin), sys.argv[1], sys.argv[2]
sys.exit(0 if any(b.get("role") == role and member in b.get("members", [])
                  for b in policy.get("bindings", [])) else 1)' "$1" "$2"
}

setup() {
  step "1. Runtime identity ${SA}"
  if quietly gcloud iam service-accounts describe "$SA" --project="$PROJECT"; then
    echo "exists"
  else
    gcloud iam service-accounts create "$SA_NAME" --project="$PROJECT" \
      --display-name="climate-risk-agent runtime (Cloud Run)"
  fi

  step "2. ${MODEL_ROLE}: aiplatform.endpoints.predict only, bound on the project"
  if quietly gcloud iam roles describe "$MODEL_ROLE" --project="$PROJECT"; then
    echo "role exists"
  else
    gcloud iam roles create "$MODEL_ROLE" --project="$PROJECT" \
      --title="Climate risk agent model invoker" \
      --description="generateContent and embedContent on Vertex AI publisher models, nothing else" \
      --permissions=aiplatform.endpoints.predict --stage=GA
  fi
  if gcloud projects get-iam-policy "$PROJECT" --format=json \
      | has_binding "projects/${PROJECT}/roles/${MODEL_ROLE}" "serviceAccount:${SA}"; then
    echo "binding exists"
  else
    gcloud projects add-iam-policy-binding "$PROJECT" \
      --member="serviceAccount:${SA}" \
      --role="projects/${PROJECT}/roles/${MODEL_ROLE}" --condition=None
  fi

  step "3. Bucket gs://${BUCKET}: ${REGION}, uniform access, no public access, no soft delete, objects deleted at 400 days"
  if quietly gcloud storage buckets describe "gs://${BUCKET}" --project="$PROJECT"; then
    echo "exists"
  else
    lifecycle=$(mktemp)
    printf '{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 400}}]}\n' > "$lifecycle"
    gcloud storage buckets create "gs://${BUCKET}" --project="$PROJECT" \
      --location="$REGION" --uniform-bucket-level-access --public-access-prevention \
      --soft-delete-duration=0 --lifecycle-file="$lifecycle"
    rm -f "$lifecycle"
  fi

  step "4. ${CACHE_ROLE}: storage.objects.get, create and delete, bound on gs://${BUCKET} only"
  if quietly gcloud iam roles describe "$CACHE_ROLE" --project="$PROJECT"; then
    echo "role exists"
  else
    gcloud iam roles create "$CACHE_ROLE" --project="$PROJECT" \
      --title="Climate risk agent cache read and write" \
      --description="Read, write and delete cache objects; bound on the cache bucket only" \
      --permissions=storage.objects.get,storage.objects.create,storage.objects.delete --stage=GA
  fi
  if gcloud storage buckets get-iam-policy "gs://${BUCKET}" --project="$PROJECT" --format=json \
      | has_binding "projects/${PROJECT}/roles/${CACHE_ROLE}" "serviceAccount:${SA}"; then
    echo "binding exists"
  else
    # A custom role made a moment ago can be refused ("does not exist in the
    # resource's hierarchy") until it has propagated, so retry for a while.
    for attempt in 1 2 3 4 5 6; do
      if gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --project="$PROJECT" \
          --member="serviceAccount:${SA}" --role="projects/${PROJECT}/roles/${CACHE_ROLE}"; then
        break
      fi
      if [[ "$attempt" == 6 ]]; then
        echo "The bucket binding still fails after 6 attempts; check the role, then run setup again." >&2
        exit 1
      fi
      echo "Binding refused (attempt ${attempt} of 6); waiting 30 s for the new role to propagate."
      sleep 30
    done
  fi

  step "5. Budget \"${BUDGET_NAME}\": gross cost of this project, alerts at 50, 90 and 100 percent"
  BILLING_ACCOUNT=$(gcloud billing projects describe "$PROJECT" --format='value(billingAccountName)')
  if [[ -z "$BILLING_ACCOUNT" ]]; then
    echo "Project ${PROJECT} has no billing account linked; no budget was created." >&2
    exit 1
  fi
  if gcloud billing budgets list --billing-account="$BILLING_ACCOUNT" --format='value(displayName)' \
      | grep -Fxq "$BUDGET_NAME"; then
    echo "exists"
  else
    # exclude-all-credits: measure gross cost. Counting credits (the default)
    # nets credit-paid spend to zero, and no alert would fire until the credit
    # is gone. A budget only notifies; it never stops spending.
    gcloud billing budgets create --billing-account="$BILLING_ACCOUNT" \
      --display-name="$BUDGET_NAME" --budget-amount=900INR \
      --credit-types-treatment=exclude-all-credits --filter-projects="projects/${PROJECT}" \
      --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 \
      --threshold-rule=percent=1.0
  fi
}

candidate() {
  # Deploy only code that carries the spend guard and the Cloud Storage cache
  # that CRG_CACHE_GCS_BUCKET configures (feat/gcs-cache).
  if [[ ! -f rag/spend_guard.py ]]; then
    echo "Run this from the repository root of a checkout that has rag/spend_guard.py." >&2
    exit 1
  fi
  if ! grep -rqs "CRG_CACHE_GCS_BUCKET" tools/; then
    echo "This checkout has no Cloud Storage cache backend (CRG_CACHE_GCS_BUCKET); merge it first." >&2
    exit 1
  fi
  step "6. A candidate revision on ${SA}: no traffic, tag candidate"
  # The first flags repeat the service's settings from DEPLOY.md's deploy command.
  gcloud run deploy "$SERVICE" --source . --region="$REGION" --project="$PROJECT" \
    --execution-environment=gen2 --port=7860 --allow-unauthenticated \
    --memory=2Gi --cpu=1 --concurrency=8 \
    --service-account="$SA" --min-instances=0 --max-instances=1 \
    --update-env-vars=CRG_CACHE_GCS_BUCKET="$BUCKET" \
    --no-traffic --tag=candidate
  echo
  echo "Check the candidate at its tagged URL (listed under Traffic below), then run: $0 promote"
  gcloud run services describe "$SERVICE" --region="$REGION" --project="$PROJECT"
}

promote() {
  step "7. Send all traffic to the checked candidate"
  gcloud run services update-traffic "$SERVICE" --region="$REGION" --project="$PROJECT" \
    --to-tags=candidate=100
}

case "${1:-}" in
  setup) setup ;;
  candidate) candidate ;;
  promote) promote ;;
  *)
    echo "usage: HARDEN_GCP_CONFIRMED=yes $0 setup|candidate|promote" >&2
    exit 2
    ;;
esac
