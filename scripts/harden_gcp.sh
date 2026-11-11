#!/usr/bin/env bash
# Harden the GCP side of the climate-risk-agent demo (DEPLOY.md, "Hardening
# the GCP side"): a dedicated runtime identity holding one permission, the
# Upstash token in Secret Manager, a redeploy that carries the spend guard, and
# a budget alert on the billing account.
#
# NEEDS ASWANI'S EXPLICIT YES BEFORE IT RUNS. Every step writes to project
# climate-risk-agent or to its billing account: it enables an API, creates a
# service account, a custom role, IAM bindings, a secret and a budget, and
# deploys a new revision of the public service. Nothing in it has been run.
#
# After that yes, run it from the root of a checkout that has the spend guard:
#   HARDEN_GCP_CONFIRMED=yes scripts/harden_gcp.sh
#
# Flags checked against the help text of gcloud 593.0.0 on 2026-11-11.
set -euo pipefail

if [[ "${HARDEN_GCP_CONFIRMED:-}" != "yes" ]]; then
  echo "Refusing to run: this changes project climate-risk-agent and its billing account." >&2
  echo "It needs Aswani's explicit yes. Read DEPLOY.md first, then run it as" >&2
  echo "  HARDEN_GCP_CONFIRMED=yes $0" >&2
  exit 1
fi

PROJECT=climate-risk-agent
REGION=us-central1
SERVICE=climate-risk-agent
SA_NAME=climate-risk-runtime
SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
ROLE_ID=crgModelInvoker
SECRET=upstash-redis-rest-token

step() { printf '\n== %s\n' "$*"; }

# Deploy only code that carries the spend guard.
if [[ ! -f rag/spend_guard.py ]]; then
  echo "Run this from the repository root of a checkout that has rag/spend_guard.py." >&2
  exit 1
fi

step "1. Enable Secret Manager (not enabled on the project as of 2026-11-11)"
gcloud services enable secretmanager.googleapis.com --project="$PROJECT"

step "2. Create the runtime service account ${SA}"
gcloud iam service-accounts create "$SA_NAME" --project="$PROJECT" \
  --display-name="climate-risk-agent runtime (Cloud Run)"

step "3. A custom role with only aiplatform.endpoints.predict, bound to ${SA}"
gcloud iam roles create "$ROLE_ID" --project="$PROJECT" \
  --title="Climate risk agent model invoker" \
  --description="generateContent and embedContent on Vertex AI publisher models, nothing else" \
  --permissions=aiplatform.endpoints.predict --stage=GA
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SA}" \
  --role="projects/${PROJECT}/roles/${ROLE_ID}" --condition=None

step "4. The Upstash token into Secret Manager (leave both prompts empty to skip Upstash)"
read -r -p "Upstash REST URL (https://<db>.upstash.io, not a secret): " UPSTASH_URL
read -rs -p "Upstash REST token (input hidden): " UPSTASH_TOKEN
echo
SECRET_FLAGS=()
ENV_VARS="CRG_DAILY_MODEL_BUDGET_USD=0.15,CRG_SESSION_FRESH_REPORTS=5"
if [[ -n "$UPSTASH_URL" && -n "$UPSTASH_TOKEN" ]]; then
  # Piped, so the token never lands in shell history, a file or a process list.
  printf '%s' "$UPSTASH_TOKEN" | gcloud secrets create "$SECRET" --project="$PROJECT" \
    --replication-policy=automatic --data-file=-
  gcloud secrets add-iam-policy-binding "$SECRET" --project="$PROJECT" \
    --member="serviceAccount:${SA}" --role="roles/secretmanager.secretAccessor"
  # Pinned to version 1: env vars resolve at instance start, and Google
  # recommends a fixed version over `latest` for them.
  SECRET_FLAGS=(--update-secrets="UPSTASH_REDIS_REST_TOKEN=${SECRET}:1")
  ENV_VARS="UPSTASH_REDIS_REST_URL=${UPSTASH_URL},${ENV_VARS}"
else
  echo "No Upstash: the spend guard counts per instance (at most 2 instances below)."
fi
unset UPSTASH_TOKEN

step "5. Redeploy on the new identity: spend limits, no minimum instance, at most 2"
# The first flags repeat the service's settings from DEPLOY.md's deploy command.
gcloud run deploy "$SERVICE" --source . --region="$REGION" --project="$PROJECT" \
  --execution-environment=gen2 --port=7860 --allow-unauthenticated \
  --memory=2Gi --cpu=1 --concurrency=8 \
  --service-account="$SA" \
  ${SECRET_FLAGS[@]+"${SECRET_FLAGS[@]}"} \
  --update-env-vars="$ENV_VARS" \
  --min-instances=0 --max-instances=2

step "6. Budget alert: INR 900 on the billing account, at 50, 90 and 100 percent"
BILLING_ACCOUNT=$(gcloud billing projects describe "$PROJECT" --format='value(billingAccountName)')
if [[ -z "$BILLING_ACCOUNT" ]]; then
  echo "Project ${PROJECT} has no billing account linked; no budget was created." >&2
  exit 1
fi
# exclude-all-credits: measure gross cost. Counting credits (the default) nets
# credit-paid spend to zero, and no alert would fire until the credit is gone.
# A budget only notifies; it never stops spending.
gcloud billing budgets create --billing-account="$BILLING_ACCOUNT" \
  --display-name="climate-risk-agent credit INR 900" \
  --budget-amount=900INR --credit-types-treatment=exclude-all-credits \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0

step "Done. Check the identity with:"
echo "  gcloud run services describe ${SERVICE} --region=${REGION} --project=${PROJECT} \\"
echo "    --format='value(spec.template.spec.serviceAccountName)'"
