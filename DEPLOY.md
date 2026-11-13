# Deploy: Docker + Google Cloud Run

## Local Docker

```bash
docker build -t climate-risk-agent .
docker run -p 7860:7860 climate-risk-agent                          # BM25-only (no key)
docker run -p 7860:7860 -e GEMINI_API_KEY=... climate-risk-agent    # hybrid + cited answers
```

Open http://localhost:7860.

- The build bakes the IPCC corpus (~50 MB, downloaded from ipcc.ch, idempotent).
- **The build fails unless the image carries every corpus vector.** It wipes
  whatever `data/cache` the build context holds, fetches the `eval-cache-v1`
  release asset pinned by sha256 (`ARG EVAL_CACHE_SHA256` in the Dockerfile; a
  mismatch fails the build), installs it through `scripts/unpack_eval_cache.py`
  (which refuses a manifest that does not match this repo), packs the vectors
  into one file, `data/cache/embeddings.pack.npy` (`scripts/pack_vectors.py`;
  the runtime reads it instead of 2,730 vector files, which took 17.5 s on
  Cloud Run gen2), and its last step, `scripts/check_baked_index.py`, fails the
  build unless every chunk in the baked `chunks.json` has its document vector
  on disk (2,730 of 2,730 today) and the same vector in the pack, bit for bit,
  so no cold container ever embeds the corpus through the API.
- Without Gemini auth the retrieval layer falls back to BM25-only **loudly**
  (measured on the 60-question dev set: 76% vs 82% headline R@3) and `answer_ipcc`-style LLM answers are
  unavailable; the UI reports both degradations honestly.

## Google Cloud Run (live demo)

The public demo runs here:
**https://climate-risk-agent-714882950125.us-central1.run.app/**

Cloud Run serves the same Docker image described above, so its embedding
cache (~9 MB) comes from the pinned release asset, not from the Cloud Build
upload, and the container starts on hybrid retrieval instead of re-embedding
the corpus (and hitting 429s) on every cold start.

Auth and models run on **Vertex AI via the `global` endpoint** (ADC through the
Cloud Run service account, so there is no API key to manage):

```bash
# Deploy from source; Cloud Build runs the docker build, then Cloud Run hosts it.
gcloud run deploy climate-risk-agent \
  --source . \
  --region us-central1 \
  --execution-environment gen2 \
  --port 7860 \
  --allow-unauthenticated \
  --min-instances 0 \
  --memory 2Gi --cpu 1 --concurrency 8 \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=climate-risk-agent,GOOGLE_CLOUD_LOCATION=global,CRG_GENERATE_MODEL=gemini-2.5-flash,CRG_EMBED_MODEL=gemini-embedding-2
```

- **`GOOGLE_CLOUD_LOCATION=global` is required.** `gemini-embedding-2` is served
  on `global` / `us` / `eu`, **not** the single region `us-central1`; a regional
  endpoint 404s the embedding call and the app silently drops to BM25-only.
  `gemini-2.5-flash` is also served on `global`.
- **`--execution-environment gen2` is required.** On 2026-10-22 a rebuild on a
  newer python:3.11-slim (Python 3.11.17, glibc 2.41-12+deb13u4) segfaulted
  (`Uncaught signal: 11`) right after geocoding on the default gen1 (gVisor)
  sandbox, while the identical image ran the full Berlin report on gen2 and
  locally. The Dockerfile now pins both base images by digest.
- `--region us-central1` is where the *Cloud Run service* (the container host)
  runs, independent of the Vertex model endpoint (`global`).
- **`--min-instances 0`** lets the service scale to zero, so idle time costs
  about $0 (see Cost below). The first request after idle pays a cold start,
  whose length is not recorded in this repo (`scripts/measure_coldstart.py`
  times one).
- Cold start itself is now attacked at the image, not the instance count: the
  Dockerfile is two-stage (no uv or package cache in the runtime layer) and the
  UI boots without scipy, langgraph or google-genai, which are imported only
  when a report is actually requested.
- `--port 7860` matches the container's `EXPOSE` / Streamlit port.
- The command above sets no `--service-account`, so the service runs as the
  Compute Engine default service account (it does today: checked read-only on
  2026-11-11). The code needs one permission, `aiplatform.endpoints.predict`;
  "Hardening the GCP side" below moves the service to a dedicated identity
  that holds only that. `roles/aiplatform.user` also works, but it carries
  464 permissions.
- The startup self-test logs `DENSE DEGRADED` (and the UI shows a banner) if the
  embedding endpoint is unreachable: a misconfig is loud, never silent.
- `CRG_BOOTSTRAP_N` (optional) sets the GEV bootstrap refits per fit for both the
  stationary (default 300) and trend-adjusted (default 200) paths. It is read
  once at import and is part of the hazard-fit cache key. A higher value buys
  tighter confidence intervals with fit time; the measured floor is 150, below
  which the tail band collapses. `scripts/prewarm.py` prints the value it
  resolved.
- `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` are **optional**. Set
  both to share one cache across replicas; with neither set every instance
  keeps its own disk cache and `tools/cache_backend.py` logs which backend it
  chose. On 2026-11-11 the live service carried neither name, so at
  `--min-instances 0` each instance's disk cache dies with it; only the entries
  shipped in the image (below) and a bucket set by `CRG_CACHE_GCS_BUCKET` (next
  item) outlive an instance. The Upstash pair also gives
  the spend guard one budget for every instance (see "Spend guard"); the
  token is a secret, so it never goes in `--set-env-vars`.
- **Shared cache on Cloud Storage: option A, chosen over Upstash by Aswani
  (2026-11-11).** Set `CRG_CACHE_GCS_BUCKET=climate-risk-agent-cache` on the
  service. Reads go shipped seed, then this instance's disk, then the bucket; a
  bucket hit fills the disk; writes go to disk and bucket; each object carries
  its expiry, enforced on read; any bucket error is logged, the bucket is left
  alone for 60 s, and the report runs on the tiers that answer. With the setting
  absent, the Upstash pair above (or disk alone) applies as before. The code
  talks to the JSON API through httpx with a google-auth token (no new
  dependency); `tools/cache_backend.py` quotes the API contract it relies on.
  Resources created on 2026-11-11 with Aswani's yes, not to be changed from here:
  - bucket `gs://climate-risk-agent-cache`: us-central1, uniform access, public
    access prevention, soft delete off, objects deleted at 400 days of age
    (longer than any cache TTL, which tops out at 365 days);
  - runtime service account
    `climate-risk-runtime@climate-risk-agent.iam.gserviceaccount.com`, holding
    on that bucket only a custom role with exactly `storage.objects.get`,
    `storage.objects.create` and `storage.objects.delete`. No list and no update:
    the code never lists, sets metadata at upload time, and overwrites under the
    delete permission (an overwrite needs it, per the objects.insert docs);
  - the service runs as that account after the next deploy, maxScale 1, min 0.
  Smoke test, 2026-11-11, with the owner's ADC under `smoke/`: write, read,
  increment twice (1 then 3), delete both, read back nothing. All passed.
- **Open-Meteo archive budget** (`tools/archive_budget.py`), in front of every
  live ERA5 request. One 1960 to 2022 request weighs about 1,644 calls against
  the free 10,000 a day and 5,000 an hour, and Cloud Run's egress addresses are
  shared. `CRG_ARCHIVE_DAILY_CALLS` (default 8,000, UTC day) and
  `CRG_ARCHIVE_HOURLY_CALLS` (default 4,000) are weighted calls; `off` disables
  one. The counts live on the bucket counter (compare-and-swap on
  `ifGenerationMatch`), counted in the process for 60 s after a bucket failure.
  Any HTTP 429 stops every archive request until the next full hour. Refused, a
  report runs without ERA5 and says so in its numbers line, for example: "ERA5
  history skipped for this place: today's free data allowance for new places is
  used up. It resets at 00:00 UTC."
- **Precomputed entries ship in the image** (`tools/cache_seed/*.json`, read
  before Redis or disk, so they survive scale to zero). Each entry sits under
  the exact cache key the live path builds, so it only answers the request and
  the code version it was computed for:
  - `scripts/precompute_climatology.py` fills `climatology.json` with ERA5 fits
    for `scripts/precompute_cities.json`. One 1960 to 2022 archive request
    counts as about 1,644 Open-Meteo calls against a free limit of 10,000 a
    day, so it runs paced and budgeted; `--plan` prints the cost first.
  - `scripts/precompute_grounding.py` fills `ipcc.json` with the IPCC answer and
    the Ch.12 projection per (hazard, AR6 region), up to 3 model calls a pair.
    `--presets` covers the six UI presets (6 pairs); no flag runs all 138.
  Both keys move when their inputs change (fit code and bootstrap counts; corpus,
  ranking code, `rag/answer.py`, `rag/prompt_safety.py`, `rag/cid.py`, model),
  so rerun the matching script after such a change and check with `--check`.
  CI fails when a shipped entry no longer matches the key the code builds
  (`tests/test_seed_freshness.py`); after a `rag/cid.py` change,
  `precompute_grounding.py --presets --only projection --prune-stale` refreshes
  the projections alone, with no answer asked again.
- `CRG_JEV`, `JEV_BASE_URL` and `JEV_API_KEY` are **optional** and Jev is off
  by default ([adr/0002](adr/0002-jev-question-triage-and-citation-support.md)).
  Turn Jev on only after `evals/run_jev_eval.py` meets that ADR's criteria.
  **`CRG_JEV=triage` is not recommended for deployment yet.** It met every
  quality gate on the dev set, and Aswani accepted its gateway latency, but the
  one-time held-out check (`evals/run_jev_heldout.py`, 2026-11-11,
  [report](evals/results/jev-triage-test-2026-11-11.md)) failed the hard gate
  of no new false refusal: one answerable background-science question (SP2-08,
  on 2020 COVID-19 emission cuts, air pollution and radiative forcing) was
  refused at the guard as air quality. The fix belongs on the dev set, and the
  default can only change after another held-out exposure passes. The commands
  below are for when it does.
  Two providers are vetted, and `JEV_BASE_URL` can only pick one of them (any
  other value is refused and Jev stays off):

  | `JEV_BASE_URL` | provider | model sent | context limit |
  | --- | --- | --- | --- |
  | unset, or `https://api.typesafe.ai` | TypeSafe direct | `jev-1.14.0` (pinned version) | 64k tokens a request |
  | `https://api.experientiallabs.ai` | Experiential Labs gateway, relaying to TypeSafe | `jev-latest` (the only Jev name it grants) | 32k tokens a request |

  The key for the chosen provider goes in `JEV_API_KEY` (`TYPESAFE_API_KEY` is
  still read as a fallback). It is a secret, so it goes through Secret Manager,
  never `--set-env-vars` (documented here, not run):

  ```bash
  # 1. Store the key, typed at a prompt so it never lands in shell history.
  read -rs KEY && printf '%s' "$KEY" | gcloud secrets create jev-api-key \
    --replication-policy=automatic --data-file=- && unset KEY

  # 2. Let the service's identity read it (the Compute Engine default service
  #    account, unless the service was deployed with --service-account).
  PROJECT_NUMBER=$(gcloud projects describe climate-risk-agent --format='value(projectNumber)')
  gcloud secrets add-iam-policy-binding jev-api-key \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

  # 3. Expose it as JEV_API_KEY: add this flag to the deploy command above.
  #    Pin a version number: env vars are resolved at instance start.
  #      --update-secrets=JEV_API_KEY=jev-api-key:1

  # 4. Choose the provider and the mode without touching the other variables.
  #    Leave out JEV_BASE_URL for TypeSafe direct.
  gcloud run services update climate-risk-agent --region us-central1 \
    --update-env-vars JEV_BASE_URL=https://api.experientiallabs.ai,CRG_JEV=triage
  ```

  To rotate, add a secret version and redeploy with its number. A refused key
  never breaks a request: the first 401, 402 or 403 pauses Jev calls for five
  minutes at a time, every caller falls back to its non-Jev behaviour, and the
  logs say so once. The gateway's free plan allows 10 requests and 250K tokens
  a minute and 1M tokens a day, shared with other models
  (https://www.experientiallabs.ai/pricing, read 2026-11-11); production does
  not pace requests, so past that rate it answers 429 and callers fall back.
  Optional tuning: `CRG_JEV_MODEL` (the provider's model by default),
  `CRG_JEV_TRIAGE_BUDGET_S` (2.5 direct, 4 through the gateway),
  `CRG_JEV_VERIFY_BUDGET_S` (6 direct, 8 through the gateway),
  `CRG_JEV_MIN_INTERVAL_S` (pacing between requests, default off), and the
  thresholds `CRG_JEV_REFUSE_P`, `CRG_JEV_HINT_P`, `CRG_JEV_NOUL_YES`,
  `CRG_JEV_NOUL_NO`, which should only move with a re-run of the eval.

## Cost

Cloud Run request-based billing charges a minimum instance at the **idle rate**
for every second of the month, whether or not it serves a request. In
`us-central1` that rate is $0.0000025 per vCPU-second and $0.0000025 per
GiB-second, so an always-on instance costs:

| min-instance shape | idle cost per instance per month |
| --- | --- |
| 2 vCPU + 4 GiB (the shape deployed on 2026-07-23) | about $39 |
| 1 vCPU + 2 GiB | about $19 |
| `--min-instances 0` | about $0 |

The service deployed on 2026-07-23 ran one min instance at the first row's
shape, a monthly floor of about $39 before a single request was served. Read on
2026-11-13 (`gcloud run services describe` and `revisions describe`), the
revision serving all traffic runs the second row's shape with no minimum
instance and maxScale 1, so its idle floor is the last row's.

Measured over 30 days on that earlier shape: 2,640 requests, memory p99 at 13%
of the 4 GiB allocation, CPU p99 at 1%. It was provisioned for a load that was
not there.

Both steps are done: the resize to the 1 vCPU + 2 GiB row, which the p99
figures already cover, and `--min-instances 0`. The condition set for the
second step, a cold start measured under 30 s (`scripts/measure_coldstart.py`
times it against a 1 vCPU / 2 GiB container), has no recorded result in this
repo, so the cold start a scale-to-zero demo pays is not known here.

## Spend guard

Every Gemini call goes through `rag/gemini_client.py`, and `rag/spend_guard.py`
holds a daily budget at that one seam for every caller: the UI, the API, the
MCP server. The defaults, the same lines as `.env.example`:

| setting | default | what it limits |
| --- | --- | --- |
| `CRG_DAILY_MODEL_BUDGET_USD` | `0.15` | US dollars of model calls per UTC day |
| `CRG_SESSION_FRESH_REPORTS` | `5` | UI reports that need new model calls, per browser session |
| `CRG_API_RATE_LIMIT` | `10` | API requests per client per window on `/report`, `/query`, `/triage` |
| `CRG_API_RATE_WINDOW_S` | `60` | that window, in seconds |

Each also takes `off`. A value that does not parse keeps the default, never
"no limit".

How it holds. Before a call is sent it reserves the most the call can cost:
the prompt at two characters a token, plus the full default thinking budget
(8,192 tokens) and 2,048 visible tokens at the output price. The call starts
only if that fits in what is left of the day, and afterwards it is charged
what its usage metadata says was billed. So the day's count stays under the
budget with calls in flight; it can pass it only by the amount one answer's
output exceeds 10,240 tokens. A refused call raises `ModelBudgetExceeded` (a
`GeminiError`), the report ships without IPCC citations exactly as on any model
failure, and its summary says why: "No new IPCC AR6 answer for this report:
The daily model budget for this deployment is used up, so no new model call
runs until it resets at 00:00 UTC. Answers already cached still carry their
citations." A session over its cap gets the same with its own sentence; a
reload starts a new session, so that cap stops one tab from spending the day
by accident, and the daily budget is the hard stop. The startup embedding
probe is exempt, and `embedding_available_once` shares one answer per process
(ten minutes when it passes, one minute when it fails), so a spent budget
never shows up as the "embedding config" banner.

Where the count lives. With `CRG_CACHE_GCS_BUCKET` set: in that Cloud Storage
bucket, one compare-and-swap on the counter object's generation per change
(`tools/cache_backend.py` `GcsCounter`), one budget for every instance, and it
survives scale to zero. Else, with `UPSTASH_REDIS_REST_URL` and
`UPSTASH_REDIS_REST_TOKEN` set: in Upstash, one `INCRBY` + `EXPIRE` transaction
per change. Without either, or for 60 s after an error of the shared counter:
in each process, so every instance has a budget of its own and a restart resets
it, and the bound below multiplies by the instances that run in a day. The
deployment sets the bucket ("Hardening the GCP side"), so its count is shared
and a cold start no longer begins the day at zero: the daily bound holds per
calendar day (UTC).

The per-client API limit (`CRG_API_RATE_LIMIT`) counts by the connecting
address uvicorn reports, and uvicorn trusts `X-Forwarded-For` only from
`FORWARDED_ALLOW_IPS` (127.0.0.1 by default in uvicorn 0.50.0). When the
API runs behind a proxy, set `FORWARDED_ALLOW_IPS` to the proxy's address,
or every caller shares one limit. The production image runs Streamlit only
(the Dockerfile's `CMD`), so this applies wherever `api/app.py` is served,
not to the live demo.

Evals run with the budget off: `evals.yml` sets `CRG_DAILY_MODEL_BUDGET_USD:
"off"`, and both eval runners default to off unless the variable is set. A
refusal halfway through an eval would be scored as wrong answers, and
`scripts/eval_gate.py` does not read `errors`.

### The arithmetic behind the defaults

Target: 30 days at the cap under about half of the INR 900 credit, so INR 450,
about US$4.80.

Prices, Vertex AI, Standard tier, at most 200K input tokens, read 2026-11-11 at
https://cloud.google.com/vertex-ai/generative-ai/pricing:

- `gemini-2.5-flash`: "Input (text, image, video) | $0.30" and "Text output
  (response and reasoning) | $2.50" per 1M tokens. Thinking bills as output.
- `gemini-embedding-2`: "Gemini Embedding 2 (Unified Multimodal, Preview) |
  Input text | ... | Online requests | $0.20" per 1M tokens.

What one answer call sends (`rag/answer.py`, `agent/graph.py`): the rules
(1,111 characters), 8 excerpts (`_IPCC_TOP_K`) of at most 1,200 characters each
(`rag/chunk.py`), and the question (166 characters at most for the templated
one the app asks; up to 1,000 from the MCP tool). With full-size excerpts that
is 11,815 characters, about 2,950 tokens at Google's "a token is equivalent to
about 4 characters" (https://ai.google.dev/gemini-api/docs/tokens), so $0.00089
of input. Output is the JSON answer plus thinking, which for `gemini-2.5-flash`
on Vertex defaults to "Auto (up to 8,192 tokens)"
(https://cloud.google.com/vertex-ai/generative-ai/docs/thinking).

| one call | basis | US$ |
| --- | --- | --- |
| answer, measured mean | dev eval of 2026-09-02: $0.204382 over 55 calls, thinking counted | 0.0037 |
| answer, worst case | 2,950 input + 8,192 thinking + 1,000 visible tokens | 0.024 |
| query embedding | about 160 characters, 40 tokens | 0.000008 |

At the cap:

- per day: US$0.15 buys about 40 fresh answers at the measured mean, or 6 if
  every call used its whole thinking budget. Embeddings are a rounding error
  at either rate.
- per 30 days: 30 x US$0.15 = US$4.50 = INR 422 at the RBI reference rate of
  INR 93.8075 per US dollar (https://www.rbi.org.in, read 2026-11-11). Google
  converts at its own rate, so the INR figure is an estimate, with about 6%
  headroom to INR 450.
- per session: 5 fresh reports, about US$0.019 at the measured mean, an
  eighth of a day.

The cap bounds model calls only. Cloud Run's own bill is separate and, with a
minimum instance, larger: an idle minimum instance in us-central1 costs
$0.0000025 per vCPU-second and $0.0000025 per GiB-second
(https://cloud.google.com/run/pricing, read 2026-11-11), so 1 vCPU + 2 GiB is
US$19.44 per 30 days (INR 1,824) and 2 vCPU + 4 GiB is US$38.88 (INR 3,647),
before any free tier (not checked here). Either one alone is more than the INR
900 credit, so while the credit pays, run `--min-instances 0`.

## Hardening the GCP side

Aswani chose Cloud Storage for the shared cache and approved these changes on
2026-11-11, when they were created. `scripts/harden_gcp.sh setup` is the record
of them: every step describes before it creates, so running it again changes
nothing that exists. Like every step of the script, it refuses to start unless
`HARDEN_GCP_CONFIRMED=yes` is set.

| what | as created on 2026-11-11 |
| --- | --- |
| runtime identity | `climate-risk-runtime@climate-risk-agent.iam.gserviceaccount.com` |
| `crgModelInvoker` | custom role, `aiplatform.endpoints.predict` only, bound on the project |
| `crgCacheReadWrite` | custom role, `storage.objects.get`, `.create` and `.delete`, bound on the bucket only |
| `gs://climate-risk-agent-cache` | `us-central1`, uniform bucket-level access, public access prevention, soft delete off, objects deleted at 400 days |
| budget "climate-risk-agent INR 900" | gross cost (`exclude-all-credits`), this project only, alerts at 50, 90 and 100 percent |

The bucket binding failed once with "does not exist in the resource's
hierarchy": the new custom role had not propagated yet. It worked on retry, so
the script retries that step, up to six times, 30 s apart.

The Cloud Storage cache backend that reads `CRG_CACHE_GCS_BUCKET` is in
`tools/cache_backend.py`, and `candidate` refuses to run from a checkout
without it. Read on 2026-11-13, revision `climate-risk-agent-00033-vuy` carries
the tag `candidate` and no traffic, and `climate-risk-agent-00031-nil` serves
all of it; which code 00031-nil runs is not recorded here. The service moves
to the new identity in two steps: `candidate` deploys a revision that takes no
traffic, under the tag `candidate`; once that revision checks out at its tagged
URL, `promote` sends it all traffic.

Read-only state before the changes (2026-11-11, names only): service
`climate-risk-agent` in `us-central1` (plus `vera-bot` in `asia-south1` in the
same project, inside the same budget), running as
`714882950125-compute@developer.gserviceaccount.com`, the Compute Engine default
service account, which Cloud Run's docs say to avoid since it "might
automatically be granted the Editor role"; environment variable names
`GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`,
`CRG_GENERATE_MODEL`, `CRG_EMBED_MODEL`.

Least privilege, from the code. The runtime calls two Vertex methods,
`generateContent` (answers) and `embedContent` (google-genai 2.11.0 sends
`gemini-embedding-2` there), and the REST reference for each says it "Requires
the following IAM permission on the `model` resource: `aiplatform.endpoints.predict`".
That permission is `SUPPORTED` in custom roles
(https://cloud.google.com/iam/docs/custom-roles-permissions-support). The cache
reads, writes and deletes objects in one bucket, so its role is bound there and
nowhere else. Vertex authenticates as the runtime identity and the bucket needs
no key, so there is no secret to store. Logs need nothing: stdout and stderr
"will be picked up automatically by Cloud Logging". No other Google API is used
(only `google.genai` and `google.auth` are imported).

The commands, in the order the script runs them (flags checked against the help
text of gcloud 593.0.0):

```bash
PROJECT=climate-risk-agent
REGION=us-central1
SERVICE=climate-risk-agent
SA="climate-risk-runtime@${PROJECT}.iam.gserviceaccount.com"
BUCKET=climate-risk-agent-cache

# setup. Each create below runs only when the describe or list before it
# (gcloud iam service-accounts describe, gcloud iam roles describe,
# gcloud projects get-iam-policy, gcloud storage buckets describe,
# gcloud storage buckets get-iam-policy, gcloud billing budgets list)
# finds nothing.
gcloud iam service-accounts create climate-risk-runtime --project="$PROJECT" \
  --display-name="climate-risk-agent runtime (Cloud Run)"
gcloud iam roles create crgModelInvoker --project="$PROJECT" \
  --title="Climate risk agent model invoker" \
  --description="generateContent and embedContent on Vertex AI publisher models, nothing else" \
  --permissions=aiplatform.endpoints.predict --stage=GA
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SA}" \
  --role="projects/${PROJECT}/roles/crgModelInvoker" --condition=None
printf '{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 400}}]}\n' > lifecycle.json
gcloud storage buckets create "gs://${BUCKET}" --project="$PROJECT" \
  --location="$REGION" --uniform-bucket-level-access --public-access-prevention \
  --soft-delete-duration=0 --lifecycle-file=lifecycle.json
gcloud iam roles create crgCacheReadWrite --project="$PROJECT" \
  --title="Climate risk agent cache read and write" \
  --description="Read, write and delete cache objects; bound on the cache bucket only" \
  --permissions=storage.objects.get,storage.objects.create,storage.objects.delete --stage=GA
# retried: a role made a moment ago can be refused until it propagates
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --project="$PROJECT" \
  --member="serviceAccount:${SA}" --role="projects/${PROJECT}/roles/crgCacheReadWrite"
BILLING_ACCOUNT=$(gcloud billing projects describe "$PROJECT" --format='value(billingAccountName)')
gcloud billing budgets create --billing-account="$BILLING_ACCOUNT" \
  --display-name="climate-risk-agent INR 900" --budget-amount=900INR \
  --credit-types-treatment=exclude-all-credits --filter-projects="projects/${PROJECT}" \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0

# candidate: a new revision on the new identity, with no traffic. The first
# flags repeat the service's settings from the deploy command at the top.
gcloud run deploy "$SERVICE" --source . --region="$REGION" --project="$PROJECT" \
  --execution-environment=gen2 --port=7860 --allow-unauthenticated \
  --memory=2Gi --cpu=1 --concurrency=8 \
  --service-account="$SA" --min-instances=0 --max-instances=1 \
  --update-env-vars=CRG_CACHE_GCS_BUCKET="$BUCKET" \
  --no-traffic --tag=candidate

# promote: only after the candidate's tagged URL has been checked.
gcloud run services update-traffic "$SERVICE" --region="$REGION" --project="$PROJECT" \
  --to-tags=candidate=100
```

Notes on what the commands do and do not do:

- `--no-traffic` keeps traffic where it was ("the revision being deployed will
  not receive traffic") and `--tag` gives the new revision its own URL.
  `--to-tags=candidate=100` then moves traffic to exactly the revision that was
  checked; `--to-latest` would instead let every later deploy take traffic at
  once.
- `--budget-amount=900INR`: "If a currency type is included, it must be the
  currency associated with the billing account", so a budget in any other
  currency fails rather than being created in the wrong one.
- `exclude-all-credits` measures gross cost: counting credits (the default)
  nets credit-paid spend to zero, and no alert would fire until the credit is
  gone. A budget sends notifications ("Default notifications are sent to those
  with Billing Account Administrator and Billing Account User IAM roles"); it
  does not stop any spending. The spend guard and `--max-instances=1` do the
  capping.
- `crgCacheReadWrite` grants no `storage.objects.list`. If the Cloud Storage
  backend ever needs to list objects, that permission has to be added to the
  role on purpose.
- Not done by the script: removing roles from the Compute Engine default service
  account. Nothing here shows which roles it holds or what else uses it
  (`vera-bot` may), so that stays a separate, checked step.
- Fallback if the custom role misses a permission the SDK turns out to need:
  bind `roles/aiplatform.user` instead of `crgModelInvoker`.

## Streamlit Community Cloud (free alternative)

As of 2026, Hugging Face gates every Python-app Space (Docker, Gradio, and the
now-deprecated Streamlit SDK) behind HF PRO; only Static is free. Streamlit
Community Cloud also hosts this app for free and is purpose-built for Streamlit,
a good no-cost alternative to the Cloud Run demo above.

1. Push the repo to public GitHub (`origin`), CI green.
2. Go to https://share.streamlit.io, sign in with GitHub, **New app**.
3. Repo `AswaniSahoo/climate-risk-agent`, branch `main`, **main file path
   `ui/app.py`**. Deploy.
4. It installs `requirements.txt`, then the app self-provisions: on first boot
   it sees no corpus and downloads the IPCC PDFs once (~50 MB, shown with a
   spinner). The only geospatial wheel left is shapely, which is
   self-contained, so no `packages.txt` is needed.
5. Optional **Advanced settings → Secrets** → add `GEMINI_API_KEY` (an
   AI-Studio key) for hybrid retrieval + cited LLM answers. Without it the app
   runs BM25-only, **loudly** (measured on the 60-question dev set: 76% vs 82% headline R@3); the UI reports
   the degradation honestly.

Note: the free tier is memory-limited. If the app OOMs on boot, run it
BM25-only (no key), which avoids loading the embedding stack, or trim optional
deps.

## Hugging Face Space: Docker SDK (requires HF PRO)

If you have HF PRO ($9/mo), the image deploys directly. Measured on the
two-stage Dockerfile with `docker image inspect --format '{{.Size}}'`: **0.31
GB**, down from 1.17 GB for the previous single-stage build (no uv or package
cache in the runtime layer, and regionmask's geopandas/rasterio/pyogrio/pyproj/
xarray stack replaced by one bundled GeoJSON read with shapely).

1. Create a Space → SDK = **Docker**. Frontmatter `sdk: docker`, `app_port: 7860`.
2. Push the repo; the Dockerfile bakes the corpus and chunk cache at build. The
   AR6 region polygons are a committed GeoJSON (`tools/ar6/`), so nothing
   downloads at runtime.
3. Add the `GEMINI_API_KEY` secret (or run BM25-only). Vertex ADC does not exist
   on Spaces.

## Publish the MCP server to the official MCP registry

The registry stores metadata only, so the image has to exist first. `server.json`
in the repo root is written and validated against the published schema, and the
ownership label lives in `Dockerfile.mcp`. Every step below needs your accounts,
so all of it is manual.

1. Build the app image, then the MCP image derived from it (the derived image
   reuses the baked corpus and chunk cache):

```bash
docker build -t climate-risk-agent .
docker build -f Dockerfile.mcp -t ghcr.io/aswanisahoo/climate-ipcc-rag-mcp:0.1.0 .
```

2. Push to GitHub Container Registry. Needs a token with `write:packages`, and
   the package must then be made public, or the registry cannot read the label.

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u AswaniSahoo --password-stdin
docker push ghcr.io/aswanisahoo/climate-ipcc-rag-mcp:0.1.0
```

3. Install `mcp-publisher` (Windows):

```powershell
$arch = if ([System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -eq "Arm64") { "arm64" } else { "amd64" }
Invoke-WebRequest -Uri "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_windows_$arch.tar.gz" -OutFile mcp-publisher.tar.gz
tar xf mcp-publisher.tar.gz mcp-publisher.exe
```

4. Authenticate and publish. `login` is a device-code flow, so it has to be run
   interactively by you:

```bash
mcp-publisher login github
mcp-publisher publish
```

5. Verify:

```bash
curl "https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.AswaniSahoo/climate-ipcc-rag"
```

Notes:

- The namespace must match your GitHub username. `server.json` uses
  `io.github.AswaniSahoo/...`; if publish rejects it, run `mcp-publisher init`
  and copy the name it generates.
- `LABEL io.modelcontextprotocol.server.name` in `Dockerfile.mcp` must
  byte-match `name` in `server.json`, or publish fails verification.
- Only the IPCC server is listed. weather-mcp is a thin Open-Meteo wrapper with
  many equivalents already in the registry, and publishing it would ship the same
  full app image (corpus and all) to serve two HTTP calls. The same pattern
  applies if you want it.

## Release gate (evals run on tag pushes, not on every commit)

`ci.yml` still runs only the unit/integration tests on every push and PR
(corpus-dependent ones auto-skip). The retrieval + e2e evals are expensive and
credential-bound, so they live in a **separate workflow** that fires exactly
when a release is being cut:

| workflow | trigger | what it runs |
| --- | --- | --- |
| `ci.yml` | every push to `main`, every PR | ruff, mypy, pytest |
| `evals.yml` | push of a `v*` tag, or manual `workflow_dispatch` | retrieval eval + e2e eval + gate |

Locally the same two runners are still the pre-tag check:

```bash
uv run python -m evals.run_retrieval_eval   # recall@k per slice vs frozen set
uv run python -m evals.run_e2e_eval         # refusal matrix, false_answer MUST be 0
uv run python -m scripts.eval_gate --eval-set dev   # the pass/fail decision, as an exit code
```

The gate (`scripts/eval_gate.py`, unit-tested in `tests/test_eval_gate.py`) fails on:

1. **`false_answer > 0`** in the e2e artifact. Confabulation blocks a release, full stop.
2. **headline R@3 more than 3 points below** the last committed run of the same
   eval set, same retriever (`evals/results/`).
3. **no `hybrid` column** in the retrieval artifact. That means the dense path was
   skipped, i.e. the embedding cache was missing and the run silently measured
   BM25-only, so its numbers are not a release.

Rule, unchanged: run both, publish the numbers in README/STATE, THEN tag/deploy.

### Repository setup (manual, once)

- **Secret `GEMINI_API_KEY`**: Settings → Secrets and variables → Actions → New
  repository secret. The e2e eval cannot run without it. `GITHUB_TOKEN` is
  supplied automatically and only needs `contents: read` to fetch the asset below.

### The eval cache release asset (manual, once, and again after any re-chunk)

The workflow must never call the embedding API for the corpus: 2,730 chunks do
not fit in a free-tier job. Instead it downloads a pre-built cache. `data/` is
git-ignored, so the cache ships as a **GitHub Release asset**:

```bash
uv run python -m scripts.pack_eval_cache        # -> eval-cache-v1.tar.gz (~13 MB)
gh release create eval-cache-v1 eval-cache-v1.tar.gz \
  --title "Eval cache v1" \
  --notes "Embedding + chunk cache for the eval workflow."
```

To replace it later (after re-chunking, a corpus refresh, or an embedding-model change):

```bash
uv run python -m scripts.pack_eval_cache
gh release upload eval-cache-v1 eval-cache-v1.tar.gz --clobber
```

The archive carries a `manifest.json` pinning the embedding model + dims, the
chunker fingerprint (a hash of the PDF sizes and `rag/chunk.py`), the PDF sizes,
and a SHA-256 of the payload bytes. `scripts/unpack_eval_cache.py` re-checks all
of it against the live repo and **refuses to install a mismatched cache**, naming
the two commands above in the failure. A stale cache that scored today's chunker
against yesterday's vectors would be worse than no cache: it would publish a
plausible, wrong number.

## Held-out test set: exposure protocol (dev/test split, 2026-07-17)

Two frozen sets exist:

- `evals/gold_set.json` (60 q), the **dev set**. It steered development
  (top_k, chunking, scope guard), so it can never claim "held-out". Run it
  freely; diagnose against it.
- `evals/gold_set_v2.json` (105 q), the **held-out test set**. Runs at
  **release gates only** (`EVAL_SET=test`, artifact gets a `-test` suffix).
  Failures found there are diagnosed on the DEV set, never by iterating
  against the test set. Every published test-set number carries its exposure
  count ("held-out, Nth exposure"). Peeking between gates burns the split.
