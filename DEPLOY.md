# Deploy: Docker + Google Cloud Run

## Local Docker

```bash
docker build -t climate-risk-agent .
docker run -p 7860:7860 climate-risk-agent                          # BM25-only (no key)
docker run -p 7860:7860 -e GEMINI_API_KEY=... climate-risk-agent    # hybrid + cited answers
```

Open http://localhost:7860.

- The build bakes the IPCC corpus (~50 MB, downloaded from ipcc.ch, idempotent).
- If `data/cache/embeddings/` exists locally it is COPYed into the image
  (`.dockerignore` deliberately does not exclude `data/`), so the container
  starts on hybrid retrieval without re-embedding.
- Without Gemini auth the retrieval layer falls back to BM25-only **loudly**
  (measured on the 60-question dev set: 76% vs 82% headline R@3) and `answer_ipcc`-style LLM answers are
  unavailable; the UI reports both degradations honestly.

## Google Cloud Run (live demo)

The public demo runs here:
**https://climate-risk-agent-714882950125.us-central1.run.app/**

Cloud Run serves the same Docker image described above. `.gcloudignore` mirrors
`.dockerignore` and deliberately keeps `data/` in the Cloud Build upload, so the
embedding cache (~9 MB) is baked into the image and the container starts on
hybrid retrieval instead of re-embedding the corpus (and hitting 429s) on every
cold start.

Auth and models run on **Vertex AI via the `global` endpoint** (ADC through the
Cloud Run service account, so there is no API key to manage):

```bash
# Deploy from source; Cloud Build runs the docker build, then Cloud Run hosts it.
gcloud run deploy climate-risk-agent \
  --source . \
  --region us-central1 \
  --port 7860 \
  --allow-unauthenticated \
  --min-instances 1 \
  --memory 2Gi --cpu 1 --concurrency 8 \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=climate-risk-agent,GOOGLE_CLOUD_LOCATION=global,CRG_GENERATE_MODEL=gemini-2.5-flash,CRG_EMBED_MODEL=gemini-embedding-2
```

- **`GOOGLE_CLOUD_LOCATION=global` is required.** `gemini-embedding-2` is served
  on `global` / `us` / `eu`, **not** the single region `us-central1`; a regional
  endpoint 404s the embedding call and the app silently drops to BM25-only.
  `gemini-2.5-flash` is also served on `global`.
- `--region us-central1` is where the *Cloud Run service* (the container host)
  runs, independent of the Vertex model endpoint (`global`).
- **`--min-instances 1`** keeps one warm instance, so demo clicks never pay a
  cold start. It is the largest line on the bill (see Cost below); drop it to 0
  once the measured cold start is acceptable.
- Cold start itself is now attacked at the image, not the instance count: the
  Dockerfile is two-stage (no uv or package cache in the runtime layer) and the
  UI boots without scipy, langgraph or google-genai, which are imported only
  when a report is actually requested.
- `--port 7860` matches the container's `EXPOSE` / Streamlit port.
- The Cloud Run service account needs the **Vertex AI User** role
  (`roles/aiplatform.user`).
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
  chose.

## Cost

Cloud Run request-based billing charges a minimum instance at the **idle rate**
for every second of the month, whether or not it serves a request. In
`us-central1` that rate is $0.0000025 per vCPU-second and $0.0000025 per
GiB-second, so an always-on instance costs:

| min-instance shape | idle cost per instance per month |
| --- | --- |
| 2 vCPU + 4 GiB (the shape deployed today) | about $39 |
| 1 vCPU + 2 GiB | about $19 |
| `--min-instances 0` | about $0 |

The service deployed on 2026-07-23 runs one min instance at the first row's
shape, so its monthly floor is about $39 before a single request is served. The
command above already asks for the second row.

Measured over 30 days: 2,640 requests, memory p99 at 13% of the 4 GiB
allocation, CPU p99 at 1%. The shape is provisioned for a load that is not
there.

Two steps, in order. Resize now, to the 1 vCPU + 2 GiB row, which the p99
figures already cover. Then set `--min-instances 0` once the two-stage image's
cold start is measured under 30 s (`scripts/measure_coldstart.py` times it
against a 1 vCPU / 2 GiB container), because below that a scale-to-zero demo
still opens fast enough not to read as broken.

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
