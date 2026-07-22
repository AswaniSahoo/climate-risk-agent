# Deploy — Docker + Google Cloud Run

## Local Docker

```bash
docker build -t climate-risk-agent .
docker run -p 7860:7860 climate-risk-agent                          # BM25-only (no key)
docker run -p 7860:7860 -e GEMINI_API_KEY=... climate-risk-agent    # hybrid + cited answers
```

Open http://localhost:7860.

- The build bakes the IPCC corpus (~50 MB, downloaded from ipcc.ch — idempotent).
- If `data/cache/embeddings/` exists locally it is COPYed into the image
  (`.dockerignore` deliberately does not exclude `data/`), so the container
  starts on hybrid retrieval without re-embedding.
- Without Gemini auth the retrieval layer falls back to BM25-only **loudly**
  (measured: 82% vs 91% headline R@3) and `answer_ipcc`-style LLM answers are
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
Cloud Run service account — no API key to manage):

```bash
# Deploy from source; Cloud Build runs the docker build, then Cloud Run hosts it.
gcloud run deploy climate-risk-agent \
  --source . \
  --region us-central1 \
  --port 7860 \
  --allow-unauthenticated \
  --min-instances 2 \
  --memory 4Gi --cpu 2 \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=climate-risk-agent,GOOGLE_CLOUD_LOCATION=global,CRG_GENERATE_MODEL=gemini-3.6-flash,CRG_EMBED_MODEL=gemini-embedding-2
```

- **`GOOGLE_CLOUD_LOCATION=global` is required.** `gemini-embedding-2` is served
  on `global` / `us` / `eu`, **not** the single region `us-central1`; a regional
  endpoint 404s the embedding call and the app silently drops to BM25-only.
  `gemini-3.6-flash` is also served on `global`.
- `--region us-central1` is where the *Cloud Run service* (the container host)
  runs — independent of the Vertex model endpoint (`global`).
- **`--min-instances 2`** keeps two warm instances, so demo clicks never pay a
  cold start.
- `--port 7860` matches the container's `EXPOSE` / Streamlit port.
- The Cloud Run service account needs the **Vertex AI User** role
  (`roles/aiplatform.user`).
- The startup self-test logs `DENSE DEGRADED` (and the UI shows a banner) if the
  embedding endpoint is unreachable — a misconfig is loud, never silent.

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
   spinner). The geospatial wheels (rasterio/shapely/pyproj) are self-contained,
   so no `packages.txt` is needed.
5. Optional **Advanced settings → Secrets** → add `GEMINI_API_KEY` (an
   AI-Studio key) for hybrid retrieval + cited LLM answers. Without it the app
   runs BM25-only, **loudly** (measured 82% vs 91% headline R@3); the UI reports
   the degradation honestly.

Note: the free tier is memory-limited. If the app OOMs on boot, run it
BM25-only (no key) — that path avoids loading the embedding stack — or trim
optional deps.

## Hugging Face Space — Docker SDK (requires HF PRO)

If you have HF PRO ($9/mo), the verified 4.76 GB image (`docker build`, boots
clean) deploys directly:

1. Create a Space → SDK = **Docker**. Frontmatter `sdk: docker`, `app_port: 7860`.
2. Push the repo; the Dockerfile bakes the corpus + AR6 polygons at build.
3. Add the `GEMINI_API_KEY` secret (or run BM25-only). Vertex ADC does not exist
   on Spaces.
   ⚠️ The 4.76 GB image may exceed the Space's storage ceiling — slim it with a
   multi-stage build first (see docs/DEBT.md) if the build is rejected.

## Release gate (evals are NOT in CI — by decision)

CI runs the unit/integration tests (corpus-dependent ones auto-skip).
The retrieval + e2e evals need the corpus, the embedding cache, and Gemini
auth, so they are a **manual pre-release gate**:

```bash
uv run python -m evals.run_retrieval_eval   # recall@k per slice vs frozen set
uv run python -m evals.run_e2e_eval         # refusal matrix — false_answer MUST be 0
```

Rule: run both, publish the numbers in README/STATE, THEN tag/deploy.
A `false_answer > 0` is a release blocker, full stop.

## Held-out test set — exposure protocol (dev/test split, 2026-07-17)

Two frozen sets exist:

- `evals/gold_set.json` (45 q) — the **dev set**. It steered development
  (top_k, chunking, scope guard), so it can never claim "held-out". Run it
  freely; diagnose against it.
- `evals/gold_set_v2.json` (105 q) — the **held-out test set**. Runs at
  **release gates only** (`EVAL_SET=test`, artifact gets a `-test` suffix).
  Failures found there are diagnosed on the DEV set — never by iterating
  against the test set. Every published test-set number carries its exposure
  count ("held-out, Nth exposure"). Peeking between gates burns the split.
