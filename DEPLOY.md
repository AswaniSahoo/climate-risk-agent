# Deploy — Docker + Hugging Face Spaces

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

## Hugging Face Space — Streamlit SDK (free, recommended)

HF gates the **Docker** SDK behind a paid tier, but the **Streamlit** SDK is
free (2 vCPU, 16 GB RAM) and runs this app natively — it already is a Streamlit
app, so no rewrite. This is the recommended path.

1. Create a Space → SDK = **Streamlit** (free). Set the License field to `mit`.
2. Put this YAML frontmatter at the **top of the Space's README.md**
   (the GitHub README stays clean):

   ```yaml
   ---
   title: Climate-Risk Analyst Agent
   emoji: 🌍
   colorFrom: blue
   colorTo: green
   sdk: streamlit
   app_file: ui/app.py
   pinned: false
   license: mit
   ---
   ```

3. Push this repo to the Space (`git remote add space https://huggingface.co/spaces/<user>/<space>` then `git push space main`). HF installs `requirements.txt` and runs `streamlit run ui/app.py`.
4. The app self-provisions: on first boot it detects the HF `SPACE_ID` env var,
   sees no baked corpus, and downloads the IPCC PDFs once (~50 MB, shown with a
   spinner). No Dockerfile bake step needed.
5. Optional **Settings → Variables and secrets** → add secret `GEMINI_API_KEY`
   (AI-Studio key; Vertex ADC does not exist on Spaces) for hybrid retrieval +
   cited LLM answers. Without it the Space runs BM25-only, **loudly** (measured
   82% vs 91% headline R@3); the UI reports the degradation honestly. First boot
   with a key embeds the corpus once (ephemeral storage, so a restart re-embeds).

## Hugging Face Space — Docker SDK (paid alternative)

Only if you have the paid Docker tier. The verified 4.76 GB image
(`docker build`, boots clean) deploys directly:

1. Create a Space → SDK = **Docker**. Frontmatter `sdk: docker`, `app_port: 7860`.
2. Push the repo; the Dockerfile bakes the corpus + AR6 polygons at build.
3. Add the `GEMINI_API_KEY` secret as above (or run BM25-only).
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
