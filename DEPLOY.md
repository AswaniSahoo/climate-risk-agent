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

## Hugging Face Space (Docker SDK)

1. Create a Space → SDK = **Docker** (blank template).
2. Add this YAML frontmatter to the **top of the Space's README.md**
   (HF requires it; keep the GitHub README clean):

   ```yaml
   ---
   title: Climate-Risk Analyst Agent
   emoji: 🌍
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

3. Push this repo to the Space (or `git remote add space https://huggingface.co/spaces/<user>/<space>` and push).
4. Space **Settings → Variables and secrets** → add secret `GEMINI_API_KEY`
   (AI-Studio key; Vertex ADC does not exist on Spaces). Without it the Space
   still runs, BM25-only.
5. First boot with a key embeds the 2730-chunk corpus once (progress in the
   container log; resumable disk cache — but note Space storage is ephemeral,
   so a Space RESTART re-embeds unless you attach persistent storage).

## Release gate (evals are NOT in CI — by decision)

CI runs the 126 unit/integration tests (corpus-dependent ones auto-skip).
The retrieval + e2e evals need the corpus, the embedding cache, and Gemini
auth, so they are a **manual pre-release gate**:

```bash
uv run python -m evals.run_retrieval_eval   # recall@k per slice vs frozen set
uv run python -m evals.run_e2e_eval         # refusal matrix — false_answer MUST be 0
```

Rule: run both, publish the numbers in README/STATE, THEN tag/deploy.
A `false_answer > 0` is a release blocker, full stop.
