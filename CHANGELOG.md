# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Enhanced dark mode styling and brand logo in Streamlit navigation header.
- Synchronized location search and suggestion chips with interactive deck.gl map markers.
- Added automated offline testing stubs for natural language queries.

## [1.0.0] - 2026-09-08

### Added

- Shared cache backend (`tools/cache_backend.py`) behind the climatology-fit,
  forecast and answer caches: Upstash Redis over REST when
  `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` are set, disk
  otherwise, with `scripts/prewarm.py` and `scripts/prewarm_cities.json` to fill
  it ahead of a demo.
- Any-location input: `agent/location.py` resolves a place name or a
  latitude/longitude pair, and `ui/app.py` accepts both instead of a fixed city
  list.
- Risk bands (`agent/risk_bands.py`) wired into the graph, and wind gusts read
  from the forecast alongside sustained wind.
- Forecast skill table (`tools/forecast_skill.py`,
  `tools/forecast_skill_table.json`) feeding `agent/verdict.compose_confidence`,
  so confidence scales with measured lead-time skill.
- Chapter 12 projected-change node: `rag/cid.py`, the `project` node in the
  graph, its contracts, and 15 `cid-table` questions in the dev set (45 rows to
  60) with `evals/gold_set.sha256` re-frozen.
- Reranking and query-rewriting seams (`rag/rerank.py`, `rag/rewrite.py`) as
  `HybridRetriever` keyword arguments, matching eval-runner flags, and an opt-in
  `eval` dependency group. Both default to off.
- Semantic scope guard, stage 2 (`rag/scope_semantic.py`,
  `rag/scope_anchors.json`) behind `CRG_SCOPE_STAGE2`, default off, with a
  matching `--scope-stage2` flag on the e2e runner.
- Evals as a release-gated workflow (`.github/workflows/evals.yml`) with
  `scripts/pack_eval_cache.py`, `scripts/unpack_eval_cache.py` and
  `scripts/eval_gate.py`.
- Contributor files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, this changelog,
  issue templates and a pull request template.
- `scripts/measure_coldstart.py` and `scripts/measure_forecast_skill.py`.
- Live progress panel in the UI: `agent/progress.py` names each of the seven
  steps, times it and badges the cache tier that served it, with one plain
  sentence per error class and a "how to read this report" explainer.

### Changed

- Dockerfile split into builder and runtime stages, with scipy, langgraph and
  google-genai imported lazily so the UI boots without them.
- AR6 reference regions resolved from the bundled
  `tools/ar6/IPCC-WGI-reference-regions-v4.geojson` with shapely, so nothing
  downloads at runtime; `requirements.txt` regenerated to match.
- `DEPLOY.md` records Cloud Run idle cost per min-instance shape, the measured
  30-day usage and the resize plan, and lists the Upstash variables as optional.
- `LIMITATIONS.md` and the Cloud Run deploy command name `gemini-2.5-flash`, the
  code default in `rag/gemini_client.py` and the model every published number
  was measured on, in place of `gemini-3.6-flash`.
- Dev-set numbers re-measured on all 60 questions and republished in
  `README.md`, `LIMITATIONS.md`, `DEPLOY.md` and `ROADMAP.md`: hybrid retrieval
  R@3 82% / @5 90% / @10 94% against 76% / 82% / 88% for BM25-only, and an
  end-to-end matrix of 48 correct answers, 11 correct refusals, 1 false refusal
  and 0 false answers. The previous 91% headline described the first 45
  questions only.
- `scripts/eval_gate.py` picks its regression baseline from runs of the same
  arm. The rerank, rewrite and scope-stage-2 runners write artifacts into the
  same directory, and the gate was grading the shipped default against a
  reranked experiment.
- Telemetry keeps a bounded in-memory ring of 5,000 events, cache reads are
  sampled out of the JSONL sink, and the answer cache no longer emits a second
  event per hit that double-counted cache hits.
- `actions/checkout` and `astral-sh/setup-uv` pinned to the same majors in
  `ci.yml` and `evals.yml` (v7 and v10, checked against the releases API on
  2026-09-02), so the two workflows cannot fail in different ways.

### Removed

- `regionmask` and the geopandas, rasterio, pyogrio, pyproj and xarray chain it
  pulled in.
- `agent/verdict.level_from_return_periods` and its tests. `agent/risk_bands.py`
  is the single source of truth for severity.

### Fixed

- Chapter 12 clause attribution: a clause now qualifies only when it carries the
  region, a direction and a calibrated confidence phrase itself, so a second
  region's assessment can no longer be read as this one's.
- The forecast cache key carries the UTC date, so an entry written late in the
  day cannot replay a window whose first day has already begun.
- Disk cache entries are written to a sibling temp file and moved with
  `os.replace`, so a crash or a concurrent write cannot leave a truncated entry.
- Question text is fenced by `rag/prompt_safety.py` before it reaches the
  rewrite and stage-2 prompts: the `<question>` tag is stripped and the text is
  capped, so a question cannot close the block and address the model directly.
- The stage-2 scope memo is a 512-entry LRU rather than an unbounded dict keyed
  by user input.
- `scripts/unpack_eval_cache.py` checks the archive is complete before moving
  anything, clears stale vectors, and reports an install failure with the
  commands that fix it instead of a traceback.
- The runtime image chowns `/app` to `appuser`, so the container can create
  `data/cache` and Streamlit's temp directory at the app root.
- `ProjectedChange.from_retrieval` reads `retrieved_chunk_ids` off the
  retriever's own output, so the validator no longer compares two lists supplied
  by the same caller.

## Phase 4: deployment, Vertex AI, and the MCP registry (2026-07-22 to 2026-08-11)

### Added

- Live demo deployed on Google Cloud Run, with `.gcloudignore` tracked so the
  embedding cache ships in the Cloud Build upload.
- Vertex AI on the `global` endpoint, an embedding self-test that reports a
  degraded dense path loudly, and a disk cache for climatology results.
- Determinism probe, plus model and telemetry provenance recorded in every eval
  artifact, so a model swap cannot pass unnoticed.
- ADR 0001 on answering-model selection: a dev-set bake-off across model
  versions, a prompt ablation, and a determinism probe.
- `server.json` and `Dockerfile.mcp` for MCP registry publishing, and the
  registry listing for `io.github.AswaniSahoo/climate-ipcc-rag` in the README.
- Tests that boot both MCP servers as real subprocesses and speak the protocol
  to them.

### Changed

- MCP servers migrated to `mcp` 2.0.0: tools annotated read-only with titles and
  world hints, output schemas published, deterministic tool order, and `.env`
  loaded when the client scrubs the environment.
- IPCC chunk cache baked at build time, polygon bake order corrected, and the
  bootstrap sample count tuned.
- Answering model reverted to the evaluated pin after the bake-off; reasoning
  tokens counted as billed output; a model with no price entry reports its cost
  as unknown instead of zero.

### Removed

- The legacy single-tool MCP server, superseded by weather-mcp.

### Fixed

- Application Default Credentials resolved once from file, so MCP stdio tools no
  longer deadlock on `gcloud`.
- MCP namespace casing corrected to match the GitHub username, and the DEPLOY.md
  verify URL updated to match.
- `requirements.txt` regenerated for `mcp` 2.0.0 so the documented install path
  works.

## Phase 3: evals v2, natural-language front door, non-stationary GEV (2026-07-15 to 2026-07-18)

### Added

- Telemetry at the single SDK seam every model call passes through: latency,
  tokens, retries, and cost per report.
- Async FastAPI service with per-request API-key access control and a `/metrics`
  endpoint.
- Eval result artifacts committed to the repository, so the published numbers
  are files rather than prose.
- Geocoding, and latitude/longitude to IPCC AR6 region mapping using the
  official Iturbide-2020 polygons.
- Natural-language front door: parse, geocode and region-map a free-text
  question into a typed report, with refusals that never guess. Free-text query
  in the UI and `POST /query`, where refusals are valid 200 responses.
- Non-stationary GEV: drifting-location MLE, a likelihood-ratio test, and
  warm-started bootstrap confidence intervals. A significance gate reports
  effective return levels at the latest year when the trend is real, and the
  trend regime is surfaced in the summary, the drivers, and the UI.
- Held-out eval v2: a 105-question test set split from the 45-question dev set,
  an `EVAL_SET` knob, an exposure-count protocol, and the first held-out gate
  artifacts (headline R@3 87%, zero false answers).
- MIT LICENSE file.
- Deployment paths for Streamlit Community Cloud and the Hugging Face Streamlit
  SDK: a `requirements.txt` install and a corpus that self-provisions on first
  boot.
- Dev Container configuration.

### Changed

- Structured logging, ruff and mypy clean and gating CI, and typed parameter
  dicts.
- README rewritten around the held-out numbers and the exposure protocol, with a
  mermaid architecture diagram, a guarantees table, the roadmap extracted to its
  own file, and UI screenshots.

### Fixed

- Marine heatwaves refused before the supported-hazard check.
- Transient Open-Meteo forecast failures retried, with the UI degrading
  gracefully instead of failing the report.

## Phase 2: RAG hardening, MCP servers, UI, and CI (2026-07-11 to 2026-07-13)

### Added

- Frozen 45-question gold eval set with machine-verified quotes.
- Zero-dependency BM25 retriever with recall@k, MRR, and Wilson confidence
  intervals.
- Dense retrieval on `gemini-embedding-2` fused with BM25 by RRF, on a
  quota-paced, resumable embedding cache.
- Cited LLM answerer, with a deterministic scope guard that runs before the
  model.
- End-to-end eval: refusal confusion matrix plus citation and numeric-provenance
  checkers, and live per-question progress.
- Split MCP servers, weather-mcp and ipcc-rag-mcp, stdio-only and typed.
- Tier-1 security controls: boundary validation, pinned outbound hosts, and
  secret-leak tests, documented in SECURITY.md alongside LIMITATIONS.md.
- A single google-genai SDK seam carrying both Vertex ADC and API-key auth, with
  models pinned.
- Research graph node wiring the IPCC RAG into `RiskReport` with real page-level
  citations.
- Streamlit one-pager over the `RiskReport` contract with headless AppTest
  tests, then an earth theme, severity badges, and a card layout.
- Self-sufficient Docker image with a Hugging Face Spaces guide, and CI running
  pytest on push with evals kept as a documented manual release gate.
- Risk verdict derived from where the forecast lands on the GEV return-level
  curve, with composed confidence.
- 90% bootstrap confidence interval on every GEV return level.
- Disk-backed answer cache, so repeat queries cost zero tokens.
- Claim-level LLM judge, and a graph-path eval that runs the real agent on live
  scenarios.
- tqdm progress, an SDK client timeout, and loud fallbacks in the long-running
  paths.

### Changed

- Row-atomic table chunks: the eval slice that had been failing entirely moved
  from 0% to 100% R@5, and headline R@3 from 76% to 82%.
- Caption-carry chunking, so table column labels reach every row chunk, which
  produced the 34/11/0/0 refusal matrix.
- HybridRetriever became the production path for the end-to-end and MCP layers.

### Fixed

- `sys.path` shim, because `streamlit run` puts `ui/` on the path rather than the
  repository root.
- Number drift across the docs corrected, and dev-set reuse disclosed.
- One-call-per-text embedding regression pinned by a test, with a thread-local
  client reset.

## Phase 1: core agent, tools, and ERA5 hazard statistics (2026-06-20 to 2026-07-10)

### Added

- uv project scaffold and dependencies.
- Typed `RiskReport` output contract.
- `get_forecast` Open-Meteo tool, later extended with `wind_speed_10m_max` in
  both the forecast and the MCP schema.
- Three-node LangGraph agent and a live end-to-end demo script.
- First MCP server exposing `get_forecast`.
- README.
- GEV return-period hazard statistics, a typed `HazardStat`, and a live ERA5
  wrapper wired into `RiskReport` with climatology confidence.
- ERA5 wind and extreme-precipitation hazard statistics.
- IPCC AR6 document fetch script, a page-aware parser and chunker, and
  Chapter 12 (regional impact-drivers) added to the corpus.

### Changed

- Hazard enum aligned to the planned hazard set.
- Hazard source replaced: the WeatherBench2 zarr path gave way to the Open-Meteo
  Archive for defensible ERA5 daily extremes, with caching, demo wiring, and
  data attribution.
