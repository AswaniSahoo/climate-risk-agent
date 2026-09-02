# Roadmap

## Shipped (each with a number attached, see README + eval outputs)

- [x] IPCC AR6 RAG with page-level citations
- [x] ERA5 hazard statistics (GEV return periods) with full provenance
- [x] Frozen eval harness: recall@k per slice + e2e refusal confusion matrix (dev + held-out test sets)
- [x] MCP servers (weather-mcp + ipcc-rag-mcp), demoed in the MCP Inspector
- [x] Hybrid dense+RRF ablation published (bm25 76% / dense 61% / hybrid 82% dev-set R@3, 60 questions)
- [x] RAG citations wired into the `RiskReport` agent path (`research` graph node)
- [x] Streamlit UI, Docker image, CI; evals as a documented release gate
- [x] Risk verdict from GEV return-level position; composed confidence; bootstrap CIs
- [x] Claim-level LLM-judge eval + graph-path (real agent) eval
- [x] Table-caption-aware chunking → perfect matrix 34/11/0/0
- [x] Observability: seam-level telemetry, cost-per-report, latency percentiles, `/metrics`
- [x] Async FastAPI service with access control
- [x] NL front door: free-text query → geocoding → hazard classification → lat/lon→AR6-region mapping (`agent/nl.py`, deterministic + prompt-injection-proof)
- [x] Non-stationary GEV (warming covariate) with a likelihood-ratio significance test, effective return levels at the latest year when the trend is real
- [x] Eval v2: held-out 105-question test set (`evals/gold_set_v2.json`) split from the 45-question dev set, with an exposure-count protocol, test-set headline R@3 87% / @5 91% / @10 96%, zero false answers
- [x] Structured logging + ruff/mypy in CI + committed eval-output artifacts (`evals/results/`)
- [x] Live demo deployed on Google Cloud Run: https://climate-risk-agent-714882950125.us-central1.run.app/
- [x] Model-selection ADR ([adr/0001](adr/0001-answering-model-selection.md)): dev-set bake-off across gemini-2.5/3.5/3.6-flash, prompt ablation, and a determinism probe; every eval artifact now records the model that produced it
- [x] Climatology-conditioned risk bands: the forecast peak's return period on the location's own curve, at 2 / 10 / 50-year brackets, replacing the fixed Day-1 cutoffs
- [x] Chapter 12 projected change quoted verbatim, with 15 `cid-table` questions added to the dev set (45 rows to 60); that slice measures hybrid R@3 67%
- [x] Forecast skill measured by lead day: 13 cities, 2024-2025, day-7 temperature extremes caught 47% of the time against 85% at day 1, and confidence weighted by it
- [x] Any location: place name or coordinates, with the AR6 region read from the bundled v4 polygons (46 land-touching regions)
- [x] Shared cache backend, Upstash Redis over REST or disk: a first fit takes 1 to 2 minutes, repeat visits return in under a second
- [x] Rerank, query rewrite and semantic scope stage 2 measured and kept off: no arm beat the hybrid baseline (R@3 88% on the 45-question dev set) and the cheapest one cost 4.0 s per question
- [x] Slim runtime image: 0.31 GB, down from 1.17 GB
- [x] Evals as a release-gated workflow on `v*` tags, with a packed embedding cache and a pass/fail gate that is an exit code
- [x] Dev-set numbers re-measured on all 60 questions: hybrid R@3 82% / @5 90% / @10 94%, e2e 48/11/1/0, $0.0034 per question, p50 5.5 s
- [x] Contributor files: CONTRIBUTING, code of conduct, changelog, three issue templates and a PR template
- [x] Live progress panel in the UI: seven named steps, each with its seconds and a cache-tier badge

## Next (ranked)

- [ ] Refresh the two UI screenshots and record a demo video against the current UI
- [ ] Promote the semantic scope guard's `embed` arm, once a held-out run confirms the refusal matrix holds
- [ ] Gemini-vision read of the Ch.12 glyph tables, validated against the chapter prose before any of it is published
- [ ] Measure Docker cold start on Cloud Run, then move to `--min-instances 0` if it comes in under 30 s
- [ ] Cross-post the build-in-public series to dev.to
