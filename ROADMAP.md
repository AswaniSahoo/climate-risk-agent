# Roadmap

## Shipped (each with a number attached — see README + eval outputs)

- [x] IPCC AR6 RAG with page-level citations
- [x] ERA5 hazard statistics (GEV return periods) with full provenance
- [x] Frozen 45-question eval harness: recall@k per slice + e2e refusal confusion matrix
- [x] MCP servers (weather-mcp + ipcc-rag-mcp), demoed in the MCP Inspector
- [x] Hybrid dense+RRF ablation published (bm25 82% / dense 71% / hybrid 91% headline R@3)
- [x] RAG citations wired into the `RiskReport` agent path (`research` graph node)
- [x] Streamlit UI, Docker image, CI; evals as a documented release gate
- [x] Risk verdict from GEV return-level position; composed confidence; bootstrap CIs
- [x] Claim-level LLM-judge eval + graph-path (real agent) eval
- [x] Table-caption-aware chunking → perfect matrix 34/11/0/0
- [x] Observability: seam-level telemetry, cost-per-report, latency percentiles, `/metrics`
- [x] Async FastAPI service with access control

## Next

- [ ] NL front door: free-text query → geocoding → hazard classification → lat/lon→AR6-region mapping (deterministic table-row targeting)
- [ ] Eval v2: 150+ questions with a dev/test split (held-out numbers)
- [ ] Non-stationary GEV (warming covariate) — retire the stationarity disclaimer
- [ ] Structured logging + ruff/mypy in CI + committed eval-output artifacts
- [ ] Deployed demo on Hugging Face Spaces
