# Roadmap

## Shipped (each with a number attached — see README + eval outputs)

- [x] IPCC AR6 RAG with page-level citations
- [x] ERA5 hazard statistics (GEV return periods) with full provenance
- [x] Frozen eval harness: recall@k per slice + e2e refusal confusion matrix (dev + held-out test sets)
- [x] MCP servers (weather-mcp + ipcc-rag-mcp), demoed in the MCP Inspector
- [x] Hybrid dense+RRF ablation published (bm25 82% / dense 71% / hybrid 91% dev-set R@3)
- [x] RAG citations wired into the `RiskReport` agent path (`research` graph node)
- [x] Streamlit UI, Docker image, CI; evals as a documented release gate
- [x] Risk verdict from GEV return-level position; composed confidence; bootstrap CIs
- [x] Claim-level LLM-judge eval + graph-path (real agent) eval
- [x] Table-caption-aware chunking → perfect matrix 34/11/0/0
- [x] Observability: seam-level telemetry, cost-per-report, latency percentiles, `/metrics`
- [x] Async FastAPI service with access control
- [x] NL front door: free-text query → geocoding → hazard classification → lat/lon→AR6-region mapping (`agent/nl.py`, deterministic + prompt-injection-proof)
- [x] Non-stationary GEV (warming covariate) with a likelihood-ratio significance test — effective return levels at the latest year when the trend is real
- [x] Eval v2: held-out 105-question test set (`evals/gold_set_v2.json`) split from the 45-question dev set, with an exposure-count protocol — test-set headline R@3 87% / @5 91% / @10 96%, zero false answers
- [x] Structured logging + ruff/mypy in CI + committed eval-output artifacts (`evals/results/`)
- [x] Live demo deployed on Google Cloud Run: https://climate-risk-agent-714882950125.us-central1.run.app/

## Next

- [ ] Semantic/LLM scope guard behind the lexical v1 gate (close the paraphrase gap)
- [ ] Automate claim-level entailment checking in the release gate (currently page-level)
- [ ] Climatology-conditioned risk levels to replace the Day-1 fixed thresholds in `agent/graph.py`
- [ ] Lift the weakest retrieval slices (regional-table 77%, premise-injection 59% on the test set)
