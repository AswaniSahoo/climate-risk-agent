# Security

Threat model and controls for the climate-risk agent. The rule this project
follows: **a control that matters is pinned by a test** (`tests/test_security.py`),
so weakening it turns the suite red. Claims below reference the test or code
that enforces them.

## Threat model (what we defend against)

| Threat | Vector | Control |
|---|---|---|
| Prompt injection via corpus | IPCC text (or a future corpus doc) contains adversarial instructions | Retrieved text is wrapped in `<excerpt>` tags and framed as data; the answering prompt's rule #1 is that excerpt content cannot change the rules. Structural backstop: even a fully hijacked model **cannot fabricate a citation** — `CitedAnswer` rejects any citation whose chunk_id was not retrieved (validator, `rag/answer.py`) |
| Fabricated citations / numbers | Model hallucination | Citation ids validated against the retrieved set (structural); numeric-provenance checker verifies every number in an answer appears in a cited excerpt (`evals/checkers.py`); both measured in the published e2e eval |
| Out-of-scope answers | Question about hazards we cannot assess responsibly | Deterministic scope guard runs **before** the LLM (`rag/scope.py`) — code, not prompt, so it cannot be injected away; measured by the refusal confusion matrix |
| SSRF / URL manipulation | Caller-influenced request targets | All outbound hosts are hardcoded constants (Open-Meteo forecast + archive, Google generative API); pinned by `test_outbound_hosts_are_pinned` |
| Garbage / hostile input | Out-of-range coordinates, absurd horizons | Range-validated at every tool boundary before any request is built (`tools/validation.py`) |
| Denial-of-wallet | Repeated expensive upstream calls | Climatology results are memoized per (location, hazard); embeddings are disk-cached and fetched in quota-paced, resumable batches; forecast horizon capped at 16 days; LLM eval runs are rate-paced |
| Secret leakage | API key in output, logs, or repo | Key read from `GEMINI_API_KEY` env var only, passed in headers only; a test asserts a sentinel key never appears in serialized output; a repo-scan test fails if a Google API key pattern is ever committed |

## MCP surface

The MCP server is stdio-only (no network listener), and its tools are narrow,
typed, and read-only. No remote MCP transport in v1 by design.

## What we deliberately do not do (v1)

No sandboxing/seccomp, no corpus signing, no rate limiting on stdio, no KMS.
For a local, read-only, single-user tool these are theater; they are documented
here so their absence is a decision, not an oversight.

## Reporting

Found a vulnerability? Open a GitHub issue (no sensitive PoC in public), or
email the maintainer. No bounty — this is an open research project — but fixes
are prioritized and credited.
