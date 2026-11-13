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
| SSRF / URL manipulation | Caller-influenced request targets | All outbound hosts are hardcoded constants (Open-Meteo forecast + archive, Google generative API, and, when `CRG_JEV` is on, one of two vetted Jev hosts, `https://api.typesafe.ai` or `https://api.experientiallabs.ai`, which `JEV_BASE_URL` can choose between but never extend); pinned by `test_outbound_targets_are_pinned` and `test_jev_targets_and_models_are_pinned` |
| Steered classifier (optional Jev) | A question written to move Jev's triage reading; TypeSafe says such content "can move the answer" | Jev reads only what the lexical stage cannot, never overrides a lexical refusal, and code thresholds its probabilities (`rag/jev_triage.py`); the worst outcome is refusing the sender's own question or routing a hazard-free question to a report that names Jev's reading. The citation check only annotates. Neither touches the answering prompt or citation validation (`tests/test_jev_wiring.py`) |
| Garbage / hostile input | Out-of-range coordinates, absurd horizons | Range-validated at every tool boundary before any request is built (`tools/validation.py`) |
| Denial-of-wallet | Repeated expensive upstream calls | A daily model budget at the one Gemini seam reserves each call's worst-case cost before it starts, counted on the shared Cloud Storage or Upstash counter when one is configured, else per process (`rag/spend_guard.py`, `CRG_DAILY_MODEL_BUDGET_USD`, default US$0.15 per UTC day; the eval runners and the grounding precompute turn it off unless it is set); each UI session gets 5 reports that need new model calls, after which its reports run with model calls paused (`ui/session_budget.py`, `CRG_SESSION_FRESH_REPORTS`); `POST /report`, `/query` and `/triage` allow 10 requests per 60 s per client and answer 429 with Retry-After past that (`api/app.py`, `CRG_API_RATE_LIMIT`, `CRG_API_RATE_WINDOW_S`); Open-Meteo Archive requests are budgeted at 8,000 weighted calls per UTC day and 4,000 per hour, and any 429 stops them until the next full hour (`tools/archive_budget.py`, `CRG_ARCHIVE_DAILY_CALLS`, `CRG_ARCHIVE_HOURLY_CALLS`). Climatology results are memoized per (location, hazard); embeddings are disk-cached and fetched in quota-paced, resumable batches; forecast horizon capped at 16 days; LLM eval runs are rate-paced |
| Secret leakage | API key in output, logs, or repo | Key read from `GEMINI_API_KEY` env var only, passed in headers only; a test asserts a sentinel key never appears in serialized output; a repo-scan test fails if a Google API key pattern is ever committed. `TYPESAFE_API_KEY` is read from the environment only, held in one private attribute, sent only in the Authorization header, and kept out of `repr`, errors, logs and telemetry; error bodies are reduced to a status and an error type because a 422 can echo the input (`tests/test_jev_client.py`) |

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
