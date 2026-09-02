# Contributing

Thanks for taking an interest in this project. It is a public, MIT-licensed
research build: an agent that turns weather data and IPCC AR6 documents into a
typed, cited risk report. Contributions of any size are welcome, from a typo fix
to a correction of a hazard number.

Before you start, please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). For
anything security-related, follow [SECURITY.md](SECURITY.md) instead of opening a
normal issue.

## Setup

Requirements: Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AswaniSahoo/climate-risk-agent.git
cd climate-risk-agent
uv sync --dev
```

Credentials are optional for most of the test suite and required for anything
that embeds text or calls the model:

```bash
cp .env.example .env
```

`.env.example` documents the two supported routes: Vertex AI via Application
Default Credentials (`GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`,
`GOOGLE_CLOUD_LOCATION`), or a single AI Studio key (`GEMINI_API_KEY`). Real
`.env` files are git-ignored. Never paste a key into an issue, a pull request, or
a commit.

The IPCC corpus is not in the repository. Fetch it once:

```bash
uv run python -m scripts.download_ipcc
```

That downloads IPCC AR6 WG1 SPM, Chapter 11 and Chapter 12 into `data/ipcc/`
(git-ignored, about 50 MB, idempotent, safe to re-run). Tests that parse the PDFs
skip themselves when the corpus is absent, so you can run the suite without it.
When the app is deployed it self-provisions the same corpus on first boot, as
described in [DEPLOY.md](DEPLOY.md).

Run the UI:

```bash
uv run streamlit run ui/app.py
```

## Tests, lint, and type-check

CI runs on every push to `main` and every pull request. Run the same four
commands locally before you open a PR, in this order:

```bash
uv sync --dev
uv run ruff check .
uv run mypy
uv run pytest -q
```

These are the exact commands in `.github/workflows/ci.yml`. Ruff and mypy gate
the build, so a lint or type error fails CI the same way a failing test does.

## Evals are the release gate

The retrieval and end-to-end evals are deliberately not in CI: they need the
corpus, the embedding cache, and model credentials. They are a manual gate, run
before a release, and the rule from [DEPLOY.md](DEPLOY.md) is:

> Rule: run both, publish the numbers in README/STATE, THEN tag/deploy.
> A `false_answer > 0` is a release blocker, full stop.

The two gate commands:

```bash
uv run python -m evals.run_retrieval_eval   # recall@k per slice against the frozen set
uv run python -m evals.run_e2e_eval         # refusal matrix, false_answer must be 0
```

Two frozen question sets exist, and the split is the point:

- `evals/gold_set.json`, 60 questions, the dev set. It steered chunking, `top_k`
  and the scope guard, so it can never be called held-out. Run it freely and
  diagnose against it. Adding questions is allowed here (the `cid-table` slice
  was added this way); re-freeze `evals/gold_set.sha256` and update the quotas
  in `tests/test_gold_set.py` in the same commit.
- `evals/gold_set_v2.json`, 105 questions, the held-out test set. It runs at
  release gates only (`EVAL_SET=test`). Failures found there are diagnosed on the
  dev set, never by iterating against the test set, and every published test-set
  number carries its exposure count. Peeking between gates burns the split.

Both sets are pinned by SHA-256. If you change a question, the hash changes and
the published numbers no longer describe the same benchmark, so say so in the PR.

Other harnesses exist for diagnosis and are not part of the gate:
`evals/run_graph_eval.py` (the real agent path), `evals/claim_judge.py`
(claim-level support), and `evals/determinism_probe.py`.

## Commit messages

The log uses Conventional Commits: one line, lowercase after the prefix,
imperative mood, no body unless the change genuinely needs one. Optional scope in
parentheses. Prefixes in use: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`,
`ci`, `perf`.

```
feat(rag): caption-carry chunking - table column labels reach every row chunk
fix(scope): refuse marine heatwaves before the supported-hazard check
docs: publish claim-judge + CI numbers, roadmap current
```

Run `git log --oneline -20` and match what is already there.

## Pull requests

Fill in [the PR template](.github/PULL_REQUEST_TEMPLATE.md). What is expected:

- Tests pass, ruff and mypy are clean, and new behaviour comes with a test.
- No eval regressions. If you touched retrieval, chunking, prompts, the scope
  guard, the model pin, or the agent graph, run the gate above and paste the
  numbers. `false_answer` must stay at 0.
- [LIMITATIONS.md](LIMITATIONS.md) is updated whenever a published number or a
  caveat changes, and README numbers are re-synced to match. A stale number in
  the README is treated as a bug, not a documentation nit.
- The PR title is a one-line conventional commit, because that is what lands in
  the log.
- No unrelated reformatting in the same diff.
- UI changes: include a screenshot.

## Data correctness is a first-class bug

This project's claim is that its numbers and citations hold up. So a wrong return
level, a citation that points at a page which does not support the claim, or a
coordinate mapped to the wrong IPCC AR6 region is a bug of the same severity as a
crash, and it gets the same priority.

Report those with the
[data correctness template](.github/ISSUE_TEMPLATE/data_correctness.md). Include
the location and hazard, what you expected against what the agent returned, and
the source you checked (the Open-Meteo archive, ERA5 directly, or the IPCC page
number). If you know which eval slice it belongs to, name it: `single_page`,
`multi_page`, `regional_table`, `premise_injection`, `duplicate_region`,
`out_of_scope_hazard`, `out_of_corpus`.

Refusal errors belong in the same category: an answer that should have been a
refusal, or a refusal for a question the agent can actually support, is measured
by the e2e refusal matrix and is worth reporting.

Everything else goes to the [bug report](.github/ISSUE_TEMPLATE/bug_report.md) or
[question](.github/ISSUE_TEMPLATE/question.md) template.
