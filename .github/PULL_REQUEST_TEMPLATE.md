## What this changes

One or two sentences, and the issue it closes if there is one.

## Checklist

- [ ] `uv run pytest -q` passes
- [ ] `uv run ruff check .` is clean
- [ ] `uv run mypy` is clean
- [ ] Evals re-run if this touches retrieval, chunking, prompts, the scope guard,
      the model pin, or the agent graph. Numbers pasted below, `false_answer` is 0.
- [ ] LIMITATIONS.md and the README numbers are in sync with any number or caveat
      this change moves
- [ ] The PR title is a one-line conventional commit, lowercase after the prefix

## Eval numbers, if re-run

```
# uv run python -m evals.run_retrieval_eval
# uv run python -m evals.run_e2e_eval
```
