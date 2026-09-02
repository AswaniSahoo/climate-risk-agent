---
name: Bug report
about: Something crashes, errors, or behaves differently from what the docs describe
title: ''
labels: bug
assignees: ''
---

## What happened

A short description of the failure.

## What you expected

## How to reproduce

Steps, or the exact command and question:

```bash
# for example
uv run streamlit run ui/app.py
```

Question or input used:

## Error output

Paste the traceback or the log lines. Redact anything that looks like a key.

## Environment

- OS:
- Python version (`uv run python -V`):
- Installed with `uv sync --dev`: yes / no
- Credential route: Vertex ADC / `GEMINI_API_KEY` / none (BM25-only fallback)
- Corpus present in `data/ipcc/`: yes / no
- Commit you are on (`git rev-parse --short HEAD`):

## Checks

- [ ] `uv run pytest -q` was run and the result is reported above
- [ ] This is not a wrong number or a wrong citation. Those go to the data
      correctness template.
