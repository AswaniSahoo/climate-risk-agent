# Climate-Risk Analyst Agent

An open-source **agent** (not a chatbot) that turns live weather data and authoritative climate documents into a **grounded, cited, structured risk report** for a location, hazard, and time horizon.

> **Status: Week 1 — core spine shipped.** The agent runs end-to-end and produces a validated `RiskReport` from live forecast data. RAG citations, ERA5 hazard statistics, evals, and MCP are on the roadmap below. Built in public.

## Why an agent, not a chatbot

A chatbot predicts plausible text — ask it about flood risk and it invents a number. This agent **plans, fetches real data, and grounds every claim**, then returns a typed report it can back up (or refuses when a question is out of scope).

## What works today

- **Typed output contract** — a Pydantic `RiskReport` (risk level, drivers, citations, data provenance, confidence, refusal). Bad output can't be constructed: the level is a 4-value enum, confidence is bounded `[0,1]`, and a report must either assert a risk *or* refuse — never both.
- **Live forecast tool** — `get_forecast` calls Open-Meteo (free, no key) and returns a typed daily series.
- **3-node LangGraph agent** — `plan → call → synthesize`, with a refusal short-circuit for unsupported hazards.
- **16 tests, all green** — HTTP is mocked, so tests are fast, offline, and deterministic.

## Architecture

```
query (location, hazard, horizon)
        │
        ▼
   LangGraph state machine
   plan ──▶ call ──▶ synthesize ──▶ RiskReport (typed, provenanced)
     │
     └─(unsupported hazard)─▶ refusal
```

## Use it over MCP

The tools are exposed as an [MCP](https://modelcontextprotocol.io) server, so any MCP client (Claude Desktop, Cursor, the MCP Inspector) can call them — no custom glue per app.

```bash
uv run mcp dev tools/mcp_server.py   # opens the MCP Inspector
```

`get_forecast` exposed as an MCP tool — the input schema is auto-generated from the Python function's type hints:

![forecast tool exposed over MCP](assets/mcp-inspector-tools.png)

Calling it live from the Inspector (real Open-Meteo data, fetched through MCP):

![calling the forecast tool over MCP](assets/mcp-inspector-run.gif)

![forecast tool result](assets/mcp-inspector-result.png)

## Quickstart

```bash
uv sync                      # install deps
uv run pytest                # run the test suite (18 green)
uv run python -m scripts.demo  # live end-to-end demo → prints a RiskReport
```

Example output (real Open-Meteo data):

```json
{
  "location": "Rourkela",
  "hazard": "extreme_precip",
  "risk_level": "low",
  "summary": "Peak daily rainfall of 12.1 mm over 7 days.",
  "confidence": 0.3,
  "provenance": [{ "source": "Open-Meteo", "retrieved_at": "..." }]
}
```

## Tech stack

Python · [uv](https://docs.astral.sh/uv/) · Pydantic v2 · LangGraph · httpx · pytest.

## Data & attribution

- **Forecasts & climatology:** [Open-Meteo](https://open-meteo.com/) (forecast + ERA5 archive APIs), licensed **CC-BY 4.0**.
- **Reanalysis:** ERA5, Copernicus Climate Change Service (C3S) / ECMWF.
- **Climate assessment:** IPCC AR6 WG1 (SPM + Chapter 11), © IPCC — reused for research under IPCC's terms.

Hazard return levels are point-interpolated ERA5 reanalysis (~25 km), not station observations — see [docs/hazard-data-source.md](docs/hazard-data-source.md) and the `representativeness` field on every `HazardStat`.

## Roadmap

- [ ] IPCC AR6 RAG with page-level citations
- [ ] ERA5 hazard statistics (return periods via extreme-value analysis) + skill-aware confidence
- [ ] Eval harness with numbers (retrieval recall@k, citation validity, numeric accuracy, latency, cost)
- [x] MCP server (tools usable from any MCP client) — `get_forecast` exposed, demoed in the MCP Inspector
- [ ] Gemini-backed synthesis, guardrails, observability
- [ ] FastAPI + Streamlit UI, Docker, deployed demo

## Limitations (honest)

Day-1 risk thresholds are crude placeholders (a rainfall/temperature cutoff, fixed low confidence). Real grounding — climate statistics and cited evidence — is what the coming weeks add. The spine ships first, then the intelligence.
