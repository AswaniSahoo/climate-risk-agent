# Climate-Risk Analyst Agent — Streamlit UI over the LangGraph agent.
#
# The IPCC corpus (~50 MB, public) is baked in at BUILD time so the container
# is self-sufficient. Retrieval at runtime:
#   - no Gemini auth        -> BM25-only (loud fallback; measured 82% R@3)
#   - GEMINI_API_KEY set    -> hybrid BM25+dense (measured 91% R@3) + cited
#                              LLM answers; first boot embeds the corpus into
#                              a resumable on-disk cache (paid tier: minutes)
#
# Build:  docker build -t climate-risk-agent .
# Run  :  docker run -p 7860:7860 -e GEMINI_API_KEY=... climate-risk-agent
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependency layer first: cache survives source-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

# Bake the corpus (idempotent: skips files already present, e.g. from a local
# data/ context copy that also carries the embedding cache).
RUN uv run python -m scripts.download_ipcc

# Bake the AR6 region polygons (~1 MB) so lat/lon->region works offline at runtime.
RUN uv run python -c "from tools.ar6_regions import _land_regions; _land_regions()"

# Cloud Run (and HF Spaces Docker SDK) route traffic to this port; runs as non-root.
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 7860

CMD ["uv", "run", "streamlit", "run", "ui/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
