# Climate-Risk Analyst Agent — Streamlit UI over the LangGraph agent.
#
# The IPCC corpus (~50 MB, public) is baked in at BUILD time so the container
# is self-sufficient. Retrieval at runtime:
#   - no Gemini auth        -> BM25-only (loud fallback; measured 76% R@3)
#   - GEMINI_API_KEY set    -> hybrid BM25+dense (measured 82% R@3) + cited
#                              LLM answers; first boot embeds the corpus into
#                              a resumable on-disk cache (paid tier: minutes)
#
# Two stages: the builder holds uv and its package cache (a full duplicate of
# .venv under UV_LINK_MODE=copy); the final image gets only /app. Cold start
# reads fewer bytes, and there is no package manager in the runtime image.
#
# Build:  docker build -t climate-risk-agent .
# Run  :  docker run -p 7860:7860 -e GEMINI_API_KEY=... climate-risk-agent

# ---------- builder ----------
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
# Pin the interpreter to the image's own: a uv-managed download would live
# outside /app and the copied .venv would point at nothing in the final stage.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=/usr/local/bin/python3.11 \
    UV_PYTHON_DOWNLOADS=never

# Dependency layer first: cache survives source-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

# The two bake steps call the venv's python directly. `uv run` re-syncs first,
# and re-syncing WITHOUT --no-dev drags mypy + ruff + pytest into the runtime
# .venv — observed in the build log for this image.
#
# Bake the corpus (idempotent: skips files already present, e.g. from a local
# data/ context copy that also carries the embedding cache).
RUN .venv/bin/python -m scripts.download_ipcc

# Pre-compute the chunk cache. Parsing 439 PDF pages into 2730 chunks costs
# ~53 s and used to run on EVERY cold container, dominating the first response
# (measured 219 s end-to-end). Baking it makes that step 0.05 s at runtime.
# The AR6 region polygons need no bake step any more: they are a GeoJSON file
# committed at tools/ar6/, read with shapely, so nothing downloads at runtime.
RUN .venv/bin/python -c "from rag.corpus import build_chunk_cache; build_chunk_cache()"

# ---------- runtime ----------
FROM python:3.11-slim

# Same path as the builder: uv writes absolute paths into .venv/bin shebangs
# and pyvenv.cfg, so /app/.venv must land at /app/.venv to stay runnable.
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# --chown on the COPY, not a later `chown -R`: that would rewrite every file
# in /app into a second layer, roughly doubling the image.
RUN useradd -m appuser
COPY --from=builder --chown=appuser:appuser /app /app
USER appuser

# Cloud Run (and HF Spaces Docker SDK) route traffic to this port.
EXPOSE 7860

# The venv's own streamlit, not `uv run`: uv would re-resolve the lock and
# re-check the environment on every container start.
CMD ["/app/.venv/bin/streamlit", "run", "ui/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
