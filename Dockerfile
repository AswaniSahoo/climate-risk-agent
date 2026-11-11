# Climate-Risk Analyst Agent — Streamlit UI over the LangGraph agent.
#
# The IPCC corpus (~50 MB, public) and its 2,730 document vectors are baked in
# at BUILD time so the container is self-sufficient. Retrieval at runtime:
#   - no Gemini auth        -> BM25-only (loud fallback; measured 76% R@3)
#   - Gemini auth           -> hybrid BM25+dense (measured 82% R@3) + cited
#                              LLM answers; only each question is embedded,
#                              never the corpus (the build fails without its vectors)
#
# Two stages: the builder holds uv and its package cache (a full duplicate of
# .venv under UV_LINK_MODE=copy); the final image gets only /app. Cold start
# reads fewer bytes, and there is no package manager in the runtime image.
#
# Build:  docker build -t climate-risk-agent .
# Run  :  docker run -p 7860:7860 -e GEMINI_API_KEY=... climate-risk-agent

# ---------- builder ----------
# Base images are pinned by digest: a floating python:3.11-slim rebuilt with a
# newer glibc crashed the app (SIGSEGV) on the Cloud Run gen1 sandbox on
# 2026-10-22. Bump the digest deliberately and redeploy to gen2.
FROM python:3.11-slim@sha256:cce12316475d7d56d605f76e1d811d0dc4e7227f2d4afd29951521264f504ff8 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff /uv /uvx /bin/

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
# Bake the corpus (idempotent: skips PDFs already present in the build context).
RUN .venv/bin/python -m scripts.download_ipcc

# Bake the corpus vectors from the eval-cache-v1 release asset, pinned by the
# sha256 GitHub reports for it, and never from whatever data/cache the build
# context happened to carry: that is wiped first, so a checkout with an empty
# or stale cache builds the same image. A digest mismatch fails the build
# (scripts/fetch_eval_cache.py); unpack_eval_cache.py then refuses an archive
# whose manifest (embedding model, dims, chunker) does not match this repo.
# After a re-chunk or an embedding-model change: rebuild and re-upload the
# asset (scripts/pack_eval_cache.py) and update both values here.
ARG EVAL_CACHE_URL=https://github.com/AswaniSahoo/climate-risk-agent/releases/download/eval-cache-v1/eval-cache-v1.tar.gz
ARG EVAL_CACHE_SHA256=7f91a5ceff12258b9ad07500129bf49d9eaa399fe4494234f023276baa525423
RUN rm -rf data/cache \
 && .venv/bin/python -m scripts.fetch_eval_cache \
        --url "$EVAL_CACHE_URL" --sha256 "$EVAL_CACHE_SHA256" --out /tmp/eval-cache.tar.gz \
 && .venv/bin/python -m scripts.unpack_eval_cache --archive /tmp/eval-cache.tar.gz \
 && rm /tmp/eval-cache.tar.gz

# Pre-compute the chunk cache. Parsing 439 PDF pages into 2730 chunks costs
# ~53 s and used to run on EVERY cold container, dominating the first response
# (measured 219 s end-to-end). Baking it makes that step 0.05 s at runtime.
# The AR6 region polygons need no bake step any more: they are a GeoJSON file
# committed at tools/ar6/, read with shapely, so nothing downloads at runtime.
RUN .venv/bin/python -c "from rag.corpus import build_chunk_cache; build_chunk_cache()"

# Fail the build unless every chunk the image will serve has its document vector
# on disk (2,730 of 2,730 today). Without them, the first report of every cold
# container would embed the missing chunks through the API, 50 at a time.
RUN .venv/bin/python -m scripts.check_baked_index

# ---------- runtime ----------
FROM python:3.11-slim@sha256:cce12316475d7d56d605f76e1d811d0dc4e7227f2d4afd29951521264f504ff8

# Same path as the builder: uv writes absolute paths into .venv/bin shebangs
# and pyvenv.cfg, so /app/.venv must land at /app/.venv to stay runnable.
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# --chown on the COPY, not a later `chown -R`: that would rewrite every file
# in /app into a second layer, roughly doubling the image.
#
# /app itself still needs chowning: WORKDIR created it as root before appuser
# existed, and `COPY --chown` sets ownership on the entries it copies, not on
# the destination directory. Without this the runtime user can read everything
# under /app but cannot CREATE a new top-level entry there — which is what
# writing data/cache or a Streamlit temp dir at the app root needs.
RUN useradd -m appuser && mkdir -p /app && chown appuser:appuser /app
COPY --from=builder --chown=appuser:appuser /app /app
USER appuser

# Cloud Run (and HF Spaces Docker SDK) route traffic to this port.
EXPOSE 7860

# The venv's own streamlit, not `uv run`: uv would re-resolve the lock and
# re-check the environment on every container start.
CMD ["/app/.venv/bin/streamlit", "run", "ui/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
