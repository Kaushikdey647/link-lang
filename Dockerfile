# Bhasha RAG API — production image for Render (or any container host).
# Frontend (frontend/) is a separate Next.js deployment, not included here.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=0

WORKDIR /app

# uv for fast, reproducible installs pinned by uv.lock
RUN pip install --no-cache-dir uv

# Copy dependency manifests first so dependency layers cache across rebuilds
# that only touch application code.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY api api
COPY pipeline pipeline
# dataset/ is imported transitively (pipeline/indexer.py) but its functions
# are only ever called by the indexing CLI, never at serving time — no
# dataset cache/download needed in this image.
COPY dataset dataset
COPY ui ui
COPY constants.py stt.py ./

# Render (and most PaaS hosts) inject the actual listen port via $PORT at
# runtime — shell form (not exec-array) so that substitution happens.
EXPOSE 8000
CMD uv run uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}
