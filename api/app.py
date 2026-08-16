"""FastAPI application entry point.

Run with:
    uvicorn api.app:app --reload --port 8000
Production (Render, see render.yaml/Dockerfile):
    uvicorn api.app:app --host 0.0.0.0 --port $PORT

Endpoints:
    POST /query        — text query → RAG answer
    POST /voice        — audio file → transcript → RAG answer
    GET  /health       — liveness check (Render healthCheckPath)
    GET  /docs         — OpenAPI UI (auto-generated)

Env vars affecting this module (see README.md for the full list):
    ALLOWED_ORIGINS  — comma-separated CORS origins; "*" (default) if unset.
    ADMIN_UI_ENABLED — "false" to skip mounting the read-only Gradio /admin
                       dashboard (e.g. to shrink prod attack surface/cold
                       start); mounted by default.
"""

import os

import truststore; truststore.inject_into_ssl()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.query import router as query_router
from api.routes.voice import router as voice_router
from api.metrics import setup_metrics

app = FastAPI(
    title="Bhasha RAG API",
    description="Voice-enabled multilingual RAG over MSMARCO-XI (Sarvam + Qdrant)",
    version="0.1.0",
)

_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allowed_origins == "*" else [o.strip() for o in _allowed_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(voice_router)

setup_metrics(app)

if os.environ.get("ADMIN_UI_ENABLED", "true").lower() != "false":
    from ui.admin_app import build_admin_ui
    import gradio as gr
    gr.mount_gradio_app(app, build_admin_ui(), path="/admin")


@app.get("/plans", tags=["infra"])
def list_plans() -> dict:
    """List all indexed plans with their embedding metadata."""
    from pipeline.index_plan import load_registry
    return {"plans": list(load_registry().values())}


@app.get("/health", tags=["infra"])
def health() -> dict:
    from qdrant_client import QdrantClient
    from pipeline.indexer import QDRANT_URL, QDRANT_API_KEY
    from pipeline.index_plan import load_registry, sync_registry_with_qdrant
    try:
        client  = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=2)
        # registry.json is local-only (gitignored) — sync from live Qdrant
        # state first so a fresh machine pointed at an already-populated
        # shared/remote cluster reports ready without needing to have run
        # indexing itself (see pipeline/index_plan.py::sync_registry_with_qdrant).
        sync_registry_with_qdrant(client)
        colls   = {c.name for c in client.get_collections().collections}
        registry = load_registry()
        # Ready if at least one registered plan's collection exists in Qdrant
        qdrant_ready = any(name in colls for name in registry)
    except Exception:
        qdrant_ready = False
    status = "ok" if qdrant_ready else "degraded"
    return {"status": status, "qdrant": qdrant_ready, "plans": list(load_registry().keys())}
