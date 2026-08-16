"""FastAPI application entry point.

Run with:
    uvicorn api.app:app --reload --port 8000

Endpoints:
    POST /query        — text query → RAG answer
    POST /voice        — audio file → transcript → RAG answer
    GET  /health       — liveness check
    GET  /docs         — OpenAPI UI (auto-generated)
"""

import truststore; truststore.inject_into_ssl()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.query import router as query_router
from api.routes.voice import router as voice_router
from api.metrics import setup_metrics
from ui.admin_app import build_admin_ui
import gradio as gr

app = FastAPI(
    title="Bhasha RAG API",
    description="Voice-enabled multilingual RAG over MSMARCO-XI (Sarvam + Qdrant)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(voice_router)

setup_metrics(app)

# Mount Gradio UI at /ui
gr.mount_gradio_app(app, build_admin_ui(), path="/admin")


@app.get("/plans", tags=["infra"])
def list_plans() -> dict:
    """List all indexed plans with their embedding metadata."""
    from pipeline.index_plan import load_registry
    return {"plans": list(load_registry().values())}


@app.get("/health", tags=["infra"])
def health() -> dict:
    from qdrant_client import QdrantClient
    from pipeline.indexer import QDRANT_URL
    from pipeline.index_plan import load_registry
    try:
        client  = QdrantClient(url=QDRANT_URL, timeout=2)
        colls   = {c.name for c in client.get_collections().collections}
        registry = load_registry()
        # Ready if at least one registered plan's collection exists in Qdrant
        qdrant_ready = any(name in colls for name in registry)
    except Exception:
        qdrant_ready = False
    status = "ok" if qdrant_ready else "degraded"
    return {"status": status, "qdrant": qdrant_ready, "plans": list(load_registry().keys())}
