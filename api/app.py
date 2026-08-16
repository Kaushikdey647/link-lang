"""Deprecated — safe to delete (frontend_only branch).

The FastAPI serving stack (api/, ui/, pipeline/rag.py, pipeline/query_engines.py,
pipeline/guardrails.py, stt.py, main.py) has been superseded by the Next.js
port under frontend/lib/server/ and frontend/app/api/ (query/voice/health
routes). Ingestion (dataset/, pipeline/chunking.py, pipeline/indexer.py,
pipeline/index_plan.py, scripts/index.py) is unaffected and stays Python.
See CHANGELOG.md.
"""
