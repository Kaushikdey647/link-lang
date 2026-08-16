"""Deprecated — safe to delete.

Local/API embedding (e5, Cohere, local MiniLM/BM25) was removed in favor of
Qdrant Cloud server-side inference (see pipeline/indexer.py's
MINILM_INFERENCE_MODEL/BM25_INFERENCE_MODEL and CHANGELOG.md). The one
constant still needed, VECTOR_DIM_FOR, now lives in pipeline/index_plan.py.
Nothing in the codebase imports this module anymore.
"""
