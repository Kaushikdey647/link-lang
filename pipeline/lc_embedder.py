"""LangChain Embeddings wrapper — delegates to pipeline.embedder backend.

Backend is selected at construction time (defaults to DEFAULT_BACKEND):
  - "cohere" → Cohere embed-multilingual-v3.0 (1024-dim, asymmetric, rate-limited)
  - "e5"     → multilingual-e5-small (384-dim, local)
"""

from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings

from pipeline.embedder import DEFAULT_BACKEND, embed_passages, embed_query as _embed_query


class ProjectEmbeddings(Embeddings):
    """Backend-agnostic LangChain embeddings for this project."""

    def __init__(self, backend: str | None = None) -> None:
        self.backend = backend or DEFAULT_BACKEND

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embed_passages(texts, backend=self.backend)

    def embed_query(self, text: str) -> List[float]:
        return _embed_query(text, backend=self.backend)

    def __repr__(self) -> str:
        return f"ProjectEmbeddings(backend={self.backend!r})"


# Keep the old name importable for any code that still uses E5Embeddings
E5Embeddings = ProjectEmbeddings
