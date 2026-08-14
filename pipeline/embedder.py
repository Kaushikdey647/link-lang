"""Singleton wrapper around multilingual-e5-small.

e5 models require a task prefix:
  - "query: <text>"   when embedding a user query
  - "passage: <text>" when embedding corpus passages
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "intfloat/multilingual-e5-small"
_VECTOR_DIM = 384

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_passages(texts: list[str], batch_size: int = 256) -> np.ndarray:
    prefixed = [f"passage: {t}" for t in texts]
    return _get_model().encode(prefixed, batch_size=batch_size, normalize_embeddings=True)


def embed_query(text: str) -> np.ndarray:
    return _get_model().encode([f"query: {text}"], normalize_embeddings=True)[0]


VECTOR_DIM: int = _VECTOR_DIM
