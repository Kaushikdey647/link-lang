"""Embedding backend selector.

Priority:
  1. Cohere embed-multilingual-v3.0  (COHERE_API_KEY set, 1024-dim)
     - Asymmetric search: embed_documents → search_document, embed_query → search_query
     - Rate-limited to Cohere's 2000 inputs/min cap via token bucket
  2. multilingual-e5-small            (local fallback, 384-dim)
     - No rate limiting (runs locally)

Each backend writes to its own Qdrant collection (name determined by IndexPlan).
"""

from __future__ import annotations

import os
import threading
import time

from dotenv import load_dotenv

# Indexing entrypoints (scripts/index.py) import this module before
# pipeline.indexer, so COHERE_API_KEY/DEFAULT_BACKEND below need .env loaded
# here too — nothing upstream of this module's first import can be relied on
# to have done it (confirmed: `uv run` does not auto-load .env).
load_dotenv()

# ── Backend registry ──────────────────────────────────────────────────────────

VECTOR_DIM_FOR: dict[str, int] = {
    "e5":      384,
    "cohere":  1024,
    "english": 384,
}

_COHERE_KEY = os.environ.get("COHERE_API_KEY", "")

DEFAULT_BACKEND: str = "cohere" if _COHERE_KEY else "e5"
# "english" is always available (local model, no API key needed)
AVAILABLE_BACKENDS: list[str] = (
    ["cohere", "e5", "english"] if _COHERE_KEY else ["e5", "english"]
)

# Backward-compat aliases (used by rag.py, metrics, etc.)
BACKEND: str = DEFAULT_BACKEND
VECTOR_DIM: int = VECTOR_DIM_FOR[DEFAULT_BACKEND]

# ── Token-bucket rate limiter (Cohere only) ───────────────────────────────────
# Cohere embed-multilingual-v3.0: 2000 inputs/min = 33.33 inputs/sec

_COHERE_RATE = 33.0   # inputs/sec — slightly under cap
_COHERE_BURST = 200   # burst: ~6 seconds of headroom


class _TokenBucket:
    """Thread-safe token bucket.  consume(n) blocks until n tokens are available."""

    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, n: int) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity,
                    self._tokens + (now - self._last) * self._rate,
                )
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                wait = (n - self._tokens) / self._rate
            time.sleep(wait)


_limiter = _TokenBucket(_COHERE_RATE, _COHERE_BURST) if _COHERE_KEY else None

# ── Lazy singletons ───────────────────────────────────────────────────────────

_e5_model = None
_minilm_model = None
_cohere_embeddings = None
_device: str | None = None


def _best_device() -> str:
    """sentence-transformers only auto-detects CUDA, not Apple's MPS — without
    this, local embedding silently runs on CPU on Apple Silicon."""
    global _device
    if _device is None:
        import torch
        if torch.backends.mps.is_available():
            _device = "mps"
        elif torch.cuda.is_available():
            _device = "cuda"
        else:
            _device = "cpu"
    return _device


def _get_e5():
    global _e5_model
    if _e5_model is None:
        from sentence_transformers import SentenceTransformer
        _e5_model = SentenceTransformer("intfloat/multilingual-e5-small", device=_best_device())
    return _e5_model


def _get_minilm():
    global _minilm_model
    if _minilm_model is None:
        from sentence_transformers import SentenceTransformer
        _minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=_best_device())
    return _minilm_model


def _get_cohere():
    global _cohere_embeddings
    if _cohere_embeddings is None:
        from langchain_cohere import CohereEmbeddings
        _cohere_embeddings = CohereEmbeddings(
            model="embed-multilingual-v3.0",
            cohere_api_key=_COHERE_KEY,
        )
    return _cohere_embeddings


# ── Public interface ──────────────────────────────────────────────────────────

def embed_passages(texts: list[str], backend: str | None = None,
                   batch_size: int = 96) -> list[list[float]]:
    """Embed passage texts for indexing.  Blocks if Cohere rate limit is close."""
    backend = backend or DEFAULT_BACKEND
    if backend == "cohere":
        if _limiter:
            _limiter.consume(len(texts))
        return _get_cohere().embed_documents(texts)
    if backend == "english":
        # Symmetric: English questions on both sides — no prefix
        return _get_minilm().encode(texts, batch_size=batch_size,
                                    normalize_embeddings=True).tolist()
    prefixed = [f"passage: {t}" for t in texts]
    return _get_e5().encode(prefixed, batch_size=batch_size,
                            normalize_embeddings=True).tolist()


def embed_query(text: str, backend: str | None = None) -> list[float]:
    """Embed a query for ANN search.  Counts as 1 input against the rate limit."""
    backend = backend or DEFAULT_BACKEND
    if backend == "cohere":
        if _limiter:
            _limiter.consume(1)
        return _get_cohere().embed_query(text)
    if backend == "english":
        return _get_minilm().encode([text], normalize_embeddings=True)[0].tolist()
    return _get_e5().encode([f"query: {text}"],
                            normalize_embeddings=True)[0].tolist()
