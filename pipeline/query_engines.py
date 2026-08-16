"""Query engine — retrieval for the one supported IndexPlan.

EnglishPivotQueryEngine is the system's one retrieval strategy: RRF fusion of
a dense search (English-translated query vs. english_query embeddings) and a
BM25 sparse search (original vernacular query vs. the vernacular passage
text). Both vectors are computed server-side by Qdrant Cloud (Document(...));
see CHANGELOG.md for why the e5/cohere vernacular-embedding VernacularQueryEngine
was removed. The vernacular query is never discarded — it's also what reaches
the generation prompt, so translation happens only internally, scoped to the
dense side of retrieval.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import truststore; truststore.inject_into_ssl()
from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, FusionQuery, Fusion, MatchAny, MatchValue,
    Prefetch,
    Document as QdrantDocument,  # aliased: langchain_core.documents.Document is already "Document" here
)

from pipeline.index_plan import IndexPlan
from pipeline.indexer import MINILM_INFERENCE_MODEL, BM25_INFERENCE_MODEL

load_dotenv()

# BCP-47 codes for Sarvam translate API
_SARVAM_LANG: dict[str, str] = {
    "hi": "hi-IN", "bn": "bn-IN", "gu": "gu-IN", "kn": "kn-IN",
    "ml": "ml-IN", "mr": "mr-IN", "ne": "ne-IN", "or": "or-IN",
    "pa": "pa-IN", "sa": "sa-IN", "ta": "ta-IN", "te": "te-IN",
    "ur": "ur-IN", "as": "as-IN",
}


def _translate_to_english(text: str, lang: str) -> str:
    """Translate a vernacular query to English via Sarvam translate API."""
    from sarvamai import SarvamAI
    client = SarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY", ""))
    r = client.text.translate(
        input=text,
        source_language_code=_SARVAM_LANG.get(lang, "auto"),
        target_language_code="en-IN",
        model="sarvam-translate:v1",
    )
    return r.translated_text


def _dedupe(hits: list[Document], top_k: int) -> list[Document]:
    """Deduplicate by passage_id, keeping the first (highest-score) occurrence."""
    seen: dict[str, Document] = {}
    for doc in hits:
        pid = doc.metadata.get("passage_id", doc.metadata.get("chunk_id"))
        if pid not in seen:
            seen[pid] = doc
        if len(seen) >= top_k:
            break
    return list(seen.values())


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseQueryEngine(ABC):
    """Retrieval strategy for one IndexPlan.

    chunk_types defaults to the plan's own chunkers but can be narrowed per
    request (e.g. api/routes/query.py's QueryRequest.chunk_types), so it's
    tracked separately rather than always reading self.plan.chunkers.
    """

    def __init__(self, plan: IndexPlan, chunk_types: list[str] | None = None):
        self.plan = plan
        self.chunk_types = chunk_types or plan.chunkers

    @abstractmethod
    def retrieve(self, query: str, lang: str, top_k: int) -> list[Document]:
        """Return top_k deduplicated Documents for this plan's collection."""

    def build_filter(self, lang: str) -> Filter:
        return Filter(must=[
            FieldCondition(key="metadata.lang", match=MatchValue(value=lang)),
            FieldCondition(key="metadata.chunk_type", match=MatchAny(any=self.chunk_types)),
        ])


# ---------------------------------------------------------------------------
# English-pivot RRF hybrid (the one supported strategy)
# ---------------------------------------------------------------------------

class EnglishPivotQueryEngine(BaseQueryEngine):
    """RRF fusion of:
      - dense: English-translated query vs. english_query embeddings
      - sparse (BM25/IDF): original vernacular query vs. parent_passage text

    Both vectors are computed server-side by Qdrant Cloud inference
    (Document(text=..., model=...)) — no local model inference. Talks to
    Qdrant directly (Prefetch/FusionQuery aren't exposed via a vectorstore
    wrapper). The vernacular `query` is never translated away from the
    caller's perspective — only this engine's internal dense prefetch sees
    the English version.
    """

    SPARSE_VECTOR_NAME = "bm25"

    def __init__(self, plan: IndexPlan, client: QdrantClient, chunk_types: list[str] | None = None):
        super().__init__(plan, chunk_types)
        self.client = client

    def _sparse_query_vector(self, text: str) -> QdrantDocument:
        return QdrantDocument(text=text, model=BM25_INFERENCE_MODEL)

    def _dense_query_vector(self, text: str) -> QdrantDocument:
        return QdrantDocument(text=text, model=MINILM_INFERENCE_MODEL)

    def retrieve(self, query: str, lang: str, top_k: int) -> list[Document]:
        english_query = _translate_to_english(query, lang)
        qfilter = self.build_filter(lang)

        results = self.client.query_points(
            collection_name=self.plan.collection_name,
            prefetch=[
                Prefetch(
                    query=self._dense_query_vector(english_query),
                    using=None,  # default/unnamed dense vector
                    filter=qfilter,
                    limit=top_k * 4,
                ),
                Prefetch(
                    query=self._sparse_query_vector(query),
                    using=self.SPARSE_VECTOR_NAME,
                    filter=qfilter,
                    limit=top_k * 4,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k * 4,
            with_payload=True,
        ).points

        docs = [
            Document(page_content=r.payload.get("page_content", ""), metadata=r.payload.get("metadata", {}))
            for r in results
        ]
        return _dedupe(docs, top_k)


def build_query_engine(
    plan: IndexPlan, *, client: QdrantClient, chunk_types: list[str] | None = None,
) -> EnglishPivotQueryEngine:
    """Only english_query plans are supported now (see CHANGELOG.md)."""
    if "english_query" not in plan.chunkers:
        raise NotImplementedError(
            f"Unsupported chunkers {plan.chunkers!r} — only plans containing "
            "'english_query' are supported now (see CHANGELOG.md)."
        )
    return EnglishPivotQueryEngine(plan, client, chunk_types)
