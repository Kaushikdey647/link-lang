"""Query engines — retrieval strategy for one IndexPlan.

Mirrors pipeline/chunking.py's BaseChunker/REGISTRY/build_chunker() pattern:
a base class, concrete strategies, and a factory that picks one from an
IndexPlan's (backend, chunkers, split) axes.

Two engines today:
  - VernacularQueryEngine — e5/cohere backends, passage/sentence/qa_pair
    chunkers. Query embedded as-is; the multilingual model handles
    cross-lingual matching. Plain dense search via the existing LangChain
    QdrantVectorStore path.
  - EnglishPivotQueryEngine — english backend + english_query chunker. RRF
    fusion of a dense search (English-translated query vs. english_query
    embeddings) and a BM25 sparse search (original vernacular query vs. the
    vernacular passage text). The vernacular query is never discarded — it's
    also what reaches the generation prompt, so translation happens only
    internally, scoped to the dense side of retrieval.

More engines (one per new retrieval strategy) register in _ENGINE_FOR_CHUNKER.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, FusionQuery, Fusion, MatchAny, MatchValue,
    Prefetch, SparseVector,
)

from pipeline.embedder import embed_query
from pipeline.index_plan import IndexPlan

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
    client = SarvamAI(api_key=os.environ.get("SARVAM_API_KEY", ""))
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
# Strategy 1: vernacular dense (e5 / cohere)
# ---------------------------------------------------------------------------

class VernacularQueryEngine(BaseQueryEngine):
    """e5 / cohere backends, passage/sentence/qa_pair chunkers — plain dense
    search, query embedded as-is (multilingual model handles cross-lingual
    matching)."""

    def __init__(self, plan: IndexPlan, vectorstore: QdrantVectorStore, chunk_types: list[str] | None = None):
        super().__init__(plan, chunk_types)
        self.vectorstore = vectorstore

    def retrieve(self, query: str, lang: str, top_k: int) -> list[Document]:
        hits = self.vectorstore.similarity_search(
            query, k=top_k * 4, filter=self.build_filter(lang),
        )
        return _dedupe(hits, top_k)


# ---------------------------------------------------------------------------
# Strategy 2: English-pivot RRF hybrid (english backend + english_query)
# ---------------------------------------------------------------------------

class EnglishPivotQueryEngine(BaseQueryEngine):
    """english backend + english_query chunker — RRF fusion of:
      - dense: English-translated query vs. english_query embeddings
      - sparse (BM25/IDF): original vernacular query vs. parent_passage text

    Talks to Qdrant directly (LangChain's vectorstore wrapper doesn't expose
    Prefetch/FusionQuery). The vernacular `query` is never translated away
    from the caller's perspective — only this engine's internal dense
    prefetch sees the English version.
    """

    SPARSE_VECTOR_NAME = "bm25"
    _SPARSE_MODEL_NAME = "Qdrant/bm25"

    def __init__(self, plan: IndexPlan, client: QdrantClient, chunk_types: list[str] | None = None):
        super().__init__(plan, chunk_types)
        self.client = client
        self._sparse_model = None

    def _get_sparse_model(self):
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding
            self._sparse_model = SparseTextEmbedding(model_name=self._SPARSE_MODEL_NAME)
        return self._sparse_model

    def _sparse_query_vector(self, text: str) -> SparseVector:
        emb = next(iter(self._get_sparse_model().query_embed(text)))
        return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())

    def _dense_query_vector(self, text: str) -> list[float]:
        vector = embed_query(text, backend=self.plan.backend)
        return vector.tolist() if hasattr(vector, "tolist") else vector

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ENGINE_FOR_CHUNKER: dict[str, type[BaseQueryEngine]] = {
    "english_query": EnglishPivotQueryEngine,
}


def build_query_engine(
    plan: IndexPlan, *,
    vectorstore: QdrantVectorStore | None = None,
    client: QdrantClient | None = None,
    chunk_types: list[str] | None = None,
) -> BaseQueryEngine:
    """Pick the retrieval strategy for a plan. Default: VernacularQueryEngine."""
    engine_cls = next(
        (cls for chunker, cls in _ENGINE_FOR_CHUNKER.items() if chunker in plan.chunkers),
        VernacularQueryEngine,
    )
    if engine_cls is EnglishPivotQueryEngine:
        return EnglishPivotQueryEngine(plan, client, chunk_types)
    return VernacularQueryEngine(plan, vectorstore, chunk_types)
