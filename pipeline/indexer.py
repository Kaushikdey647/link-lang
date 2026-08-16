"""Embed passage chunks for a language and upsert into Qdrant via LangChain.

Every indexing run is described by an IndexPlan (backend × chunkers × split).
The plan determines the Qdrant collection name deterministically:

    msmarco_xi__{backend}__{chunkers_sorted}__{split}

Usage:
    python -m pipeline.indexer --lang hi --backend english --chunkers english_query
    python -m pipeline.indexer --lang hi --backend cohere --chunkers passage sentence qa_pair
"""

from __future__ import annotations

import argparse
import uuid
from itertools import islice
from typing import Iterable, Iterator

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PayloadSchemaType, SparseVectorParams, Modifier,
    PointVectors, SparseVector,
)
from tqdm import tqdm

from dataset import iter_passages, load_language
from pipeline.chunking import BaseChunker, Chunk, DEFAULT as DEFAULT_CHUNKER
from pipeline.lc_embedder import ProjectEmbeddings
from pipeline.index_plan import IndexPlan
from pipeline.embedder import DEFAULT_BACKEND, AVAILABLE_BACKENDS

# Sparse (BM25/IDF) vector space, only used by english_query plans for RRF
# hybrid retrieval (pipeline/query_engines.py:EnglishPivotQueryEngine).
SPARSE_VECTOR_NAME = "bm25"
_sparse_model = None


def _get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_model


def sparse_vectors_for(texts: list[str]) -> list[SparseVector]:
    """BM25 sparse vectors for a batch of (vernacular passage) texts."""
    embeddings = list(_get_sparse_model().embed(texts))
    return [SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in embeddings]

QDRANT_URL = "http://localhost:6333"
# qdrant-client defaults to a 5s request timeout, which large embed+upsert
# batches (or a payload-index rebuild on a big collection) can exceed under
# load — that raised a timeout that killed the whole language indexing run
# instead of just one slow request.
QDRANT_INDEXING_TIMEOUT = 60


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=QDRANT_INDEXING_TIMEOUT)


def ensure_collection(client: QdrantClient, plan: IndexPlan) -> None:
    """Create the collection for this plan if it doesn't exist; ensure payload indexes."""
    collection  = plan.collection_name
    needs_bm25  = "english_query" in plan.chunkers
    # get_collections() only lists physical collections, not aliases — a
    # migrated collection (scripts/migrate_bm25.py) lives under an aliased
    # name, so without this, ensure_collection would think it doesn't exist
    # and fail trying to create_collection() over an already-aliased name
    # (confirmed empirically: Qdrant rejects that with a 400).
    existing = {c.name for c in client.get_collections().collections}
    existing |= {a.alias_name for a in client.get_aliases().aliases}

    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=plan.vector_dim, distance=Distance.COSINE),
            sparse_vectors_config=(
                {SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)}
                if needs_bm25 else None
            ),
        )
    elif needs_bm25:
        # Qdrant can't add a brand-new named vector to a collection that was
        # created without one (confirmed empirically — update_collection only
        # tunes params of vectors already in the schema). A collection from
        # before the RRF hybrid strategy existed needs an explicit one-time
        # migration, not something to attempt silently mid-indexing-run.
        info = client.get_collection(collection)
        if not info.config.params.sparse_vectors or SPARSE_VECTOR_NAME not in info.config.params.sparse_vectors:
            raise RuntimeError(
                f"Collection {collection!r} predates the RRF hybrid strategy and is "
                f"missing the {SPARSE_VECTOR_NAME!r} sparse vector space. Run "
                f"`python -m scripts.migrate_bm25 --collection {collection}` first."
            )

    for field_name, schema in [
        ("metadata.lang",        PayloadSchemaType.KEYWORD),
        ("metadata.chunk_type",  PayloadSchemaType.KEYWORD),
        ("metadata.query_id",    PayloadSchemaType.INTEGER),
        ("metadata.is_selected", PayloadSchemaType.BOOL),
    ]:
        client.create_payload_index(collection, field_name, schema)


def attach_sparse_vectors(client: QdrantClient, plan: IndexPlan,
                          chunks: list[Chunk], ids: list[str]) -> None:
    """For english_query plans, compute + push BM25 sparse vectors onto
    already-upserted point IDs, sourced from each chunk's vernacular
    parent_passage text (falling back to chunk.text if absent). No-op for
    plans that don't use the RRF hybrid strategy."""
    if "english_query" not in plan.chunkers:
        return
    texts   = [c.parent_passage or c.text for c in chunks]
    vectors = sparse_vectors_for(texts)
    client.update_vectors(
        collection_name=plan.collection_name,
        points=[
            PointVectors(id=pid, vector={SPARSE_VECTOR_NAME: vec})
            for pid, vec in zip(ids, vectors)
        ],
    )


def _chunk_to_document(chunk: Chunk) -> Document:
    meta: dict = {
        "chunk_id":       chunk.chunk_id,
        "chunk_type":     chunk.chunk_type,
        "lang":           chunk.lang,
        "passage_id":     chunk.passage_id,
        "query_id":       chunk.query_id,
        "is_selected":    chunk.is_selected,
        "query":          chunk.query,
        "answer":         chunk.answer,
        "query_type":     chunk.query_type,
        "sentence_index": chunk.sentence_index,
    }
    if chunk.parent_passage:
        meta["parent_passage"] = chunk.parent_passage
    return Document(page_content=chunk.text, metadata=meta)


def _batched(it: Iterable, n: int) -> Iterator[list]:
    it = iter(it)
    while batch := list(islice(it, n)):
        yield batch


def index_language(
    lang: str,
    plan: IndexPlan,
    batch_size: int = 256,
    chunker: BaseChunker | None = None,
) -> QdrantVectorStore:
    """Embed and upsert all passage chunks for one language according to the plan.

    Args:
        lang:       2-letter language code.
        plan:       IndexPlan describing backend, chunkers, and split.
        batch_size: Documents per upsert batch.
        chunker:    Pre-built chunker. If None, built from plan.chunkers.

    Returns:
        Configured QdrantVectorStore instance.
    """
    from pipeline.chunking import build_chunker
    if chunker is None:
        chunker = build_chunker(plan.chunkers)

    qdrant_client = _get_qdrant_client()
    ensure_collection(qdrant_client, plan)

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=plan.collection_name,
        embedding=ProjectEmbeddings(backend=plan.backend),
    )

    ds       = load_language(lang, splits=(plan.split,))
    passages = iter_passages(ds[plan.split], lang, translated=True)

    all_chunks: Iterator[Chunk] = (
        chunk for record in passages for chunk in chunker.chunk(record)
    )

    total = 0
    for batch in tqdm(_batched(all_chunks, batch_size),
                      desc=f"Indexing {lang}/{plan.split} [{plan.collection_name}]",
                      unit="batch"):
        docs = [_chunk_to_document(c) for c in batch]
        ids  = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.chunk_id)) for c in batch]
        vectorstore.add_documents(docs, ids=ids)
        attach_sparse_vectors(qdrant_client, plan, batch, ids)
        total += len(docs)

    print(f"Done. {total:,} chunks for lang={lang!r} in {plan.collection_name!r}.")
    return vectorstore


def get_vectorstore(plan: IndexPlan) -> QdrantVectorStore:
    """Return a QdrantVectorStore for an existing collection described by plan."""
    return QdrantVectorStore(
        client=_get_qdrant_client(),
        collection_name=plan.collection_name,
        embedding=ProjectEmbeddings(backend=plan.backend),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",     default="hi")
    parser.add_argument("--backend",  default=DEFAULT_BACKEND, choices=AVAILABLE_BACKENDS)
    parser.add_argument("--chunkers", nargs="+", default=["passage", "sentence", "qa_pair"])
    parser.add_argument("--split",    default="train")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    plan = IndexPlan(backend=args.backend, chunkers=args.chunkers, split=args.split)
    print(f"Collection: {plan.collection_name}")
    index_language(args.lang, plan, args.batch_size)
