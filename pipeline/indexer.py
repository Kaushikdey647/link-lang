"""Embed passage chunks and upsert into Qdrant. CLI-only, by design.

Every indexing run is described by an IndexPlan (backend x chunkers x split).
The plan determines the Qdrant collection name deterministically:

    msmarco_xi__{backend}__{chunkers_sorted}__{split}

Indexing is intentionally not reachable from the running API/admin server —
no HTTP route triggers it and no admin-UI control exists for it. It's an
occasional, operator-driven, potentially long-running task; keeping it a
separate CLI process means the server has zero control surface for it, and
each language can be resumed/retried/parallelized independently of server
uptime.

CLI entrypoint is scripts/index.py (this module is deliberately import-only —
see the note above run_indexing() for why: ProcessPoolExecutor's "spawn"
start method can't safely re-import a worker function from a module whose
identity is __main__).

Usage:
    uv run python -m scripts.index --langs hi bn --backend english --chunkers english_query
    uv run python -m scripts.index --langs all --workers 4 --backend e5 --chunkers passage sentence qa_pair
    uv run python -m scripts.index --langs hi --limit 5000   # quick test run
"""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PayloadSchemaType, SparseVectorParams, Modifier,
    PointStruct, PointVectors, SparseVector,
    Document as QdrantDocument,  # aliased: langchain_core.documents.Document is already "Document" in this file
)
from tqdm import tqdm

from dataset import count_language_rows, iter_language_rows, iter_passages
from pipeline.chunking import BaseChunker, Chunk, build_chunker
from pipeline.embedder import embed_passages
from pipeline.lc_embedder import ProjectEmbeddings
from pipeline.index_plan import IndexPlan, register_plan, sync_registry_with_qdrant

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_CLUSTER_ENDPOINT") or "http://localhost:6333"
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
# True only when pointed at a real Qdrant Cloud cluster (API key present) —
# local/self-hosted Qdrant doesn't support server-side cloud inference, so
# QDRANT_API_KEY's presence is what gates the Document(text=..., model=...)
# code paths below vs. the local sentence-transformers/fastembed ones.
QDRANT_CLOUD_INFERENCE = bool(QDRANT_API_KEY)

# Qdrant's inference model registry identifiers — exact casing matters, this
# is what actually gets sent to Qdrant Cloud, distinct from the human-readable
# MODEL_NAME_FOR strings in pipeline/index_plan.py.
MINILM_INFERENCE_MODEL = "sentence-transformers/all-minilm-l6-v2"
BM25_INFERENCE_MODEL = "qdrant/bm25"

# qdrant-client defaults to a 5s request timeout, which large embed+upsert
# batches (or a payload-index rebuild on a big collection) can exceed under
# load — that raised a timeout that killed the whole language indexing run
# instead of just one slow request. Cloud inference computes embeddings
# synchronously inside the upsert call, so the remote cluster gets extra
# headroom over the local-embedding case.
QDRANT_INDEXING_TIMEOUT = 120 if QDRANT_CLOUD_INFERENCE else 60


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_INDEXING_TIMEOUT,
        cloud_inference=QDRANT_CLOUD_INFERENCE,
    )


# ---------------------------------------------------------------------------
# Sparse (BM25/IDF) vector space, only used by english_query plans for RRF
# hybrid retrieval (pipeline/query_engines.py:EnglishPivotQueryEngine).
# ---------------------------------------------------------------------------

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
                f"`uv run python -m scripts.migrate_bm25 --collection {collection}` first."
            )

    for field_name, schema in [
        ("metadata.lang",        PayloadSchemaType.KEYWORD),
        ("metadata.chunk_type",  PayloadSchemaType.KEYWORD),
        ("metadata.query_id",    PayloadSchemaType.INTEGER),
        ("metadata.is_selected", PayloadSchemaType.BOOL),
    ]:
        client.create_payload_index(collection, field_name, schema)


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


def _upsert_batch(client: QdrantClient, plan: IndexPlan,
                  vectorstore: QdrantVectorStore, chunks: list[Chunk]) -> None:
    """Embed + upsert one batch. For english_query plans this does dense AND
    sparse in a single Qdrant round-trip (previously: LangChain's
    add_documents() for dense, then a second update_vectors() call for
    sparse) — halves network round-trips for that plan type."""
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.chunk_id)) for c in chunks]
    if "english_query" in plan.chunkers:
        docs = [_chunk_to_document(c) for c in chunks]
        if QDRANT_CLOUD_INFERENCE:
            # Server-side embedding: raw text goes to Qdrant Cloud, which
            # computes both the dense (MiniLM) and sparse (BM25) vectors —
            # no local sentence-transformers/fastembed inference at all.
            points = [
                PointStruct(
                    id=pid,
                    vector={
                        "": QdrantDocument(text=c.text, model=MINILM_INFERENCE_MODEL),
                        SPARSE_VECTOR_NAME: QdrantDocument(
                            text=c.parent_passage or c.text, model=BM25_INFERENCE_MODEL,
                        ),
                    },
                    payload={"page_content": doc.page_content, "metadata": doc.metadata},
                )
                for pid, c, doc in zip(ids, chunks, docs)
            ]
        else:
            dense  = embed_passages([c.text for c in chunks], backend=plan.backend)
            sparse = sparse_vectors_for([c.parent_passage or c.text for c in chunks])
            points = [
                PointStruct(
                    id=pid,
                    vector={"": dv, SPARSE_VECTOR_NAME: sv},
                    payload={"page_content": doc.page_content, "metadata": doc.metadata},
                )
                for pid, dv, sv, doc in zip(ids, dense, sparse, docs)
            ]
        client.upsert(collection_name=plan.collection_name, points=points)
    else:
        docs = [_chunk_to_document(c) for c in chunks]
        vectorstore.add_documents(docs, ids=ids)


def _batched(it: Iterable, n: int) -> Iterator[list]:
    it = iter(it)
    while batch := list(islice(it, n)):
        yield batch


# ---------------------------------------------------------------------------
# Checkpoints — {collection_name}__{lang}.json, keyed by plan + language.
# Ported from ui/indexing.py (same file format/convention — anything already
# on disk from the old UI-driven runs is still valid) so the CLI can resume a
# killed/interrupted run per language.
# ---------------------------------------------------------------------------

_CHECKPOINT_DIR = Path(".indexer_checkpoints")


def _checkpoint_path(plan: IndexPlan, lang: str) -> Path:
    _CHECKPOINT_DIR.mkdir(exist_ok=True)
    return _CHECKPOINT_DIR / f"{plan.collection_name}__{lang}.json"


def load_checkpoint(plan: IndexPlan, lang: str) -> dict:
    p = _checkpoint_path(plan, lang)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"passages_done": 0, "chunks_done": 0}


def _save_checkpoint(plan: IndexPlan, lang: str, passages_done: int, chunks_done: int) -> None:
    _checkpoint_path(plan, lang).write_text(
        json.dumps({"passages_done": passages_done, "chunks_done": chunks_done})
    )


def _clear_checkpoint(plan: IndexPlan, lang: str) -> None:
    p = _checkpoint_path(plan, lang)
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_language(
    lang: str,
    plan: IndexPlan,
    batch_size: int = 256,
    chunker: BaseChunker | None = None,
    limit: int | None = None,
) -> int:
    """Embed and upsert passage chunks for one language, resuming from its
    checkpoint if one exists. `limit` caps total passages processed this call
    (for quick test runs) — the checkpoint still records exactly how far it
    got, so a later un-limited run picks up from there.

    Returns the total chunks indexed for this language (including anything
    from a prior resumed run).
    """
    if chunker is None:
        chunker = build_chunker(plan.chunkers)

    client = _get_qdrant_client()
    ensure_collection(client, plan)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=plan.collection_name,
        embedding=ProjectEmbeddings(backend=plan.backend),
    )

    ckpt          = load_checkpoint(plan, lang)
    start_passage = ckpt["passages_done"]
    done_chunks   = ckpt["chunks_done"]

    num_rows = count_language_rows(lang, plan.split)
    passages = iter_passages(iter_language_rows(lang, plan.split), lang, translated=True)
    if start_passage:
        for _ in islice(passages, start_passage):
            pass
    if limit is not None:
        passages = islice(passages, max(0, limit - start_passage))
        total_for_bar = min(num_rows, limit)
    else:
        total_for_bar = num_rows

    passage_idx = start_passage
    batch: list[Chunk] = []
    desc = f"{lang}/{plan.split} [{plan.collection_name}]"

    with tqdm(total=total_for_bar, initial=start_passage, desc=desc, unit="passage") as pbar:
        for rec in passages:
            batch.extend(chunker.chunk(rec))
            passage_idx += 1
            pbar.update(1)
            if len(batch) >= batch_size:
                _upsert_batch(client, plan, vectorstore, batch)
                done_chunks += len(batch)
                _save_checkpoint(plan, lang, passage_idx, done_chunks)
                batch = []
        if batch:
            _upsert_batch(client, plan, vectorstore, batch)
            done_chunks += len(batch)
            _save_checkpoint(plan, lang, passage_idx, done_chunks)

    if limit is not None and passage_idx < num_rows:
        # Partial by design (a test run) — leave the checkpoint in place so
        # a later full run resumes from here instead of starting over.
        print(f"[{lang}] stopped at {passage_idx:,}/{num_rows:,} passages (--limit); checkpoint saved.")
    else:
        _clear_checkpoint(plan, lang)
        register_plan(plan, {lang: done_chunks})
        print(f"[{lang}] done — {done_chunks:,} chunks in {plan.collection_name!r}.")

    return done_chunks


def run_indexing(langs: list[str], plan: IndexPlan, batch_size: int,
                 workers: int = 1, limit: int | None = None) -> None:
    print(f"Collection: {plan.collection_name}")
    if workers <= 1:
        for lang in langs:
            index_language(lang, plan, batch_size, limit=limit)
    else:
        # Separate OS processes, not threads — embedding is CPU-bound and
        # GIL-bound, and each language is already fully independent (own
        # dataset slice, own checkpoint file, own point IDs post lang-prefix
        # fix in dataset/passages.py).
        #
        # Default "spawn" start method — NOT "fork": once MPS/Metal has been
        # touched (device selection in pipeline/embedder.py), forking crashes
        # ("MPSGraphObject initialize... Crashing instead", an Apple/Metal +
        # Objective-C runtime limitation, not fixable from Python). spawn
        # avoids that by starting genuinely fresh interpreters — which is
        # exactly why the CLI entrypoint lives in scripts/index.py rather than
        # this module's own __main__: spawn needs to re-import the worker
        # function (index_language) from a real module path, and a module
        # executed as `python -m pipeline.indexer` registers itself as
        # __main__, which spawn cannot safely re-import in child processes.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(index_language, lang, plan, batch_size, None, limit): lang
                for lang in langs
            }
            for future in as_completed(futures):
                lang = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[{lang}] FAILED: {exc}")

    sync_registry_with_qdrant(_get_qdrant_client())


def get_vectorstore(plan: IndexPlan) -> QdrantVectorStore:
    """Return a QdrantVectorStore for an existing collection described by plan."""
    return QdrantVectorStore(
        client=_get_qdrant_client(),
        collection_name=plan.collection_name,
        embedding=ProjectEmbeddings(backend=plan.backend),
    )


# No `if __name__ == "__main__":` here on purpose — the CLI entrypoint is
# scripts/index.py. Running this module directly via `python -m
# pipeline.indexer` would register it as __main__, and --workers>1 uses
# ProcessPoolExecutor's default "spawn" start method, which needs to
# re-import index_language from a real module path in each child process;
# it can't safely do that if the defining module's identity is __main__
# instead of pipeline.indexer (confirmed empirically — every worker crashed
# immediately). Keeping this file import-only sidesteps that entirely.
