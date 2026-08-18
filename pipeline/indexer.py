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
    uv run python -m scripts.index --langs hi bn
    uv run python -m scripts.index --langs all --workers 4
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
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PayloadSchemaType, SparseVectorParams, Modifier,
    PointStruct,
    Document as QdrantDocument,  # aliased: langchain_core.documents.Document is already "Document" in this file
)
from tqdm import tqdm

from dataset import count_language_rows, iter_language_rows, iter_passages
from pipeline.chunking import BaseChunker, Chunk, build_chunker
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
E5_SMALL_INFERENCE_MODEL = "intfloat/multilingual-e5-small"

# Which inference model backs each backend's dense vector.
DENSE_INFERENCE_MODEL_FOR = {
    "english": MINILM_INFERENCE_MODEL,
    "multilingual_e5_small": E5_SMALL_INFERENCE_MODEL,
}

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
# Sparse (BM25/IDF) vector space — computed server-side by Qdrant Cloud
# (see MINILM_INFERENCE_MODEL/BM25_INFERENCE_MODEL above), used for RRF
# hybrid retrieval (frontend/lib/server/retrieval.ts for the english_query plan).
# ---------------------------------------------------------------------------

SPARSE_VECTOR_NAME = "bm25"


def ensure_collection(client: QdrantClient, plan: IndexPlan) -> None:
    """Create the collection for this plan if it doesn't exist; ensure payload indexes."""
    collection  = plan.collection_name
    needs_bm25  = "english_query" in plan.chunkers
    # get_collections() only lists physical collections, not aliases — a
    # A collection reachable only via an alias would otherwise look "missing"
    # here, and create_collection() over an already-aliased name fails with a
    # 400 (confirmed empirically).
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
        # tunes params of vectors already in the schema). Every collection
        # created by this code always includes the sparse vector space from
        # the start, so this only fires against an externally-created or
        # otherwise malformed collection — fail loudly rather than silently
        # indexing dense-only.
        info = client.get_collection(collection)
        if not info.config.params.sparse_vectors or SPARSE_VECTOR_NAME not in info.config.params.sparse_vectors:
            raise RuntimeError(
                f"Collection {collection!r} is missing the {SPARSE_VECTOR_NAME!r} "
                "sparse vector space required for RRF hybrid retrieval — recreate it."
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


def _e5_text(text: str) -> str:
    # e5 models are trained with an asymmetric "query: "/"passage: " prefix
    # convention — omitting it measurably hurts retrieval quality. qa_pair
    # chunks are indexed (retrieval-target) documents, so "passage: " applies
    # here; query-time embedding (frontend/lib/server/retrieval.ts) prefixes
    # with "query: ".
    return f"passage: {text}"


def _upsert_batch(client: QdrantClient, plan: IndexPlan, chunks: list[Chunk]) -> None:
    """Embed + upsert one batch via Qdrant Cloud server-side inference — all
    vectors are computed remotely from raw text in a single Qdrant round-trip;
    no local model inference at all. Only the two registered plans (see
    pipeline/index_plan.py) are supported."""
    ids  = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.chunk_id)) for c in chunks]
    docs = [_chunk_to_document(c) for c in chunks]

    if plan.chunkers == ["english_query"]:
        vectors = [
            {
                "": QdrantDocument(text=c.text, model=MINILM_INFERENCE_MODEL),
                SPARSE_VECTOR_NAME: QdrantDocument(
                    text=c.parent_passage or c.text, model=BM25_INFERENCE_MODEL,
                ),
            }
            for c in chunks
        ]
    elif plan.chunkers == ["qa_pair"]:
        model = DENSE_INFERENCE_MODEL_FOR[plan.backend]
        vectors = [
            {"": QdrantDocument(text=_e5_text(c.text), model=model)}
            for c in chunks
        ]
    else:
        raise NotImplementedError(
            f"Unsupported chunkers {plan.chunkers!r} — see pipeline/index_plan.py "
            "for the registered (backend, chunkers) plans."
        )

    points = [
        PointStruct(
            id=pid, vector=vec,
            payload={"page_content": doc.page_content, "metadata": doc.metadata},
        )
        for pid, vec, doc in zip(ids, vectors, docs)
    ]
    client.upsert(collection_name=plan.collection_name, points=points)


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
                _upsert_batch(client, plan, batch)
                done_chunks += len(batch)
                _save_checkpoint(plan, lang, passage_idx, done_chunks)
                batch = []
        if batch:
            _upsert_batch(client, plan, batch)
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
        # Separate OS processes, not threads — network I/O to Qdrant Cloud
        # releases the GIL either way, but each language is already fully
        # independent (own dataset slice, own checkpoint file, own point IDs
        # post lang-prefix fix in dataset/passages.py), so process isolation
        # costs nothing extra.
        #
        # Default "spawn" start method: needed regardless of local-model
        # concerns because the CLI entrypoint lives in scripts/index.py
        # rather than this module's own __main__ — spawn needs to re-import
        # the worker function (index_language) from a real module path, and
        # a module executed as `python -m pipeline.indexer` registers itself
        # as __main__, which spawn cannot safely re-import in child processes.
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


# No `if __name__ == "__main__":` here on purpose — the CLI entrypoint is
# scripts/index.py. Running this module directly via `python -m
# pipeline.indexer` would register it as __main__, and --workers>1 uses
# ProcessPoolExecutor's default "spawn" start method, which needs to
# re-import index_language from a real module path in each child process;
# it can't safely do that if the defining module's identity is __main__
# instead of pipeline.indexer (confirmed empirically — every worker crashed
# immediately). Keeping this file import-only sidesteps that entirely.
