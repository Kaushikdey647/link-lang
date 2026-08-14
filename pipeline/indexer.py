"""Embed all passages for a language and upsert into Qdrant.

Usage:
    python -m pipeline.indexer --lang hi
    python -m pipeline.indexer --lang hi --split train --batch-size 512
"""

from __future__ import annotations

import argparse
import uuid
from itertools import islice
from typing import Iterable, Iterator

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from dataset import iter_passages, load_language
from pipeline.chunking import BaseChunker, Chunk, DEFAULT as DEFAULT_CHUNKER
from pipeline.embedder import VECTOR_DIM, embed_passages

COLLECTION = "msmarco_xi"
QDRANT_URL = "http://localhost:6333"


def _get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        # Payload indexes for fast filtered search
        for field, schema in [
            ("lang", PayloadSchemaType.KEYWORD),
            ("chunk_type", PayloadSchemaType.KEYWORD),
            ("query_id", PayloadSchemaType.INTEGER),
            ("is_selected", PayloadSchemaType.BOOL),
        ]:
            client.create_payload_index(COLLECTION, field, schema)


def _batched(it: Iterable, n: int) -> Iterator[list]:
    it = iter(it)
    while batch := list(islice(it, n)):
        yield batch


def _chunk_id_to_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def index_language(
    lang: str,
    split: str = "train",
    batch_size: int = 256,
    chunker: BaseChunker | None = None,
) -> None:
    """Embed and upsert all passage chunks for one language split.

    Args:
        lang: 2-letter language code.
        split: "train" or "validation".
        batch_size: Embedding + upsert batch size.
        chunker: Chunking strategy. Defaults to CompositeChunker (all three strategies).
    """
    if chunker is None:
        chunker = DEFAULT_CHUNKER

    client = _get_client()
    ensure_collection(client)

    ds = load_language(lang, splits=(split,))
    passages = iter_passages(ds[split], lang, translated=True)

    all_chunks: Iterator[Chunk] = (
        chunk
        for record in passages
        for chunk in chunker.chunk(record)
    )

    total_upserted = 0
    for batch in tqdm(_batched(all_chunks, batch_size), desc=f"Indexing {lang}/{split}", unit="batch"):
        texts = [c.text for c in batch]
        vectors = embed_passages(texts, batch_size=batch_size)

        points = [
            PointStruct(
                id=_chunk_id_to_uuid(c.chunk_id),
                vector=vectors[i].tolist(),
                payload={
                    "chunk_id": c.chunk_id,
                    "chunk_type": c.chunk_type,
                    "lang": c.lang,
                    "passage_id": c.passage_id,
                    "query_id": c.query_id,
                    "is_selected": c.is_selected,
                    "text": c.text,
                    "query": c.query,
                    "answer": c.answer,
                    "query_type": c.query_type,
                    "sentence_index": c.sentence_index,
                },
            )
            for i, c in enumerate(batch)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        total_upserted += len(points)

    print(f"Done. Upserted {total_upserted:,} chunks for lang={lang!r} split={split!r}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    index_language(args.lang, args.split, args.batch_size)
