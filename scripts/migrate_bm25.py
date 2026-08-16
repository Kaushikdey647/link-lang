"""One-time migration: add the BM25 sparse vector space to an existing
english_query collection that predates the RRF hybrid retrieval strategy
(pipeline/query_engines.py:EnglishPivotQueryEngine).

Qdrant can't add a new named vector to a collection that doesn't already
have it — update_collection only tunes params of vectors already in the
schema (confirmed empirically; see indexing-controls plan notes). So this:

  1. Creates a new physical collection with the correct dense+sparse schema.
  2. Scrolls the old collection (with_vectors=True, with_payload=True),
     reusing its already-computed dense vectors — no re-embedding.
  3. Computes a BM25 sparse vector per point from metadata.parent_passage
     and upserts both vectors + payload into the new collection.
  4. Deletes the old collection and points a Qdrant alias at the new one
     under the original name, so every existing call site (which always
     references the collection purely by name) keeps working unchanged.

Safe to re-run: if the collection already has the sparse vector space,
this is a no-op.

Usage:
    uv run python -m scripts.migrate_bm25 --collection msmarco_xi__english__english_query__train
"""

from __future__ import annotations

import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias, CreateAliasOperation, Modifier, PayloadSchemaType,
    PointStruct, SparseVectorParams, VectorParams,
)

from pipeline.indexer import (
    QDRANT_INDEXING_TIMEOUT, QDRANT_URL, SPARSE_VECTOR_NAME, sparse_vectors_for,
)

_PAYLOAD_INDEXES = [
    ("metadata.lang",        PayloadSchemaType.KEYWORD),
    ("metadata.chunk_type",  PayloadSchemaType.KEYWORD),
    ("metadata.query_id",    PayloadSchemaType.INTEGER),
    ("metadata.is_selected", PayloadSchemaType.BOOL),
]


def migrate(collection: str, batch_size: int = 256) -> None:
    client = QdrantClient(url=QDRANT_URL, timeout=QDRANT_INDEXING_TIMEOUT)

    info = client.get_collection(collection)
    if info.config.params.sparse_vectors and SPARSE_VECTOR_NAME in info.config.params.sparse_vectors:
        print(f"{collection!r} already has the {SPARSE_VECTOR_NAME!r} sparse vector — nothing to do.")
        return

    dim      = info.config.params.vectors.size
    distance = info.config.params.vectors.distance
    total    = info.points_count
    new_name = f"{collection}__bm25migrated"

    if new_name in {c.name for c in client.get_collections().collections}:
        client.delete_collection(new_name)

    client.create_collection(
        collection_name=new_name,
        vectors_config=VectorParams(size=dim, distance=distance),
        sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)},
    )
    for field_name, schema in _PAYLOAD_INDEXES:
        client.create_payload_index(new_name, field_name, schema)
    print(f"Created {new_name!r} ({dim}-dim, {distance.value}) with sparse vector {SPARSE_VECTOR_NAME!r}.")

    migrated    = 0
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=next_offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        texts = [
            p.payload.get("metadata", {}).get("parent_passage") or p.payload.get("page_content", "")
            for p in points
        ]
        sparse_vecs = sparse_vectors_for(texts)

        client.upsert(
            collection_name=new_name,
            points=[
                PointStruct(id=p.id, vector={"": p.vector, SPARSE_VECTOR_NAME: sv}, payload=p.payload)
                for p, sv in zip(points, sparse_vecs)
            ],
        )
        migrated += len(points)
        print(f"  migrated {migrated:,}/{total:,}")

        if next_offset is None:
            break

    client.delete_collection(collection)
    client.update_collection_aliases(change_aliases_operations=[
        CreateAliasOperation(create_alias=CreateAlias(collection_name=new_name, alias_name=collection)),
    ])
    print(f"Done. {migrated:,} points migrated. Alias {collection!r} -> {new_name!r}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    migrate(args.collection, args.batch_size)
