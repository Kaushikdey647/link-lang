"""Query the Qdrant collection and return ranked passages.

Small-to-big retrieval:
  1. Search at sentence or passage level (configurable).
  2. If a sentence chunk matches, return its full parent passage text instead
     (the parent is also indexed as chunk_type="passage" with the same passage_id).
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from pipeline.embedder import embed_query
from pipeline.index_plan import best_available_plan, IndexPlan
from pipeline.indexer import QDRANT_URL


@dataclass
class RetrievedPassage:
    passage_id: str
    text: str           # full passage text (not the sentence sub-chunk)
    score: float
    lang: str
    query_id: int
    is_selected: bool   # ground-truth label — useful for offline eval
    answer: str
    query_type: str


def retrieve(
    query: str,
    lang: str,
    *,
    top_k: int = 5,
    chunk_types: list[str] | None = None,
    plan: IndexPlan | None = None,
) -> list[RetrievedPassage]:
    """Embed query and return top_k deduplicated passages.

    Args:
        query: The user's question (in any language; e5 is multilingual).
        lang: Filter results to this language code.
        top_k: Number of distinct passages to return.
        chunk_types: Which chunk strategies to search. Defaults to all three.
                     Pass ["passage"] for fastest search, ["sentence"] for highest precision.
    """
    resolved_plan = plan or best_available_plan()
    collection    = resolved_plan.collection_name if resolved_plan else "msmarco_xi__e5__passage_sentence_qa_pair__train"
    backend       = resolved_plan.backend if resolved_plan else "e5"

    if chunk_types is None:
        chunk_types = resolved_plan.chunkers if resolved_plan else ["passage", "sentence", "qa_pair"]

    vector = embed_query(query, backend=backend)
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    client = QdrantClient(url=QDRANT_URL)

    query_filter = Filter(must=[
        FieldCondition(key="lang", match=MatchValue(value=lang)),
        FieldCondition(key="chunk_type", match=MatchAny(any=chunk_types)),
    ])

    # Over-fetch to allow deduplication by passage_id
    hits = client.query_points(
        collection_name=collection,
        query=vector,
        query_filter=query_filter,
        limit=top_k * 4,
        with_payload=True,
    ).points

    # Deduplicate: keep highest-scoring hit per passage_id
    seen: dict[str, RetrievedPassage] = {}
    for hit in hits:
        p = hit.payload
        pid = p["passage_id"]

        # Small-to-big: prefer the full passage text for context
        text = p["text"]
        if p["chunk_type"] == "sentence":
            # The passage-level chunk for this passage_id has the full text.
            # We stored it in the same collection — fetch it if not already seen.
            text = _fetch_passage_text(client, pid) or text

        if pid not in seen or hit.score > seen[pid].score:
            seen[pid] = RetrievedPassage(
                passage_id=pid,
                text=text,
                score=hit.score,
                lang=p["lang"],
                query_id=p["query_id"],
                is_selected=p.get("is_selected", False),
                answer=p.get("answer", ""),
                query_type=p.get("query_type", ""),
            )
        if len(seen) >= top_k:
            break

    return sorted(seen.values(), key=lambda r: r.score, reverse=True)[:top_k]


def _fetch_passage_text(client: QdrantClient, passage_id: str) -> str | None:
    """Look up the full-passage chunk for a given passage_id (payload filter, no vector)."""
    results, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[
            FieldCondition(key="passage_id", match=MatchValue(value=passage_id)),
            FieldCondition(key="chunk_type", match=MatchValue(value="passage")),
        ]),
        limit=1,
        with_payload=["text"],
        with_vectors=False,
    )
    return results[0].payload["text"] if results else None
