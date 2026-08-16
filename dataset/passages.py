"""Iterators that unpack the dataset into flat records for embedding/indexing."""

from typing import Iterable, Iterator

from .types import PassageRecord, QueryRecord


def iter_passages(
    dataset: Iterable[dict],
    lang: str,
    *,
    translated: bool = True,
) -> Iterator[PassageRecord]:
    """Yield one PassageRecord per passage in the dataset.

    Args:
        dataset: Any iterable of row dicts — a single split Dataset (e.g.
            ds["train"]) or a streaming source like iter_language_rows().
        lang: 2-letter language code used to tag records.
        translated: If True, yield Translated_passages text + translated query/answer.
                    If False, yield English_passages text + English query/answer.
    """
    text_key = "Translated_passages" if translated else "English_passages"
    query_key = "query" if translated else "Eng_Query"
    answer_key = "Answer" if translated else "Eng_Answer"

    for row in dataset:
        passages = row["passages"]
        texts: list[str] = passages[text_key]
        selected: list[int] = passages["is_selected"]
        query: str = row[query_key] or ""
        eng_query: str = row["Eng_Query"] or ""
        answer: str = row[answer_key] or ""
        query_id: int = row["query_id"]
        query_type: str = row.get("query_type") or ""

        for idx, (text, sel) in enumerate(zip(texts, selected)):
            yield PassageRecord(
                # lang-prefixed: query_id is shared across all 14 language
                # translations of the same underlying MSMARCO question, so
                # without the lang prefix, indexing language B after A
                # silently overwrites A's Qdrant points wherever query_id+idx
                # match (chunk_id / point-ID both derive from passage_id).
                passage_id=f"{lang}_{query_id}_{idx}",
                query_id=query_id,
                text=text or "",
                lang=lang,
                is_selected=bool(sel),
                query=query,
                eng_query=eng_query,
                answer=answer,
                query_type=query_type,
            )


def iter_queries(
    dataset: Iterable[dict],
    lang: str,
    *,
    translated: bool = True,
) -> Iterator[QueryRecord]:
    """Yield one QueryRecord per row — useful for building an evaluation set."""
    query_key = "query" if translated else "Eng_Query"
    answer_key = "Answer" if translated else "Eng_Answer"

    for row in dataset:
        yield QueryRecord(
            query_id=row["query_id"],
            query=row[query_key] or "",
            answer=row[answer_key] or "",
            lang=lang,
            query_type=row.get("query_type") or "",
        )
