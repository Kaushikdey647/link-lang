"""Iterators that unpack the dataset into flat records for embedding/indexing."""

from typing import Iterator

from datasets import Dataset

from .types import PassageRecord, QueryRecord


def iter_passages(
    dataset: Dataset,
    lang: str,
    *,
    translated: bool = True,
) -> Iterator[PassageRecord]:
    """Yield one PassageRecord per passage in the dataset.

    Args:
        dataset: A single split Dataset (e.g. ds["train"]).
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
        answer: str = row[answer_key] or ""
        query_id: int = row["query_id"]
        query_type: str = row.get("query_type") or ""

        for idx, (text, sel) in enumerate(zip(texts, selected)):
            yield PassageRecord(
                passage_id=f"{query_id}_{idx}",
                query_id=query_id,
                text=text or "",
                lang=lang,
                is_selected=bool(sel),
                query=query,
                answer=answer,
                query_type=query_type,
            )


def iter_queries(
    dataset: Dataset,
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
