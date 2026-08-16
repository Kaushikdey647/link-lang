"""Chunking for MSMARCO-XI passages.

Two supported strategies, each paired with one backend/collection (see
pipeline/index_plan.py):
  - EnglishQueryChunker ("english_query")  -> "english" backend (MiniLM+BM25)
  - QAPairChunker       ("qa_pair")        -> "multilingual_e5_small" backend

QAPairChunker was previously removed (see CHANGELOG.md) when the project
collapsed to english-pivot only; it's back as an additional, separate
collection rather than a replacement.

    chunker.chunk(record: PassageRecord) -> list[Chunk]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from dataset.types import PassageRecord


# ---------------------------------------------------------------------------
# Shared output type
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    text: str
    chunk_type: str           # "english_query" | "qa_pair"
    lang: str
    passage_id: str
    query_id: int
    is_selected: bool
    query: str
    answer: str
    query_type: str
    sentence_index: int = -1  # unused (no sentence-level chunker anymore); kept for payload-shape stability
    parent_passage: str = ""  # vernacular passage text — returned as LLM context


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, record: PassageRecord) -> list[Chunk]:
        """Return zero or more Chunk objects for the given passage record."""
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# English-pivot (embed English question, store vernacular passage)
# ---------------------------------------------------------------------------

class EnglishQueryChunker(BaseChunker):
    """One chunk per record: text = English question (embedded), parent_passage = vernacular passage (returned).

    Use with the "english" embedding backend. Requires record.eng_query to be non-empty
    (always true for MSMARCO-XI records — Eng_Query is populated regardless of `translated`).
    """

    def chunk(self, record: PassageRecord) -> list[Chunk]:
        if not record.eng_query:
            return []
        return [Chunk(
            chunk_id=f"{record.passage_id}__enq",
            text=record.eng_query,      # English question → embedded
            chunk_type="english_query",
            parent_passage=record.text, # vernacular passage → returned as context
            **_base(record),
        )]


class QAPairChunker(BaseChunker):
    """Concatenate the query with its selected passage.

    Biases the embedding space toward the task distribution — these points
    are the closest thing to "golden" retrieval units in the dataset.
    Only emits a chunk when is_selected=True AND the query is non-empty.

    Use with the "multilingual_e5_small" backend — unlike EnglishQueryChunker,
    the embedded text is vernacular, so it needs a multilingual embedding
    model rather than the English-pivot MiniLM one.
    """

    def chunk(self, record: PassageRecord) -> list[Chunk]:
        if not record.is_selected or not record.query:
            return []
        return [Chunk(
            chunk_id=f"{record.passage_id}__qa",
            text=f"{record.query} {record.text}",
            chunk_type="qa_pair",
            parent_passage=record.text,   # full passage stored in payload
            **_base(record),
        )]


_REGISTRY: dict[str, type[BaseChunker]] = {
    "english_query": EnglishQueryChunker,
    "qa_pair":        QAPairChunker,
}


def build_chunker(names: list[str] | None = None) -> BaseChunker:
    """names must be a single-item list naming one of _REGISTRY's strategies —
    validated explicitly so a stale/misconfigured caller fails loudly instead
    of silently getting the wrong chunker."""
    if names is None or len(names) != 1 or names[0] not in _REGISTRY:
        raise ValueError(
            f"Unsupported chunkers {names!r} — must be one of {list(_REGISTRY)!r}."
        )
    return _REGISTRY[names[0]]()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base(record: PassageRecord) -> dict:
    return dict(
        lang=record.lang,
        passage_id=record.passage_id,
        query_id=record.query_id,
        is_selected=record.is_selected,
        query=record.query,
        answer=record.answer,
        query_type=record.query_type,
    )
