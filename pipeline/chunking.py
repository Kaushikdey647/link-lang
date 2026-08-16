"""Chunking for MSMARCO-XI passages.

english-pivot (EnglishQueryChunker) is the system's one supported chunking
strategy — see CHUNKING.md for the strategies that were prototyped and
evaluated (passage/sentence/qa_pair chunking, e5/cohere embedding backends)
before this single-strategy, latency-focused, Qdrant-Cloud-inference-only
architecture was chosen. That prior code lived here; see CHANGELOG.md for
when/why it was removed rather than kept as unused-but-present.

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
    chunk_type: str           # always "english_query"
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


def build_chunker(names: list[str] | None = None) -> BaseChunker:
    """english_query is the only supported strategy — validated explicitly so
    a stale/misconfigured caller fails loudly instead of silently getting the
    wrong chunker."""
    if names is not None and names != ["english_query"]:
        raise ValueError(
            f"Unsupported chunkers {names!r} — only ['english_query'] is supported now "
            "(see CHANGELOG.md for why the other chunking strategies were removed)."
        )
    return EnglishQueryChunker()


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
