"""Pluggable chunking strategies for MSMARCO-XI passages.

All chunkers share the same interface:

    chunker.chunk(record: PassageRecord) -> list[Chunk]

Compose strategies with CompositeChunker:

    chunker = CompositeChunker([PassageChunker(), SentenceChunker(), QAPairChunker()])

Or use the pre-built DEFAULT (all three strategies).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from dataset.types import PassageRecord


# ---------------------------------------------------------------------------
# Shared output type
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    text: str
    chunk_type: str           # "passage" | "sentence" | "qa_pair" | "english_query"
    lang: str
    passage_id: str
    query_id: int
    is_selected: bool
    query: str
    answer: str
    query_type: str
    sentence_index: int = -1  # only set for chunk_type == "sentence"
    parent_passage: str = ""  # full parent text; set on sentence/qa_pair chunks


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
# Strategy 1: whole passage
# ---------------------------------------------------------------------------

class PassageChunker(BaseChunker):
    """One chunk per passage — the primary retrieval unit."""

    def chunk(self, record: PassageRecord) -> list[Chunk]:
        return [Chunk(
            chunk_id=f"{record.passage_id}__passage",
            text=record.text,
            chunk_type="passage",
            **_base(record),
        )]


# ---------------------------------------------------------------------------
# Strategy 2: sentence-level sub-chunks
# ---------------------------------------------------------------------------

# Sentence delimiters: Latin (.!?) + Devanagari danda (।) + double danda (॥)
_SENT_RE = re.compile(r"(?<=[.!?।॥])\s+")


class SentenceChunker(BaseChunker):
    """Split each passage at sentence boundaries.

    Small-to-big retrieval: at query time, a sentence match can be expanded
    back to its parent passage for fuller context (retriever handles this).

    Args:
        min_words: Discard sentences shorter than this — avoids indexing fragments.
    """

    def __init__(self, min_words: int = 4):
        self.min_words = min_words

    def chunk(self, record: PassageRecord) -> list[Chunk]:
        sentences = [
            s.strip()
            for s in _SENT_RE.split(record.text)
            if len(s.split()) >= self.min_words
        ]
        if not sentences:
            sentences = [record.text]  # fall back to the full passage

        base = _base(record)
        return [
            Chunk(
                chunk_id=f"{record.passage_id}__sent_{i}",
                text=sent,
                chunk_type="sentence",
                sentence_index=i,
                parent_passage=record.text,  # full passage stored in payload
                **base,
            )
            for i, sent in enumerate(sentences)
        ]


# ---------------------------------------------------------------------------
# Strategy 3: query-anchored chunk (only for ground-truth positives)
# ---------------------------------------------------------------------------

class QAPairChunker(BaseChunker):
    """Concatenate the query with its selected passage.

    Biases the embedding space toward the task distribution — these points
    are the closest thing to "golden" retrieval units in the dataset.
    Only emits a chunk when is_selected=True AND the query is non-empty.
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


# ---------------------------------------------------------------------------
# Strategy 4: English-pivot (embed English question, store vernacular passage)
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


# ---------------------------------------------------------------------------
# Composite: run multiple strategies and merge
# ---------------------------------------------------------------------------

class CompositeChunker(BaseChunker):
    """Run several chunkers and concatenate their outputs.

    Example:
        chunker = CompositeChunker([PassageChunker(), SentenceChunker()])
    """

    def __init__(self, chunkers: list[BaseChunker]):
        if not chunkers:
            raise ValueError("CompositeChunker requires at least one chunker")
        self.chunkers = chunkers

    def chunk(self, record: PassageRecord) -> list[Chunk]:
        return [c for chunker in self.chunkers for c in chunker.chunk(record)]

    def __repr__(self) -> str:
        return f"CompositeChunker({self.chunkers!r})"


# ---------------------------------------------------------------------------
# Named registry — look up by string key
# ---------------------------------------------------------------------------

# Leaf strategies only — CompositeChunker is not selectable by name
# because it requires a list of chunkers as an argument.
REGISTRY: dict[str, type[BaseChunker]] = {
    "passage":       PassageChunker,
    "sentence":      SentenceChunker,
    "qa_pair":       QAPairChunker,
    "english_query": EnglishQueryChunker,
}


def build_chunker(names: list[str] | None = None) -> BaseChunker:
    """Build a chunker from a list of strategy names.

    Args:
        names: e.g. ["passage", "sentence", "qa_pair"]. None → all three.

    Example:
        chunker = build_chunker(["passage", "sentence"])
    """
    if names is None:
        names = ["passage", "sentence", "qa_pair"]
    chunkers = [REGISTRY[n]() for n in names]
    return chunkers[0] if len(chunkers) == 1 else CompositeChunker(chunkers)


# Default: all three strategies
DEFAULT: BaseChunker = CompositeChunker([PassageChunker(), SentenceChunker(), QAPairChunker()])


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
