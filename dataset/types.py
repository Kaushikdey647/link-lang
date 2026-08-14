from dataclasses import dataclass


@dataclass
class PassageRecord:
    """A single passage ready to embed and index into a vector store."""
    passage_id: str       # "{query_id}_{passage_idx}" — stable unique key
    query_id: int
    text: str
    lang: str             # 2-letter code: "hi", "bn", …
    is_selected: bool     # ground-truth relevance label
    query: str            # associated query (same language as text)
    answer: str           # associated answer (same language as text)
    query_type: str       # e.g. "DESCRIPTION", "NUMERIC", …


@dataclass
class QueryRecord:
    """A query + answer pair for evaluation or retrieval."""
    query_id: int
    query: str
    answer: str
    lang: str
    query_type: str
