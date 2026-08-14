from .loader import load_language
from .passages import iter_passages, iter_queries
from .types import PassageRecord, QueryRecord

__all__ = [
    "load_language",
    "iter_passages",
    "iter_queries",
    "PassageRecord",
    "QueryRecord",
]
