"""IndexPlan — the authoritative descriptor of how a Qdrant collection was built.

A plan = (backend × chunkers × split) → deterministic collection name.
The registry persists plan metadata so the query layer can discover what
embeddings are available and which model to load for each collection.

Collection name format:  msmarco_xi__{backend}__{chunkers_sorted}__{split}
Examples:
  msmarco_xi__english__english_query__train
  msmarco_xi__cohere__passage_qa_pair_sentence__train
  msmarco_xi__e5__passage__validation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.chunking import REGISTRY
from pipeline.embedder import AVAILABLE_BACKENDS, VECTOR_DIM_FOR

# Human-readable model identifiers stored in the registry
MODEL_NAME_FOR: dict[str, str] = {
    "e5":      "intfloat/multilingual-e5-small",
    "cohere":  "Cohere/embed-multilingual-v3.0",
    "english": "sentence-transformers/all-MiniLM-L6-v2",
}

# Backend preference for auto-selection (higher index = preferred)
_BACKEND_PREFERENCE = ["e5", "english", "cohere"]

_REGISTRY_PATH = Path(".indexer_checkpoints/registry.json")


# ---------------------------------------------------------------------------
# IndexPlan
# ---------------------------------------------------------------------------

@dataclass
class IndexPlan:
    backend:  str
    chunkers: list[str]
    split:    str = "train"
    # Only set for pre-refactor collections whose Qdrant name doesn't follow
    # the deterministic format below (see register_legacy_collection).
    _collection_name_override: Optional[str] = field(default=None, repr=False, compare=False)

    @property
    def collection_name(self) -> str:
        if self._collection_name_override:
            return self._collection_name_override
        key = "_".join(sorted(self.chunkers))
        return f"msmarco_xi__{self.backend}__{key}__{self.split}"

    @property
    def vector_dim(self) -> int:
        return VECTOR_DIM_FOR[self.backend]

    @property
    def model_name(self) -> str:
        return MODEL_NAME_FOR.get(self.backend, self.backend)

    def to_dict(self) -> dict:
        return {
            "collection_name": self.collection_name,
            "backend":         self.backend,
            "model_name":      self.model_name,
            "vector_dim":      self.vector_dim,
            "chunkers":        self.chunkers,
            "split":           self.split,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IndexPlan":
        plan = cls(
            backend=d["backend"],
            chunkers=d["chunkers"],
            split=d.get("split", "train"),
        )
        stored_name = d.get("collection_name")
        if stored_name and stored_name != plan.collection_name:
            plan._collection_name_override = stored_name
        return plan

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexPlan):
            return NotImplemented
        return self.collection_name == other.collection_name

    def __hash__(self) -> int:
        return hash(self.collection_name)


# ---------------------------------------------------------------------------
# Registry CRUD
# ---------------------------------------------------------------------------

def load_registry() -> dict[str, dict]:
    """Load the full plan registry. Returns {} if the file doesn't exist."""
    if _REGISTRY_PATH.exists():
        try:
            return json.loads(_REGISTRY_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_registry(registry: dict) -> None:
    _REGISTRY_PATH.parent.mkdir(exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def prune_missing_collections(existing_names: set[str]) -> list[str]:
    """Remove registry entries whose backing Qdrant collection no longer
    exists (e.g. it was deleted/recreated outside the app). Returns the
    collection names that were pruned."""
    registry = load_registry()
    stale = [name for name in registry if name not in existing_names]
    for name in stale:
        del registry[name]
    if stale:
        _save_registry(registry)
    return stale


def register_plan(plan: IndexPlan, lang_counts: dict[str, int] | None = None) -> None:
    """Upsert a plan into the registry. Call after successful indexing."""
    registry = load_registry()
    entry = registry.get(plan.collection_name, plan.to_dict())
    if lang_counts:
        existing = entry.get("lang_counts", {})
        existing.update({k: v for k, v in lang_counts.items() if v > 0})
        entry["lang_counts"] = existing
    registry[plan.collection_name] = entry
    _save_registry(registry)


def register_legacy_collection(collection_name: str, backend: str, chunkers: list[str],
                               split: str, lang_counts: dict[str, int] | None = None) -> None:
    """Register a pre-refactor collection whose name doesn't follow the
    deterministic IndexPlan.collection_name format (e.g. `msmarco_xi_e5`),
    so it still shows up in the registry and query layer instead of being
    permanently invisible to `all_plans()`/`best_available_plan()`."""
    registry = load_registry()
    entry = registry.get(collection_name, {
        "collection_name": collection_name,
        "backend":         backend,
        "model_name":      MODEL_NAME_FOR.get(backend, backend),
        "vector_dim":      VECTOR_DIM_FOR[backend],
        "chunkers":        chunkers,
        "split":           split,
    })
    if lang_counts:
        existing = entry.get("lang_counts", {})
        existing.update({k: v for k, v in lang_counts.items() if v > 0})
        entry["lang_counts"] = existing
    registry[collection_name] = entry
    _save_registry(registry)


def _split_chunkers(key: str) -> Optional[list[str]]:
    """Reverse of '_'.join(sorted(chunkers)) — greedy longest-match against
    REGISTRY keys, since some keys (qa_pair, english_query) contain '_'."""
    tokens = key.split("_")
    chunkers: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        for j in range(n, i, -1):
            candidate = "_".join(tokens[i:j])
            if candidate in REGISTRY:
                chunkers.append(candidate)
                i = j
                break
        else:
            return None
    return chunkers


def parse_collection_name(name: str) -> Optional[IndexPlan]:
    """Reverse of IndexPlan.collection_name. Returns None for names that
    don't follow the deterministic msmarco_xi__{backend}__{chunkers}__{split}
    format (e.g. pre-refactor collections like msmarco_xi_e5 — look those up
    via get_plan_by_collection() instead, which honors the stored override)."""
    prefix = "msmarco_xi__"
    if not name.startswith(prefix):
        return None
    parts = name[len(prefix):].split("__")
    if len(parts) != 3:
        return None
    backend, chunkers_key, split = parts
    if backend not in AVAILABLE_BACKENDS:
        return None
    chunkers = _split_chunkers(chunkers_key)
    if chunkers is None:
        return None
    plan = IndexPlan(backend=backend, chunkers=chunkers, split=split)
    return plan if plan.collection_name == name else None


def get_plan_by_collection(collection_name: str) -> Optional[IndexPlan]:
    """Look up a plan by its collection name. Returns None if not registered."""
    registry = load_registry()
    entry = registry.get(collection_name)
    if entry is None:
        return None
    return IndexPlan.from_dict(entry)


def best_available_plan() -> Optional[IndexPlan]:
    """Return the best registered plan: prefer cohere > english > e5."""
    registry = load_registry()
    if not registry:
        return None
    # Sort by backend preference (highest index = most preferred)
    def _rank(entry: dict) -> int:
        backend = entry.get("backend", "")
        try:
            return _BACKEND_PREFERENCE.index(backend)
        except ValueError:
            return -1

    best_entry = max(registry.values(), key=_rank)
    return IndexPlan.from_dict(best_entry)


def all_plans() -> list[IndexPlan]:
    """Return all registered plans, best-first."""
    registry = load_registry()
    entries = sorted(registry.values(),
                     key=lambda e: _BACKEND_PREFERENCE.index(e.get("backend", ""))
                     if e.get("backend") in _BACKEND_PREFERENCE else -1,
                     reverse=True)
    return [IndexPlan.from_dict(e) for e in entries]
