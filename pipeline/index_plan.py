"""IndexPlan — the authoritative descriptor of how a Qdrant collection was built.

A plan = (backend × chunkers × split) → deterministic collection name. Kept
as a dataclass of (currently fixed) values rather than hardcoded constants so
the collection-name derivation and registry persistence stay unchanged from
before the single-strategy collapse (see CHANGELOG.md) — the remote Qdrant
Cloud collection populated under the old scheme keeps resolving correctly.

Collection name format:  msmarco_xi__{backend}__{chunkers_sorted}__{split}
Two valid (backend, chunkers) pairings today, each its own collection:
  - msmarco_xi__english__english_query__train
    (MiniLM dense + BM25 sparse, both computed server-side by Qdrant Cloud)
  - msmarco_xi__multilingual_e5_small__qa_pair__train
    (multilingual-e5-small dense only, computed server-side by Qdrant Cloud)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from constants import LANG_CODE_MAP

# "english" (MiniLM dense + BM25 sparse) and "multilingual_e5_small" (dense
# only) — both computed server-side via Qdrant Cloud inference. See
# CHANGELOG.md for why e5/cohere were originally removed and later why
# multilingual_e5_small came back as the qa_pair collection's backend.
AVAILABLE_BACKENDS: list[str] = ["english", "multilingual_e5_small"]
VECTOR_DIM_FOR: dict[str, int] = {"english": 384, "multilingual_e5_small": 384}

# Human-readable model identifier stored in the registry
MODEL_NAME_FOR: dict[str, str] = {
    "english": "sentence-transformers/all-MiniLM-L6-v2",
    "multilingual_e5_small": "intfloat/multilingual-e5-small",
}

_BACKEND_PREFERENCE = ["multilingual_e5_small", "english"]

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


_VALID_CHUNKER_KEYS = {"english_query", "qa_pair"}


def _split_chunkers(key: str) -> Optional[list[str]]:
    """Reverse of '_'.join(sorted(chunkers)) — trivial since both valid keys
    are single-chunker names (no multi-chunker plan exists)."""
    return [key] if key in _VALID_CHUNKER_KEYS else None


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
    """Return the best registered plan, ranked by _BACKEND_PREFERENCE."""
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


# ---------------------------------------------------------------------------
# Registry <-> Qdrant reconciliation (used by the CLI after each language
# completes, and by the read-only Ingestion tab on refresh — moved here from
# ui/indexing.py so both can share it without either depending on the UI).
# ---------------------------------------------------------------------------

def _lang_counts_for(client: QdrantClient, collection: str) -> dict[str, int]:
    lang_counts: dict[str, int] = {}
    for code in LANG_CODE_MAP:
        try:
            r = client.count(
                collection_name=collection,
                count_filter=Filter(must=[
                    FieldCondition(key="metadata.lang", match=MatchValue(value=code))
                ]),
                exact=False,
            )
            if r.count > 0:
                lang_counts[code] = r.count
        except Exception:
            continue
    return lang_counts


def sync_registry_with_qdrant(client: QdrantClient) -> None:
    """Reconcile registry.json against what's actually in Qdrant: drop entries
    whose collection no longer exists (e.g. deleted/recreated outside the app,
    which would otherwise show as a permanent ghost entry), and auto-register
    any deterministically-named IndexPlan collection that's live in Qdrant but
    missing from the *local* registry.

    That last case matters once Qdrant is a shared remote cluster (Qdrant
    Cloud): registry.json is local-only (gitignored — it's per-machine run
    state, not source), so a fresh clone/redeploy/teammate's machine pointed
    at an already-populated cluster would otherwise see qdrant_ready=False
    forever (GET /health, best_available_plan()) despite the data genuinely
    being there — this is what lets it "discover" the shared cluster's state
    instead of depending on this one machine's indexing history."""
    try:
        existing = {c.name for c in client.get_collections().collections}
    except Exception:
        return

    prune_missing_collections(existing)

    registry = load_registry()
    for name in existing:
        if name in registry:
            continue
        plan = parse_collection_name(name)
        if plan is None:
            continue
        lang_counts = _lang_counts_for(client, name)
        if lang_counts:
            register_plan(plan, lang_counts)


def all_plans() -> list[IndexPlan]:
    """Return all registered plans, best-first."""
    registry = load_registry()
    entries = sorted(registry.values(),
                     key=lambda e: _BACKEND_PREFERENCE.index(e.get("backend", ""))
                     if e.get("backend") in _BACKEND_PREFERENCE else -1,
                     reverse=True)
    return [IndexPlan.from_dict(e) for e in entries]
