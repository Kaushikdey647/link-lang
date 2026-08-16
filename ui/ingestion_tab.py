"""Ingestion tab — read-only Qdrant observability.

Indexing runs via CLI only (`python -m pipeline.indexer`) now — there is no
start/stop/resume control anywhere in this tab, by design. It shows exactly
what's actually in Qdrant: every collection, its alias(es), size on disk,
vector config (dense dim + sparse presence), which IndexPlan it maps to, and
registry status — plus an on-demand per-collection language breakdown.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from constants import LANG_CODE_MAP
from pipeline.index_plan import (
    get_plan_by_collection, load_registry, parse_collection_name,
    sync_registry_with_qdrant,
)
from pipeline.indexer import QDRANT_URL

_STATS_TIMEOUT = 30  # read-only queries; generous but not indexing-length

# Local bind-mount for the Qdrant container's storage (confirmed via
# `docker inspect` against the qdrant/qdrant container this repo runs).
# Override via env var if Qdrant runs somewhere else.
QDRANT_STORAGE_PATH = Path(os.environ.get("QDRANT_STORAGE_PATH", "qdrant_storage"))


def _client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=_STATS_TIMEOUT)


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _format_size(num_bytes: int) -> str:
    gb = num_bytes / (1024 ** 3)
    if gb >= 0.01:
        return f"{gb:.2f} GB"
    return f"{num_bytes / (1024 ** 2):.1f} MB"


def _collection_names() -> list[str]:
    try:
        return sorted(c.name for c in _client().get_collections().collections)
    except Exception:
        return []


def _collections_table() -> list[list]:
    try:
        client = _client()
        sync_registry_with_qdrant(client)
        names   = sorted(c.name for c in client.get_collections().collections)
        aliases = client.get_aliases().aliases
    except Exception as e:
        return [[f"⚠️ Qdrant unreachable: {e}", "", "", "", "", "", "", "", "", ""]]

    alias_map: dict[str, list[str]] = {}
    for a in aliases:
        alias_map.setdefault(a.collection_name, []).append(a.alias_name)
    registry = load_registry()

    rows = []
    for name in names:
        info      = client.get_collection(name)
        vectors   = info.config.params.vectors
        dense_dim = getattr(vectors, "size", None)
        distance  = getattr(vectors, "distance", None)
        sparse    = bool(info.config.params.sparse_vectors)
        alias_names = alias_map.get(name, [])
        # Plans/registry entries for aliased collections are keyed by the
        # alias name (the deterministic/override collection_name), not the
        # underlying physical name — look up whichever applies.
        lookup_name = alias_names[0] if alias_names else name
        plan = get_plan_by_collection(lookup_name) or parse_collection_name(name)
        reg_entry = registry.get(lookup_name)
        langs_done = f"{len(reg_entry.get('lang_counts', {}))}/14" if reg_entry else "not registered"
        size = _format_size(_dir_size_bytes(QDRANT_STORAGE_PATH / "collections" / name))

        rows.append([
            name,
            ", ".join(alias_names) or "—",
            f"{info.points_count:,}",
            size,
            f"{dense_dim}-dim ({distance.value})" if dense_dim and distance else "—",
            "yes" if sparse else "no",
            plan.backend if plan else "?",
            ", ".join(plan.chunkers) if plan else "?",
            plan.split if plan else "?",
            langs_done,
        ])
    return rows or [["No collections yet", "", "", "", "", "", "", "", "", ""]]


def _language_distribution(collection_name: str) -> list[list]:
    if not collection_name:
        return [["Select a collection above", ""]]
    client = _client()
    rows = []
    for code in LANG_CODE_MAP:
        try:
            r = client.count(
                collection_name=collection_name,
                count_filter=Filter(must=[
                    FieldCondition(key="metadata.lang", match=MatchValue(value=code))
                ]),
                exact=False,
            )
            if r.count > 0:
                rows.append([code, f"{r.count:,}"])
        except Exception:
            continue
    return rows or [["No data for this collection", ""]]


def build_ingestion_tab() -> None:
    gr.Markdown(
        "Read-only view of what's actually in Qdrant. Indexing runs via CLI only "
        "(`python -m pipeline.indexer`) — nothing on this page starts, stops, or "
        "resumes anything."
    )

    gr.Markdown("### Collections")
    with gr.Group():
        collections_table = gr.Dataframe(
            headers=["Collection", "Alias", "Points", "Size", "Dense", "Sparse",
                     "Backend", "Chunkers", "Split", "Registered"],
            value=_collections_table(),
            interactive=False, wrap=True,
        )
        refresh_btn = gr.Button("↻ Refresh", size="sm", scale=0)

    gr.Markdown("### Language Distribution")
    with gr.Group():
        with gr.Row():
            collection_dd = gr.Dropdown(choices=_collection_names(), label="Collection", scale=3)
            load_btn = gr.Button("Load", scale=0)
        lang_table = gr.Dataframe(headers=["Language", "Points"], value=[], interactive=False)

    def _refresh():
        return _collections_table(), gr.update(choices=_collection_names())

    refresh_btn.click(fn=_refresh, outputs=[collections_table, collection_dd])
    load_btn.click(fn=_language_distribution, inputs=[collection_dd], outputs=[lang_table])

    timer = gr.Timer(value=30)
    timer.tick(fn=_refresh, outputs=[collections_table, collection_dd])
