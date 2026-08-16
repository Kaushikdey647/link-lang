"""Indexing tab — timer-driven live UI.

Architecture:
  - on_index_all / on_resume_paused / on_stop are fire-and-forget (return immediately).
  - A single background worker thread drains the language queue sequentially.
  - gr.Timer(1s) calls _poll_state() which reads _state and returns all component updates.
  - Backend (model) and chunking strategy (what to embed) are independent axes.
  - Their combination forms an IndexPlan with a deterministic collection name.
  - A plan registry persists metadata so the query layer can discover what's available.

Layout: a global run banner (what's happening, regardless of the selected plan),
a collapsible plan selector, one merged status table per selected plan (recorded
vs. live Qdrant counts + drift, replacing two previously-redundant panels), an
action bar, and a live-run detail panel that only appears while something is
actually running.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import gradio as gr
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from pipeline.chunking import REGISTRY
from pipeline.embedder import AVAILABLE_BACKENDS, DEFAULT_BACKEND
from pipeline.index_plan import (
    IndexPlan, load_registry, register_plan, register_legacy_collection, prune_missing_collections,
    get_plan_by_collection, parse_collection_name, MODEL_NAME_FOR,
)
from pipeline.indexer import QDRANT_URL, QDRANT_INDEXING_TIMEOUT

_QDRANT_STATS_TIMEOUT = 30  # read-only count queries; generous but shorter than indexing
from ui.theme import dark_figure, INDIGO, AMBER

# ---------------------------------------------------------------------------
# Language map
# ---------------------------------------------------------------------------

LANGUAGES: dict[str, str] = {
    "Hindi (हिंदी)":         "hi",
    "Bengali (বাংলা)":       "bn",
    "Gujarati (ગુજરાતી)":    "gu",
    "Kannada (ಕನ್ನಡ)":       "kn",
    "Malayalam (മലയാളം)":    "ml",
    "Marathi (मराठी)":       "mr",
    "Nepali (नेपाली)":       "ne",
    "Odia (ଓଡ଼ିଆ)":          "or",
    "Punjabi (ਪੰਜਾਬੀ)":      "pa",
    "Sanskrit (संस्कृतम्)":  "sa",
    "Tamil (தமிழ்)":         "ta",
    "Telugu (తెలుగు)":       "te",
    "Urdu (اردو)":           "ur",
    "Assamese (অসমীয়া)":    "as",
}

_LANG_DISPLAY = {v: k for k, v in LANGUAGES.items()}

_BACKEND_LABELS: dict[str, str] = {
    "e5":      "multilingual-e5-small (local, 384-dim)",
    "cohere":  "Cohere embed-multilingual-v3.0 (API, 1024-dim)",
    "english": "all-MiniLM-L6-v2 English pivot (local, 384-dim)",
}
_BACKEND_CHOICES = [(_BACKEND_LABELS.get(b, b), b) for b in AVAILABLE_BACKENDS]

# ---------------------------------------------------------------------------
# Checkpoint helpers  (keyed by plan.collection_name + lang)
# ---------------------------------------------------------------------------

_CHECKPOINT_DIR = Path(".indexer_checkpoints")


def _checkpoint_path(plan: IndexPlan, lang: str) -> Path:
    _CHECKPOINT_DIR.mkdir(exist_ok=True)
    return _CHECKPOINT_DIR / f"{plan.collection_name}__{lang}.json"


def _load_checkpoint(plan: IndexPlan, lang: str) -> dict:
    p = _checkpoint_path(plan, lang)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"passages_done": 0, "chunks_done": 0}


def _save_checkpoint(plan: IndexPlan, lang: str,
                     passages_done: int, chunks_done: int) -> None:
    _checkpoint_path(plan, lang).write_text(
        json.dumps({"passages_done": passages_done, "chunks_done": chunks_done})
    )


def _clear_checkpoint(plan: IndexPlan, lang: str) -> None:
    p = _checkpoint_path(plan, lang)
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------------------
# Chunk estimates
# ---------------------------------------------------------------------------

# Every MSMARCO-XI row has exactly 10 candidate passages (dataset/passages.py
# zips over a fixed 10-element is_selected array), and each chunker fires per
# PASSAGE, not per row — so "chunks per row" = (chunks per passage) x 10.
# qa_pair is the one exception: it only fires for the single is_selected=True
# passage per row.
_PASSAGES_PER_ROW = 10
_CHUNKS_PER_ROW = {"passage": 10, "sentence": 25, "qa_pair": 1, "english_query": 10}


def _estimate_total(num_rows: int, chunker_names: list[str]) -> int:
    return num_rows * sum(_CHUNKS_PER_ROW.get(n, 10) for n in chunker_names)


# ---------------------------------------------------------------------------
# Per-language status
# ---------------------------------------------------------------------------

_STATUS_ICON = {
    "not_started": "⬜",
    "queued":      "⏳",
    "running":     "🔄",
    "done":        "✅",
    "failed":      "❌",
    "paused":      "⏸",
}


@dataclass
class _LangStatus:
    status:        str = "not_started"
    chunks_done:   int = 0
    passages_done: int = 0
    error:         str = ""


def _default_lang_status() -> dict[str, _LangStatus]:
    return {code: _LangStatus() for code in LANGUAGES.values()}


# ---------------------------------------------------------------------------
# Shared state (written by worker thread, read by poll)
# ---------------------------------------------------------------------------

@dataclass
class _State:
    running:       bool  = False
    current_lang:  str   = ""
    current_plan:  object = None   # IndexPlan | None
    queue:         list  = field(default_factory=list)
    lang_status:   dict  = field(default_factory=_default_lang_status)
    # Langs the user explicitly reset via "↺ Reset status display" — held back
    # from disk-reconciliation until a new run queues them again, otherwise
    # the reconcile step (which reads checkpoints/registry every poll) would
    # immediately restore the pre-reset status and make the button a no-op.
    reset_overrides: set = field(default_factory=set)

    done_chunks:     int   = 0
    total_chunks:    int   = 0
    total_passages:  int   = 0   # exact passage count (num_rows x 10) — drives real progress %
    history:         list  = field(default_factory=list)
    log:          list  = field(default_factory=list)
    start_time:   float = 0.0
    stop_event:   threading.Event = field(default_factory=threading.Event)

    def add_log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")
        if len(self.log) > 500:
            self.log = self.log[-500:]

    def sample(self) -> None:
        passages_done = (
            self.lang_status[self.current_lang].passages_done if self.current_lang else 0
        )
        self.history.append((time.perf_counter() - self.start_time, self.done_chunks, passages_done))


_state = _State()
_lock  = threading.Lock()


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

def _current_rate() -> float:
    """Chunks/min over the last few samples (drives the throughput plot)."""
    h = _state.history
    if len(h) < 2:
        return 0.0
    w  = h[-5:]
    dt = (w[-1][0] - w[0][0]) / 60
    dc = w[-1][1] - w[0][1]
    return dc / dt if dt > 0 else 0.0


def _current_passage_rate() -> float:
    """Passages/min over the last few samples — drives ETA (exact, not a chunk-yield estimate)."""
    h = _state.history
    if len(h) < 2:
        return 0.0
    w  = h[-5:]
    dt = (w[-1][0] - w[0][0]) / 60
    dp = w[-1][2] - w[0][2]
    return dp / dt if dt > 0 else 0.0


def _index_one_lang(lang: str, plan: IndexPlan, batch_size: int,
                    stop_event: threading.Event) -> None:
    import uuid
    from itertools import islice as _islice
    from langchain_qdrant import QdrantVectorStore
    from dataset import iter_passages, load_language
    from pipeline.indexer import ensure_collection, _chunk_to_document, attach_sparse_vectors
    from pipeline.lc_embedder import ProjectEmbeddings
    from pipeline.chunking import build_chunker
    from api.metrics import (
        INDEXING_RUNNING, INDEXING_CHUNKS_DONE,
        INDEXING_CHUNKS_TARGET, INDEXING_THROUGHPUT,
    )
    _lbl = {"lang": lang, "split": plan.split}
    ckpt          = _load_checkpoint(plan, lang)
    start_passage = ckpt["passages_done"]
    start_chunks  = ckpt["chunks_done"]

    with _lock:
        _state.current_lang                    = lang
        _state.done_chunks                     = start_chunks
        _state.total_chunks                    = _estimate_total(100_000, plan.chunkers)
        _state.total_passages                  = 0   # unknown until dataset loads
        _state.history                         = []
        _state.start_time                      = time.perf_counter()
        _state.lang_status[lang].status        = "running"
        _state.lang_status[lang].chunks_done   = start_chunks
        _state.lang_status[lang].passages_done = start_passage
        verb = "Resuming" if start_passage else "Starting"
        _state.add_log(
            f"{verb} {lang!r}  collection={plan.collection_name!r}"
            + (f"  from passage {start_passage:,}" if start_passage else "")
        )
        _state.sample()

    try:
        ds             = load_language(lang, splits=(plan.split,))
        num_rows       = ds[plan.split].num_rows
        total_passages = num_rows * _PASSAGES_PER_ROW

        with _lock:
            _state.total_chunks   = _estimate_total(num_rows, plan.chunkers)
            _state.total_passages = total_passages
            _state.add_log(
                f"  {num_rows:,} rows ({total_passages:,} passages) · "
                f"est. {_state.total_chunks:,} chunks"
            )

        chunker = build_chunker(plan.chunkers)
        qdrant  = QdrantClient(url=QDRANT_URL, timeout=QDRANT_INDEXING_TIMEOUT)
        ensure_collection(qdrant, plan)
        vs = QdrantVectorStore(
            client=qdrant,
            collection_name=plan.collection_name,
            embedding=ProjectEmbeddings(backend=plan.backend),
        )

        INDEXING_RUNNING.set(1)
        INDEXING_CHUNKS_TARGET.labels(**_lbl).set(_state.total_chunks)
        INDEXING_CHUNKS_DONE.labels(**_lbl).set(_state.done_chunks)

        passages = iter_passages(ds[plan.split], lang, translated=True)
        if start_passage:
            for _ in _islice(passages, start_passage):
                pass

        batch: list = []
        passage_idx = start_passage
        batch_num   = 0

        for rec in passages:
            if stop_event.is_set():
                break
            batch.extend(chunker.chunk(rec))
            passage_idx += 1

            if len(batch) >= batch_size:
                docs = [_chunk_to_document(c) for c in batch]
                ids  = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.chunk_id)) for c in batch]
                vs.add_documents(docs, ids=ids)
                attach_sparse_vectors(qdrant, plan, batch, ids)
                batch_num += 1

                with _lock:
                    _state.done_chunks                     += len(batch)
                    _state.lang_status[lang].chunks_done   = _state.done_chunks
                    _state.lang_status[lang].passages_done = passage_idx
                    _state.sample()
                    rate = _current_rate()

                INDEXING_CHUNKS_DONE.labels(**_lbl).set(_state.done_chunks)
                INDEXING_THROUGHPUT.labels(**_lbl).set(rate)
                _save_checkpoint(plan, lang, passage_idx, _state.done_chunks)

                if batch_num % 10 == 0:
                    with _lock:
                        _state.add_log(
                            f"  [{lang}] batch {batch_num}"
                            f" · {_state.done_chunks:,} chunks"
                            f" · {rate:,.0f}/min"
                            f" · passage {passage_idx:,}/{total_passages:,}"
                        )
                batch = []

        # flush tail
        if batch and not stop_event.is_set():
            docs = [_chunk_to_document(c) for c in batch]
            ids  = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.chunk_id)) for c in batch]
            vs.add_documents(docs, ids=ids)
            attach_sparse_vectors(qdrant, plan, batch, ids)
            with _lock:
                _state.done_chunks                     += len(batch)
                _state.lang_status[lang].chunks_done   = _state.done_chunks
                _state.lang_status[lang].passages_done = passage_idx
                _state.sample()
            _save_checkpoint(plan, lang, passage_idx, _state.done_chunks)

        INDEXING_RUNNING.set(0)
        INDEXING_THROUGHPUT.labels(**_lbl).set(0)

        with _lock:
            if stop_event.is_set():
                _state.lang_status[lang].status = "paused"
                _state.add_log(
                    f"  [{lang}] ⏸ paused at passage {passage_idx:,}/{total_passages:,}"
                    f" · {_state.done_chunks:,} chunks · checkpoint saved"
                )
            else:
                _clear_checkpoint(plan, lang)
                _state.lang_status[lang].status = "done"
                _state.add_log(f"  [{lang}] ✅ {_state.done_chunks:,} chunks indexed")
                # Update the registry with this language's chunk count
                register_plan(plan, {lang: _state.done_chunks})

    except Exception as exc:
        INDEXING_RUNNING.set(0)
        with _lock:
            _state.lang_status[lang].status = "failed"
            _state.lang_status[lang].error  = str(exc)
            _state.add_log(f"  [{lang}] ❌ {exc}")


def _worker(plan: IndexPlan, batch_size: int, stop_event: threading.Event) -> None:
    while True:
        with _lock:
            if not _state.queue or stop_event.is_set():
                break
            lang = _state.queue.pop(0)

        _index_one_lang(lang, plan, batch_size, stop_event)

        if stop_event.is_set():
            with _lock:
                for queued_lang in _state.queue:
                    if _state.lang_status[queued_lang].status == "queued":
                        ckpt = _load_checkpoint(plan, queued_lang)
                        _state.lang_status[queued_lang].status = (
                            "paused" if ckpt["passages_done"] > 0 else "not_started"
                        )
                _state.queue.clear()
            break

    with _lock:
        _state.running      = False
        _state.current_lang = ""
        done_count = sum(1 for ls in _state.lang_status.values() if ls.status == "done")
        _state.add_log(f"Worker finished · {done_count} languages complete")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_plt_lock = threading.Lock()


def _make_figure():
    with _lock:
        history = list(_state.history)
        total   = _state.total_chunks

    with _plt_lock:
        import matplotlib.ticker as ticker

        fig, ax1 = dark_figure(7, 2.8, dpi=100)

        if len(history) < 2:
            ax1.text(0.5, 0.5, "Waiting for data…", ha="center", va="center",
                     transform=ax1.transAxes, fontsize=11, color="#4b5563")
            ax1.set_axis_off()
            fig.tight_layout()
            return fig

        times  = [t for t, _, _ in history]
        chunks = [c for _, c, _ in history]

        ax1.fill_between(times, chunks, alpha=0.12, color=INDIGO)
        ax1.plot(times, chunks, color=INDIGO, linewidth=2)
        if total > 0:
            ax1.axhline(total, color=INDIGO, linestyle="--", linewidth=1, alpha=0.35)
            ax1.text(times[-1], total * 1.02, f"  est. {total:,}",
                     color=INDIGO, fontsize=7, alpha=0.6, va="bottom")

        ax1.set_xlabel("elapsed (s)", fontsize=7, color="#9ca3af")
        ax1.set_ylabel("chunks indexed", color=INDIGO, fontsize=7)
        ax1.tick_params(axis="y", labelcolor=INDIGO)
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

        if len(history) >= 3:
            rates, rtimes = [], []
            for i in range(1, len(history)):
                dt = (history[i][0] - history[i-1][0]) / 60
                dc = history[i][1] - history[i-1][1]
                rates.append(dc / dt if dt > 0 else 0)
                rtimes.append(history[i][0])
            sm = rates[:]
            for i in range(1, len(rates) - 1):
                sm[i] = (rates[i-1] + rates[i] + rates[i+1]) / 3
            ax2 = ax1.twinx()
            ax2.set_facecolor("#111827")
            ax2.plot(rtimes, sm, color=AMBER, linewidth=1.5, linestyle="--", alpha=0.85)
            ax2.set_ylabel("chunks/min", color=AMBER, fontsize=7)
            ax2.tick_params(axis="y", labelcolor=AMBER, labelsize=7)
            ax2.spines[["top", "right", "left", "bottom"]].set_color("#1f2937")
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

        fig.tight_layout(pad=1.0)
        return fig


# ---------------------------------------------------------------------------
# Reconcile in-memory status from on-disk checkpoints + registry.
#
# _state is a process-local singleton and starts blank on every restart. This
# rehydrates it from the two on-disk stores (checkpoints, registry.json)
# before rendering, so a restart doesn't lose track of real progress. It never
# touches the language actively running in this process — live counters win.
# ---------------------------------------------------------------------------

def _reconcile_lang_status(plan: IndexPlan) -> None:
    lang_counts = load_registry().get(plan.collection_name, {}).get("lang_counts", {})
    with _lock:
        active = _state.current_lang if _state.running else None
        for code in LANGUAGES.values():
            if code == active or code in _state.reset_overrides:
                continue
            ls = _state.lang_status[code]
            if code in lang_counts:
                ls.status      = "done"
                ls.chunks_done = lang_counts[code]
                continue
            if ls.status in ("running", "failed", "queued"):
                continue  # reflects this session already; disk has nothing newer
            ckpt = _load_checkpoint(plan, code)
            if ckpt["passages_done"] > 0:
                ls.status        = "paused"
                ls.chunks_done   = ckpt["chunks_done"]
                ls.passages_done = ckpt["passages_done"]
            else:
                ls.status = "not_started"


# ---------------------------------------------------------------------------
# Live Qdrant counts — cached per collection, refreshed on a slow cadence
# (30s timer / manual refresh / plan change) so the 1s status-table poll never
# hits Qdrant directly.
# ---------------------------------------------------------------------------

_qdrant_cache: dict[str, tuple[dict[str, int], int, bool]] = {}
_qdrant_cache_lock = threading.Lock()


def _fetch_live_qdrant_counts(plan: IndexPlan) -> tuple[dict[str, int], int, bool]:
    """(per-language counts, total points, collection_exists). Never raises."""
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=_QDRANT_STATS_TIMEOUT)
        names  = {c.name for c in client.get_collections().collections}
        if plan.collection_name not in names:
            return {}, 0, False
        info   = client.get_collection(plan.collection_name)
        total  = info.points_count or 0
        counts = {}
        for code in LANGUAGES.values():
            r = client.count(
                collection_name=plan.collection_name,
                count_filter=Filter(must=[
                    FieldCondition(key="metadata.lang", match=MatchValue(value=code))
                ]),
                exact=False,
            )
            if r.count > 0:
                counts[code] = r.count
        return counts, total, True
    except Exception:
        return {}, 0, False


def _refresh_qdrant_cache(plan: IndexPlan) -> None:
    result = _fetch_live_qdrant_counts(plan)
    with _qdrant_cache_lock:
        _qdrant_cache[plan.collection_name] = result


def _cached_qdrant(plan: IndexPlan) -> tuple[dict[str, int], int, bool]:
    with _qdrant_cache_lock:
        cached = _qdrant_cache.get(plan.collection_name)
    if cached is None:
        _refresh_qdrant_cache(plan)
        with _qdrant_cache_lock:
            cached = _qdrant_cache.get(plan.collection_name, ({}, 0, False))
    return cached


# ---------------------------------------------------------------------------
# Selected-plan status: one merged table (recorded + live Qdrant + drift),
# replacing the old separate Language Status / Collection Point Counts panels.
# ---------------------------------------------------------------------------

def _plan_header(plan: IndexPlan) -> str:
    _, total, exists = _cached_qdrant(plan)
    if not exists:
        return f"`{plan.collection_name}` &nbsp;·&nbsp; **{plan.model_name}** ({plan.vector_dim}-dim) &nbsp;·&nbsp; _not yet created in Qdrant_"
    return f"`{plan.collection_name}` &nbsp;·&nbsp; **{plan.model_name}** ({plan.vector_dim}-dim) &nbsp;·&nbsp; **{total:,}** total points"


def _build_status_table(plan: IndexPlan) -> list[list]:
    _reconcile_lang_status(plan)
    live_counts, _, exists = _cached_qdrant(plan)
    rows = []
    with _lock:
        for display, code in LANGUAGES.items():
            ls     = _state.lang_status[code]
            status = ls.status
            icon   = _STATUS_ICON.get(status, "❓")
            label  = f"{icon} {status.replace('_', ' ').title()}"
            passages = f"{ls.passages_done:,}" if ls.passages_done > 0 else "—"
            chunks   = f"{ls.chunks_done:,}" if ls.chunks_done > 0 else "—"
            live     = live_counts.get(code, 0)
            if not exists:
                live_str = "?"
            else:
                live_str = f"{live:,}" if live > 0 else "—"
            drift = ""
            if exists and ls.chunks_done > 0:
                if live == 0:
                    drift = "⚠️ MISSING"
                elif abs(live - ls.chunks_done) / ls.chunks_done > 0.05:
                    drift = "⚠️ drift"
            error = ls.error[:80] if status == "failed" and ls.error else ""
            rows.append([display, label, passages, chunks, live_str, drift, error])
    return rows


# ---------------------------------------------------------------------------
# Registry table (cross-plan discovery — "All Indexed Plans" accordion)
# ---------------------------------------------------------------------------

def _build_registry_table() -> list[list]:
    registry = load_registry()
    rows = []
    for entry in registry.values():
        lang_counts = entry.get("lang_counts", {})
        langs_done  = len(lang_counts)
        total_chunks = sum(lang_counts.values())
        rows.append([
            entry.get("collection_name", ""),
            entry.get("backend", ""),
            entry.get("model_name", ""),
            ", ".join(entry.get("chunkers", [])),
            entry.get("split", ""),
            f"{langs_done}/14",
            f"{total_chunks:,}" if total_chunks else "—",
        ])
    return rows or [["No plans indexed yet", "", "", "", "", "", ""]]


# ---------------------------------------------------------------------------
# Legacy collections: pre-refactor Qdrant collections whose names don't follow
# the deterministic IndexPlan.collection_name format, so they'd otherwise
# never appear in the registry / Indexed Plans table.
# ---------------------------------------------------------------------------

_LEGACY_COLLECTION_DEFAULTS: dict[str, dict] = {
    "msmarco_xi_e5": {"backend": "e5", "chunkers": ["passage", "sentence", "qa_pair"], "split": "train"},
}


def _sync_registry_with_qdrant() -> None:
    """Reconcile registry.json against what's actually in Qdrant: drop entries
    whose collection no longer exists (e.g. deleted/recreated outside the app,
    which would otherwise show as a permanent ghost entry), and auto-register
    known legacy collections that exist but aren't registered yet."""
    registry = load_registry()
    to_check = {n: d for n, d in _LEGACY_COLLECTION_DEFAULTS.items() if n not in registry}
    if not registry and not to_check:
        return
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=_QDRANT_STATS_TIMEOUT)
        existing = {c.name for c in client.get_collections().collections}
    except Exception:
        return

    prune_missing_collections(existing)

    for name, defaults in to_check.items():
        if name not in existing:
            continue
        lang_counts = {}
        for code in LANGUAGES.values():
            try:
                r = client.count(
                    collection_name=name,
                    count_filter=Filter(must=[
                        FieldCondition(key="metadata.lang", match=MatchValue(value=code))
                    ]),
                    exact=False,
                )
                if r.count > 0:
                    lang_counts[code] = r.count
            except Exception:
                continue
        register_legacy_collection(name, lang_counts=lang_counts, **defaults)


# ---------------------------------------------------------------------------
# Known collections — quick-select so the user never has to reconstruct
# which backend/chunkers produced a given collection name from memory.
# ---------------------------------------------------------------------------

_MAX_KNOWN = 6


def _known_collections() -> list[tuple[str, str]]:
    """(collection_name, label) for every collection with any record on disk —
    registry entries plus new-format checkpoints not yet registered."""
    registry = load_registry()
    names = set(registry.keys())
    for p in _CHECKPOINT_DIR.glob("*__*.json"):
        names.add(p.stem.rsplit("__", 1)[0])

    results = []
    for name in sorted(names):
        entry = registry.get(name, {})
        lang_counts = entry.get("lang_counts", {})
        tag = f"{len(lang_counts)}/14 done" if lang_counts else "paused"
        results.append((name, f"{name} · {tag}"))
    return results[:_MAX_KNOWN]


def _resolve_known(name: str) -> IndexPlan | None:
    return get_plan_by_collection(name) or parse_collection_name(name)


# ---------------------------------------------------------------------------
# Collection name preview (computed from UI state)
# ---------------------------------------------------------------------------

def _collection_preview(backend: str, chunker: str, split: str) -> str:
    if not chunker:
        return "_Select a chunking strategy._"
    plan = IndexPlan(backend=backend, chunkers=[chunker], split=split)
    model = MODEL_NAME_FOR.get(backend, backend)
    return f"`{plan.collection_name}`\n\nModel: **{model}** · Dim: **{plan.vector_dim}**"


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

def _make_plan(backend: str, chunker: str, split: str) -> IndexPlan | None:
    if not chunker:
        return None
    return IndexPlan(backend=backend, chunkers=[chunker], split=split)


def on_index_all(backend: str, split: str, batch_size: int, chunker: str) -> str:
    if not chunker:
        return "⚠️ Select a chunking strategy."
    with _lock:
        if _state.running:
            return "Already running — press ⏹ Stop first."

    plan = IndexPlan(backend=backend, chunkers=[chunker], split=split)

    with _lock:
        langs_to_queue = [
            code for code in LANGUAGES.values()
            if _state.lang_status[code].status not in ("done", "running")
        ]
        if not langs_to_queue:
            return "All languages already marked done. Use ↺ Reset to re-index."

        for code in langs_to_queue:
            _state.lang_status[code].status = "queued"
            _state.reset_overrides.discard(code)
        _state.queue        = list(langs_to_queue)
        _state.running      = True
        _state.current_plan = plan
        _state.stop_event   = threading.Event()
        _state.add_log(
            f"Queued {len(langs_to_queue)} languages · {plan.collection_name}"
        )

    stop_ev = _state.stop_event
    threading.Thread(
        target=_worker,
        args=(plan, batch_size, stop_ev),
        daemon=True,
    ).start()
    return ""


def on_resume_paused(backend: str, split: str, batch_size: int, chunker: str) -> str:
    if not chunker:
        return "⚠️ Select a chunking strategy."
    with _lock:
        if _state.running:
            return "Already running — press ⏹ Stop first."

    plan = IndexPlan(backend=backend, chunkers=[chunker], split=split)

    with _lock:
        paused = []
        for code in LANGUAGES.values():
            ckpt = _load_checkpoint(plan, code)
            if ckpt["passages_done"] > 0 and _state.lang_status[code].status != "done":
                paused.append(code)
                _state.lang_status[code].status = "queued"
                _state.reset_overrides.discard(code)
        if not paused:
            return "No paused checkpoints found for this plan."

        _state.queue        = list(paused)
        _state.running      = True
        _state.current_plan = plan
        _state.stop_event   = threading.Event()
        _state.add_log(f"Resuming {len(paused)} paused languages · {plan.collection_name}")

    stop_ev = _state.stop_event
    threading.Thread(
        target=_worker,
        args=(plan, batch_size, stop_ev),
        daemon=True,
    ).start()
    return ""


def on_stop() -> str:
    with _lock:
        if not _state.running:
            return "Nothing is running."
        _state.stop_event.set()
        _state.add_log("⏹ Stop requested — finishing current batch…")
    return "Stop signal sent — finishing current batch…"


def on_reset_statuses() -> str:
    with _lock:
        if _state.running:
            return "⚠️ Stop indexing before resetting."
        for code, ls in _state.lang_status.items():
            ls.status        = "not_started"
            ls.chunks_done   = 0
            ls.passages_done = 0
            _state.reset_overrides.add(code)
    return "Status display reset (checkpoints preserved)."


# ---------------------------------------------------------------------------
# Run banner — the single glanceable answer to "what's happening right now,
# anywhere", independent of whichever plan is selected in the dropdowns.
# ---------------------------------------------------------------------------

def _run_banner() -> str:
    with _lock:
        running      = _state.running
        current_lang = _state.current_lang
        current_plan = _state.current_plan
        queue_len    = len(_state.queue)
        done         = _state.done_chunks
    if not running:
        return "⚫ **Idle** — no run in progress"
    display = _LANG_DISPLAY.get(current_lang, current_lang or "…")
    collection = current_plan.collection_name if current_plan else "?"
    queue_note = f" · **{queue_len}** queued" if queue_len else ""
    return f"🔄 **Running: {display}** under `{collection}` · {done:,} chunks so far{queue_note}"


# ---------------------------------------------------------------------------
# Timer poll
# ---------------------------------------------------------------------------

def _poll_state(backend: str, chunker: str, split: str) -> tuple:
    plan = _make_plan(backend, chunker, split)

    # Reconciles _state.lang_status against checkpoints/registry first, so the
    # status table reflects real on-disk state, not just whatever survived in
    # memory since the last restart.
    status_table = _build_status_table(plan) if plan else []
    plan_header  = _plan_header(plan) if plan else "_Select at least one chunking strategy._"

    with _lock:
        running       = _state.running
        current_lang  = _state.current_lang
        current_plan  = _state.current_plan
        done           = _state.done_chunks
        total_est      = _state.total_chunks
        total_passages = _state.total_passages
        passages_done  = _state.lang_status[current_lang].passages_done if current_lang else 0
        log_txt        = "\n".join(_state.log[-80:])
        rate           = _current_rate() if running else 0.0
        passage_rate   = _current_passage_rate() if running else 0.0
        queue_len      = len(_state.queue)

    # Progress % is passage-based (exact: passages_done/total_passages), not
    # the old chunk-count estimate — chunk yield per row varies wildly by
    # language (e.g. Sanskrit ~1.2 chunks/row vs. a flat heuristic of ~36),
    # which used to push the displayed count past 100%. total_passages is
    # num_rows x 10 (every row has exactly 10 candidate passages) — comparing
    # passages_done against the raw row count instead was the earlier bug
    # that made languages read "100% done" at ~10% real completion.
    pct = min(100.0, round(passages_done / total_passages * 100, 1)) if total_passages > 0 else 0.0
    # Self-correcting total-chunks estimate for display: once real chunks/passage
    # ratio is observed, extrapolate from it instead of the static heuristic.
    display_total = round(done / passages_done * total_passages) if passages_done > 0 and total_passages > 0 else total_est

    banner = _run_banner()

    if running and current_lang:
        display = _LANG_DISPLAY.get(current_lang, current_lang)
        eta_str = ""
        if passage_rate > 0 and total_passages > passages_done:
            eta_str = f" · ETA ~{(total_passages - passages_done) / passage_rate:.0f} min"
        run_detail = (
            f"🔄 **{display}** · passages {passages_done:,}/{total_passages:,} "
            f"· {done:,} / ~{display_total:,} chunks "
            f"({pct:.1f}%) · {rate:,.0f} chunks/min{eta_str}"
        )
    else:
        run_detail = ""

    action_note = "⚙️ Stop indexing to change options or select a different plan." if running else ""

    selector_upd = gr.update(interactive=not running)  # backend / chunker / split / batch size
    all_upd      = gr.update(interactive=not running)
    resume_upd   = gr.update(interactive=not running)
    if running:
        stop_label = f"⏹ Stop ({_LANG_DISPLAY.get(current_lang, current_lang)} · {current_plan.collection_name if current_plan else '?'})"
    else:
        stop_label = "⏹ Stop"
    stop_upd = gr.update(value=stop_label, interactive=running)

    return (
        banner,
        plan_header,
        status_table,
        gr.update(visible=running),
        run_detail,
        pct,
        _make_figure(),
        log_txt,
        action_note,
        all_upd,
        resume_upd,
        stop_upd,
        selector_upd,
        selector_upd,
        selector_upd,
        selector_upd,
    )


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_indexing_tab() -> None:
    gr.Markdown(
        "Index MSMARCO-XI passages into Qdrant. "
        "**Embedding model** and **what to embed** are independent axes — "
        "their combination defines a named collection."
    )

    # ── Run banner: always visible, independent of the plan selected below ──
    run_banner_md = gr.Markdown(_run_banner())

    # ── Plan definition ───────────────────────────────────────────────────
    with gr.Accordion("Define what to index", open=True) as plan_accordion:
        with gr.Row():
            backend_dd  = gr.Dropdown(
                choices=_BACKEND_CHOICES, value=DEFAULT_BACKEND,
                label="Embedding model (which)", scale=2,
            )
            split_radio = gr.Radio(
                ["train", "validation"], value="train",
                label="Dataset split", scale=1,
            )

        # One chunking strategy + one embedding model maps to exactly one
        # collection — single-select, not a checkbox group.
        chunker_dd = gr.Dropdown(
            choices=list(REGISTRY.keys()),
            value="english_query",
            label="What to embed (chunking strategy)",
        )

        collection_preview = gr.Markdown(
            _collection_preview(DEFAULT_BACKEND, "english_query", "train"),
            label="→ Collection that will be created / updated",
        )

        batch_slider = gr.Slider(64, 512, value=256, step=64, label="Batch size")

        gr.Markdown("**Known collections** — click to load a previously-used plan:")
        _initial_known = _known_collections()
        known_buttons = []
        with gr.Row():
            for i in range(_MAX_KNOWN):
                visible = i < len(_initial_known)
                label   = _initial_known[i][1] if visible else ""
                known_buttons.append(gr.Button(label, visible=visible, size="sm", scale=1))

    # ── Selected-plan status: one merged table, no duplication ─────────────
    gr.Markdown("### Selected Plan")
    plan_header_md = gr.Markdown(
        _plan_header(IndexPlan(DEFAULT_BACKEND, ["english_query"]))
    )
    status_table = gr.Dataframe(
        headers=["Language", "Status", "Passages", "Chunks", "Qdrant", "Drift", "Error"],
        value=_build_status_table(
            IndexPlan(DEFAULT_BACKEND, ["english_query"])
        ),
        interactive=False, wrap=True,
    )

    # ── Action bar ───────────────────────────────────────────────────────
    with gr.Row():
        all_btn    = gr.Button("▶ Index All",      variant="primary",   scale=3)
        resume_btn = gr.Button("⏸ Resume Paused", variant="secondary", scale=2)
        stop_btn   = gr.Button("⏹ Stop",           variant="stop",      scale=3,
                                interactive=False)
    action_note = gr.Markdown()

    # ── Live run detail — only visible while something is running ─────────
    with gr.Group(visible=False) as live_group:
        gr.Markdown("### Current Run")
        run_detail_md = gr.Markdown()
        progress_bar  = gr.Slider(0, 100, value=0, step=0.1,
                                  label="Progress (%)", interactive=False)
        with gr.Row():
            with gr.Column(scale=3):
                throughput_plot = gr.Plot(label="Throughput")
            with gr.Column(scale=2):
                log_box = gr.Textbox(label="Log", lines=14, max_lines=14,
                                     interactive=False, autoscroll=True)

    # ── All Indexed Plans (cross-plan registry — collapsed, discovery only) ─
    with gr.Accordion("📚 All Indexed Plans (queryable via API)", open=False):
        _sync_registry_with_qdrant()
        registry_table = gr.Dataframe(
            headers=["Collection", "Backend", "Model", "Chunkers",
                     "Split", "Langs Done", "Total Chunks"],
            value=_build_registry_table(),
            interactive=False, wrap=True,
        )
        with gr.Row():
            refresh_btn = gr.Button("↻ Refresh stats", size="sm", scale=0)
            reset_btn   = gr.Button("↺ Reset status display", size="sm", scale=0)

    # ── Timer: drives all live components ─────────────────────────────────
    _POLL_OUTPUTS = [
        run_banner_md, plan_header_md, status_table, live_group, run_detail_md,
        progress_bar, throughput_plot, log_box, action_note,
        all_btn, resume_btn, stop_btn,
        backend_dd, chunker_dd, split_radio, batch_slider,
    ]
    timer = gr.Timer(value=1)
    timer.tick(
        fn=_poll_state,
        inputs=[backend_dd, chunker_dd, split_radio],
        outputs=_POLL_OUTPUTS,
    )

    # ── Collection name preview: update on any plan change ────────────────
    def _update_preview(backend, chunker, split):
        return _collection_preview(backend, chunker, split)

    for component in [backend_dd, chunker_dd, split_radio]:
        component.change(
            fn=_update_preview,
            inputs=[backend_dd, chunker_dd, split_radio],
            outputs=[collection_preview],
        )

    # ── Known-collection quick-select ───────────────────────────────────────
    def _known_updates():
        known = _known_collections()
        upds = []
        for i in range(_MAX_KNOWN):
            if i < len(known):
                upds.append(gr.update(value=known[i][1], visible=True))
            else:
                upds.append(gr.update(visible=False))
        return upds

    def _select_known(index: int):
        def _handler():
            known = _known_collections()
            if index >= len(known):
                return gr.update(), gr.update(), gr.update()
            plan = _resolve_known(known[index][0])
            if plan is None or not plan.chunkers:
                return gr.update(), gr.update(), gr.update()
            # Single-select dropdown: a legacy multi-chunker collection is
            # represented by its first chunker (closest single equivalent).
            return plan.backend, plan.split, plan.chunkers[0]
        return _handler

    for i, btn in enumerate(known_buttons):
        btn.click(fn=_select_known(i), outputs=[backend_dd, split_radio, chunker_dd])

    # ── Buttons ───────────────────────────────────────────────────────────
    def _index_all_and_collapse(backend, split, batch_size, chunker):
        note = on_index_all(backend, split, batch_size, chunker)
        return note, gr.update(open=False)

    def _resume_and_collapse(backend, split, batch_size, chunker):
        note = on_resume_paused(backend, split, batch_size, chunker)
        return note, gr.update(open=False)

    all_btn.click(
        fn=_index_all_and_collapse,
        inputs=[backend_dd, split_radio, batch_slider, chunker_dd],
        outputs=[action_note, plan_accordion],
    )
    resume_btn.click(
        fn=_resume_and_collapse,
        inputs=[backend_dd, split_radio, batch_slider, chunker_dd],
        outputs=[action_note, plan_accordion],
    )
    stop_btn.click(fn=on_stop, outputs=[action_note])
    reset_btn.click(fn=on_reset_statuses, outputs=[action_note])

    # Stats: updates when plan selection changes + manual refresh + 30 s timer
    def _refresh_stats(backend, chunker, split):
        plan = _make_plan(backend, chunker, split)
        if plan:
            _refresh_qdrant_cache(plan)
        _sync_registry_with_qdrant()
        return (_build_registry_table(), *_known_updates())

    _STATS_OUTPUTS = [registry_table, *known_buttons]

    for component in [backend_dd, chunker_dd, split_radio]:
        component.change(
            fn=_refresh_stats,
            inputs=[backend_dd, chunker_dd, split_radio],
            outputs=_STATS_OUTPUTS,
        )
    refresh_btn.click(
        fn=_refresh_stats,
        inputs=[backend_dd, chunker_dd, split_radio],
        outputs=_STATS_OUTPUTS,
    )
    stats_timer = gr.Timer(value=30)
    stats_timer.tick(
        fn=_refresh_stats,
        inputs=[backend_dd, chunker_dd, split_radio],
        outputs=_STATS_OUTPUTS,
    )
