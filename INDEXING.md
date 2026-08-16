# Indexing

Indexing is **CLI-only**. There is no button, endpoint, or admin control that
starts, stops, or resumes an indexing run — the only way to index is:

```bash
uv run python -m scripts.index --langs hi bn
```

This is deliberate: indexing is a heavy, occasional, operator-driven task
(minutes to hours per language), and keeping it out-of-process means the
serving side (`frontend/` — see CHANGELOG.md's `frontend_only` entry) has zero
control surface for it. Nothing in the Next.js serving app can start, stop,
or trigger an indexing run.

english-pivot (MiniLM dense + BM25 sparse, both computed server-side by
Qdrant Cloud) is the system's one supported chunking/embedding strategy — see
`CHUNKING.md` for what else was prototyped and why. There's no `--backend`/
`--chunkers` flag to choose; every run uses `IndexPlan(backend="english",
chunkers=["english_query"])`.

---

## Quick start

```bash
# Index Hindi
uv run python -m scripts.index --langs hi

# Index everything, four languages at a time
uv run python -m scripts.index --langs all --workers 4

# Quick smoke test — 5,000 passages per language, no commitment
uv run python -m scripts.index --langs hi --limit 5000
```

Prerequisite: Qdrant reachable — either local at `http://localhost:6333`
(fallback, local-only dev, see `README.md`) or a Qdrant Cloud cluster via
`QDRANT_CLUSTER_ENDPOINT`/`QDRANT_API_KEY` in `.env` (see "Remote (Qdrant
Cloud) inference" below — **required** for embeddings to actually work, since
there's no local embedding fallback anymore).

---

## CLI reference (`scripts/index.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--langs` | `hi` | One or more 2-letter codes, or `all` for all 14 (`as bn gu hi kn ml mr ne or pa sa ta te ur`). |
| `--split` | `train` | `train` or `validation`. |
| `--batch-size` | `256` | Passages embedded + upserted per Qdrant round-trip. |
| `--workers` | `1` | Parallel language processes. `1` = sequential (today's default). `>1` runs that many languages concurrently, each in its own OS process. |
| `--limit` | none | Stop after N passages *per language* — for quick test runs. The checkpoint still records exactly where it stopped, so a later un-limited run continues from there. |

---

## Collections & IndexPlan

Every run is described by an `IndexPlan` (`pipeline/index_plan.py`):
`(backend, chunkers, split)`, always `("english", ["english_query"], split)`
today. The collection name is derived deterministically:

```
msmarco_xi__english__english_query__{split}
```

Running the same `--split` always targets the same collection — that's how
`--langs hi` today and `--langs bn` tomorrow end up in one multi-language
collection, and how re-running an interrupted language resumes into the same
place instead of creating a duplicate.

---

## Use cases

### 1. Index one language
```bash
uv run python -m scripts.index --langs hi
```
Sequential by default (`--workers 1`). Safe to `Ctrl+C` at any point — see Checkpoints below.

### 2. Index several languages faster (parallel)
```bash
uv run python -m scripts.index --langs hi bn gu ta --workers 4
```
Each language runs in its own process (`concurrent.futures.ProcessPoolExecutor`), fully independent — own dataset slice, own checkpoint file, own point IDs. Pick `--workers` based on available CPU cores/network concurrency to Qdrant Cloud; there's no cross-language shared state to contend over.

### 3. Index everything
```bash
uv run python -m scripts.index --langs all --workers 4
```
`all` expands to all 14 language codes (`constants.LANG_CODE_MAP`).

### 4. Resume an interrupted run
Just re-run the exact same command. Each language reads its checkpoint (`.indexer_checkpoints/{collection_name}__{lang}.json`) and picks up from the last completed batch — no flag needed, this is always-on behavior, not opt-in.

```bash
# First attempt gets killed partway through
uv run python -m scripts.index --langs hi
# ^C

# Re-run the identical command — resumes from the checkpoint, doesn't restart
uv run python -m scripts.index --langs hi
```

### 5. Quick test runs without committing to a full language
```bash
uv run python -m scripts.index --langs hi bn --limit 5000
```
Stops after 5,000 passages per language. The checkpoint is left in place (not cleared, not registered as done), so:
- A later un-limited run continues from passage 5,000 rather than starting over.
- `registry.json` correctly shows it as "not registered" rather than falsely complete.

---

## Checkpoints

`.indexer_checkpoints/{collection_name}__{lang}.json`:
```json
{"passages_done": 338944, "chunks_done": 338944}
```
Written after every batch. Deleted only when a language finishes cleanly (not stopped early, no `--limit` cutoff) — at that point it also registers into `.indexer_checkpoints/registry.json` (see below). A file existing means that language is genuinely incomplete for that exact collection; if it's missing, the language is either untouched or fully done.

---

## Registry (`registry.json`)

`.indexer_checkpoints/registry.json` tracks, per collection: backend, model, chunkers, split, and `lang_counts` (chunks indexed per language that's actually finished). This is ingestion-side bookkeeping only now — the Next.js serving side (`frontend/`) does **not** read this file; it resolves the live collection directly from Qdrant and caches it in memory per warm instance (`frontend/lib/server/qdrant.ts::getLiveCollection()`), since a serverless deploy's filesystem doesn't survive across invocations. See CHANGELOG.md's `frontend_only` entry.

It's local-only (gitignored — per-machine run state, not source), so it stays in sync automatically: `run_indexing()` calls `sync_registry_with_qdrant()` after every run — any process that hits Qdrant auto-discovers deterministically-named collections it doesn't yet know about and registers them, and prunes entries for collections that no longer exist. You never edit this file by hand.

---

## Performance notes

- **Streaming dataset load**: `dataset/loader.py::iter_language_rows()` reads each language's parquet file batch-by-batch (`pq.ParquetFile.iter_batches()`), never materializing a full-file Arrow Table — memory stays bounded by batch size, not file size (each language's train parquet is 3.3-4.0GB on disk). This is what `pipeline/indexer.py::index_language()` uses; `--limit` genuinely bounds how much gets read from disk, and `--workers > 1` doesn't multiply full-file loads across processes.
- **No local embedding model** — dense (MiniLM) and sparse (BM25) vectors are both computed server-side by Qdrant Cloud; nothing to load, no device selection, no local inference cost. See "Remote (Qdrant Cloud) inference" below.
- **Merged upsert**: dense + sparse vectors are computed and upserted in a single Qdrant call per batch, not two.
- **Parallelism caveat**: always invoke via `uv run python -m scripts.index`, not `python -m pipeline.indexer` directly. `--workers > 1` uses `ProcessPoolExecutor`'s `spawn` start method, which needs to re-import the worker function from a real module path in each child process — running `pipeline/indexer.py` itself as `__main__` breaks that (confirmed: every worker crashes instantly). `scripts/index.py` exists specifically so `pipeline.indexer` is always imported normally.

---

## Remote (Qdrant Cloud) inference

`QDRANT_CLUSTER_ENDPOINT` + `QDRANT_API_KEY` in `.env` control where Qdrant lives *and* who computes embeddings — there's no local embedding fallback, so `QDRANT_API_KEY` is effectively required for the system to do anything useful (`pipeline/indexer.py::QDRANT_CLOUD_INFERENCE`):

1. **Where Qdrant lives** — every `QdrantClient` in the app (`pipeline/indexer.py`, `pipeline/rag.py`, `ui/ingestion_tab.py`, `api/app.py`) connects to the cloud cluster instead of `localhost:6333`.
2. **Who computes the embedding** — the client sends raw text as a Qdrant `Document(text=..., model=...)` at both index time (`pipeline/indexer.py::_upsert_batch`) and query time (`pipeline/query_engines.py::EnglishPivotQueryEngine`), and Qdrant Cloud computes MiniLM dense + BM25 sparse vectors server-side (`cloud_inference=True` on the client). If `QDRANT_API_KEY` is unset, indexing/serving still runs against local Qdrant, but embedding fails loudly with a Qdrant-side error rather than silently falling back to anything local.

Nothing else about indexing changes: the dataset still streams from the local parquet cache (`dataset/loader.py::iter_language_rows`), chunking still happens locally (`pipeline/chunking.py`) — only the embedding *computation* moves server-side.

### SOP: testing remote embeddings

This can't be verified from every network (the cluster may be unreachable from some VPNs/firewalls) — run these from a network that can actually reach your `QDRANT_CLUSTER_ENDPOINT`, after `.env` has both vars set.

1. **Connectivity + auth check** (no embedding involved yet):
   ```bash
   uv run python -c "from pipeline.indexer import _get_qdrant_client; print(_get_qdrant_client().get_collections())"
   ```
   A `401`/`403` here means `QDRANT_API_KEY` is wrong; a connection error means `QDRANT_CLUSTER_ENDPOINT` is wrong or the network can't reach it.

2. **Tiny smoke-test ingest** — exercises the cloud-inference upsert path end to end:
   ```bash
   uv run python -m scripts.index --langs hi --limit 50
   ```
   `ensure_collection()` will create the collection on the remote cluster. If the model name doesn't match Qdrant's registry exactly, or your cluster's plan/tier doesn't include cloud inference, this fails here with a Qdrant-side error naming the problem — it does not silently fall back to local embedding.

3. **Confirm real vectors landed** (not just accepted):
   ```bash
   uv run python -c "
   from pipeline.indexer import _get_qdrant_client
   from pipeline.index_plan import IndexPlan
   plan = IndexPlan(backend='english', chunkers=['english_query'], split='train')
   pts, _ = _get_qdrant_client().scroll(plan.collection_name, limit=1, with_vectors=True)
   print(pts[0].vector.keys() if isinstance(pts[0].vector, dict) else pts[0].vector)
   "
   ```
   Should print both the unnamed dense vector and the `bm25` sparse vector — non-empty.

4. **Retrieval-only query test** (no `SARVAM_API_KEY` needed — skips generation):
   ```bash
   uv run python -c "
   from pipeline.rag import RAGChain
   chain = RAGChain(lang='hi')
   for d in chain.retrieve_only('what was the manhattan project?'):
       print(d.metadata.get('parent_passage') or d.page_content)
   "
   ```
   This round-trips the query-time `Document(...)`-based dense + sparse Prefetch/RRF fusion — the part that's hardest to reason about without live access.

**Troubleshooting**:
- Auth errors (401/403) → check `QDRANT_API_KEY`.
- "Inference not available"/plan-tier errors → your Qdrant Cloud cluster's plan may not include cloud inference; check the cluster's plan/tier.
- Model-not-found errors → the model identifier must match Qdrant's registry exactly, including casing (`sentence-transformers/all-minilm-l6-v2`, `qdrant/bm25` — both lowercase, per `pipeline/indexer.py::MINILM_INFERENCE_MODEL`/`BM25_INFERENCE_MODEL`).
- Upsert timeouts → embedding now happens synchronously inside the upsert call, so a large `--batch-size` (default 256) may need lowering (e.g. `--batch-size 64`) over a slower/higher-latency connection to the cluster.

### This task's ingestion command

10,000 passages per language, all 14 languages **except Telugu** (`te`'s train parquet isn't cached locally):
```bash
uv run python -m scripts.index --langs as bn gu hi kn ml mr ne or pa sa ta ur --limit 10000 --workers 4
```

---

## Observability

The Gradio admin dashboard (Ingestion/Serving tabs) that used to show this is retired along with the rest of the Python serving stack (see CHANGELOG.md's `frontend_only` entry) — a Next.js equivalent is a planned follow-up, not yet built. In the meantime, check indexing progress directly:
- `GET /api/health` (`frontend/`) — degraded if the live collection doesn't exist/is empty in Qdrant.
- Qdrant's own dashboard/API (`GET /collections/{name}`, `POST /collections/{name}/points/count`) for point counts and per-language breakdowns.
- `.indexer_checkpoints/registry.json` and the per-language checkpoint files, directly.
