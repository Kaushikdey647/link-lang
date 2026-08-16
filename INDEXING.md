# Indexing

Indexing is **CLI-only**. There is no button, endpoint, or admin control that
starts, stops, or resumes an indexing run — the only way to index is:

```bash
uv run python -m scripts.index --langs hi bn --backend english --chunkers english_query
```

This is deliberate: indexing is a heavy, occasional, operator-driven task
(minutes to hours per language), and keeping it out-of-process means the
running API/admin server has zero control surface for it. The admin UI's
**Ingestion** tab (`/admin`) is read-only — it shows you what's actually in
Qdrant (collections, sizes, per-language counts, mapped plan), never a way to
trigger anything.

---

## Quick start

```bash
# Index Hindi into the default (e5, vernacular) plan
uv run python -m scripts.index --langs hi

# Index two languages with the English-pivot strategy
uv run python -m scripts.index --langs hi bn --backend english --chunkers english_query

# Index everything, four languages at a time
uv run python -m scripts.index --langs all --workers 4 --backend e5 --chunkers passage sentence qa_pair

# Quick smoke test — 5,000 passages per language, no commitment
uv run python -m scripts.index --langs hi --limit 5000
```

Prerequisite: Qdrant reachable — either local at `http://localhost:6333` (default, see `README.md`) or a Qdrant Cloud cluster via `QDRANT_CLUSTER_ENDPOINT`/`QDRANT_API_KEY` in `.env` (see "Remote (Qdrant Cloud) inference" below).

---

## CLI reference (`scripts/index.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--langs` | `hi` | One or more 2-letter codes, or `all` for all 14 (`as bn gu hi kn ml mr ne or pa sa ta te ur`). |
| `--backend` | `cohere` if `COHERE_API_KEY` set, else `e5` | Embedding model: `e5` (multilingual-e5-small, local, 384-dim), `english` (all-MiniLM-L6-v2, local, 384-dim, English-only), `cohere` (embed-multilingual-v3.0, API, 1024-dim). |
| `--chunkers` | `passage sentence qa_pair` | One or more of: `passage`, `sentence`, `qa_pair`, `english_query`. See `CHUNKING.md` for what each embeds. |
| `--split` | `train` | `train` or `validation`. |
| `--batch-size` | `256` | Passages embedded + upserted per Qdrant round-trip. |
| `--workers` | `1` | Parallel language processes. `1` = sequential (today's default). `>1` runs that many languages concurrently, each in its own OS process. |
| `--limit` | none | Stop after N passages *per language* — for quick test runs. The checkpoint still records exactly where it stopped, so a later un-limited run continues from there. |

`--backend` and `--chunkers` are independent axes — see "Collections & IndexPlan" below for how they combine into a collection name.

---

## Collections & IndexPlan

Every run is described by an `IndexPlan` (`pipeline/index_plan.py`): `(backend, chunkers, split)`. The collection name is derived deterministically, never chosen manually:

```
msmarco_xi__{backend}__{sorted chunkers}__{split}
```

Examples:
- `msmarco_xi__e5__passage_qa_pair_sentence__train` — vernacular, full recall
- `msmarco_xi__english__english_query__train` — English-pivot (RRF hybrid)
- `msmarco_xi__cohere__passage__validation` — passage-only, Cohere

Running the same `--backend`/`--chunkers`/`--split` combination always targets the same collection — that's how `--langs hi` today and `--langs bn` tomorrow end up in one multi-language collection, and how re-running an interrupted language resumes into the same place instead of creating a duplicate.

---

## Use cases

### 1. Index one language, one strategy
```bash
uv run python -m scripts.index --langs hi --backend e5 --chunkers passage sentence qa_pair
```
Sequential by default (`--workers 1`). Safe to `Ctrl+C` at any point — see Checkpoints below.

### 2. Index several languages faster (parallel)
```bash
uv run python -m scripts.index --langs hi bn gu ta --workers 4 --backend english --chunkers english_query
```
Each language runs in its own process (`concurrent.futures.ProcessPoolExecutor`), fully independent — own dataset slice, own checkpoint file, own point IDs. Pick `--workers` based on available CPU cores; there's no cross-language shared state to contend over.

### 3. Index everything
```bash
uv run python -m scripts.index --langs all --workers 4 --backend e5 --chunkers passage sentence qa_pair
```
`all` expands to all 14 language codes (`constants.LANG_CODE_MAP`).

### 4. Resume an interrupted run
Just re-run the exact same command. Each language reads its checkpoint (`.indexer_checkpoints/{collection_name}__{lang}.json`) and picks up from the last completed batch — no flag needed, this is always-on behavior, not opt-in.

```bash
# First attempt gets killed partway through
uv run python -m scripts.index --langs hi --backend e5 --chunkers passage
# ^C

# Re-run the identical command — resumes from the checkpoint, doesn't restart
uv run python -m scripts.index --langs hi --backend e5 --chunkers passage
```

### 5. Quick test runs without committing to a full language
```bash
uv run python -m scripts.index --langs hi bn --limit 5000
```
Stops after 5,000 passages per language. The checkpoint is left in place (not cleared, not registered as done), so:
- A later un-limited run continues from passage 5,000 rather than starting over.
- The Ingestion tab / `registry.json` correctly show it as "not registered" rather than falsely complete.

### 6. English-pivot / RRF hybrid retrieval
```bash
uv run python -m scripts.index --langs hi --backend english --chunkers english_query
```
This is the only chunker that also builds a BM25 sparse vector (`pipeline/query_engines.py::EnglishPivotQueryEngine` fuses it with the dense vector via Qdrant's server-side RRF at query time). Indexing computes both vectors in a single upsert per batch — no separate pass needed.

If you have an **existing** `english_query` collection that predates this (dense-only, no sparse vector space), `ensure_collection()` will refuse to index into it with a clear error instead of silently degrading:
```
RuntimeError: Collection '...' predates the RRF hybrid strategy and is missing
the 'bm25' sparse vector space. Run `uv run python -m scripts.migrate_bm25 --collection ...` first.
```
Run the suggested migration once (reuses existing dense vectors, no re-embedding, no downtime — see `scripts/migrate_bm25.py`'s docstring for exactly what it does), then index normally.

### 7. Choosing a backend/chunker combination
See `CHUNKING.md` for the full tradeoffs. Quick guide:

| Goal | Backend | Chunkers |
|---|---|---|
| Best all-around recall, any of 14 languages | `e5` | `passage sentence qa_pair` |
| Fastest, cheapest, cross-lingual via English pivot + BM25 | `english` | `english_query` |
| Highest recall, willing to pay per-token | `cohere` | `passage sentence qa_pair` |
| Fast baseline only | any | `passage` |

---

## Checkpoints

`.indexer_checkpoints/{collection_name}__{lang}.json`:
```json
{"passages_done": 338944, "chunks_done": 338944}
```
Written after every batch. Deleted only when a language finishes cleanly (not stopped early, no `--limit` cutoff) — at that point it also registers into `.indexer_checkpoints/registry.json` (see below). A file existing means that language is genuinely incomplete for that exact collection; if it's missing, the language is either untouched or fully done.

---

## Registry (`registry.json`)

`.indexer_checkpoints/registry.json` tracks, per collection: backend, model, chunkers, split, and `lang_counts` (chunks indexed per language that's actually finished). This is what backs:
- `GET /plans` and the "Registered" column in the Ingestion tab.
- `best_available_plan()` / `get_plan_by_collection()` (`pipeline/index_plan.py`), used by `RAGChain` to pick a collection at query time when none is specified.

It stays in sync automatically — `run_indexing()` calls `sync_registry_with_qdrant()` after every run, which also prunes entries for collections that no longer exist (e.g. deleted/recreated outside the app) and auto-registers known pre-refactor collections. You never edit this file by hand.

---

## Performance notes

- **Streaming dataset load**: `dataset/loader.py::iter_language_rows()` reads each language's parquet file batch-by-batch (`pq.ParquetFile.iter_batches()`), never materializing a full-file Arrow Table — memory stays bounded by batch size, not file size (each language's train parquet is 3.3-4.0GB on disk). This is what `pipeline/indexer.py::index_language()` uses; `--limit` now genuinely bounds how much gets read from disk (previously it only limited iteration *after* the whole file had already been loaded), and `--workers > 1` no longer multiplies full-file loads across processes.
- **Device selection**: `pipeline/embedder.py` auto-picks the best available accelerator (`mps` → `cuda` → `cpu`) for the local models (`e5`, `english`). Nothing to configure.
- **Merged upsert**: `english_query` batches compute dense + sparse vectors and upsert both in one Qdrant call, not two.
- **Parallelism caveat**: always invoke via `uv run python -m scripts.index`, not `python -m pipeline.indexer` directly. `--workers > 1` uses `ProcessPoolExecutor`'s `spawn` start method, which needs to re-import the worker function from a real module path in each child process — running `pipeline/indexer.py` itself as `__main__` breaks that (confirmed: every worker crashes instantly). `scripts/index.py` exists specifically so `pipeline.indexer` is always imported normally. (`fork` was tried as an alternative and rejected: it crashes once MPS/Metal has been touched — an Apple/Objective-C runtime limitation, not fixable from Python.)

---

## Remote (Qdrant Cloud) inference

Setting `QDRANT_CLUSTER_ENDPOINT` + `QDRANT_API_KEY` in `.env` switches two things at once, both gated by the same `QDRANT_API_KEY` presence check (`pipeline/indexer.py::QDRANT_CLOUD_INFERENCE`):

1. **Where Qdrant lives** — every `QdrantClient` in the app (`pipeline/indexer.py`, `pipeline/rag.py`, `ui/ingestion_tab.py`, `api/app.py`, `scripts/migrate_bm25.py`) connects to the cloud cluster instead of `localhost:6333`.
2. **Who computes the embedding, for the `english`/`english_query` plan only** — instead of running `sentence-transformers/all-MiniLM-L6-v2` and `fastembed`'s BM25 model locally, the client sends raw text as a Qdrant `Document(text=..., model=...)` at both index time (`pipeline/indexer.py::_upsert_batch`) and query time (`pipeline/query_engines.py::EnglishPivotQueryEngine`), and Qdrant Cloud computes the vector server-side (`cloud_inference=True` on the client). The `e5` and `cohere` backends are untouched — they're not part of this switch.

Nothing else about indexing changes: the dataset still streams from the local parquet cache (`dataset/loader.py::iter_language_rows`), chunking still happens locally (`pipeline/chunking.py`) — only the embedding *computation* moves server-side. If `QDRANT_API_KEY` is unset, everything behaves exactly as before (local Qdrant, local embedding) — this is an additive, env-gated switch, not a hard requirement.

### SOP: testing remote embeddings

This can't be verified from every network (the cluster may be unreachable from some VPNs/firewalls) — run these from a network that can actually reach your `QDRANT_CLUSTER_ENDPOINT`, after `.env` has both vars set.

1. **Connectivity + auth check** (no embedding involved yet):
   ```bash
   uv run python -c "from pipeline.indexer import _get_qdrant_client; print(_get_qdrant_client().get_collections())"
   ```
   A `401`/`403` here means `QDRANT_API_KEY` is wrong; a connection error means `QDRANT_CLUSTER_ENDPOINT` is wrong or the network can't reach it.

2. **Tiny smoke-test ingest** — exercises the cloud-inference upsert path end to end:
   ```bash
   uv run python -m scripts.index --langs hi --backend english --chunkers english_query --limit 50
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

10,000 passages per language, all 14 languages **except Telugu** (`te`'s train parquet isn't cached locally — confirmed separately):
```bash
uv run python -m scripts.index --langs as bn gu hi kn ml mr ne or pa sa ta ur \
  --backend english --chunkers english_query --limit 10000 --workers 4
```

---

## Observability

Everything indexing writes is visible without touching the CLI again:
- **Ingestion tab** (`/admin`) — every collection, alias, point count, on-disk size, vector config, mapped plan, registry status, and on-demand per-language breakdown.
- `GET /plans` — registry contents as JSON.
- `GET /health` — degraded if no registered plan's collection actually exists in Qdrant.
