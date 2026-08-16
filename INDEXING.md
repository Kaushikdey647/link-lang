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

Prerequisite: Qdrant reachable at `http://localhost:6333` (see `README.md`).

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

## Observability

Everything indexing writes is visible without touching the CLI again:
- **Ingestion tab** (`/admin`) — every collection, alias, point count, on-disk size, vector config, mapped plan, registry status, and on-demand per-language breakdown.
- `GET /plans` — registry contents as JSON.
- `GET /health` — degraded if no registered plan's collection actually exists in Qdrant.
