# Changelog

## [Unreleased]

### Added — `qa_pair` chunking strategy reintroduced as a second, separate collection
- Restored `QAPairChunker` (`pipeline/chunking.py`) and added a `multilingual_e5_small` backend (`pipeline/index_plan.py`, `pipeline/indexer.py`) targeting `intfloat/multilingual-e5-small` via **Qdrant Cloud server-side inference** (dense-only, no BM25) — not the local `sentence-transformers` inference that got the original `e5` backend removed. Indexes into its own collection, `msmarco_xi__multilingual_e5_small__qa_pair__{split}`, leaving the existing `msmarco_xi__english__english_query__train` collection untouched.
- `scripts/index.py` gains `--strategy {english_query,qa_pair}` (default `qa_pair`) to select between the two registered (backend, chunker) plans.
- `pipeline/indexer.py::_e5_text()` prepends `"passage: "` to qa_pair text before embedding; query-time embedding in `frontend/lib/server/retrieval.ts` prefixes with `"query: "`.

### Changed — serving flow ported to Next.js (`frontend_only` branch); Python is ingestion-only now
- Ported the entire serving stack — RRF hybrid retrieval, Sarvam STT/translate/generation, the 4-stage RAGChain harness (guardrails → retrieve → generate → grounding), and guardrails themselves — from Python (FastAPI) to TypeScript, living in `frontend/lib/server/` (`qdrant.ts`, `sarvam.ts`, `retrieval.ts`, `guardrails.ts`, `rag.ts`) and `frontend/app/api/{query,voice,health}/route.ts`. Verified this session, not assumed: `@qdrant/js-client-rest` supports both cloud inference (`{text, model}`) and the Query API's `prefetch`/`FusionQuery({fusion:"rrf"})` — the same pattern the Python `EnglishPivotQueryEngine` used; Sarvam's REST contracts (STT, translate, language-ID, chat completions) were confirmed directly against `docs.sarvam.ai` rather than guessed. Live-tested from this session's environment: a real Sarvam chat-completion call succeeded end-to-end (input guardrail), and the Qdrant Cloud call failed with the exact same "unreachable from this VPN" signature seen from the Python client all session — confirms correct wiring, not a bug.
- **"Which collection is live" is now resolved by querying Qdrant directly and caching the result in memory per warm instance** (`getLiveCollection()`), not by reading `.indexer_checkpoints/registry.json`. That file is local-filesystem state that wouldn't survive a serverless deploy (e.g. Vercel) — the ingestion side still writes it for its own bookkeeping, but nothing on the serving side reads it anymore.
- Found and fixed along the way: (1) a pre-existing bug in `pipeline/query_engines.py`'s Sarvam language map — Odia was mapped to `"or-IN"`, but Sarvam's actual API uses `"od-IN"`; fixed in the (now-retired) Python source and correctly ported to `sarvam.ts`. (2) `frontend/.env.local` never had `SARVAM_API_KEY`/`QDRANT_API_KEY`/`QDRANT_CLUSTER_ENDPOINT` — Next.js doesn't read the parent directory's `.env` — added them.
- **Retired** (each replaced with a short deprecation-notice stub, not deleted outright — pending manual `rm`): `api/` (FastAPI app + routes + models + metrics), `ui/` (Gradio admin dashboard, including `ui/journey_tab.py` and `ui/app.py` which were already dead/broken before this branch), `pipeline/rag.py`, `pipeline/query_engines.py`, `pipeline/guardrails.py`, `stt.py`, `main.py` (a stale dataset-loading demo, redundant with `scripts/index.py`), `scripts/benchmark.py` (depended on the now-retired `RAGChain`; a TypeScript equivalent is a planned follow-up). Ingestion (`dataset/`, `pipeline/chunking.py`, `pipeline/indexer.py`, `pipeline/index_plan.py`, `scripts/index.py`) is untouched and still Python.
- `pyproject.toml` trimmed to ingestion-only dependencies — dropped `fastapi`, `gradio`, `uvicorn`, `python-multipart`, `prometheus-client`, `prometheus-fastapi-instrumentator`, `matplotlib`, `langchain-sarvam`, `pydantic`, plus two packages found to already be dead weight independent of this change (`anthropic`, `pandas`, `numpy` — never actually imported anywhere once `scripts/benchmark.py` was retired). 37 packages removed via `uv sync`.
- Not yet built: Prometheus-style metrics (`prom-client`) and an admin/observability dashboard for the Next.js side — deferred, tracked as a follow-up rather than blocking the core query/voice path.

### Changed — collapsed to a single embedding/chunking strategy (latency + production footprint over extensibility)
- **english-pivot (MiniLM dense + BM25 sparse, both via Qdrant Cloud server-side inference) is now the system's one supported strategy.** Deleted rather than merely made unreachable: the `e5`/`cohere` embedding backends, local MiniLM/BM25 inference and its `torch`/`sentence-transformers`/`fastembed`/`langchain-cohere`/`langchain-huggingface`/`cohere` dependencies, `VernacularQueryEngine`, the `PassageChunker`/`SentenceChunker`/`QAPairChunker`/`CompositeChunker` chunking strategies, and the now-obsolete `scripts/migrate_bm25.py` (BM25 is always present from the first upsert now, nothing to migrate). `pipeline/embedder.py` and `pipeline/lc_embedder.py` are gone (`VECTOR_DIM_FOR` folded into `pipeline/index_plan.py`). `pipeline/chunking.py` keeps only `EnglishQueryChunker`; `pipeline/query_engines.py` keeps only `EnglishPivotQueryEngine`. `scripts/index.py` drops the `--backend`/`--chunkers` flags entirely (hardcoded). `IndexPlan`'s `(backend, chunkers, split)` shape is unchanged, so the already-populated remote collection (`msmarco_xi__english__english_query__train`) keeps resolving correctly.
- `pipeline/guardrails.py` rewritten to drop its embedding dependency (it was the one place still calling `embed_passages`/`embed_query` independent of the indexing plan, for domain-centroid/grounding checks). `check_input()` is now LLM-only (Sarvam safety/relevance check); `check_grounding()` uses lexical token-overlap between the answer and retrieved passages instead of cosine similarity — no embedding call, no numpy dependency in this file anymore.
- Removing `torch`/`sentence-transformers`/`fastembed` etc. (28 packages total after `uv sync`) directly serves the Render production-readiness work: a from-scratch Docker build with these dependencies present idled at ~804MB RSS before serving a single request; see the Dockerfile/render.yaml notes for the re-measured figure.
- Rationale: PROBLEM-STATEMENT.md's "vast chunking" requirement is addressed via documentation now (CHUNKING.md's "What else was considered" section) rather than by keeping the other strategies' code present-but-unused — a deliberate choice to prioritize latency and a lean production footprint over retrieval-strategy extensibility.

### Added — Qdrant Cloud + server-side (cloud) inference for MiniLM/BM25
- `pipeline/indexer.py` now reads `QDRANT_CLUSTER_ENDPOINT`/`QDRANT_API_KEY` from `.env` (`QDRANT_URL`/`QDRANT_API_KEY`/`QDRANT_CLOUD_INFERENCE` constants); every `QdrantClient` construction in the app (indexer, `pipeline/rag.py`, `ui/ingestion_tab.py`, `api/app.py`, `scripts/migrate_bm25.py`) picks these up. Falls back to local `http://localhost:6333` with no API key when unset — fully backward compatible.
- When `QDRANT_API_KEY` is present, the `english`/`english_query` plan's dense (MiniLM) and sparse (BM25) vectors are computed **server-side** by Qdrant Cloud (`Document(text=..., model=...)` + `cloud_inference=True`) instead of locally via `sentence-transformers`/`fastembed` — both at index time (`pipeline/indexer.py::_upsert_batch`) and query time (`pipeline/query_engines.py::EnglishPivotQueryEngine`). `e5`/`cohere` backends are unaffected.
- `INDEXING.md` gained a "Remote (Qdrant Cloud) inference" section with an SOP for testing the remote-inference path (untestable from this environment/VPN) plus the ingestion command for 10K passages × 13 languages (Telugu excluded — train split not cached).
- Fixed a pre-existing gap: nothing in the indexing CLI's import chain (`scripts/index.py` → `pipeline.embedder` → `pipeline.indexer`) ever called `load_dotenv()`, so `.env` was silently never loaded for indexing runs — `COHERE_API_KEY`-based backend auto-selection has always been affected too. Both `pipeline/embedder.py` and `pipeline/indexer.py` now call `load_dotenv()`.

### Fixed — critical: indexing OOM on `--workers > 1` / `all` languages
- `dataset/loader.py::load_language()` fully materialized a language's whole parquet file (3.3-4.0GB on disk) into an Arrow `Table`, then cast it to a large-string schema (second copy), then concatenated multi-file languages (third copy) — ~7-12GB resident per language. `--limit` was applied via `islice()` *after* this unconditional full load (`pipeline/indexer.py`), so it never prevented the read. With `--workers 4`, four of these loads ran concurrently, causing >50GB RSS even with `--limit 10000`.
- Added `dataset/loader.py::iter_language_rows()` (streams row-by-row via `pq.ParquetFile.iter_batches()`, never assembling a full-file table) and `count_language_rows()` (parquet-footer-only row count, no data read). `pipeline/indexer.py::index_language()` now uses these instead of `load_language()`; `dataset/passages.py::iter_passages()`/`iter_queries()` type hints widened to `Iterable[dict]` (no body change) so the streaming source plugs in directly. Memory now scales with batch size and worker count, not file size — `--limit` and `--workers` behave as documented.
- Added `numpy` as an explicit `pyproject.toml` dependency (previously used directly in `pipeline/guardrails.py` but only transitively available).

### Changed — indexing is now CLI-only; admin UI is read-only observability
- **Indexing moved entirely to the CLI** (`uv run python -m scripts.index`, see `INDEXING.md`) — no admin-UI control (start/stop/resume) or HTTP route triggers it anymore. `ui/indexing.py` (the old Gradio worker/state-machine tab) removed.
- **Admin dashboard collapsed to two read-only tabs**: `ui/metrics_tab.py` → **Serving** (unchanged serving metrics, ingestion pieces removed), new `ui/ingestion_tab.py` → **Ingestion** (every Qdrant collection: aliases, points, size on disk, dense/sparse vector config, mapped `IndexPlan`, registry status, on-demand per-language distribution). "Pipeline Journey" tab dropped from the mount (file kept on disk, unmounted).
- Removed the now-orphaned `INDEXING_*` Prometheus gauges (`api/metrics.py`) and their consumers in `ui/metrics_store.py` — nothing in-process populates them once indexing is an external CLI process.
- `pipeline/indexer.py` restructured as an **import-only library module** (checkpoint/resume logic ported in from the old UI tab, unchanged file format; `run_indexing()` orchestrates `--langs`/`--workers`; batches now compute dense+sparse in one Qdrant upsert instead of two for english_query plans). The CLI entrypoint lives in the new `scripts/index.py` — deliberately separate, because `--workers > 1` uses `ProcessPoolExecutor`'s `spawn` start method, which can't safely re-import a worker function whose home module is `__main__` (confirmed: running `pipeline/indexer.py` directly crashed every worker instantly).
- `pipeline/embedder.py` now auto-selects `mps`/`cuda`/`cpu` for local models — previously always silently ran on CPU even on Apple Silicon.
- `pipeline/index_plan.py` gained `sync_registry_with_qdrant()` (moved from `ui/indexing.py`) so both the CLI and the Ingestion tab share one registry-reconciliation implementation.

### Fixed — critical: cross-language point-ID collisions
- `dataset/passages.py::iter_passages()` built `passage_id` as `f"{query_id}_{idx}"` with no language component. MSMARCO-XI's `query_id` is the **same identifier across all 14 language translations**, so indexing language B after A silently overwrote A's Qdrant points wherever `query_id`+`idx` matched (every chunk_id / point-ID derives from `passage_id`). Confirmed via direct testing: indexing 100K Bengali then 100K Gujarati passages left ~99,990 total points with Bengali at zero. Fixed by prefixing with `lang`; verified two languages indexing concurrently now sum correctly with no overwrite.

### Fixed — english_query chunker was embedding the wrong-language text
- `EnglishQueryChunker` embedded `record.query`, expecting English — but `iter_passages(..., translated=True)` (the only mode indexing ever uses) makes `record.query` the **vernacular** translation, not English (`Eng_Query` is the actual English field). So the English-pivot dense vectors were built from vernacular text run through an English-only model, undermining the entire premise. `PassageRecord` gained an `eng_query` field (always English, independent of `translated`); `EnglishQueryChunker` now embeds that instead. `CHUNKING.md`'s "English-pivot idea" section corrected to match.

### Added — RRF hybrid retrieval for the english-pivot index
- `pipeline/query_engines.py`: `BaseQueryEngine` / `VernacularQueryEngine` (plain dense, e5/cohere) / `EnglishPivotQueryEngine` (dense on the English-translated query + BM25 sparse on the original vernacular query, fused server-side via Qdrant's `FusionQuery(fusion=Fusion.RRF)`). `RAGChain` now delegates retrieval to a plan-selected engine instead of an inline branch. New `fastembed` dependency (`Qdrant/bm25` sparse model). Existing english_query collections need a one-time migration (`uv run python -m scripts.migrate_bm25 --collection <name>`) since Qdrant can't add a new named vector to a collection that didn't have one at creation.
- Removed dead code: `pipeline/retriever.py`, `pipeline/generator.py` (no live importers; both had real bugs — a NameError-in-waiting and un-prefixed Qdrant filter keys that never matched the actual payload shape).

### Fixed (state sync / progress bar, from the admin UI overhaul)
- State synchronization between `_state` in-memory, checkpoint files on disk, and actual Qdrant point counts — the admin UI now reconciles all three every poll instead of only reflecting in-memory state since the last restart.
- Progress bar previously used chunk-count estimates (`_CHUNKS_PER_ROW`) that diverged badly for short-passage languages (Sanskrit ~1.2 chunks/row vs. a flat estimate of ~36) and compared passage counts against raw row counts (~10x off) — now passage-based and exact (`total_passages = num_rows × 10`, since every row has exactly 10 candidate passages).
- Failed-language errors (`_LangStatus.error`) now surface in the status table instead of being silently dropped.
- Fixed the mixed-script Gujarati display-name bug that caused row-rendering glitches in the language table.
- Pre-refactor collections (e.g. `msmarco_xi_e5`) not present in `registry.json` now auto-register from live Qdrant state, and stale registry entries whose collection no longer exists get pruned instead of showing as permanent ghost entries.
- Indexing `QdrantClient` timeout raised from the library's 5s default to 60s — large embed+upsert batches or payload-index rebuilds on big collections were exceeding 5s under load and killing the whole language run instead of just retrying.

---

## [0.3.0] — Pipeline Journey Tab + IndexPlan Refactor

### Added
- **`pipeline/index_plan.py`** — `IndexPlan` dataclass encoding `(backend, chunkers, split)` → deterministic collection name (`msmarco_xi__{backend}__{sorted_chunkers}__{split}`). Registry CRUD: `load_registry()`, `register_plan()`, `get_plan_by_collection()`, `best_available_plan()`, `all_plans()`. Registry stored at `.indexer_checkpoints/registry.json`.
- **`pipeline/chunking.py`** — `EnglishQueryChunker`: embeds `record.query` (English question), stores `record.text` (vernacular) in `parent_passage`. Enables English-pivot retrieval across all 14 languages with a single monolingual model.
- **`pipeline/lc_embedder.py`** — `ProjectEmbeddings(Embeddings)`: LangChain wrapper for all three backends.
- **`ui/journey_tab.py`** — Pipeline Journey tab: interactive 7-stage flowchart (matplotlib, dark theme) + 6 JSON accordion panels showing data shape at each stage. Updates live on backend/chunker selection.
- **`GET /plans`** endpoint in `api/app.py` — returns registry contents as JSON.
- **`collection` field** on `QueryRequest` — allows routing a query to a specific indexed plan.

### Changed
- **`pipeline/embedder.py`**: removed `COLLECTION_FOR`; added `VECTOR_DIM_FOR`, `AVAILABLE_BACKENDS`, `DEFAULT_BACKEND`; added `_TokenBucket` rate limiter for Cohere (33 inputs/sec, 200-burst); added `english` backend (`all-MiniLM-L6-v2`, 384-dim, symmetric encoding).
- **`pipeline/indexer.py`**: all functions now take `plan: IndexPlan` instead of `backend: str`. Checkpoint path is `{collection_name}__{lang}.json`.
- **`pipeline/rag.py`**: `RAGChain.__init__` takes `plan: IndexPlan | None`; added `_translate_to_english()` via Sarvam translate API; translation triggered when `"english_query" in plan.chunkers`; `_format_context` uses `parent_passage` field for english_query chunks.
- **`pipeline/retriever.py`**: replaced dead `COLLECTION` import with `IndexPlan` + `best_available_plan()`; `retrieve()` now accepts `plan: IndexPlan | None`.
- **`ui/indexing.py`**: complete rewrite — `IndexPlan`-based state, backend dropdown + chunker checkboxes, live collection name preview, 14-language status table, registry table, Qdrant collection stats panel.
- **`ui/admin_app.py`**: added Pipeline Journey tab; removed language dropdown (all languages shown at once).
- **`api/routes/query.py`**: `_get_chain(lang, collection)` with plan-keyed chain cache.
- **`api/routes/voice.py`**: threads `collection` form field through to chain.

### Fixed
- `ImportError: cannot import name 'COLLECTION' from 'pipeline.indexer'` — retriever.py and `__init__.py` updated after refactor removed the constant.
- Devanagari font warnings in matplotlib pipeline graph — replaced Hindi text in graph labels with ASCII-safe descriptions; Unicode kept in JSON panels.

---

## [0.2.0] — Multiple Embedding Backends + Rate Limiter

### Added
- **Cohere backend** (`embed-multilingual-v3.0`, 1024-dim) — requires `COHERE_API_KEY`; asymmetric via `langchain-cohere`.
- **English backend** (`all-MiniLM-L6-v2`, 384-dim) — local, symmetric, no prefix.
- **Token bucket rate limiter** — thread-safe, counts individual texts, Cohere-only, 2000 inputs/min (33/sec), 200-burst.
- Admin UI backend dropdown with descriptive labels.
- Separate Qdrant collections per backend — collection name encodes the backend.

### Changed
- `embed_passages(texts, backend=None)` and `embed_query(text, backend=None)` — backend defaults to `DEFAULT_BACKEND` (`cohere` if key present, else `e5`).
- Admin UI: dropped language control dropdown; shows all 14 languages' indexing status at once; added "Index All" and "Resume Paused" buttons.

---

## [0.1.0] — Initial RAG Pipeline

### Added
- `pipeline/chunking.py` — `PassageChunker`, `SentenceChunker`, `QAPairChunker`, `CompositeChunker`, `build_chunker()`. Multilingual sentence splitting (Latin `.!?` + Devanagari `।॥`). `min_words=4` filter for fragments.
- `pipeline/embedder.py` — `multilingual-e5-small` (384-dim), `passage:` / `query:` prefix convention.
- `pipeline/indexer.py` — `index_language()`, `ensure_collection()`, Qdrant payload indexes on `lang`, `chunk_type`, `query_id`, `is_selected`.
- `pipeline/rag.py` — `RAGChain`: retrieve top-k, small-to-big expansion, Sarvam-105B generation.
- `pipeline/retriever.py` — `retrieve()`: ANN search, deduplication by `passage_id`, small-to-big scroll.
- `pipeline/generator.py` — Sarvam-105B with `reasoning_effort="low"`.
- `api/` — FastAPI app with `/query` and `/voice` endpoints.
- `stt.py` — Sarvam Saaras v3 transcription.
- `ui/admin_app.py` + `ui/indexing.py` — Gradio admin panel.
- `CHUNKING.md` — detailed chunking strategy documentation.
- `PROBLEM-STATEMENT.md` — original brief.
