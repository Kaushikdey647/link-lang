# Changelog

## [Unreleased]

### In Progress
- **RRF hybrid retrieval for the english-pivot index** — `pipeline/query_engines.py`: `BaseQueryEngine` / `VernacularQueryEngine` (plain dense, e5/cohere) / `EnglishPivotQueryEngine` (dense on the English-translated query + BM25 sparse on the original vernacular query, fused server-side via Qdrant's `FusionQuery(fusion=Fusion.RRF)`). `RAGChain` now delegates retrieval to a plan-selected engine instead of an inline branch. New `fastembed` dependency (`Qdrant/bm25` sparse model). Existing english_query collections need a one-time migration (`python -m scripts.migrate_bm25 --collection <name>`) since Qdrant can't add a new named vector to a collection that didn't have one at creation.
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
