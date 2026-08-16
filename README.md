# Link-Lang

Multilingual voice RAG system for MSMARCO-XI. Speak a question in any of 14 Indic languages, get a grounded answer in the same language.

```
Voice → Sarvam STT → embed → Qdrant ANN → Sarvam-105B → answer
```

---

## Architecture

### Components

| Component | Technology | Role |
|---|---|---|
| STT | Sarvam Saaras v3 | Vernacular speech → text |
| Embedding | `all-MiniLM-L6-v2` (dense) + BM25 (sparse) — both computed server-side by Qdrant Cloud | Text → vectors |
| Vector DB | Qdrant Cloud | ANN retrieval + payload filter + RRF fusion |
| Translation | Sarvam `sarvam-translate:v1` | Query → English (english-pivot retrieval) |
| Generation | `sarvam-m` / `sarvam-105b` | Grounded answer in target language |
| Admin UI | Gradio | Read-only observability (Serving + Ingestion tabs) |
| API | FastAPI | `/query`, `/voice`, `/plans`, `/health` |

### Dataset: MSMARCO-XI

Each row contains:
- `query` — original English question (MSMARCO)
- `text` — vernacular passage (translated)
- `query_id`, `passage_id`, `lang`, `is_selected`, `answer`, `query_type`

14 languages: Hindi, Bengali, Tamil, Telugu, Marathi, Kannada, Malayalam, Gujarati, Punjabi, Urdu, Assamese, Oriya, Sanskrit, Sinhala.

---

## Indexing

### IndexPlan

english-pivot is the system's one supported strategy (see CHUNKING.md for what
else was prototyped and why this was chosen) — `backend="english"`,
`chunkers=["english_query"]` is the only valid `IndexPlan`, giving one
deterministic collection name:

```
msmarco_xi__english__english_query__{split}
```

### Chunking + embedding strategy

| | |
|---|---|
| Chunker | `english_query` — embeds the English question (`Eng_Query`), returns the vernacular passage as context |
| Dense embedding | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim — computed server-side by Qdrant Cloud |
| Sparse embedding | BM25/IDF — also computed server-side by Qdrant Cloud |
| Retrieval | RRF fusion of the dense + sparse results (`pipeline/query_engines.py::EnglishPivotQueryEngine`) |

See [CHUNKING.md](./CHUNKING.md) for the full rationale, including the other
chunking strategies and embedding backends (vernacular passage/sentence/qa_pair
chunking; local e5; Cohere API) that were prototyped and evaluated before this
single-strategy, Qdrant-Cloud-inference-only architecture was chosen.

### Registry

`.indexer_checkpoints/registry.json` — persists plan metadata so the query layer can discover collections and load the correct model.

---

## Running

### Prerequisites

```bash
uv sync
# Qdrant: either local (fallback, no env vars needed, local-only dev)...
docker run -p 6333:6333 qdrant/qdrant
# ...or a Qdrant Cloud cluster — set QDRANT_CLUSTER_ENDPOINT + QDRANT_API_KEY
# in .env instead. Required for the embedding to actually work: Qdrant Cloud's
# server-side inference is the only embedding mechanism now (see INDEXING.md).
```

Environment variables:
```
SARVAM_API_KEY=...
QDRANT_CLUSTER_ENDPOINT=...     # Qdrant Cloud cluster URL — falls back to http://localhost:6333 if unset
QDRANT_API_KEY=...              # required for embeddings to work at all (server-side MiniLM+BM25 inference)
```

### Index a language

Indexing is CLI-only — see `INDEXING.md` for the full reference (multi-language, parallel workers, resuming).

```bash
uv run python -m scripts.index --langs hi
```

### Start the API + Admin UI

```bash
uv run python main.py
# API:      http://localhost:8000
# Admin UI: http://localhost:8000/admin
```

### Admin UI tabs

Read-only observability — no controls that start, stop, or resume anything (indexing is CLI-only, see `INDEXING.md`):

- **Serving** — live query latency and retrieval quality metrics
- **Ingestion** — every Qdrant collection (aliases, points, size on disk, vector config, mapped plan, registry status) plus on-demand per-language document counts

---

## API

```
POST /query
  { "query": "...", "lang": "hi", "collection": "msmarco_xi__english__english_query__train" }
  → { "answer": "...", "passages": [...] }

POST /voice          (multipart/form-data: audio + lang + collection)
  → { "transcript": "...", "answer": "..." }

GET  /plans          → list of indexed plans with model metadata
GET  /health         → { "status": "ok"|"degraded", "plans": [...] }
```

---

## Project Structure

```
pipeline/
  index_plan.py    — IndexPlan dataclass + registry CRUD + sync_registry_with_qdrant
  chunking.py      — EnglishQueryChunker (the one supported chunker)
  indexer.py       — ensure_collection, index_language, run_indexing (import-only, see scripts/index.py)
  rag.py           — RAGChain: guardrails + retrieve (via query_engines) + generate
  query_engines.py — EnglishPivotQueryEngine (RRF hybrid, Qdrant Cloud server-side inference)
  guardrails.py    — LLM-based input check + lexical-overlap grounding check (no embedding calls)

scripts/
  index.py         — the indexing CLI entrypoint (see INDEXING.md)

api/
  app.py           — FastAPI app, /plans, /health
  routes/query.py  — /query endpoint, plan-keyed chain cache
  routes/voice.py  — /voice endpoint, STT + chain

ui/
  admin_app.py     — Gradio Blocks root (Serving + Ingestion tabs)
  metrics_tab.py   — Serving tab (Prometheus-backed)
  ingestion_tab.py — Ingestion tab (read-only Qdrant/registry observability)

.indexer_checkpoints/
  registry.json           — registered IndexPlans
  {collection}__{lang}.json — per-language checkpoints (resumable)
```

---

See [INDEXING.md](./INDEXING.md) for the indexing CLI reference and [CHANGELOG.md](./CHANGELOG.md) for full history.
