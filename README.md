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
| Embedding | `multilingual-e5-small` / `all-MiniLM-L6-v2` / Cohere v3 | Text → vectors |
| Vector DB | Qdrant (local or remote) | ANN retrieval + payload filter |
| Translation | Sarvam `sarvam-translate:v1` | Query → English (english-pivot path only) |
| Generation | `sarvam-m` / `sarvam-105b` | Grounded answer in target language |
| Admin UI | Gradio | Indexing control + pipeline visualization |
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

Every index is described by an `IndexPlan = (backend, chunkers, split)` which determines a deterministic Qdrant collection name:

```
msmarco_xi__{backend}__{sorted_chunkers}__{split}
```

Examples:
- `msmarco_xi__english__english_query__train` — English-pivot, all-MiniLM (recommended)
- `msmarco_xi__e5__passage_sentence_qa_pair__train` — full vernacular, e5
- `msmarco_xi__cohere__passage__train` — passage-only, Cohere

### Chunking Strategies

| Strategy | What gets embedded | Use case |
|---|---|---|
| `passage` | Full vernacular passage | Baseline, good recall |
| `sentence` | Individual sentences | High precision + small-to-big expansion |
| `qa_pair` | English query + vernacular passage | In-distribution query bias |
| `english_query` | English question only | English-pivot; all 14 langs, one model |

See [CHUNKING.md](./CHUNKING.md) for detailed strategy documentation.

### Embedding Backends

| Backend | Model | Dim | Notes |
|---|---|---|---|
| `e5` | `intfloat/multilingual-e5-small` | 384 | Local, all 14 languages, asymmetric prefix |
| `english` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Local, English only, symmetric |
| `cohere` | `Cohere/embed-multilingual-v3.0` | 1024 | API, best recall, rate-limited (2000 inputs/min) |

### Registry

`.indexer_checkpoints/registry.json` — persists plan metadata so the query layer can discover collections and load the correct model.

---

## Running

### Prerequisites

```bash
uv sync
# Qdrant: either local (default, no env vars needed)...
docker run -p 6333:6333 qdrant/qdrant
# ...or a Qdrant Cloud cluster — set QDRANT_CLUSTER_ENDPOINT + QDRANT_API_KEY
# in .env instead; also switches MiniLM/BM25 embedding to server-side
# inference (see INDEXING.md).
```

Environment variables:
```
SARVAM_API_KEY=...
COHERE_API_KEY=...              # optional; enables cohere backend
QDRANT_CLUSTER_ENDPOINT=...     # optional; Qdrant Cloud cluster URL — falls back to http://localhost:6333 if unset
QDRANT_API_KEY=...              # optional; presence also enables Qdrant Cloud server-side inference
                                 # (MiniLM + BM25 embedding computed remotely instead of locally — see INDEXING.md)
```

### Index a language

Indexing is CLI-only — see `INDEXING.md` for the full reference (multi-language, parallel workers, resuming, English-pivot/RRF, migrations).

```bash
uv run python -m scripts.index --langs hi --backend english --chunkers english_query
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
  embedder.py      — three backends + device auto-selection (mps/cuda/cpu), rate limiter for Cohere
  chunking.py      — PassageChunker, SentenceChunker, QAPairChunker, EnglishQueryChunker
  indexer.py       — ensure_collection, index_language, run_indexing, get_vectorstore (import-only, see scripts/index.py)
  rag.py           — RAGChain: guardrails + retrieve (via query_engines) + generate
  query_engines.py — BaseQueryEngine / VernacularQueryEngine / EnglishPivotQueryEngine (RRF hybrid)
  lc_embedder.py   — LangChain Embeddings wrapper for all backends

scripts/
  index.py         — the indexing CLI entrypoint (see INDEXING.md)
  migrate_bm25.py  — one-time sparse-vector migration for pre-RRF english_query collections

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
