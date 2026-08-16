# Bhasha (link-lang)

Multilingual voice RAG system for MSMARCO-XI. Speak a question in any of 14 Indic languages, get a grounded answer in the same language.

```
Voice → Sarvam STT → embed → Qdrant ANN → Sarvam-105B → answer
```

**This is the `frontend_only` branch**: the Python repo is ingestion-only now — it streams the dataset, chunks it, and writes directly into Qdrant Cloud (`scripts/index.py`). The entire serving flow (voice/text query, guardrails, RRF retrieval, generation) has been ported to Next.js and lives in `frontend/` — see `frontend/README.md` and `CHANGELOG.md` for what moved and why. The old FastAPI/Gradio serving stack (`api/`, `ui/`, `pipeline/rag.py`, `pipeline/query_engines.py`, `pipeline/guardrails.py`, `stt.py`, `main.py`, `scripts/benchmark.py`) is retired — each file is a short deprecation stub pointing at its Next.js replacement, kept only until you delete them.

---

## Architecture

### Components

| Component | Technology | Role | Lives in |
|---|---|---|---|
| Ingestion | Python (`scripts/index.py`) | Stream MSMARCO-XI → chunk → upsert into Qdrant Cloud | this repo |
| STT | Sarvam Saaras v3 | Vernacular speech → text | `frontend/lib/server/sarvam.ts` |
| Embedding | `all-MiniLM-L6-v2` (dense) + BM25 (sparse) — both computed server-side by Qdrant Cloud | Text → vectors | Qdrant Cloud (no local/API embedding anywhere) |
| Vector DB | Qdrant Cloud | ANN retrieval + payload filter + RRF fusion | `frontend/lib/server/retrieval.ts` |
| Translation | Sarvam `sarvam-translate:v1` | Query → English (english-pivot retrieval) | `frontend/lib/server/sarvam.ts` |
| Generation | `sarvam-105b` | Grounded answer in target language | `frontend/lib/server/rag.ts` |
| Guardrails | LLM safety check + lexical grounding check | Off-topic/unsafe rejection, hallucination check | `frontend/lib/server/guardrails.ts` |
| Serving API | Next.js Route Handlers | `/api/query`, `/api/voice`, `/api/health` | `frontend/app/api/` |

### Dataset: MSMARCO-XI

Each row contains:
- `query` — original English question (MSMARCO)
- `text` — vernacular passage (translated)
- `query_id`, `passage_id`, `lang`, `is_selected`, `answer`, `query_type`

14 languages: Hindi, Bengali, Tamil, Telugu, Marathi, Kannada, Malayalam, Gujarati, Punjabi, Urdu, Assamese, Oriya, Sanskrit, Sinhala.

---

## Indexing (Python — unchanged by the frontend port)

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
| Retrieval | RRF fusion of the dense + sparse results (`frontend/lib/server/retrieval.ts`) |

See [CHUNKING.md](./CHUNKING.md) for the full rationale, including the other
chunking strategies and embedding backends (vernacular passage/sentence/qa_pair
chunking; local e5; Cohere API) that were prototyped and evaluated before this
single-strategy, Qdrant-Cloud-inference-only architecture was chosen.

### Registry

`.indexer_checkpoints/registry.json` — persists plan metadata for the ingestion CLI's own bookkeeping (which languages are done). The Next.js serving side does **not** read this file — it resolves the live collection directly from Qdrant and caches it in memory per warm instance (`frontend/lib/server/qdrant.ts::getLiveCollection()`), since a serverless deploy's filesystem doesn't survive across invocations.

---

## Running

### Ingestion (this repo)

```bash
uv sync
```

Environment variables (`.env`):
```
QDRANT_CLUSTER_ENDPOINT=...     # Qdrant Cloud cluster URL
QDRANT_API_KEY=...              # required — no local embedding fallback anymore
```

Indexing is CLI-only — see `INDEXING.md` for the full reference (multi-language, parallel workers, resuming).

```bash
uv run python -m scripts.index --langs hi
```

### Serving (frontend/)

```bash
cd frontend
npm install
npm run dev
# http://localhost:3000
```

Environment variables (`frontend/.env.local`):
```
SARVAM_API_KEY=...
QDRANT_CLUSTER_ENDPOINT=...
QDRANT_API_KEY=...
```
(Next.js doesn't read the parent directory's `.env` — these need to be set in `frontend/.env.local` separately from the ingestion side's `.env`.)

---

## API (frontend/app/api/)

```
POST /api/query
  { "query": "...", "lang": "hi" }
  → { "answer": "...", "passages": [...], "latency": {...}, "guardrails": {...} }

POST /api/voice      (multipart/form-data: audio, top_k)
  → { "transcript": "...", "detected_lang": "hi", "answer": "...", "passages": [...] }

GET  /api/health     → { "status": "ok"|"degraded", "qdrant": true|false }
```

---

## Project Structure

```
dataset/           — MSMARCO-XI loading (streaming parquet reader)
pipeline/
  chunking.py       — EnglishQueryChunker (the one supported chunker)
  index_plan.py     — IndexPlan dataclass + registry CRUD + sync_registry_with_qdrant
  indexer.py        — ensure_collection, index_language, run_indexing (import-only, see scripts/index.py)
scripts/
  index.py          — the indexing CLI entrypoint (see INDEXING.md)

frontend/
  app/api/          — query/voice/health Route Handlers
  lib/server/       — qdrant.ts, sarvam.ts, retrieval.ts, guardrails.ts, rag.ts
  lib/api.ts        — client-side fetch wrappers (same-origin /api/*)

.indexer_checkpoints/
  registry.json            — ingestion-side bookkeeping only (not read by the serving side)
  {collection}__{lang}.json — per-language checkpoints (resumable)
```

Retired (deprecation stubs, safe to delete): `api/`, `ui/`, `pipeline/rag.py`, `pipeline/query_engines.py`, `pipeline/guardrails.py`, `stt.py`, `main.py`, `scripts/benchmark.py`.

---

See [INDEXING.md](./INDEXING.md) for the indexing CLI reference and [CHANGELOG.md](./CHANGELOG.md) for full history.
