# Bhasha (link-lang)

Multilingual voice RAG system for MSMARCO-XI. Speak a question in any of 14 Indic languages, get a grounded answer in the same language.

```
Voice → Sarvam STT → embed → Qdrant ANN → Sarvam-105B → answer
```

**This is the `frontend_only` branch**: the Python repo is ingestion-only now — it streams the dataset, chunks it, and writes directly into Qdrant Cloud (`scripts/index.py`). The entire serving flow (voice/text query, guardrails, dense e5 retrieval, generation) has been ported to Next.js and lives in `frontend/` — see `frontend/README.md` and `CHANGELOG.md` for what moved and why. The old FastAPI/Gradio serving stack (`api/`, `ui/`, `pipeline/rag.py`, `pipeline/query_engines.py`, `pipeline/guardrails.py`, `stt.py`, `main.py`, `scripts/benchmark.py`) is retired — each file is a short deprecation stub pointing at its Next.js replacement, kept only until you delete them.

---

## Architecture

### Components

| Component | Technology | Role | Lives in |
|---|---|---|---|
| Ingestion | Python (`scripts/index.py`) | Stream MSMARCO-XI → chunk → upsert into Qdrant Cloud | this repo |
| STT | Sarvam Saaras v3 | Vernacular speech → text | `frontend/lib/server/sarvam.ts` |
| Embedding | `intfloat/multilingual-e5-small` (dense, Qdrant Cloud inference) | Vernacular query/passage → vectors | Qdrant Cloud |
| Vector DB | Qdrant Cloud | ANN retrieval + payload filter | `frontend/lib/server/retrieval.ts` |
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

Serving + default indexing use `qa_pair` + `multilingual_e5_small`. The older
english-pivot collection is still indexable via `--strategy english_query`.

```
msmarco_xi__multilingual_e5_small__qa_pair__{split}   # default / serving
msmarco_xi__english__english_query__{split}           # optional
```

### Chunking + embedding strategy

| | |
|---|---|
| Chunker | `qa_pair` — embeds vernacular `query + passage`, returns the passage as context |
| Dense embedding | `intfloat/multilingual-e5-small`, 384-dim — Qdrant Cloud (`passage:` at index, `query:` at serve) |
| Retrieval | dense ANN on the vernacular query (`frontend/lib/server/retrieval.ts`) |

See [CHUNKING.md](./CHUNKING.md) for the english-pivot alternative and earlier prototypes.

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

Indexing is CLI-only — see `INDEXING.md` for the full reference (multi-language, parallel workers, resuming). Dataset access is **local-first** (HF cache parquet) with **Hub streaming fallback** per language shard, so `--limit` on a new machine does not require downloading all ~55 GB.

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
QDRANT_COLLECTION_NAME=msmarco_xi__multilingual_e5_small__qa_pair__train
QDRANT_EMBEDDING_MODEL=intfloat/multilingual-e5-small
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
  chunking.py       — QAPairChunker / EnglishQueryChunker
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
