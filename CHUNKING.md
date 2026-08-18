# Chunking Strategy

## What is Chunking?

In a RAG pipeline, every piece of text that might be relevant to a query gets embedded into a vector and stored in a vector database. **Chunking** is the decision of what unit of text gets one embedding vector.

This matters because:
- **Too large** → one chunk covers multiple facts; a query about one fact retrieves irrelevant content from the same chunk
- **Too small** → a chunk lacks enough context for the LLM to produce a grounded answer

The goal is to match the **retrieval granularity** to the **semantic granularity** of the questions being asked.

---

## The MSMARCO-XI Data Shape

Each row in the dataset looks like this:

```json
{
  "query_id": 1185869,
  "query_type": "DESCRIPTION",
  "query": "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव क्या था?",
  "Eng_Query": "what was the immediate impact of the success of the manhattan project?",
  "Answer": "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव...",
  "passages": {
    "is_selected": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "English_passages": ["The presence of communication amid scientific minds was equally important...", "..."],
    "Translated_passages": ["वैज्ञानिक दिमाग के बीच संचार की उपस्थिति...", "..."]
  }
}
```

Key observations:
- Each row has **10 candidate passages**, one of which is marked `is_selected=1` (ground-truth relevant)
- Passages are **already semantically bounded** — they were scraped from the web as discrete snippets, not continuous prose
- Passages are **parallel**: every `Translated_passages[i]` is the translation of `English_passages[i]`
- Every row also carries `Eng_Query` — the original English question, always present regardless of translation

---

## english-pivot: the system's one chunking + embedding strategy

**Priority: latency and production footprint over retrieval-strategy extensibility.** Multiple chunking strategies and embedding backends were prototyped (see "What else was considered" below) — english-pivot was chosen as the sole production strategy and everything else was removed, not just made unreachable.

**The idea:** MSMARCO-XI contains both the original English question (`record.eng_query`) and its translated vernacular passage (`record.text`). Instead of embedding the vernacular passage, embed the *English question* using a small English-only model (`sentence-transformers/all-MiniLM-L6-v2`). At query time, the user's vernacular voice query is translated to English before embedding — giving cross-lingual retrieval with a single monolingual model, across all 14 languages.

**What gets embedded:** `record.eng_query` (English question — clean, concise, already high quality)
**What gets returned:** `record.text` (vernacular passage — stored in `parent_passage`, returned as LLM context)

```
chunk_id:       "hi_1185869_0__enq"
chunk_type:     "english_query"
text:           "what was the immediate impact of the success of the manhattan project?"   ← embedded
parent_passage: "वैज्ञानिक दिमाग के बीच संचार की उपस्थिति मैनहट्टन परियोजना की सफलता..."  ← returned
```

**Query-time flow:**
```
User speaks (Hindi): "मैनहट्टन परियोजना में संचार की क्या भूमिका थी?"
                              ↓  Sarvam translate (hi → en, ~50ms)
Translated query:   "What was the role of communication in the Manhattan Project?"
                              ↓  MiniLM embed — server-side, Qdrant Cloud
Vector search → top-k english_query chunks (RRF-fused with a BM25 sparse search
                on the original vernacular query, also computed server-side)
                              ↓  payload["parent_passage"]
Context returned:   "वैज्ञानिक दिमाग के बीच संचार की उपस्थिति..."
                              ↓  Sarvam-105B
Answer (Hindi):     "संचार की भूमिका मैनहट्टन परियोजना में अत्यंत महत्वपूर्ण थी।"
```

Only fires when `record.eng_query` is non-empty — always true for MSMARCO-XI (`Eng_Query` is populated regardless of the `translated` flag `iter_passages()` was called with).

Implementation: `pipeline/chunking.py::EnglishQueryChunker` (the only chunker in that file).

---

## Embeddings: Qdrant Cloud server-side inference, no local models

Both vectors are computed **server-side by Qdrant Cloud**, not by any model running in this codebase:

| Vector | Model | Dim | Computed by |
|---|---|---|---|
| Dense | `sentence-transformers/all-minilm-l6-v2` | 384 | Qdrant Cloud (inference API) |
| Sparse | `qdrant/bm25` (IDF-weighted) | — | Qdrant Cloud (inference API) |

At index and query time, raw text is sent as a Qdrant `Document(text=..., model=...)` object instead of a pre-computed vector — `qdrant_client.QdrantClient(..., cloud_inference=True)` sends it to Qdrant Cloud, which returns the embedded point. No `sentence-transformers`, `torch`, `fastembed`, or Cohere API dependency exists in this codebase anymore (removed along with the other backends — see CHANGELOG.md). This is enforced by `QDRANT_API_KEY` being required: with no local fallback, indexing/serving without it fails loudly with a clear Qdrant-side error rather than silently degrading.

See `INDEXING.md`'s "Remote (Qdrant Cloud) inference" section for the connection details, env vars, and an SOP for testing this end to end.

### Retrieval: RRF hybrid

`frontend/lib/server/retrieval.ts` (ported from the now-retired `pipeline/query_engines.py::EnglishPivotQueryEngine`) fuses two searches server-side via Qdrant's `FusionQuery(fusion=Fusion.RRF)`:
- **Dense**: the vernacular query, translated to English (Sarvam), embedded via MiniLM
- **Sparse**: the original vernacular query, embedded via BM25

The vernacular query is never discarded from the caller's perspective — only the internal dense prefetch sees the English-translated version. `parent_passage` is already inline in the payload (no extra lookup needed to get the full vernacular passage back for LLM context).

---

## What else was considered (and removed)

Earlier iterations of this project chunked each passage three additional ways, crossed with three embedding backends — this is what a genuinely "vast" chunking exploration looked like before the collapse to a single strategy:

| Strategy | What it did | Why it was dropped |
|---|---|---|
| `PassageChunker` | One chunk per full vernacular passage | Needed a vernacular-capable embedding model (e5 or Cohere) — removed along with those backends |
| `SentenceChunker` | Split each passage at sentence boundaries (Latin `.!?` + Devanagari `।॥`), small-to-big expansion back to the parent passage at query time | Same vernacular-embedding dependency; also added a second Qdrant round-trip (scroll for the parent) that english-pivot avoids entirely (`parent_passage` is already inline) |
| `e5` backend (local) | `intfloat/multilingual-e5-small`, local, all 14 languages | Local model inference (`sentence-transformers` + `torch`) — real, measured memory/cold-start cost in production (idle Docker RSS before this removal: ~800MB from imports alone) |
| `cohere` backend | `Cohere/embed-multilingual-v3.0`, API, 1024-dim | Extra API dependency/cost/rate-limiting for no retrieval-quality requirement this project actually needed once english-pivot covers all 14 languages with one model |

The tradeoff: english-pivot is English-only on the embedding side (by design — that's the entire point of the pivot), so it can't do pure vernacular semantic matching the way `e5`/`cohere` could. In exchange: one model, one collection, no local inference, materially lower latency and memory footprint, and cross-lingual retrieval across all 14 languages without needing a multilingual embedding model at all. Given the <200ms retrieval latency target and a production deployment with real memory constraints, that tradeoff was made deliberately — see CHANGELOG.md for when.

### `QAPairChunker` + `multilingual_e5_small` — reintroduced as a second, separate collection

Unlike the local `e5` backend above, this uses **Qdrant Cloud's server-side inference** for `intfloat/multilingual-e5-small` (dense-only). It is the default indexing strategy (`scripts/index.py --strategy qa_pair`) and the Next.js serving path.

e5's asymmetric prefixes: index time `pipeline/indexer.py::_e5_text()` prepends `"passage: "`; query time `frontend/lib/server/retrieval.ts` prepends `"query: "` to the vernacular query (no English translate hop).

---

## IndexPlan

Every index is described by an `IndexPlan` (`pipeline/index_plan.py`): `(backend, chunkers, split)`. Two combinations are valid today, each its own deterministic collection name:

```
msmarco_xi__english__english_query__{split}
msmarco_xi__multilingual_e5_small__qa_pair__{split}
```

`IndexPlan` kept its `(backend, chunkers, split)` shape rather than collapsing to a hardcoded constant specifically so the collection-name derivation stays compatible with data already indexed under this scheme.

---

## Qdrant: Storage and Retrieval

### Collection Structure

```
Collection: msmarco_xi__english__english_query__train
  Dense vector (unnamed):  384-dim, cosine distance
  Sparse vector ("bm25"):  IDF-weighted
  Payload indexes:
    lang        → KEYWORD   (filter by language at query time)
    chunk_type  → KEYWORD   (always "english_query")
    query_id    → INTEGER   (group chunks from the same row)
    is_selected → BOOL      (filter to ground-truth positives)
```

Each point's payload:
```json
{
  "chunk_id":       "hi_1185869_0__enq",
  "chunk_type":     "english_query",
  "lang":           "hi",
  "passage_id":     "hi_1185869_0",
  "query_id":       1185869,
  "is_selected":    true,
  "text":           "what was the immediate impact of the success of the manhattan project?",
  "parent_passage": "वैज्ञानिक दिमाग के बीच संचार की उपस्थिति...",
  "query":          "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव क्या था?",
  "answer":         "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव...",
  "query_type":     "DESCRIPTION"
}
```

### How ANN Search Works

Qdrant does **Approximate Nearest Neighbour (ANN)** search using the HNSW graph index — it does not scan every vector, it navigates the graph in logarithmic time. The payload filter (`lang=hi`) is applied after the ANN shortlist, not before — Qdrant's filtered search is approximate but very fast.

---

## End-to-End RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Voice Input (user speaks)                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ audio file
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Sarvam Saaras v3  (STT)                                            │
│  stt.transcribe(audio) → transcript + detected language              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ vernacular text query
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Sarvam translate (hi → en, ~50ms)                                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ English text
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Qdrant Cloud — RRF hybrid retrieval                                 │
│  Prefetch(dense: MiniLM on English query) +                          │
│  Prefetch(sparse: BM25 on original vernacular query) → RRF fusion    │
│  Deduplication by passage_id → top-k distinct passages                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ list[Document] (parent_passage already inline)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Sarvam-105B  (Answer Generation)                                    │
│  System: "Answer in {lang} using ONLY the provided passages."        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ answer text
                               ▼
                        User sees answer
```

### Latency Budget

The <200ms target applies to the **retrieval sub-pipeline** (translate + embed + Qdrant RRF fusion). Full end-to-end including generation is longer (dominated by Sarvam-105B's reasoning phase, not retrieval). The Python P50/P70/P100 benchmark script (`scripts/benchmark.py`) is retired along with `pipeline/rag.py` — a TypeScript equivalent against `frontend/lib/server/rag.ts` is a planned follow-up (see CHANGELOG.md's `frontend_only` entry).
