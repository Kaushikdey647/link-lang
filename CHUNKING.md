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

## Default / serving: `qa_pair` + `intfloat/multilingual-e5-small`

**This is the production path** for both indexing (`scripts/index.py`, default `--strategy qa_pair`) and serving (Next.js `link-lang-frontend`).

**The idea:** Embed the vernacular `query + passage` pair with a multilingual dense model. At query time, embed the user's vernacular STT transcript directly — **no English translate hop**.

**What gets embedded:** `f"{record.query} {record.text}"` (vernacular question + selected passage), with e5's `passage: ` prefix at index time  
**What gets returned:** `record.text` (vernacular passage — stored in `parent_passage`, used as LLM context)

```
chunk_id:       "hi_1185869_0__qa"
chunk_type:     "qa_pair"
text:           "<vernacular query> <vernacular passage>"   ← embedded (as "passage: …")
parent_passage: "<vernacular passage>"                     ← returned to the LLM
```

Only fires when `record.is_selected` and `record.query` are set. Indexing always uses `iter_passages(..., translated=True)`, so both sides of the pair are vernacular.

Implementation: `pipeline/chunking.py::QAPairChunker`.

**Query-time flow (serving):**
```
User speaks (Hindi): "मैनहट्टन परियोजना में संचार की क्या भूमिका थी?"
                              ↓  Sarvam Saaras v3 STT
Vernacular transcript (+ lang from STT / text-LID)
                              ↓  e5 embed — "query: …", Qdrant Cloud
Dense search → top-k qa_pair chunks (filter lang + chunk_type)
                              ↓  payload["parent_passage"]
Context → Sarvam-105B (input refusal folded into system prompt;
          grounding check post-stream, lexical)
Answer (same language): …
```

e5 asymmetric prefixes:
- Index: `pipeline/indexer.py::_e5_text()` → `"passage: "`
- Query: `link-lang-frontend/lib/server/retrieval.ts` → `"query: "`

---

## Embeddings: Qdrant Cloud server-side inference, no local models

Vectors are computed **server-side by Qdrant Cloud**, not by any model running in this codebase:

| Plan | Dense model | Sparse | Used when |
|---|---|---|---|
| **qa_pair (default)** | `intfloat/multilingual-e5-small` (384-dim) | none | CLI default + Next.js serving |
| `english_query` (optional) | `sentence-transformers/all-minilm-l6-v2` | `qdrant/bm25` | `--strategy english_query` only |

At index time, raw text is sent as a Qdrant `Document(text=..., model=...)` with `cloud_inference=True`. No `sentence-transformers`, `torch`, or `fastembed` dependency in this repo.

See `INDEXING.md` for env vars and end-to-end SOP.

---

## Optional alternate: `english_query` (english-pivot)

Still indexable via `uv run python -m scripts.index --strategy english_query`. **Not** the serving path.

Embeds `record.eng_query` (English) with MiniLM; stores vernacular `parent_passage`. Would need an English-translate hop at query time (removed from the Next.js serving stack). Kept as a separate collection so existing data stays valid.

```
msmarco_xi__multilingual_e5_small__qa_pair__{split}   # default / serving
msmarco_xi__english__english_query__{split}           # optional
```

---

## What else was considered (and removed)

Earlier iterations also tried:

| Strategy | What it did | Why it was dropped |
|---|---|---|
| `PassageChunker` | One chunk per full vernacular passage | Needed vernacular embeddings; superseded by qa_pair |
| `SentenceChunker` | Split at sentence boundaries + small-to-big parent lookup | Extra Qdrant round-trip; same model dependency |
| Local `e5` backend | `intfloat/multilingual-e5-small` via sentence-transformers | Memory/cold-start cost (~800MB idle RSS) — replaced by Qdrant Cloud inference |
| `cohere` backend | `Cohere/embed-multilingual-v3.0` | Extra API dependency for no required quality gain |

---

## IndexPlan

Every index is described by an `IndexPlan` (`pipeline/index_plan.py`): `(backend, chunkers, split)`. Two combinations are valid today, each its own deterministic collection name:

```
msmarco_xi__multilingual_e5_small__qa_pair__{split}
msmarco_xi__english__english_query__{split}
```

---

## Qdrant: Storage and Retrieval (default collection)

```
Collection: msmarco_xi__multilingual_e5_small__qa_pair__train
  Dense vector (unnamed):  384-dim, cosine distance
  Payload indexes:
    lang        → KEYWORD
    chunk_type  → KEYWORD   (always "qa_pair")
    query_id    → INTEGER
    is_selected → BOOL
```

Each point's payload (shape):
```json
{
  "chunk_id":       "hi_1185869_0__qa",
  "chunk_type":     "qa_pair",
  "lang":           "hi",
  "passage_id":     "hi_1185869_0",
  "query_id":       1185869,
  "is_selected":    true,
  "text":           "<vernacular query> <vernacular passage>",
  "parent_passage": "वैज्ञानिक दिमाग के बीच संचार की उपस्थिति...",
  "query":          "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव क्या था?",
  "answer":         "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव...",
  "query_type":     "DESCRIPTION"
}
```

### How ANN Search Works

Qdrant does **Approximate Nearest Neighbour (ANN)** search using the HNSW graph index. The payload filter (`lang=hi`, `chunk_type=qa_pair`) is applied on the ANN shortlist.

---

## End-to-End RAG Pipeline (serving)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Voice Input (user speaks)                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ audio
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Sarvam Saaras v3  (STT) — /api/voice                                │
│  → transcript + language                                             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ vernacular text (no translate)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Qdrant Cloud — dense retrieval                                      │
│  e5 "query: …" → top-k qa_pair chunks (lang + chunk_type filter)     │
│  Deduplicate by passage_id → parent_passage as LLM context           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Sarvam-105B  (streamed generation)                                  │
│  System prompt: answer in {lang} from passages only;                 │
│  refuse harmful/off-topic with "REFUSED: …" (input guardrail)        │
│  Post-stream: lexical grounding check                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                        User sees answer
```

### Latency Budget

The <200ms target applies to the **retrieval sub-pipeline** (e5 embed + Qdrant dense query). Full end-to-end including generation is longer (dominated by Sarvam-105B). There is **no translate stage** on this path.
