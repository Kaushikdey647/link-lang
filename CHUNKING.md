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
    "English_passages": [
      "The presence of communication amid scientific minds was equally important...",
      "The Manhattan Project and its atomic bomb helped bring an end to World War II...",
      ...
    ],
    "Translated_passages": [
      "वैज्ञानिक दिमाग के बीच संचार की उपस्थिति मैनहट्टन परियोजना की सफलता के लिए...",
      "मैनहट्टन परियोजना और इसके परमाणु बम ने द्वितीय विश्व युद्ध को समाप्त करने में मदद की...",
      ...
    ]
  }
}
```

Key observations:

- Each row has **10 candidate passages**, one of which is marked `is_selected=1` (ground-truth relevant)
- Passages are **already semantically bounded** — they were scraped from the web as discrete snippets, not continuous prose
- Passages are **parallel**: every `Translated_passages[i]` is the translation of `English_passages[i]`
- The `query_type` field (`DESCRIPTION`, `NUMERIC`, `ENTITY`, etc.) tells us what kind of answer is expected

This means the passages **are** the natural chunk boundary. We don't re-chunk across passages — we apply multiple strategies *within and around* each passage.

---

## Three Chunking Strategies

All strategies share a unified interface:

```python
class BaseChunker(ABC):
    def chunk(self, record: PassageRecord) -> list[Chunk]
```

Every `Chunk` carries the same metadata regardless of strategy:

| Field | Description |
|---|---|
| `chunk_id` | Stable unique key: `"{passage_id}__{strategy}"` |
| `chunk_type` | `"passage"` / `"sentence"` / `"qa_pair"` |
| `text` | The actual text to embed |
| `passage_id` | Parent passage (links sentences back to their source) |
| `query_id` | The row this came from |
| `is_selected` | Ground-truth relevance label |
| `lang` | 2-letter language code |
| `sentence_index` | Index within parent passage (sentence chunks only) |

---

### Strategy 1 — Passage (`PassageChunker`)

The full translated passage is stored as a single chunk.

**Why:** This is the primary retrieval unit. Passages are 2–6 sentences, typically 80–300 tokens — the right size for both precise retrieval and sufficient LLM context.

**Input passage** (`passage_id = "1185869_0"`, `is_selected = True`):
```
वैज्ञानिक दिमाग के बीच संचार की उपस्थिति मैनहट्टन परियोजना की सफलता के लिए उतनी ही
महत्वपूर्ण थी जितनी कि वैज्ञानिक बुद्धिमत्ता थी। परमाणु शोधकर्ताओं और इंजीनियरों की
प्रभावशाली उपलब्धि पर लटकता एकमात्र बादल उनकी सफलता का वास्तव में क्या अर्थ था;
सैकड़ों हजारों निर्दोष जीवन का विनाश।
```

**Output — 1 chunk:**

```
chunk_id:   "1185869_0__passage"
chunk_type: "passage"
text:       "वैज्ञानिक दिमाग के बीच संचार की उपस्थिति मैनहट्टन परियोजना की सफलता के
             लिए उतनी ही महत्वपूर्ण थी जितनी कि वैज्ञानिक बुद्धिमत्ता थी। परमाणु
             शोधकर्ताओं और इंजीनियरों की प्रभावशाली उपलब्धि पर लटकता एकमात्र बादल
             उनकी सफलता का वास्तव में क्या अर्थ था; सैकड़ों हजारों निर्दोष जीवन का
             विनाश।"
```

---

### Strategy 2 — Sentence (`SentenceChunker`)

Each passage is split at sentence boundaries into individual sentence chunks. Sentence delimiters handled:

| Script | Delimiters |
|---|---|
| Latin | `.` `!` `?` |
| Devanagari | `।` (danda) `॥` (double danda) |

Sentences with fewer than `min_words` words (default: 4) are discarded as fragments. If splitting produces nothing, the whole passage is kept as a single sentence chunk.

**Why:** Sentence-level chunks allow **more precise retrieval** — a query about a specific fact matches the one sentence that contains it, not a multi-sentence passage where it's buried. This enables *small-to-big* retrieval: match at sentence level, but return the full parent passage as LLM context (the retriever expands via `passage_id`).

**Same input passage** → **3 sentence chunks:**

```
chunk_id:        "1185869_0__sent_0"
chunk_type:      "sentence"
sentence_index:  0
text:            "वैज्ञानिक दिमाग के बीच संचार की उपस्थिति मैनहट्टन परियोजना की
                  सफलता के लिए उतनी ही महत्वपूर्ण थी जितनी कि वैज्ञानिक बुद्धिमत्ता थी।"

chunk_id:        "1185869_0__sent_1"
chunk_type:      "sentence"
sentence_index:  1
text:            "परमाणु शोधकर्ताओं और इंजीनियरों की प्रभावशाली उपलब्धि पर लटकता
                  एकमात्र बादल उनकी सफलता का वास्तव में क्या अर्थ था;"

chunk_id:        "1185869_0__sent_2"
chunk_type:      "sentence"
sentence_index:  2
text:            "सैकड़ों हजारों निर्दोष जीवन का विनाश।"
```

**Small-to-big expansion at retrieval time:**

```
Query: "मैनहट्टन परियोजना में किसका योगदान अधिक था?"
                              ↓
Vector search matches "sent_0" (communication vs scientific intellect)
                              ↓
Retriever fetches parent passage_id "1185869_0__passage"
                              ↓
Full passage (all 3 sentences) sent to LLM as context
```

This way the LLM gets the full surrounding context even though retrieval was precise.

---

### Strategy 3 — QA Pair (`QAPairChunker`)

For passages where `is_selected = True`, the query is **prepended to the passage text** and indexed as a single chunk.

**Why:** This biases the embedding space toward the task distribution. A query-passage pair sits in a region of the vector space where *similar queries* will land at retrieval time. These are the closest thing to "golden" training examples in the dataset — we exploit the ground-truth signal for free.

Only fires when `is_selected=True` AND `query` is non-empty. For `is_selected=False` passages (9 out of 10 per row), no QA chunk is emitted.

**Same input passage** (`is_selected = True`) → **1 QA chunk:**

```
chunk_id:   "1185869_0__qa"
chunk_type: "qa_pair"
text:       "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव क्या था?
             वैज्ञानिक दिमाग के बीच संचार की उपस्थिति मैनहट्टन परियोजना
             की सफलता के लिए उतनी ही महत्वपूर्ण थी जितनी कि वैज्ञानिक
             बुद्धिमत्ता थी। परमाणु शोधकर्ताओं और इंजीनियरों की प्रभावशाली
             उपलब्धि पर लटकता एकमात्र बादल उनकी सफलता का वास्तव में क्या
             अर्थ था; सैकड़ों हजारों निर्दोष जीवन का विनाश।"
```

---

## Full Example: One Passage → All Chunks

Starting from `passage_id = "1185869_0"` with `is_selected = True`:

```
Input:
  query_id   = 1185869
  passage_id = "1185869_0"
  is_selected = True
  query      = "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव क्या था?"
  text       = "वैज्ञानिक दिमाग के बीच संचार... सैकड़ों हजारों निर्दोष जीवन का विनाश।"

CompositeChunker output (5 chunks):
┌──────────────────────────────┬────────────┬───────────────────────┐
│ chunk_id                     │ chunk_type │ text (truncated)      │
├──────────────────────────────┼────────────┼───────────────────────┤
│ 1185869_0__passage           │ passage    │ वैज्ञानिक दिमाग...   │
│ 1185869_0__sent_0            │ sentence   │ वैज्ञानिक दिमाग...   │
│ 1185869_0__sent_1            │ sentence   │ परमाणु शोधकर्ताओं... │
│ 1185869_0__sent_2            │ sentence   │ सैकड़ों हजारों...    │
│ 1185869_0__qa                │ qa_pair    │ मैनहट्टन परियोजना... │
└──────────────────────────────┴────────────┴───────────────────────┘

For is_selected=False passages (9 out of 10 per row):
┌──────────────────────────────┬────────────┐
│ chunk_id                     │ chunk_type │
├──────────────────────────────┼────────────┤
│ 1185869_1__passage           │ passage    │
│ 1185869_1__sent_0            │ sentence   │
│ 1185869_1__sent_1            │ sentence   │
│   (no qa_pair — not selected)│            │
└──────────────────────────────┴────────────┘
```

---

## Scale Estimate (Hindi)

| Split | Rows | Passages | Passage chunks | Sentence chunks (est. 3/passage) | QA chunks (1 per row) | **Total** |
|---|---|---|---|---|---|---|
| train | 778,638 | 7,786,380 | 7,786,380 | 23,359,140 | 778,638 | **~31.9M** |
| validation | 97,941 | 979,410 | 979,410 | 2,938,230 | 97,941 | **~4.0M** |

In practice most rows have fewer than 3 splittable sentences per passage, so the real number is lower. Passage-only indexing (~7.8M for Hindi train) is the minimum viable index.

---

## Plugging in a New Strategy

Implement `BaseChunker` and optionally register it:

```python
from pipeline.chunking import BaseChunker, Chunk, REGISTRY
from dataset.types import PassageRecord

class OverlapChunker(BaseChunker):
    """Sliding window over sentences with 1-sentence overlap."""

    def __init__(self, window: int = 2, step: int = 1):
        self.window = window
        self.step = step

    def chunk(self, record: PassageRecord) -> list[Chunk]:
        sentences = record.text.split("।")
        chunks = []
        for i in range(0, len(sentences), self.step):
            window_text = "।".join(sentences[i:i + self.window]).strip()
            if window_text:
                chunks.append(Chunk(
                    chunk_id=f"{record.passage_id}__overlap_{i}",
                    text=window_text,
                    chunk_type="overlap",
                    sentence_index=i,
                    lang=record.lang,
                    passage_id=record.passage_id,
                    query_id=record.query_id,
                    is_selected=record.is_selected,
                    query=record.query,
                    answer=record.answer,
                    query_type=record.query_type,
                ))
        return chunks

# Register for use with build_chunker()
REGISTRY["overlap"] = OverlapChunker
```

Then use it anywhere:

```python
from pipeline.chunking import build_chunker

chunker = build_chunker(["passage", "overlap"])
chunks = chunker.chunk(record)
```

---

## Retrieval Strategy per Chunk Type

At query time, the retriever can filter by `chunk_type` to change precision/recall behaviour:

| `chunk_types` filter | Behaviour |
|---|---|
| `["passage"]` | Fast, balanced. Good default. |
| `["sentence"]` | Highest precision; expands to full passage via `passage_id` lookup. |
| `["qa_pair"]` | Biased toward queries seen during indexing; good for in-distribution queries. |
| `["passage", "sentence", "qa_pair"]` | Maximum recall; deduplication by `passage_id` keeps top-k clean. |

```python
from pipeline import retrieve

# Fast retrieval (passage only)
results = retrieve(query, lang="hi", chunk_types=["passage"])

# High-precision retrieval (sentence → expanded to passage)
results = retrieve(query, lang="hi", chunk_types=["sentence"])

# Default: all strategies, deduplicated
results = retrieve(query, lang="hi")
```

---

## Embeddings

### Model: `multilingual-e5-small`

Every chunk's text is converted to a **384-dimensional float vector** using [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small). This model covers all 14 Indic languages in the dataset (Hindi, Bengali, Tamil, Telugu, etc.) plus English — a single model, no per-language switching.

**Why e5-small over other options:**
- Sarvam does not offer an embedding API — their platform covers STT, chat, translation, and TTS only
- OpenAI / Cohere embeddings add ~50ms network RTT and per-token cost at query time
- e5-small runs on CPU in ~25ms locally — well within the 200ms retrieval budget
- "small" (117M params) is the right size: larger e5 variants improve recall by ~1-2% but double latency

### The e5 Prefix Convention

e5 models are trained with task-specific prefixes that shift vectors into the correct region of the embedding space:

```
Indexing time  →  "passage: <chunk text>"
Query time     →  "query: <user question>"
```

Without the prefix, cosine similarity degrades significantly. Our `embedder.py` handles this automatically:

```python
def embed_passages(texts):          # called by indexer
    return model.encode(["passage: " + t for t in texts])

def embed_query(text):              # called by retriever
    return model.encode(["query: " + text])
```

### Each Chunk Gets Its Own Vector

Every `Chunk` object produced by the chunker becomes one independent point in Qdrant with its own 384-dim embedding:

```
passage_id "1185869_0"  (is_selected=True, 3 sentences)

  Chunk                    Text fed to e5-small                     Vector
  ─────────────────────    ──────────────────────────────────────   ──────────────
  1185869_0__passage   →  "passage: वैज्ञानिक दिमाग के बीच..."  → [0.12, -0.34, ...]
  1185869_0__sent_0    →  "passage: वैज्ञानिक दिमाग के बीच..."  → [0.09, -0.31, ...]  ← similar to passage
  1185869_0__sent_1    →  "passage: परमाणु शोधकर्ताओं और..."    → [0.41,  0.02, ...]  ← shifted (different topic)
  1185869_0__sent_2    →  "passage: सैकड़ों हजारों निर्दोष..."  → [-0.1,  0.55, ...]  ← very different
  1185869_0__qa        →  "passage: मैनहट्टन परियोजना...       → [0.18, -0.28, ...]  ← query-shifted
                                     वैज्ञानिक दिमाग के बीच..."
```

The sentence chunks and passage chunk are **similar but not identical** — the model sees different token windows and produces slightly different vectors. The QA chunk shifts more noticeably because the prepended query text pulls it toward the query-side of the embedding space, making it easier to match at retrieval time.

---

## Qdrant: Storage and Retrieval

### Collection Structure

All chunks across all languages land in a single Qdrant collection `msmarco_xi`. Language is a payload field, not a separate collection — this lets you do cross-lingual retrieval later if needed.

```
Collection: msmarco_xi
  Vector size:  384  (cosine distance)
  Payload indexes:
    lang        → KEYWORD   (filter by language at query time)
    chunk_type  → KEYWORD   (filter by strategy)
    query_id    → INTEGER   (group chunks from the same row)
    is_selected → BOOL      (filter to ground-truth positives)
```

Each point in Qdrant looks like:

```json
{
  "id": "b3f2a1...",
  "vector": [0.12, -0.34, 0.09, ...],
  "payload": {
    "chunk_id":       "1185869_0__passage",
    "chunk_type":     "passage",
    "lang":           "hi",
    "passage_id":     "1185869_0",
    "query_id":       1185869,
    "is_selected":    true,
    "text":           "वैज्ञानिक दिमाग के बीच संचार...",
    "query":          "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव क्या था?",
    "answer":         "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव...",
    "query_type":     "DESCRIPTION",
    "sentence_index": -1
  }
}
```

### How ANN Search Works

When a query arrives, Qdrant does **Approximate Nearest Neighbour (ANN)** search using the HNSW graph index. It does not scan all 31.9M vectors — it navigates the graph in logarithmic time to find vectors with the highest cosine similarity to the query vector.

```
Query vector  →  HNSW graph traversal  →  Top-K candidates  →  Payload filter applied
```

The payload filter (`lang=hi`, `chunk_type=passage`) is applied **after** the ANN shortlist, not before — Qdrant's filtered search is approximate but very fast (~10-20ms at this scale).

### Small-to-Big Expansion in Qdrant

When sentence chunks match at retrieval, the retriever does a `scroll` (payload filter, no vector search) to fetch the full parent passage text:

```
ANN search hits "1185869_0__sent_1"  (score: 0.87)
                        ↓
payload["passage_id"] = "1185869_0"
                        ↓
scroll(filter: passage_id="1185869_0" AND chunk_type="passage")
                        ↓
returns full passage text → sent to LLM
```

This two-step lookup adds ~2ms and gives the LLM the full context around the matched sentence.

---

## End-to-End RAG Pipeline

### Component Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Voice Input (user speaks)                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ audio file
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Sarvam Saaras v3  (STT)                                            │
│  stt.transcribe(audio, language_code="hi-IN")                       │
│  → transcript: "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव क्या था?"│
└──────────────────────────────┬──────────────────────────────────────┘
                               │ text query
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  multilingual-e5-small  (Embedding)                                 │
│  embed_query("query: मैनहट्टन परियोजना...")                         │
│  → [0.18, -0.41, 0.03, ...]   384-dim vector                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ query vector
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Qdrant  (ANN Retrieval)                                            │
│  query_points(vector, filter: lang=hi, chunk_type=[...], limit=20) │
│  → top-20 candidate chunks ranked by cosine similarity              │
│                                                                     │
│  Small-to-big expansion:                                            │
│  sentence hits → scroll → parent passage text                       │
│                                                                     │
│  Deduplication by passage_id → top-5 distinct passages             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ list[RetrievedPassage]
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Sarvam-105B  (Answer Generation)                                   │
│                                                                     │
│  System: "Answer in Hindi using ONLY the provided passages."        │
│  User:   "[1] वैज्ञानिक दिमाग के बीच...\n\n[2] मैनहट्टन...\n\n     │
│           Question: मैनहट्टन परियोजना की सफलता का प्रभाव क्या था?" │
│                                                                     │
│  reasoning_content: <chain-of-thought, not shown to user>           │
│  content: "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव..."         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ answer text
                               ▼
                        User sees answer
```

### Why Sarvam-105B for Generation (Not for Embedding)

Sarvam's platform has a clear role split:

| Task | Why Sarvam | Why NOT Sarvam |
|---|---|---|
| STT | Saaras v3 is purpose-built for Indic-accented speech; best-in-class for Indian languages | — |
| Embedding | — | No embedding API exists; e5-small is faster, cheaper, local |
| Generation | sarvam-105b natively generates fluent Hindi/Bengali/Tamil/etc. without translation overhead; 128K context window fits many passages | Reasoning overhead means 3-8s latency per call |

A general-purpose model like GPT-4o would need to "think" in English and translate back. sarvam-105b reasons directly in the target language, producing more natural output for Indic queries.

### Latency Budget

The 200ms target from the problem statement applies to the **retrieval sub-pipeline** (embed + Qdrant). Full end-to-end including generation is longer:

```
Step                          Target      Notes
──────────────────────────────────────────────────────────────────
STT (Sarvam Saaras v3)        ~1-3s       Network + model inference
Embed query (e5-small local)   ~25ms      CPU, single vector
Qdrant ANN search              ~15ms      HNSW index, payload filter
Small-to-big scroll            ~2ms       Only on sentence hits
                              ──────
Retrieval subtotal             ~42ms      ✓ well under 200ms

Sarvam-105B generation         ~3-8s      Reasoning model (CoT)
                              ──────
Full pipeline                  ~5-12s     STT + retrieval + generation
```

The retrieval stage meets 200ms comfortably. Generation latency is dominated by sarvam-105b's reasoning phase — unavoidable with the only available Sarvam chat model. Setting `reasoning_effort="low"` keeps this closer to 3s than 8s.
