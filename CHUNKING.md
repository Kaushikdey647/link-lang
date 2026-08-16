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

## Four Chunking Strategies

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

### Strategy 4 — English Query (`EnglishQueryChunker`)

**The English-pivot idea:** MSMARCO-XI contains both the original English question (`record.eng_query`) and its translated vernacular passage (`record.text`). Instead of embedding the vernacular passage, we embed the *English question* using a lightweight English sentence model. At query time, the user's vernacular voice query is translated to English before embedding — giving us cross-lingual retrieval with a single monolingual model.

Note: `record.query` is the *vernacular* translation of the question (same language as `record.text`) — it's `record.eng_query` that always holds the original English, independent of the `translated` flag `iter_passages()` was called with.

**Why:** A good English embedding model covers a much smaller vocabulary than a multilingual model. `all-MiniLM-L6-v2` (22M params) outperforms `multilingual-e5-small` (117M params) on English-only retrieval benchmarks while running 3× faster. Since the original MSMARCO questions are high-quality English, and Sarvam's translate API can convert user queries to English in ~50ms, this gives us the best of both worlds: English model quality + vernacular UI.

**What gets embedded:** `record.eng_query` (English question — clean, concise, already high quality)

**What gets returned:** `record.text` (vernacular passage — stored in `parent_passage` field, returned as LLM context)

**Same input passage** (`is_selected = True`) → **1 english_query chunk:**

```
chunk_id:       "1185869_0__enq"
chunk_type:     "english_query"
text:           "what was the immediate impact of the success of the manhattan project?"   ← embedded
parent_passage: "वैज्ञानिक दिमाग के बीच संचार की उपस्थिति मैनहट्टन परियोजना की सफलता..."  ← returned
```

**Query-time flow:**
```
User speaks (Hindi): "मैनहट्टन परियोजना में संचार की क्या भूमिका थी?"
                              ↓  Sarvam translate (hi → en, ~50ms)
Translated query:   "What was the role of communication in the Manhattan Project?"
                              ↓  all-MiniLM-L6-v2 embed (~8ms)
Vector search → top-k english_query chunks
                              ↓  payload["parent_passage"]
Context returned:   "वैज्ञानिक दिमाग के बीच संचार की उपस्थिति..."
                              ↓  Sarvam-105B
Answer (Hindi):     "संचार की भूमिका मैनहट्टन परियोजना में अत्यंत महत्वपूर्ण थी।"
```

Chunks are only emitted when `record.query` is non-empty (all `is_selected` passages have a query; `is_selected=False` passages skip QA chunk but still have `record.query` in the dataset — both get an `english_query` chunk).

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

CompositeChunker output — all strategies (6 chunks, is_selected=True):
┌──────────────────────────────┬────────────────┬─────────────────────────────────────────────────┐
│ chunk_id                     │ chunk_type     │ text field (what gets embedded)                 │
├──────────────────────────────┼────────────────┼─────────────────────────────────────────────────┤
│ 1185869_0__passage           │ passage        │ वैज्ञानिक दिमाग... (vernacular passage)         │
│ 1185869_0__sent_0            │ sentence       │ वैज्ञानिक दिमाग... (first sentence)             │
│ 1185869_0__sent_1            │ sentence       │ परमाणु शोधकर्ताओं... (second sentence)          │
│ 1185869_0__sent_2            │ sentence       │ सैकड़ों हजारों... (third sentence)              │
│ 1185869_0__qa                │ qa_pair        │ मैनहट्टन परियोजना? + वैज्ञानिक दिमाग...        │
│ 1185869_0__enq               │ english_query  │ "what was the immediate impact..." (English!)   │
└──────────────────────────────┴────────────────┴─────────────────────────────────────────────────┘

For is_selected=False passages (9 out of 10 per row):
┌──────────────────────────────┬────────────────┐
│ chunk_id                     │ chunk_type     │
├──────────────────────────────┼────────────────┤
│ 1185869_1__passage           │ passage        │
│ 1185869_1__sent_0            │ sentence       │
│ 1185869_1__sent_1            │ sentence       │
│ 1185869_1__enq               │ english_query  │  ← still has English query; no qa_pair
│  (no qa_pair — not selected) │                │
└──────────────────────────────┴────────────────┘
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
| `["english_query"]` | Requires translate-to-English at query time; returns `parent_passage` as context. |
| `["passage", "sentence", "qa_pair"]` | Maximum recall (vernacular embedding); deduplication by `passage_id` keeps top-k clean. |

```python
from pipeline import retrieve

# Fast retrieval (passage only)
results = retrieve(query, lang="hi", chunk_types=["passage"])

# English-pivot: query translated to English, vernacular passage returned
results = retrieve(query, lang="hi", chunk_types=["english_query"])

# Default: all strategies, deduplicated
results = retrieve(query, lang="hi")
```

---

## Embeddings

### Three Backends

The embedding backend is configured independently of the chunking strategy via `IndexPlan`. Each backend creates its own Qdrant collection.

| Backend | Model | Dim | Prefix | Language support | Cost |
|---|---|---|---|---|---|
| `e5` | `intfloat/multilingual-e5-small` | 384 | `passage:` / `query:` | All 14 Indic + English | Local, free |
| `english` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | None (symmetric) | English only | Local, free |
| `cohere` | `Cohere/embed-multilingual-v3.0` | 1024 | Input type flag | 100+ languages | API, ~$0.10/1M tokens |

**When to use which backend:**
- `e5` — default for vernacular chunking strategies (`passage`, `sentence`, `qa_pair`); no API dependency
- `english` — use with `english_query` chunks; faster, smaller model, English-only data embeds well here
- `cohere` — best recall for mixed-language or high-stakes deployments; requires `COHERE_API_KEY`

The `english` backend with `english_query` chunker is the recommended production setup: Sarvam translates queries to English (~50ms), and a small 22M-param model produces high-quality vectors, with vernacular passages returned via `parent_passage`.

### The e5 Prefix Convention

e5 models are trained with task-specific prefixes that shift vectors into the correct region of the embedding space:

```
Indexing time  →  "passage: <chunk text>"
Query time     →  "query: <user question>"
```

Without the prefix, cosine similarity degrades significantly. `embedder.py` handles this automatically. The `english` backend (all-MiniLM-L6-v2) is symmetric — no prefix needed.

### Cohere Rate Limiter

Cohere's embed API has a 2,000-input/minute limit on free-tier keys. `embedder.py` implements a thread-safe token bucket (33 inputs/sec, 200-burst capacity) that counts individual texts, not batch requests. This allows full parallelism up to the limit without explicit sleep calls.

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

## IndexPlan: Decoupled Indexing Strategy

An `IndexPlan` ties together the backend (which model) and chunkers (what to embed) into a deterministic collection name:

```python
@dataclass
class IndexPlan:
    backend:  str          # "e5" | "english" | "cohere"
    chunkers: list[str]    # e.g. ["passage", "sentence", "qa_pair"]
    split:    str = "train"

    @property
    def collection_name(self) -> str:
        key = "_".join(sorted(self.chunkers))   # sorted → always deterministic
        return f"msmarco_xi__{self.backend}__{key}__{self.split}"
```

Example collection names:
- `msmarco_xi__english__english_query__train` — English-pivot, all-MiniLM
- `msmarco_xi__e5__passage_sentence_qa_pair__train` — full vernacular strategy, e5
- `msmarco_xi__cohere__passage__train` — passage-only, Cohere

A registry at `.indexer_checkpoints/registry.json` records every successfully indexed plan with its model metadata. The query layer calls `best_available_plan()` to pick from what's available (preference: cohere > english > e5), or the caller can specify a `collection` name explicitly.

## Qdrant: Storage and Retrieval

### Collection Structure

Each `IndexPlan` gets its own Qdrant collection. Language is a payload field within each collection — all 14 languages share one collection per plan.

```
Collection: msmarco_xi__english__english_query__train
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

Two query paths depending on the active `IndexPlan`:

### Path A — Vernacular embedding (e5 / cohere backend)

```
User speaks (Hindi) → Sarvam STT → vernacular query text
                                          ↓
                           embed_query("query: <text>", backend="e5")
                                          ↓
                     Qdrant ANN (filter: lang=hi, chunk_type=[...])
                                          ↓
                     sentence hits → scroll → parent passage expansion
                                          ↓
                           Sarvam-105B → answer in Hindi
```

### Path B — English-pivot (english backend + english_query chunker)

```
User speaks (Hindi) → Sarvam STT → vernacular query text
                                          ↓
                         Sarvam translate (hi → en, ~50ms)
                                          ↓
                           embed_query(english_text, backend="english")
                                          ↓
                     Qdrant ANN (filter: lang=hi, chunk_type=english_query)
                                          ↓
                     payload["parent_passage"] → vernacular passage context
                                          ↓
                           Sarvam-105B → answer in Hindi
```

### Detailed Component Map (Path A)

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
Step                               Path A (e5)   Path B (english-pivot)   Notes
────────────────────────────────────────────────────────────────────────────────
STT (Sarvam Saaras v3)             ~1-3s         ~1-3s                    Network + model inference
Sarvam translate (hi → en)         —             ~50ms                    Only in english-pivot path
Embed query (local)                ~25ms         ~8ms                     e5-small vs all-MiniLM
Qdrant ANN search                  ~15ms         ~15ms                    HNSW index, payload filter
Small-to-big scroll / parent_pass  ~2ms          ~0ms (payload field)     parent_passage already in payload
                                   ──────        ──────
Retrieval subtotal                 ~42ms         ~73ms                    ✓ well under 200ms (both paths)

Sarvam-105B generation             ~3-8s         ~3-8s                    Reasoning model (CoT)
                                   ──────        ──────
Full pipeline                      ~5-12s        ~5-12s                   STT + retrieval + generation
```

The retrieval stage meets 200ms comfortably on both paths. Generation latency is dominated by sarvam-105b's reasoning phase. Setting `reasoning_effort="low"` keeps this closer to 3s than 8s.
