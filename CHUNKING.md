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
