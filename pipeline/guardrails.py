"""Guardrails for the RAG pipeline — no embedding calls.

Two checks:
  1. InputGuardrail    — LLM-based safety/relevance check (off-topic or unsafe).
  2. GroundingGuardrail — lexical token-overlap between the generated answer
                          and the retrieved passages.

Previously both checks embedded text (domain-centroid cosine similarity for
input, per-sentence cosine similarity for grounding) via whichever local/API
backend was configured (e5 locally, or Cohere). That embedding dependency was
removed along with the e5/cohere backends — Qdrant Cloud's server-side
inference (pipeline/indexer.py) is scoped to index/query operations, not a
standalone "embed this text" call, so it isn't a drop-in replacement here.
See CHANGELOG.md.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import truststore; truststore.inject_into_ssl()
from dotenv import load_dotenv
from langchain_sarvam import ChatSarvam
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Fraction of an answer sentence's content words that must appear somewhere
# in the retrieved passages for that sentence to count as grounded.
_GROUNDING_OVERLAP_THRESHOLD = 0.35

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # unicode letters — works across Indic scripts
_MIN_WORD_LEN = 3  # crude stopword filter: skip very short tokens (articles/particles)


@dataclass
class GuardrailResult:
    passed: bool
    reason: str   # empty string when passed=True


# ---------------------------------------------------------------------------
# Input guardrail
# ---------------------------------------------------------------------------

def check_input(query: str) -> GuardrailResult:
    """LLM-based safety/relevance check — rejects harmful, abusive, or
    completely off-topic queries."""
    llm = ChatSarvam(
        api_key=os.environ.get("SARVAM_API_KEY", ""),
        model_name="sarvam-105b",
        max_tokens=50,
        reasoning_effort="low",
    )
    messages = [
        SystemMessage(content=(
            "You are a safety and relevance filter. Reply with only 'SAFE' or 'UNSAFE'.\n"
            "Mark UNSAFE if the query is harmful, abusive, or completely unrelated "
            "to information retrieval / question answering."
        )),
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)
    verdict = (response.content or "").strip().upper()
    if "UNSAFE" in verdict:
        return GuardrailResult(passed=False, reason="Query flagged as unsafe or off-topic.")
    return GuardrailResult(passed=True, reason="")


# ---------------------------------------------------------------------------
# Grounding guardrail
# ---------------------------------------------------------------------------

def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) >= _MIN_WORD_LEN}


def check_grounding(answer: str, passage_texts: list[str]) -> GuardrailResult:
    """Verify the answer is grounded in the retrieved passages via lexical
    token overlap (no embedding call).

    Splits the answer into sentences; each sentence must have at least
    _GROUNDING_OVERLAP_THRESHOLD of its content words appearing somewhere in
    the retrieved passages. Rejects if more than half the sentences fail.
    """
    if not answer.strip() or not passage_texts:
        return GuardrailResult(passed=False, reason="Empty answer or no passages.")

    sentences = [s.strip() for s in re.split(r"(?<=[.!?।॥])\s+", answer) if len(s.split()) >= 3]
    if not sentences:
        return GuardrailResult(passed=True, reason="")

    context_words: set[str] = set()
    for text in passage_texts:
        context_words |= _content_words(text)

    ungrounded = 0
    for sentence in sentences:
        s_words = _content_words(sentence)
        if not s_words:
            continue
        overlap = len(s_words & context_words) / len(s_words)
        if overlap < _GROUNDING_OVERLAP_THRESHOLD:
            ungrounded += 1

    ungrounded_ratio = ungrounded / len(sentences)
    if ungrounded_ratio > 0.5:
        return GuardrailResult(
            passed=False,
            reason=f"{ungrounded}/{len(sentences)} answer sentences not grounded in retrieved passages.",
        )
    return GuardrailResult(passed=True, reason="")
