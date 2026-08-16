"""Guardrails for the RAG pipeline.

Two checks:
  1. InputGuardrail  — rejects off-topic or unsafe queries before retrieval.
  2. GroundingGuardrail — rejects answers not supported by retrieved passages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import truststore; truststore.inject_into_ssl()
from dotenv import load_dotenv
from langchain_sarvam import ChatSarvam
from langchain_core.messages import HumanMessage, SystemMessage

from pipeline.embedder import embed_query, embed_passages

load_dotenv()

# Cosine similarity threshold below which a query is considered off-topic.
# Computed against the mean vector of a small sample of in-domain passages.
_DOMAIN_THRESHOLD = 0.20

# Minimum cosine similarity between answer sentences and the best retrieved passage.
_GROUNDING_THRESHOLD = 0.35


@dataclass
class GuardrailResult:
    passed: bool
    reason: str   # empty string when passed=True


# ---------------------------------------------------------------------------
# Input guardrail
# ---------------------------------------------------------------------------

# Domain centroid is computed lazily from the first batch of indexed passages
# and cached in memory. For a production system this would be stored.
_domain_centroid: np.ndarray | None = None


def set_domain_centroid(passage_texts: list[str]) -> None:
    """Pre-compute and cache the domain centroid from a representative sample."""
    global _domain_centroid
    vecs = np.array(embed_passages(passage_texts))
    _domain_centroid = vecs.mean(axis=0)
    norm = np.linalg.norm(_domain_centroid)
    if norm > 0:
        _domain_centroid /= norm


def ensure_domain_centroid(sample_texts: list[str]) -> None:
    """Idempotent entry point: sets the domain centroid on first call only, so
    repeated RAGChain construction (or multiple languages sharing a process)
    doesn't re-embed a sample every time. No-ops if sample_texts is empty."""
    if _domain_centroid is not None or not sample_texts:
        return
    set_domain_centroid(sample_texts)


def check_input(query: str) -> GuardrailResult:
    """Reject queries that are off-topic or contain unsafe content.

    Strategy:
      1. Cosine similarity of query vector to domain centroid (fast, local).
         If similarity < threshold → off-topic.
      2. If no centroid is cached, falls back to a lightweight LLM safety check.
    """
    if _domain_centroid is not None:
        q_vec = embed_query(query)
        similarity = float(np.dot(q_vec, _domain_centroid))
        if similarity < _DOMAIN_THRESHOLD:
            return GuardrailResult(
                passed=False,
                reason=f"Query appears off-topic (domain similarity={similarity:.2f} < {_DOMAIN_THRESHOLD}).",
            )
        return GuardrailResult(passed=True, reason="")

    # Fallback: LLM-based safety check (only when no centroid is loaded)
    return _llm_safety_check(query)


def _llm_safety_check(query: str) -> GuardrailResult:
    llm = ChatSarvam(
        api_key=os.environ.get("SARVAM_API_KEY", ""),
        model_name="sarvam-105b",
        max_tokens=50,
        reasoning_effort="low",
    )
    messages = [
        SystemMessage(content=(
            "You are a safety filter. Reply with only 'SAFE' or 'UNSAFE'.\n"
            "Mark UNSAFE if the query is harmful, abusive, or completely unrelated "
            "to information retrieval / question answering."
        )),
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)
    verdict = (response.content or "").strip().upper()
    if "UNSAFE" in verdict:
        return GuardrailResult(passed=False, reason="Query flagged as unsafe.")
    return GuardrailResult(passed=True, reason="")


# ---------------------------------------------------------------------------
# Grounding guardrail
# ---------------------------------------------------------------------------

def check_grounding(answer: str, passage_texts: list[str]) -> GuardrailResult:
    """Verify the answer is grounded in the retrieved passages.

    Splits the answer into sentences and checks that each has at least one
    passage with cosine similarity >= GROUNDING_THRESHOLD. If more than half
    the sentences are ungrounded, the answer is rejected.
    """
    if not answer.strip() or not passage_texts:
        return GuardrailResult(passed=False, reason="Empty answer or no passages.")

    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।॥])\s+", answer) if len(s.split()) >= 3]
    if not sentences:
        return GuardrailResult(passed=True, reason="")

    passage_vecs = np.array(embed_passages(passage_texts))
    ungrounded = 0

    for sentence in sentences:
        s_vec = np.array(embed_query(sentence))
        sims = passage_vecs @ s_vec
        if float(sims.max()) < _GROUNDING_THRESHOLD:
            ungrounded += 1

    ungrounded_ratio = ungrounded / len(sentences)
    if ungrounded_ratio > 0.5:
        return GuardrailResult(
            passed=False,
            reason=f"{ungrounded}/{len(sentences)} answer sentences not grounded in retrieved passages.",
        )
    return GuardrailResult(passed=True, reason="")
