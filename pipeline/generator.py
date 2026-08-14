"""Answer generation via Sarvam-105B.

sarvam-105b is a reasoning model: it emits reasoning_content (chain-of-thought)
then content (the final answer). We surface both so callers can log/inspect reasoning.

Latency note: reasoning adds ~1-3s on top of network RTT. The 200ms target covers
retrieval only; full pipeline (retrieval + generation) will be 3-10s end-to-end.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import truststore; truststore.inject_into_ssl()
from dotenv import load_dotenv
from sarvamai import SarvamAI

from pipeline.retriever import RetrievedPassage

load_dotenv()

_client: SarvamAI | None = None

# Language code → display name for the system prompt
_LANG_NAMES: dict[str, str] = {
    "hi": "Hindi", "bn": "Bengali", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali", "or": "Odia",
    "pa": "Punjabi", "sa": "Sanskrit", "ta": "Tamil", "te": "Telugu",
    "ur": "Urdu", "as": "Assamese",
}


@dataclass
class GenerationResult:
    answer: str
    reasoning: str           # chain-of-thought (for logging/debugging)
    passages_used: list[str] # passage_ids that were injected as context
    latency_ms: float
    finish_reason: str
    tokens_used: int


def _get_client() -> SarvamAI:
    global _client
    if _client is None:
        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise RuntimeError("SARVAM_API_KEY not set")
        _client = SarvamAI(api_subscription_key=key)
    return _client


def _build_context(passages: list[RetrievedPassage]) -> str:
    parts = []
    for i, p in enumerate(passages, 1):
        parts.append(f"[{i}] {p.text}")
    return "\n\n".join(parts)


def _system_prompt(lang: str) -> str:
    lang_name = _LANG_NAMES.get(lang, lang)
    return (
        f"You are a helpful assistant that answers questions in {lang_name}. "
        "You are given numbered context passages retrieved from a document corpus. "
        "Answer the user's question using ONLY information present in the provided passages. "
        "If the passages do not contain enough information to answer, say so clearly. "
        "Keep your answer concise and grounded in the context. "
        "Do not use any external knowledge beyond what is in the passages."
    )


def generate(
    query: str,
    passages: list[RetrievedPassage],
    lang: str,
    *,
    reasoning_effort: str = "low",
    max_tokens: int = 2048,
) -> GenerationResult:
    """Generate an answer grounded in the retrieved passages.

    Args:
        query: The user's question (in the target language).
        passages: Retrieved passages from Qdrant (ordered by score descending).
        lang: 2-letter language code — used to set the response language.
        reasoning_effort: "low" | "medium" | "high". Low reduces latency.
        max_tokens: Upper bound on total tokens (reasoning + answer combined).
    """
    if not passages:
        return GenerationResult(
            answer="संदर्भ में पर्याप्त जानकारी नहीं है।" if lang == "hi" else "No relevant context found.",
            reasoning="",
            passages_used=[],
            latency_ms=0.0,
            finish_reason="no_context",
            tokens_used=0,
        )

    context = _build_context(passages)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    client = _get_client()
    t0 = time.perf_counter()

    resp = client.chat.completions(
        model="sarvam-105b",
        messages=[
            {"role": "system", "content": _system_prompt(lang)},
            {"role": "user", "content": user_message},
        ],
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    choice = resp.choices[0]

    return GenerationResult(
        answer=choice.message.content or "",
        reasoning=choice.message.reasoning_content or "",
        passages_used=[p.passage_id for p in passages],
        latency_ms=latency_ms,
        finish_reason=choice.finish_reason or "",
        tokens_used=resp.usage.total_tokens if resp.usage else 0,
    )
