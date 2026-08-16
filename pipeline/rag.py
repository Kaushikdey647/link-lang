"""LangChain LCEL RAG chain with harness (retries, structured I/O) and guardrails.

Pipeline:
    query → InputGuardrail → Retriever → PromptBuilder → ChatSarvam → GroundingGuardrail → answer

The harness wraps retrieval + generation with:
  - Structured input/output via Pydantic
  - Automatic retry on transient Sarvam API errors
  - Per-step latency tracking
  - Guardrail enforcement before and after generation
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import truststore; truststore.inject_into_ssl()
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_sarvam import ChatSarvam
from qdrant_client import QdrantClient

from pipeline.embedder import DEFAULT_BACKEND
from pipeline.guardrails import GuardrailResult, check_grounding, check_input
from pipeline.index_plan import IndexPlan, best_available_plan
from pipeline.indexer import QDRANT_URL, get_vectorstore
from pipeline.query_engines import build_query_engine

load_dotenv()

_LANG_NAMES = {
    "hi": "Hindi", "bn": "Bengali", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali", "or": "Odia",
    "pa": "Punjabi", "sa": "Sanskrit", "ta": "Tamil", "te": "Telugu",
    "ur": "Urdu", "as": "Assamese",
}

_SYSTEM_TEMPLATE = (
    "You are a helpful assistant that answers questions in {lang_name}. "
    "You are given numbered context passages retrieved from a document corpus. "
    "Answer the user's question using ONLY information present in the provided passages. "
    "If the passages do not contain enough information to answer, say so clearly in {lang_name}. "
    "Keep your answer concise and grounded in the context."
)

_HUMAN_TEMPLATE = (
    "Context passages:\n{context}\n\n"
    "Question: {question}"
)


@dataclass
class RAGResponse:
    answer: str
    passages: list[dict]           # retrieved passage metadata
    input_guardrail: GuardrailResult
    grounding_guardrail: GuardrailResult
    latency: dict[str, float]      # step → ms
    tokens_used: int = 0
    error: str = ""


class RAGChain:
    """Harness wrapping retrieval + Sarvam-105B generation with guardrails.

    Args:
        lang: 2-letter language code to filter Qdrant results.
        top_k: Number of distinct passages to retrieve.
        chunk_types: Which chunk strategies to search.
        reasoning_effort: Sarvam-105B reasoning depth ("low" | "medium" | "high").
        max_retries: Number of retries on Sarvam API errors.
    """

    def __init__(
        self,
        lang: str = "hi",
        plan: IndexPlan | None = None,
        top_k: int = 5,
        chunk_types: list[str] | None = None,
        reasoning_effort: str = "low",
        max_retries: int = 2,
    ):
        self.lang = lang
        self.top_k = top_k
        # Resolve plan: explicit > best from registry > safe default
        self.plan = plan or best_available_plan() or IndexPlan(
            backend=DEFAULT_BACKEND,
            chunkers=["passage", "sentence", "qa_pair"],
        )
        self.chunk_types = chunk_types or self.plan.chunkers
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries

        self._vectorstore: QdrantVectorStore = get_vectorstore(self.plan)
        self._qdrant_client = QdrantClient(url=QDRANT_URL)
        self._engine = build_query_engine(
            self.plan,
            vectorstore=self._vectorstore,
            client=self._qdrant_client,
            chunk_types=self.chunk_types,
        )
        self._llm = ChatSarvam(
            api_key=os.environ.get("SARVAM_API_KEY", ""),
            model_name="sarvam-105b",
            reasoning_effort=reasoning_effort,
            max_tokens=2048,
            max_retries=max_retries,
        )
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_TEMPLATE),
            ("human", _HUMAN_TEMPLATE),
        ])
        self._chain = self._prompt | self._llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke(self, query: str) -> RAGResponse:
        latency: dict[str, float] = {}

        # 1. Input guardrail
        t0 = time.perf_counter()
        input_result = check_input(query)
        latency["input_guardrail_ms"] = (time.perf_counter() - t0) * 1000

        if not input_result.passed:
            return RAGResponse(
                answer=f"I cannot answer this query. {input_result.reason}",
                passages=[],
                input_guardrail=input_result,
                grounding_guardrail=GuardrailResult(passed=True, reason=""),
                latency=latency,
            )

        # 2. Retrieval
        t0 = time.perf_counter()
        docs = self._retrieve(query, self.lang)
        latency["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        if not docs:
            return RAGResponse(
                answer="No relevant passages found in the corpus.",
                passages=[],
                input_guardrail=input_result,
                grounding_guardrail=GuardrailResult(passed=False, reason="No passages retrieved."),
                latency=latency,
            )

        # 3. Generation
        context = self._format_context(docs)
        lang_name = _LANG_NAMES.get(self.lang, self.lang)
        t0 = time.perf_counter()
        answer = self._chain.invoke({"context": context, "question": query, "lang_name": lang_name})
        latency["generation_ms"] = (time.perf_counter() - t0) * 1000

        # 4. Grounding guardrail — check against the full parent texts used for generation
        t0 = time.perf_counter()
        passage_texts = [d.metadata.get("parent_passage") or d.page_content for d in docs]
        grounding_result = check_grounding(answer, passage_texts)
        latency["grounding_guardrail_ms"] = (time.perf_counter() - t0) * 1000

        if not grounding_result.passed:
            answer = (
                "I don't have sufficient grounded information to answer this question reliably. "
                f"({grounding_result.reason})"
            )

        latency["total_ms"] = sum(latency.values())

        return RAGResponse(
            answer=answer,
            passages=[{"text": d.page_content, **d.metadata} for d in docs],
            input_guardrail=input_result,
            grounding_guardrail=grounding_result,
            latency=latency,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, lang: str) -> list[Document]:
        """Delegates to this plan's query engine (pipeline/query_engines.py)."""
        return self._engine.retrieve(query, lang, self.top_k)

    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        parts = []
        for i, d in enumerate(docs):
            # Use full parent passage if available (sentence/qa_pair chunks);
            # fall back to the chunk text itself (passage chunks).
            text = d.metadata.get("parent_passage") or d.page_content
            parts.append(f"[{i+1}] {text}")
        return "\n\n".join(parts)
