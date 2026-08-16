"""Prometheus metrics for the RAG pipeline.

HTTP-level metrics (request count, latency, status codes) are handled
automatically by prometheus-fastapi-instrumentator.

Custom RAG metrics defined here cover the internal pipeline steps that
HTTP metrics can't see: retrieval latency, generation latency, guardrail
outcomes, and per-language request distribution.

Usage:
    from api.metrics import record_rag_result

    record_rag_result(result, lang="hi")
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI

# ---------------------------------------------------------------------------
# Latency histograms — one per pipeline step
# ---------------------------------------------------------------------------

STT_LATENCY = Histogram(
    "rag_stt_latency_seconds",
    "Time spent on Sarvam speech-to-text transcription",
    ["lang"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Time spent on Qdrant ANN search + small-to-big expansion",
    ["lang"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0),
)

GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds",
    "Time spent on Sarvam-105B generation (includes reasoning)",
    ["lang"],
    buckets=(0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0),
)

GUARDRAIL_LATENCY = Histogram(
    "rag_guardrail_latency_seconds",
    "Time spent on input + grounding guardrail checks combined",
    ["lang"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

PIPELINE_LATENCY = Histogram(
    "rag_pipeline_latency_seconds",
    "Total end-to-end latency (retrieval + generation + guardrails)",
    ["lang"],
    buckets=(1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 30.0),
)

# ---------------------------------------------------------------------------
# Counters — guardrail blocks and per-language traffic
# ---------------------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total RAG queries processed",
    ["lang", "endpoint"],          # endpoint: "text" | "voice"
)

INPUT_GUARDRAIL_BLOCKED = Counter(
    "rag_input_guardrail_blocked_total",
    "Queries rejected by the input guardrail (off-topic / unsafe)",
    ["lang"],
)

GROUNDING_GUARDRAIL_BLOCKED = Counter(
    "rag_grounding_guardrail_blocked_total",
    "Answers rejected by the grounding guardrail (not grounded in retrieved passages)",
    ["lang"],
)

PASSAGES_RETRIEVED = Histogram(
    "rag_passages_retrieved",
    "Number of distinct passages returned per query",
    ["lang"],
    buckets=(1, 2, 3, 4, 5, 8, 10, 20),
)

# ---------------------------------------------------------------------------
# Ingestion gauges — updated by the indexing worker in ui/indexing.py
# ---------------------------------------------------------------------------

INDEXING_RUNNING = Gauge(
    "indexing_running",
    "1 if an indexing job is active, 0 otherwise",
)

INDEXING_CHUNKS_DONE = Gauge(
    "indexing_chunks_done",
    "Chunks upserted into Qdrant in the current/last run",
    ["lang", "split"],
)

INDEXING_CHUNKS_TARGET = Gauge(
    "indexing_chunks_target",
    "Estimated total chunks for the current/last run",
    ["lang", "split"],
)

INDEXING_THROUGHPUT = Gauge(
    "indexing_throughput_chunks_per_min",
    "Rolling chunk throughput for the active indexing run",
    ["lang", "split"],
)


# ---------------------------------------------------------------------------
# Convenience recorder — call this at the end of every RAG invoke()
# ---------------------------------------------------------------------------

def record_rag_result(result, lang: str, endpoint: str = "text") -> None:
    """Push all metrics for a completed RAGResponse to Prometheus."""
    latency = result.latency

    REQUESTS_TOTAL.labels(lang=lang, endpoint=endpoint).inc()

    RETRIEVAL_LATENCY.labels(lang=lang).observe(latency.get("retrieval_ms", 0) / 1000)
    GENERATION_LATENCY.labels(lang=lang).observe(latency.get("generation_ms", 0) / 1000)

    guardrail_ms = (
        latency.get("input_guardrail_ms", 0) + latency.get("grounding_guardrail_ms", 0)
    )
    GUARDRAIL_LATENCY.labels(lang=lang).observe(guardrail_ms / 1000)
    PIPELINE_LATENCY.labels(lang=lang).observe(latency.get("total_ms", 0) / 1000)

    PASSAGES_RETRIEVED.labels(lang=lang).observe(len(result.passages))

    if not result.input_guardrail.passed:
        INPUT_GUARDRAIL_BLOCKED.labels(lang=lang).inc()
    if not result.grounding_guardrail.passed:
        GROUNDING_GUARDRAIL_BLOCKED.labels(lang=lang).inc()


# ---------------------------------------------------------------------------
# Instrumentator setup
# ---------------------------------------------------------------------------

def setup_metrics(app: FastAPI) -> None:
    """Attach prometheus-fastapi-instrumentator and expose /metrics."""
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        excluded_handlers=["/metrics", "/health", "/docs", "/openapi.json"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
