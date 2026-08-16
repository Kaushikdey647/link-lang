from fastapi import APIRouter, HTTPException
from api.models import QueryRequest, QueryResponse, LatencyBreakdown
from pipeline.rag import RAGChain
from pipeline.index_plan import get_plan_by_collection, best_available_plan
from api.metrics import record_rag_result
from stt import identify_language

router = APIRouter(prefix="/query", tags=["query"])

# Cache one RAGChain per (lang, collection_name) combo
_chains: dict[str, RAGChain] = {}


def _get_chain(lang: str, collection: str | None = None,
               chunk_types: list[str] | None = None) -> RAGChain:
    plan = (
        get_plan_by_collection(collection)
        if collection else best_available_plan()
    )
    if plan is None:
        raise HTTPException(
            status_code=503,
            detail="No indexed plan found. Index at least one language via /admin first.",
        )
    key = f"{lang}:{plan.collection_name}"
    if key not in _chains:
        _chains[key] = RAGChain(lang=lang, plan=plan, chunk_types=chunk_types)
    return _chains[key]


@router.post("", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Run the full RAG pipeline for a text query.

    - `lang`: auto-detected when omitted.
    - `collection`: which indexed plan to query; uses best available when omitted.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    lang  = req.lang or identify_language(req.query)
    chain = _get_chain(lang, req.collection, req.chunk_types)
    result = chain.invoke(req.query)
    record_rag_result(result, lang=lang, endpoint="text")

    return QueryResponse(
        answer=result.answer,
        passages=result.passages,
        latency=LatencyBreakdown(**result.latency),
        guardrails={
            "input_passed":    result.input_guardrail.passed,
            "input_reason":    result.input_guardrail.reason,
            "grounding_passed": result.grounding_guardrail.passed,
            "grounding_reason": result.grounding_guardrail.reason,
        },
    )
