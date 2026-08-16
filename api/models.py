from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    query: str = Field(..., description="User's question in any supported language")
    lang: Optional[str] = Field(
        None,
        description="2-letter ISO 639-1 code (hi, bn, ta …). Omit to auto-detect.",
    )
    top_k: int = Field(5, ge=1, le=20)
    chunk_types: Optional[list[str]] = Field(
        None,
        description="Chunk strategies to search: passage, sentence, qa_pair. Defaults to plan default.",
    )
    collection: Optional[str] = Field(
        None,
        description="Qdrant collection name to query (e.g. msmarco_xi__english__english_query__train). "
                    "Omit to use the best available indexed plan.",
    )


class PassageResult(BaseModel):
    passage_id: str
    chunk_type: str
    text: str
    score: Optional[float] = None
    is_selected: Optional[bool] = None


class LatencyBreakdown(BaseModel):
    input_guardrail_ms: float
    retrieval_ms: float
    generation_ms: float
    grounding_guardrail_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    answer: str
    passages: list[dict]
    latency: LatencyBreakdown
    guardrails: dict[str, bool | str]   # passed + reason for each guardrail


class VoiceQueryResponse(QueryResponse):
    transcript: str
    detected_lang: str   # 2-letter code returned by saaras LID
