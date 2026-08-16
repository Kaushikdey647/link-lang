import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.models import LatencyBreakdown, VoiceQueryResponse
from api.routes.query import _get_chain
from api.metrics import record_rag_result, STT_LATENCY
from stt import transcribe

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("", response_model=VoiceQueryResponse)
async def voice_query(
    audio: UploadFile = File(..., description="Audio file (WAV/MP3, best at 16kHz)"),
    top_k: int = Form(5),
    collection: str | None = Form(None, description="Collection to query; omit for best available"),
) -> VoiceQueryResponse:
    """Transcribe audio via Sarvam saaras:v3 (auto-detects language), then run RAG."""
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()
        transcript, lang = transcribe(tmp_path)   # LID + STT in one call
        STT_LATENCY.labels(lang=lang).observe(time.perf_counter() - t0)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"STT failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not transcript.strip():
        raise HTTPException(status_code=422, detail="STT returned empty transcript.")

    try:
        chain = _get_chain(lang, collection)
        result = chain.invoke(transcript)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Query pipeline unavailable: {e}")
    record_rag_result(result, lang=lang, endpoint="voice")

    return VoiceQueryResponse(
        transcript=transcript,
        detected_lang=lang,
        answer=result.answer,
        passages=result.passages,
        latency=LatencyBreakdown(**result.latency),
        guardrails={
            "input_passed": result.input_guardrail.passed,
            "input_reason": result.input_guardrail.reason,
            "grounding_passed": result.grounding_guardrail.passed,
            "grounding_reason": result.grounding_guardrail.reason,
        },
    )
