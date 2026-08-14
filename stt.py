"""Speech-to-text via Sarvam Saaras v3."""

import os
from pathlib import Path

import truststore; truststore.inject_into_ssl()
from sarvamai import SarvamAI

_client: SarvamAI | None = None


def _get_client() -> SarvamAI:
    global _client
    if _client is None:
        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise RuntimeError("Set SARVAM_API_KEY environment variable")
        _client = SarvamAI(api_subscription_key=key)
    return _client


def transcribe(audio_path: str | Path, language_code: str = "unknown") -> str:
    """Transcribe an audio file, returning the transcript string.

    Args:
        audio_path: Path to audio file (WAV/MP3/OGG/FLAC/etc., best at 16kHz).
        language_code: BCP-47 code e.g. "hi-IN", "ta-IN". "unknown" = auto-detect.
    """
    client = _get_client()
    with open(audio_path, "rb") as f:
        response = client.speech_to_text.transcribe(
            file=f,
            model="saaras:v3",
            language_code=language_code,
        )
    return response.transcript
