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


def transcribe(audio_path: str | Path) -> tuple[str, str]:
    """Transcribe audio, auto-detecting language via saaras:v3.

    Returns:
        (transcript, lang) where lang is a 2-letter ISO 639-1 code (e.g. "hi").
    """
    client = _get_client()
    with open(audio_path, "rb") as f:
        response = client.speech_to_text.transcribe(
            file=f,
            model="saaras:v3",
            language_code="unknown",   # triggers Sarvam LID
        )
    bcp47 = response.language_code or "hi-IN"
    lang  = bcp47.split("-")[0]   # "hi-IN" → "hi"
    return response.transcript, lang


def identify_language(text: str) -> str:
    """Detect the language of a text string.

    Returns a 2-letter ISO 639-1 code; defaults to "hi" on failure.
    """
    try:
        resp = _get_client().text.identify_language(input=text)
        bcp47 = resp.language_code or "hi-IN"
        return bcp47.split("-")[0]
    except Exception:
        return "hi"
