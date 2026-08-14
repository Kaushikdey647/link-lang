"""Quick smoke-test for Sarvam API — validates auth and STT setup.

Costs: language-ID call is text-only (negligible). STT call uses a
1-second silent WAV generated in-memory (shortest billable unit).
Run with: uv run python test_sarvam.py
"""

import io
import os
import ssl
import struct
import wave

import truststore
from dotenv import load_dotenv
from sarvamai import SarvamAI

# Inject macOS system keychain (picks up corporate proxy certs)
truststore.inject_into_ssl()

load_dotenv()

key = os.environ.get("SARVAM_API_KEY")
if not key:
    raise SystemExit("SARVAM_API_KEY not found in .env")

client = SarvamAI(api_subscription_key=key)


def _silent_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a minimal silent WAV in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        n_frames = int(sample_rate * duration_s)
        w.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return buf.getvalue()


# --- Test 1: auth via language-ID (text only, no audio cost) ---
print("Test 1: Language identification (text)...")
lid = client.text.identify_language(input="नमस्ते आप कैसे हैं")
print(f"  language_code={lid.language_code}  script_code={lid.script_code}")
assert lid.language_code == "hi-IN", f"Unexpected: {lid.language_code}"
print("  PASS")

# --- Test 2: STT on 1-second silent WAV ---
print("Test 2: Speech-to-text (1s silent WAV)...")
wav_bytes = _silent_wav_bytes()
response = client.speech_to_text.transcribe(
    file=("silent.wav", wav_bytes, "audio/wav"),
    model="saaras:v3",
    language_code="hi-IN",
)
print(f"  transcript={response.transcript!r}  language={getattr(response, 'language_code', 'n/a')}")
print("  PASS (empty transcript expected for silent audio)")

print("\nAll tests passed. Sarvam API is wired up correctly.")
