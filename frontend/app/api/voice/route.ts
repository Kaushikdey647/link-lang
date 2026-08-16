import { NextRequest, NextResponse } from "next/server";
import { ragInvoke } from "@/lib/server/rag";
import { transcribe } from "@/lib/server/sarvam";

/** Port of POST /voice (api/routes/voice.py). Transcribes audio via Sarvam
 * saaras:v3 (auto-detects language), then runs the same RAG path as /query. */
export async function POST(req: NextRequest) {
  const form = await req.formData();
  const audio = form.get("audio");
  const topK = Number(form.get("top_k") ?? 5);

  if (!(audio instanceof Blob)) {
    return NextResponse.json({ detail: "audio field is required" }, { status: 400 });
  }

  let transcript: string;
  let lang: string;
  try {
    const result = await transcribe(audio, "recording.webm");
    transcript = result.transcript;
    lang = result.lang;
  } catch (e) {
    return NextResponse.json({ detail: `STT failed: ${e}` }, { status: 502 });
  }

  if (!transcript.trim()) {
    return NextResponse.json({ detail: "STT returned empty transcript." }, { status: 422 });
  }

  try {
    const result = await ragInvoke(transcript, { lang, topK });

    return NextResponse.json({
      transcript,
      detected_lang: lang,
      answer: result.answer,
      passages: result.passages,
      latency: result.latency,
      guardrails: {
        input_passed: result.inputGuardrail.passed,
        input_reason: result.inputGuardrail.reason,
        grounding_passed: result.groundingGuardrail.passed,
        grounding_reason: result.groundingGuardrail.reason,
      },
    });
  } catch (e) {
    return NextResponse.json(
      { detail: `Query pipeline unavailable: ${e}` },
      { status: 503 },
    );
  }
}
