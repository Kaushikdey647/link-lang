import { NextRequest, NextResponse } from "next/server";
import { ragInvoke } from "@/lib/server/rag";
import { identifyLanguage } from "@/lib/server/sarvam";

/** Port of POST /query (api/routes/query.py). */
export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => null);
  const query: string | undefined = body?.query;
  if (!query || !query.trim()) {
    return NextResponse.json({ detail: "query must not be empty" }, { status: 400 });
  }

  const topK: number = body?.top_k ?? 5;
  const chunkTypes: string[] | undefined = body?.chunk_types;

  let lang: string = body?.lang;
  try {
    if (!lang) lang = await identifyLanguage(query);

    const result = await ragInvoke(query, { lang, topK, chunkTypes });

    return NextResponse.json({
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
