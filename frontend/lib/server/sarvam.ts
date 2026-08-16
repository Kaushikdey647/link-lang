import "server-only";

/**
 * Thin fetch wrappers around Sarvam's REST API — ports stt.py, the translate
 * call in pipeline/query_engines.py, and the ChatSarvam usage in
 * pipeline/rag.py / pipeline/guardrails.py. Contracts confirmed directly
 * against docs.sarvam.ai (not guessed) — see CHANGELOG.md.
 *
 * Note: STT/translate/language-ID use the `api-subscription-key` header;
 * chat completions uses `Authorization: Bearer` — this isn't a mistake,
 * it's what each endpoint's own reference page documents.
 */

const SARVAM_API_KEY = process.env.SARVAM_API_KEY || "";
const SARVAM_BASE = "https://api.sarvam.ai";

async function sarvamFetch(path: string, init: RequestInit): Promise<Response> {
  const res = await fetch(`${SARVAM_BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Sarvam ${path} failed: ${res.status} ${body}`);
  }
  return res;
}

export interface TranscribeResult {
  transcript: string;
  lang: string; // 2-letter ISO 639-1 code, e.g. "hi"
}

/** "hi-IN" -> "hi"; defaults to "hi" if missing (matches stt.py's fallback). */
function to2Letter(bcp47: string | null | undefined): string {
  return (bcp47 || "hi-IN").split("-")[0];
}

/** Speech-to-text via Saaras v3, auto-detecting language. Max 30s audio (Sarvam's own limit). */
export async function transcribe(audio: Blob, filename = "audio.wav"): Promise<TranscribeResult> {
  const form = new FormData();
  form.append("file", audio, filename);
  form.append("model", "saaras:v3");
  form.append("language_code", "unknown"); // triggers Sarvam's auto-LID, matches stt.py

  const res = await sarvamFetch("/speech-to-text", {
    method: "POST",
    headers: { "api-subscription-key": SARVAM_API_KEY },
    body: form,
  });
  const data = await res.json();
  return { transcript: data.transcript ?? "", lang: to2Letter(data.language_code) };
}

/** Text language identification — used to route text queries without an explicit lang.
 * Defaults to "hi" on any failure (matches stt.py::identify_language). */
export async function identifyLanguage(text: string): Promise<string> {
  try {
    const res = await sarvamFetch("/text-lid", {
      method: "POST",
      headers: {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input: text }),
    });
    const data = await res.json();
    return to2Letter(data.language_code);
  } catch {
    return "hi";
  }
}

// BCP-47 codes for Sarvam's API — mirrors pipeline/query_engines.py::_SARVAM_LANG.
const SARVAM_LANG: Record<string, string> = {
  hi: "hi-IN", bn: "bn-IN", gu: "gu-IN", kn: "kn-IN",
  ml: "ml-IN", mr: "mr-IN", ne: "ne-IN", or: "od-IN",
  pa: "pa-IN", sa: "sa-IN", ta: "ta-IN", te: "te-IN",
  ur: "ur-IN", as: "as-IN",
};

/** Translate a vernacular query to English (used for the english-pivot dense search). */
export async function translateToEnglish(text: string, lang: string): Promise<string> {
  const res = await sarvamFetch("/translate", {
    method: "POST",
    headers: {
      "api-subscription-key": SARVAM_API_KEY,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      input: text,
      source_language_code: SARVAM_LANG[lang] ?? "auto",
      target_language_code: "en-IN",
      model: "sarvam-translate:v1",
    }),
  });
  const data = await res.json();
  return data.translated_text ?? text;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatCompletionOptions {
  maxTokens?: number;
  reasoningEffort?: "low" | "medium" | "high";
}

/** Sarvam-105B chat completion — used for both generation and the guardrail safety check. */
export async function chatCompletion(
  messages: ChatMessage[],
  opts: ChatCompletionOptions = {},
): Promise<string> {
  const res = await sarvamFetch("/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${SARVAM_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "sarvam-105b",
      messages,
      max_tokens: opts.maxTokens ?? 2048,
      reasoning_effort: opts.reasoningEffort ?? "low",
    }),
  });
  const data = await res.json();
  return data.choices?.[0]?.message?.content ?? "";
}
