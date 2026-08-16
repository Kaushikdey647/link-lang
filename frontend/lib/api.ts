// Same-origin now that the serving flow is Next.js API routes
// (frontend/app/api/*) instead of a separate FastAPI backend.

export interface Passage {
  passage_id: string;
  chunk_type: string;
  text: string;
  is_selected: boolean;
}

export interface Guardrails {
  input_passed: boolean;
  input_reason: string;
  grounding_passed: boolean;
  grounding_reason: string;
}

export interface QueryResult {
  answer: string;
  passages: Passage[];
  guardrails: Guardrails;
}

export interface VoiceResult extends QueryResult {
  transcript: string;
  detected_lang: string;
}

/** Text query — language is auto-detected by the backend. */
export async function queryText(query: string, topK = 5): Promise<QueryResult> {
  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

/** Voice query — language detected from audio by saaras:v3. */
export async function queryVoice(audioBlob: Blob, topK = 5): Promise<VoiceResult> {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.webm");
  form.append("top_k", String(topK));

  const res = await fetch("/api/voice", {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}
