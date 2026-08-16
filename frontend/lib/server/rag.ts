import "server-only";
import { retrieve, type RetrievedDoc } from "./retrieval";
import { checkInput, checkGrounding, type GuardrailResult } from "./guardrails";
import { chatCompletion } from "./sarvam";

/**
 * Port of pipeline/rag.py::RAGChain — same 4-stage harness (input guardrail
 * → retrieve → generate → grounding guardrail), same per-stage try/catch
 * (each stage fails closed into a structured response instead of throwing),
 * same per-stage latency timing.
 */

const LANG_NAMES: Record<string, string> = {
  hi: "Hindi", bn: "Bengali", gu: "Gujarati", kn: "Kannada",
  ml: "Malayalam", mr: "Marathi", ne: "Nepali", or: "Odia",
  pa: "Punjabi", sa: "Sanskrit", ta: "Tamil", te: "Telugu",
  ur: "Urdu", as: "Assamese",
};

const SYSTEM_TEMPLATE = (langName: string) =>
  `You are a helpful assistant that answers questions in ${langName}. ` +
  "You are given numbered context passages retrieved from a document corpus. " +
  "Answer the user's question using ONLY information present in the provided passages. " +
  `If the passages do not contain enough information to answer, say so clearly in ${langName}. ` +
  "Keep your answer concise and grounded in the context.";

export interface RagResponse {
  answer: string;
  passages: Array<{ text: string } & Record<string, unknown>>;
  inputGuardrail: GuardrailResult;
  groundingGuardrail: GuardrailResult;
  latency: Record<string, number>; // step -> ms
  error?: string;
}

function formatContext(docs: RetrievedDoc[]): string {
  return docs
    .map((d, i) => {
      // english_query chunks embed the English question but return the
      // vernacular passage (parent_passage) as LLM context.
      const text = (d.metadata.parent_passage as string) || d.pageContent;
      return `[${i + 1}] ${text}`;
    })
    .join("\n\n");
}

function toPassages(docs: RetrievedDoc[]) {
  return docs.map((d) => ({ text: d.pageContent, ...d.metadata }));
}

export interface RagOptions {
  lang?: string;
  topK?: number;
  chunkTypes?: string[];
  reasoningEffort?: "low" | "medium" | "high";
}

export async function ragInvoke(query: string, opts: RagOptions = {}): Promise<RagResponse> {
  const lang = opts.lang ?? "hi";
  const topK = opts.topK ?? 5;
  const chunkTypes = opts.chunkTypes ?? ["english_query"];
  const reasoningEffort = opts.reasoningEffort ?? "low";

  const latency: Record<string, number> = {};

  // 1. Input guardrail — fails closed rather than propagating, since an
  // unhandled error here would otherwise skip safety screening entirely.
  let t0 = performance.now();
  let inputResult: GuardrailResult;
  try {
    inputResult = await checkInput(query);
  } catch (e) {
    inputResult = { passed: false, reason: `Input guardrail check failed: ${e}` };
  }
  latency.input_guardrail_ms = performance.now() - t0;

  if (!inputResult.passed) {
    return {
      answer: `I cannot answer this query. ${inputResult.reason}`,
      passages: [],
      inputGuardrail: inputResult,
      groundingGuardrail: { passed: true, reason: "" },
      latency,
    };
  }

  // 2. Retrieval
  t0 = performance.now();
  let docs: RetrievedDoc[];
  try {
    docs = await retrieve(query, lang, topK, chunkTypes);
  } catch (e) {
    latency.retrieval_ms = performance.now() - t0;
    latency.total_ms = Object.values(latency).reduce((a, b) => a + b, 0);
    return {
      answer: "I ran into an error trying to retrieve context for this question — please try again.",
      passages: [],
      inputGuardrail: inputResult,
      groundingGuardrail: { passed: false, reason: "Retrieval failed." },
      latency,
      error: `retrieval_error: ${e}`,
    };
  }
  latency.retrieval_ms = performance.now() - t0;

  if (docs.length === 0) {
    return {
      answer: "No relevant passages found in the corpus.",
      passages: [],
      inputGuardrail: inputResult,
      groundingGuardrail: { passed: false, reason: "No passages retrieved." },
      latency,
    };
  }

  // 3. Generation
  const context = formatContext(docs);
  const langName = LANG_NAMES[lang] ?? lang;
  t0 = performance.now();
  let answer: string;
  try {
    answer = await chatCompletion(
      [
        { role: "system", content: SYSTEM_TEMPLATE(langName) },
        { role: "user", content: `Context passages:\n${context}\n\nQuestion: ${query}` },
      ],
      { maxTokens: 2048, reasoningEffort },
    );
  } catch (e) {
    latency.generation_ms = performance.now() - t0;
    latency.total_ms = Object.values(latency).reduce((a, b) => a + b, 0);
    return {
      answer: "I ran into an error trying to generate an answer for this question — please try again.",
      passages: toPassages(docs),
      inputGuardrail: inputResult,
      groundingGuardrail: { passed: false, reason: "Generation failed." },
      latency,
      error: `generation_error: ${e}`,
    };
  }
  latency.generation_ms = performance.now() - t0;

  // 4. Grounding guardrail — check against the full parent texts used for
  // generation. Fails closed rather than propagating, so a transient error
  // doesn't discard an already-generated answer.
  t0 = performance.now();
  const passageTexts = docs.map((d) => (d.metadata.parent_passage as string) || d.pageContent);
  let groundingResult: GuardrailResult;
  try {
    groundingResult = checkGrounding(answer, passageTexts);
  } catch (e) {
    groundingResult = { passed: false, reason: `Grounding check failed: ${e}` };
  }
  latency.grounding_guardrail_ms = performance.now() - t0;

  if (!groundingResult.passed) {
    answer =
      "I don't have sufficient grounded information to answer this question reliably. " +
      `(${groundingResult.reason})`;
  }

  latency.total_ms = Object.values(latency).reduce((a, b) => a + b, 0);

  return {
    answer,
    passages: toPassages(docs),
    inputGuardrail: inputResult,
    groundingGuardrail: groundingResult,
    latency,
  };
}

/** Retrieval-only path — the sub-pipeline PROBLEM-STATEMENT.md's <200ms
 * latency target applies to. Skips guardrails/generation entirely. */
export async function retrieveOnly(
  query: string,
  lang = "hi",
  topK = 5,
  chunkTypes: string[] = ["english_query"],
): Promise<RetrievedDoc[]> {
  return retrieve(query, lang, topK, chunkTypes);
}
