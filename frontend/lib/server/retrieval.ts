import "server-only";
import {
  getQdrantClient,
  getLiveCollection,
  MINILM_INFERENCE_MODEL,
  BM25_INFERENCE_MODEL,
  SPARSE_VECTOR_NAME,
} from "./qdrant";
import { translateToEnglish } from "./sarvam";

/**
 * Port of pipeline/query_engines.py::EnglishPivotQueryEngine — RRF fusion of:
 *   - dense: English-translated query vs. english_query embeddings
 *   - sparse (BM25/IDF): original vernacular query vs. parent_passage text
 *
 * Both vectors are computed server-side by Qdrant Cloud inference
 * ({text, model} in place of a raw vector) — no local model inference. The
 * vernacular query is never discarded from the caller's perspective — only
 * the dense prefetch sees the English-translated version.
 */

export interface RetrievedDoc {
  pageContent: string;
  metadata: Record<string, unknown>;
}

function dedupe(hits: RetrievedDoc[], topK: number): RetrievedDoc[] {
  const seen = new Map<string, RetrievedDoc>();
  for (const doc of hits) {
    const pid = (doc.metadata.passage_id ?? doc.metadata.chunk_id) as string | undefined;
    if (pid && !seen.has(pid)) seen.set(pid, doc);
    if (seen.size >= topK) break;
  }
  return [...seen.values()];
}

function buildFilter(lang: string, chunkTypes: string[]) {
  return {
    must: [
      { key: "metadata.lang", match: { value: lang } },
      { key: "metadata.chunk_type", match: { any: chunkTypes } },
    ],
  };
}

export async function retrieve(
  query: string,
  lang: string,
  topK: number,
  chunkTypes: string[] = ["english_query"],
): Promise<RetrievedDoc[]> {
  const client = getQdrantClient();
  const collection = await getLiveCollection();
  const englishQuery = await translateToEnglish(query, lang);
  const qfilter = buildFilter(lang, chunkTypes);

  const result = await client.query(collection, {
    prefetch: [
      {
        // no `using` — targets the default/unnamed dense vector
        query: { text: englishQuery, model: MINILM_INFERENCE_MODEL },
        filter: qfilter,
        limit: topK * 4,
      },
      {
        query: { text: query, model: BM25_INFERENCE_MODEL },
        using: SPARSE_VECTOR_NAME,
        filter: qfilter,
        limit: topK * 4,
      },
    ],
    query: { fusion: "rrf" },
    limit: topK * 4,
    with_payload: true,
  });

  const docs: RetrievedDoc[] = result.points.map((p) => {
    const payload = (p.payload ?? {}) as Record<string, unknown>;
    return {
      pageContent: (payload.page_content as string) ?? "",
      metadata: (payload.metadata as Record<string, unknown>) ?? {},
    };
  });

  return dedupe(docs, topK);
}
