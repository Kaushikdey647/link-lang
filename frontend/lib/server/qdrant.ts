import "server-only";
import { QdrantClient } from "@qdrant/js-client-rest";

/**
 * Qdrant connection + live-collection resolution.
 *
 * Ports pipeline/indexer.py's QDRANT_URL/QDRANT_API_KEY constants and the
 * registry-based collection lookup — but the "which collection is live"
 * resolution is cached in-memory per warm serverless instance (a module
 * scoped variable) instead of persisted to a local registry.json file. That
 * file is per-machine, local-filesystem state; a serverless deploy's
 * filesystem doesn't survive across invocations, so this looks the
 * collection up once per warm instance and re-resolves on cold start.
 *
 * Ingestion (scripts/index.py, staying Python) is the only writer of data —
 * this module only ever reads.
 */

const QDRANT_URL = process.env.QDRANT_CLUSTER_ENDPOINT || "http://localhost:6333";
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;

// Matches pipeline/indexer.py's MINILM_INFERENCE_MODEL/BM25_INFERENCE_MODEL —
// exact Qdrant Cloud inference model registry identifiers, casing matters.
export const MINILM_INFERENCE_MODEL = "sentence-transformers/all-minilm-l6-v2";
export const BM25_INFERENCE_MODEL = "qdrant/bm25";
export const SPARSE_VECTOR_NAME = "bm25";

// IndexPlan(backend="english", chunkers=["english_query"], split="train").collection_name
// — the one supported plan (see CHANGELOG.md's single-strategy collapse).
export const COLLECTION_NAME = "msmarco_xi__english__english_query__train";

let client: QdrantClient | null = null;

export function getQdrantClient(): QdrantClient {
  if (!client) {
    client = new QdrantClient({ url: QDRANT_URL, apiKey: QDRANT_API_KEY });
  }
  return client;
}

// Cached per warm instance — populated on first use, reset on cold start.
let cachedCollection: string | null = null;

/** Resolve + cache the live collection name. Throws if it doesn't exist or is empty. */
export async function getLiveCollection(): Promise<string> {
  if (cachedCollection) return cachedCollection;

  const client = getQdrantClient();
  const { collections } = await client.getCollections();
  const found = collections.some((c) => c.name === COLLECTION_NAME);
  if (!found) {
    throw new Error(
      `Collection "${COLLECTION_NAME}" not found in Qdrant — has it been indexed? See INDEXING.md.`,
    );
  }
  const countResult = await client.count(COLLECTION_NAME, { exact: false });
  if (countResult.count === 0) {
    throw new Error(`Collection "${COLLECTION_NAME}" exists but has no points.`);
  }
  cachedCollection = COLLECTION_NAME;
  return cachedCollection;
}

/** Non-throwing readiness check for /api/health. */
export async function isLive(): Promise<boolean> {
  try {
    await getLiveCollection();
    return true;
  } catch {
    return false;
  }
}
