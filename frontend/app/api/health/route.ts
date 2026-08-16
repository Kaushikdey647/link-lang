import { NextResponse } from "next/server";
import { isLive } from "@/lib/server/qdrant";

/** Port of GET /health (api/app.py). */
export async function GET() {
  const ready = await isLive();
  return NextResponse.json({ status: ready ? "ok" : "degraded", qdrant: ready });
}
