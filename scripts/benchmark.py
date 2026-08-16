"""Latency benchmark: exact P50/P70/P100 over real MSMARCO-XI queries.

PROBLEM-STATEMENT.md requires reporting P50/P70/P100 latency across a
reasonable number of test queries, and that the sub-pipeline "chunking +
vector DB retrieval" completes under 200ms. Prometheus (api/metrics.py,
ui/metrics_tab.py) only gives bucket-approximate p50/p90/p99 — this script
computes exact order-statistic percentiles over a fixed query set instead.

Measures two numbers per query:
  - retrieval_ms: RAGChain.retrieve_only() — embed_query (+ translation for
                  the english-pivot plan) + Qdrant ANN search. This is the
                  sub-pipeline the <200ms target applies to.
  - total_ms:     full RAGChain.invoke() (guardrails + retrieval + Sarvam-105B
                  generation), reported for completeness. Dominated by
                  generation's reasoning latency (see CHUNKING.md's latency
                  budget table) — expect several seconds, not milliseconds.

Usage:
    uv run python -m scripts.benchmark --lang hi --n-queries 300
    uv run python -m scripts.benchmark --collection msmarco_xi__english__english_query__train --lang hi
    uv run python -m scripts.benchmark --lang hi --retrieval-only   # skip generation, fast
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dataset import iter_queries, load_language
from pipeline.index_plan import best_available_plan, get_plan_by_collection
from pipeline.rag import RAGChain

_BENCH_DIR = Path("benchmarks")


def _percentiles(samples: list[float]) -> dict:
    arr = np.array(samples)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p70": float(np.percentile(arr, 70)),
        "p100": float(np.percentile(arr, 100)),
        "mean": float(arr.mean()),
        "n": len(samples),
    }


def run_benchmark(
    lang: str, collection: str | None, split: str, n_queries: int, retrieval_only: bool,
) -> dict:
    plan = get_plan_by_collection(collection) if collection else best_available_plan()
    if plan is None:
        raise SystemExit("No indexed plan found — index at least one language first (see INDEXING.md).")

    chain = RAGChain(lang=lang, plan=plan)

    ds = load_language(lang, splits=(split,))
    queries = [q.query for q in iter_queries(ds[split], lang) if q.query.strip()][:n_queries]
    if not queries:
        raise SystemExit(f"No queries found for lang={lang!r} split={split!r}.")

    # Warmup — first call pays for cold model/client loads (embedding model,
    # sparse BM25 model, Sarvam client init); excluded from measured samples.
    chain.invoke(queries[0])

    retrieval_ms: list[float] = []
    total_ms: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        chain.retrieve_only(q)
        retrieval_ms.append((time.perf_counter() - t0) * 1000)

        if not retrieval_only:
            t0 = time.perf_counter()
            chain.invoke(q)
            total_ms.append((time.perf_counter() - t0) * 1000)

    summary = {
        "collection": plan.collection_name,
        "lang": lang,
        "n_queries": len(queries),
        "retrieval_only_ms": _percentiles(retrieval_ms),
    }
    if total_ms:
        summary["full_pipeline_ms"] = _percentiles(total_ms)
    return summary


def _print_summary(summary: dict) -> None:
    print(f"Collection: {summary['collection']}  lang={summary['lang']}  n={summary['n_queries']}")
    for key in ("retrieval_only_ms", "full_pipeline_ms"):
        p = summary.get(key)
        if not p:
            continue
        print(
            f"  {key:18s}  P50={p['p50']:.1f}ms  P70={p['p70']:.1f}ms  "
            f"P100={p['p100']:.1f}ms  mean={p['mean']:.1f}ms"
        )
    target_p100 = summary["retrieval_only_ms"]["p100"]
    verdict = "OK" if target_p100 < 200 else "OVER"
    print(f"\n<200ms retrieval target: P100={target_p100:.1f}ms [{verdict}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--collection", default=None, help="Collection name; defaults to best_available_plan()")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--n-queries", type=int, default=300)
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip full-pipeline (generation) timing — fast, retrieval percentiles only")
    args = parser.parse_args()

    summary = run_benchmark(args.lang, args.collection, args.split, args.n_queries, args.retrieval_only)
    _print_summary(summary)

    _BENCH_DIR.mkdir(exist_ok=True)
    out_path = _BENCH_DIR / f"latency_{summary['collection']}_{summary['lang']}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Written: {out_path}")
