"""CLI entrypoint for indexing. This is the only place indexing runs from —
not the API, not the admin UI (see ui/ingestion_tab.py, which is read-only).

Kept separate from pipeline/indexer.py (where the actual logic lives) on
purpose: --workers > 1 uses ProcessPoolExecutor's default "spawn" start
method, which needs to re-import the worker function (index_language) from
a real module path in each child process. Running pipeline/indexer.py
directly via `python -m pipeline.indexer` would register that module as
__main__, and spawn cannot safely re-import a function whose home module is
__main__ (confirmed empirically — every worker crashed instantly). Because
this script is a different module, pipeline.indexer is always imported
normally, so spawn works.

Usage:
    uv run python -m scripts.index --langs hi bn --backend english --chunkers english_query
    uv run python -m scripts.index --langs all --workers 4 --backend e5 --chunkers passage sentence qa_pair
    uv run python -m scripts.index --langs hi --limit 5000   # quick test run
"""

from __future__ import annotations

import argparse

from constants import LANG_CODE_MAP
from pipeline.embedder import AVAILABLE_BACKENDS, DEFAULT_BACKEND
from pipeline.index_plan import IndexPlan
from pipeline.indexer import run_indexing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["hi"],
                       help="Language codes to index, or 'all' for all 14")
    parser.add_argument("--backend",  default=DEFAULT_BACKEND, choices=AVAILABLE_BACKENDS)
    parser.add_argument("--chunkers", nargs="+", default=["passage", "sentence", "qa_pair"])
    parser.add_argument("--split",    default="train")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1,
                       help="Parallel language processes (default 1 = sequential)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Stop after N passages per language (for quick test runs)")
    args = parser.parse_args()

    langs = list(LANG_CODE_MAP.keys()) if args.langs == ["all"] else args.langs
    plan = IndexPlan(backend=args.backend, chunkers=args.chunkers, split=args.split)
    run_indexing(langs, plan, args.batch_size, args.workers, args.limit)
