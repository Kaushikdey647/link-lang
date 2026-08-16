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
    uv run python -m scripts.index --langs hi bn
    uv run python -m scripts.index --langs all --workers 4
    uv run python -m scripts.index --langs hi --limit 5000   # quick test run
    uv run python -m scripts.index --langs hi --strategy qa_pair

--strategy selects one of the two registered (backend, chunkers) plans (see
pipeline/index_plan.py) — each is its own Qdrant collection, so switching
strategies never touches the other collection's data.
"""

from __future__ import annotations

import argparse

from constants import LANG_CODE_MAP
from pipeline.index_plan import IndexPlan
from pipeline.indexer import run_indexing

# strategy name -> (backend, chunker)
STRATEGIES: dict[str, tuple[str, str]] = {
    "english_query": ("english", "english_query"),
    "qa_pair":        ("multilingual_e5_small", "qa_pair"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["hi"],
                       help="Language codes to index, or 'all' for all 14")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="english_query",
                       help="Which (backend, chunker) plan to index into")
    parser.add_argument("--split",    default="train")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1,
                       help="Parallel language processes (default 1 = sequential)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Stop after N passages per language (for quick test runs)")
    args = parser.parse_args()

    langs = list(LANG_CODE_MAP.keys()) if args.langs == ["all"] else args.langs
    backend, chunker = STRATEGIES[args.strategy]
    plan = IndexPlan(backend=backend, chunkers=[chunker], split=args.split)
    run_indexing(langs, plan, args.batch_size, args.workers, args.limit)
