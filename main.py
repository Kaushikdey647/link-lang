"""Default ingestion entrypoint — equivalent to:

    uv run python -m scripts.index \\
      --langs as bn gu hi kn ml mr ne or pa sa ta ur \\
      --strategy qa_pair \\
      --limit 1000000 \\
      --batch-size 512 \\
      --workers 4

For ad-hoc flags, use `python -m scripts.index` instead.
Telugu (`te`) is omitted: there is no train parquet on the Hub.
"""

from __future__ import annotations

from pipeline.index_plan import IndexPlan
from pipeline.indexer import run_indexing

LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "ur"]

if __name__ == "__main__":
    plan = IndexPlan(backend="multilingual_e5_small", chunkers=["qa_pair"], split="train")
    run_indexing(LANGS, plan, batch_size=512, workers=4, limit=1_000_000)
