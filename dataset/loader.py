import glob
import os
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, load_dataset

from constants import DATASET_NAME, LANG_CODE_MAP

_HUB_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--"
    + DATASET_NAME.replace("/", "--")
    + "/snapshots"
)


def _snapshot() -> str | None:
    if not os.path.isdir(_HUB_CACHE):
        return None
    snaps = sorted(os.listdir(_HUB_CACHE))
    if not snaps:
        return None
    return os.path.join(_HUB_CACHE, snaps[-1])


def _parquet_relpath(lang: str, split: str) -> str:
    """Hub-relative path for one language shard, e.g. train/hintrain.parquet."""
    prefix = LANG_CODE_MAP.get(lang)
    if prefix is None:
        raise ValueError(f"Unknown language {lang!r}. Valid codes: {sorted(LANG_CODE_MAP)}")
    if split == "train":
        return f"train/{prefix}train.parquet"
    if split == "validation":
        return f"validation/{prefix}val.parquet"
    raise ValueError(f"Unknown split {split!r}; expected 'train' or 'validation'")


def _cast_to_large(t: pa.DataType) -> pa.DataType:
    if t in (pa.string(), pa.utf8(), pa.large_utf8(), pa.large_string()):
        return pa.large_string()
    if pa.types.is_list(t):
        return pa.large_list(_cast_to_large(t.value_type))
    if pa.types.is_struct(t):
        return pa.struct([pa.field(t.field(i).name, _cast_to_large(t.field(i).type)) for i in range(t.num_fields)])
    return t


def _large_schema(schema: pa.Schema) -> pa.Schema:
    return pa.schema([pa.field(f.name, _cast_to_large(f.type)) for f in schema])


def _read_parquet(path: str) -> pa.Table:
    # pq.read_table / ParquetFile.read fail on PyArrow 25 for list<string> columns
    # nested inside a struct. iter_batches returns RecordBatch (not ChunkedArray)
    # and avoids the conversion bug. Cast to large_string/large_list so
    # datasets' combine_chunks() doesn't hit the 2GB int32 offset limit.
    pf = pq.ParquetFile(path)
    table = pa.Table.from_batches(pf.iter_batches())
    return table.cast(_large_schema(table.schema))


def _resolve_files(lang: str, split: str) -> list[str]:
    """Local cached parquet paths for lang/split, or [] if nothing is cached."""
    prefix = LANG_CODE_MAP.get(lang)
    if prefix is None and lang != "all":
        raise ValueError(f"Unknown language {lang!r}. Valid codes: {sorted(LANG_CODE_MAP)}")

    snap = _snapshot()
    if snap is None:
        return []

    split_dir = os.path.join(snap, split)
    if not os.path.isdir(split_dir):
        return []

    pattern = "*.parquet" if lang == "all" else f"{prefix}*.parquet"
    return sorted(
        p for p in glob.glob(os.path.join(split_dir, pattern))
        if os.path.isfile(p) and os.path.getsize(p) > 0
    )


def _iter_hub_rows(lang: str, split: str) -> Iterator[dict]:
    """Stream one language shard from the Hub without downloading the full dataset.

    Uses the real parquet paths (train/hintrain.parquet), not the stale
    BuilderConfig language configs that point at missing *train.jsonl files.
    """
    if lang == "all":
        raise ValueError(
            "Hub streaming does not support lang='all' — pass individual language codes "
            "so each process streams a single shard."
        )
    rel = _parquet_relpath(lang, split)
    uri = f"hf://datasets/{DATASET_NAME}/{rel}"
    try:
        ds = load_dataset("parquet", data_files=uri, split="train", streaming=True)
    except Exception as e:
        raise FileNotFoundError(
            f"No Hub parquet for lang={lang!r} split={split!r} at {uri} "
            f"(e.g. te has no train shard on the Hub). Underlying error: {e}"
        ) from e
    yield from ds


def load_language(lang: str, splits: tuple[str, ...] = ("train", "validation")) -> DatasetDict:
    """Load MSMARCO-XI for one Indic language from the local HuggingFace cache.

    Fully materializes each split into memory — fine for small/ad-hoc use but
    NOT for indexing; the CLI uses iter_language_rows()/count_language_rows().
    Requires a populated local cache (no Hub fallback).

    Args:
        lang: 2-letter language code ("hi", "bn", …) or "all" for every language.
        splits: Subset of ("train", "validation") to load.

    Returns:
        DatasetDict with a Dataset per requested split.
    """
    result: dict[str, Dataset] = {}
    for split in splits:
        files = _resolve_files(lang, split)
        if not files:
            raise FileNotFoundError(
                f"No cached parquet files for lang={lang!r} split={split!r} — "
                f"load_language requires a local HF cache (use iter_language_rows for Hub streaming)."
            )
        table = pa.concat_tables([_read_parquet(f) for f in files])
        result[split] = Dataset(table)

    return DatasetDict(result)


def count_language_rows(lang: str, split: str) -> int | None:
    """Row count for a language/split from local parquet footers, or None when
    falling back to Hub streaming (no cheap remote count without downloading)."""
    files = _resolve_files(lang, split)
    if not files:
        return None
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def iter_language_rows(lang: str, split: str, batch_size: int = 5000) -> Iterator[dict]:
    """Stream rows for a language/split without materializing a full-file Arrow Table.

    Local-first: if a non-empty language parquet is in the HF hub cache, read it
    batch-by-batch via pq.ParquetFile.iter_batches(). Otherwise stream that single
    shard from the Hub (hf://datasets/.../{prefix}{train|val}.parquet) so --limit
    only transfers what is iterated — not the full ~55GB dataset.
    """
    files = _resolve_files(lang, split)
    if files:
        for path in files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=batch_size):
                yield from batch.to_pylist()
        return

    yield from _iter_hub_rows(lang, split)
