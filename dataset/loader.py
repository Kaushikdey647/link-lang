import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict

from constants import DATASET_NAME, LANG_CODE_MAP

_HUB_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--"
    + DATASET_NAME.replace("/", "--")
    + "/snapshots"
)


def _snapshot() -> str:
    return os.path.join(_HUB_CACHE, sorted(os.listdir(_HUB_CACHE))[-1])


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


def load_language(lang: str, splits: tuple[str, ...] = ("train", "validation")) -> DatasetDict:
    """Load MSMARCO-XI for one Indic language from the local HuggingFace cache.

    Args:
        lang: 2-letter language code ("hi", "bn", …) or "all" for every language.
        splits: Subset of ("train", "validation") to load.

    Returns:
        DatasetDict with a Dataset per requested split.
    """
    prefix = LANG_CODE_MAP.get(lang)
    if prefix is None and lang != "all":
        raise ValueError(f"Unknown language {lang!r}. Valid codes: {sorted(LANG_CODE_MAP)}")

    base = _snapshot()
    result: dict[str, Dataset] = {}
    for split in splits:
        split_dir = os.path.join(base, split)
        pattern = "*.parquet" if lang == "all" else f"{prefix}*.parquet"
        files = sorted(glob.glob(os.path.join(split_dir, pattern)))
        if not files:
            raise FileNotFoundError(
                f"No cached parquet files for lang={lang!r} split={split!r} in {split_dir}"
            )
        table = pa.concat_tables([_read_parquet(f) for f in files])
        result[split] = Dataset(table)

    return DatasetDict(result)
