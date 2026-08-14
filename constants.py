DATASET_NAME = "ai4bharat/MSMARCO-XI"

# Maps 2-letter HuggingFace config codes -> 3-letter parquet filename prefix
LANG_CODE_MAP: dict[str, str] = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
}