"""Shared utilities for the comment-lint training pipeline.

Centralizes JSONL I/O, feature schema constants, and type coercion
helpers used across training scripts. Feature order MUST match
crates/comment-lint-ml/src/tensor.rs exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Feature order MUST match tensor.rs exactly.
# See crates/comment-lint-ml/src/tensor.rs for the authoritative mapping.
FEATURE_NAMES = [
    "token_overlap_jaccard",         # 0  - direct f32
    "identifier_substring_ratio",    # 1  - direct f32
    "comment_token_count",           # 2  - usize as f32
    "is_doc_comment",                # 3  - bool -> 0/1
    "is_before_declaration",         # 4  - bool -> 0/1
    "is_inline",                     # 5  - bool -> 0/1
    "nesting_depth",                 # 6  - usize as f32
    "has_why_indicator",             # 7  - bool -> 0/1
    "has_external_ref",              # 8  - bool -> 0/1
    "imperative_verb_noun",          # 9  - bool -> 0/1
    "is_section_label",              # 10 - bool -> 0/1
    "contains_literal_values",       # 11 - bool -> 0/1
    "references_other_files",        # 12 - bool -> 0/1
    "references_specific_functions", # 13 - bool -> 0/1
    "mirrors_data_structure",        # 14 - bool -> 0/1
    "comment_code_age_ratio",        # 15 - Option -> value or 0.0
]

FEATURE_DIM = 16
OUTPUT_DIM = 2

# Boolean feature names (for encoding True/False -> 1.0/0.0)
BOOL_FEATURES = {
    "is_doc_comment",
    "is_before_declaration",
    "is_inline",
    "has_why_indicator",
    "has_external_ref",
    "imperative_verb_noun",
    "is_section_label",
    "contains_literal_values",
    "references_other_files",
    "references_specific_functions",
    "mirrors_data_structure",
}


def load_jsonl(path: str | Path) -> list[dict]:
    """Load records from a JSONL file, skipping empty lines.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON records.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write records as JSONL to a file.

    Args:
        records: List of dicts to serialize.
        path: Output file path.
    """
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def to_bool(value: Any) -> bool:
    """Coerce incoming values to bool.

    Handles bool, int/float (nonzero = True), and common string
    representations ("true", "1", "yes", "y").
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def to_int(value: Any, default: int = 0) -> int:
    """Coerce to int with fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    """Coerce to float with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
