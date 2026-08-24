"""Shared selection logic: score a trajectory's reasoning tokens, keep the best few.

Method-dispatched — `jlens`, `logitlens` and `random` are entries in `methods.METHODS`, not
branches in each consumer. Adding a lens is a registry entry.

Stdlib only — no torch — so the two standalone scripts that decide what lands on disk can
import it without pulling in the model stack, and so the selection is unit-testable on a
laptop. See README.md for how the pieces fit together.
"""

from .jlens_csv import (
    DEFAULT_JLENS_CSV_SUFFIX,
    DIRECTION_CLASSES,
    TokenScore,
    csv_layers,
    jlens_csv_path,
    load_direction_tokens,
    output_start,
    read_direction_counts,
    step_folder_index,
)
from .methods import (
    DEFAULT_METHODS,
    METHODS,
    SelectionMethod,
    analysis_csv_path,
    get_method,
    method_names,
    parse_methods,
    scored_methods,
)
from .record import (
    RECORD_FORMAT_VERSION,
    RECORD_SUFFIX,
    build_record,
    merge_records,
    normalize_record,
    read_raw_record,
    read_selection_record,
    record_path,
    write_selection_record,
)
from .top_filter import (
    DEFAULT_ALWAYS_LAYERS,
    Arm,
    KeptTokens,
    TokenPick,
    arm_seed,
    rank_layers_by_direction,
    rank_tokens,
    to_disk_coords,
    top_filter,
)

__all__ = [
    "DEFAULT_ALWAYS_LAYERS",
    "DEFAULT_JLENS_CSV_SUFFIX",
    "DEFAULT_METHODS",
    "DIRECTION_CLASSES",
    "METHODS",
    "RECORD_FORMAT_VERSION",
    "RECORD_SUFFIX",
    "Arm",
    "KeptTokens",
    "SelectionMethod",
    "TokenPick",
    "TokenScore",
    "analysis_csv_path",
    "arm_seed",
    "build_record",
    "csv_layers",
    "get_method",
    "jlens_csv_path",
    "load_direction_tokens",
    "merge_records",
    "method_names",
    "normalize_record",
    "output_start",
    "parse_methods",
    "rank_layers_by_direction",
    "rank_tokens",
    "read_direction_counts",
    "read_raw_record",
    "read_selection_record",
    "record_path",
    "scored_methods",
    "step_folder_index",
    "to_disk_coords",
    "top_filter",
    "write_selection_record",
]
