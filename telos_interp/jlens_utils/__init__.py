"""Shared jlens selection logic: score a trajectory's reasoning tokens, keep the best few.

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
from .record import (
    RECORD_FORMAT_VERSION,
    RECORD_SUFFIX,
    build_record,
    read_selection_record,
    record_path,
    write_selection_record,
)
from .top_filter import (
    DEFAULT_ALWAYS_LAYERS,
    KeptTokens,
    TokenPick,
    jlens_top_filter,
    rank_layers_by_direction,
    to_disk_coords,
)

__all__ = [
    "DEFAULT_ALWAYS_LAYERS",
    "DEFAULT_JLENS_CSV_SUFFIX",
    "DIRECTION_CLASSES",
    "KeptTokens",
    "RECORD_FORMAT_VERSION",
    "RECORD_SUFFIX",
    "TokenPick",
    "TokenScore",
    "build_record",
    "csv_layers",
    "jlens_csv_path",
    "jlens_top_filter",
    "load_direction_tokens",
    "output_start",
    "rank_layers_by_direction",
    "read_direction_counts",
    "read_selection_record",
    "record_path",
    "step_folder_index",
    "to_disk_coords",
    "write_selection_record",
]
