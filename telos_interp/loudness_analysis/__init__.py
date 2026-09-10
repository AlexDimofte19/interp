"""The lens -> probe -> loudness pipeline. See README.md.

Only the LIBRARY layer is re-exported here -- the registries, the column convention, the
statistics, the tree readers and the provenance record. The pipeline scripts are not, because
importing this package would then pull in torch and matplotlib, and the analysis layer is
deliberately able to run without either.
"""

from .columns import (
    LENS_LABEL,
    axis_label,
    loudness_column,
    membership_column,
    prob_column,
    score_columns,
)
from .lens_io import (
    check_vocabulary,
    find_act_folder,
    load_trajectory,
    read_lens_tables,
    read_mass_columns,
    trajectory_dirs,
)
from .probes import PROBE_TYPES, ProbeType, get_probe_type, probe_type_names
from .provenance import RunConfig
from .signals import SIGNALS, Signal, get_signal, signal_names
from .stats import (
    bal_acc,
    bal_acc_from_counts,
    boot_bal_acc,
    clustered_band,
    qbin,
    spearman,
)

__all__ = [
    "LENS_LABEL",
    "PROBE_TYPES",
    "SIGNALS",
    "ProbeType",
    "RunConfig",
    "Signal",
    "axis_label",
    "bal_acc",
    "bal_acc_from_counts",
    "boot_bal_acc",
    "check_vocabulary",
    "clustered_band",
    "find_act_folder",
    "get_probe_type",
    "get_signal",
    "load_trajectory",
    "loudness_column",
    "membership_column",
    "prob_column",
    "probe_type_names",
    "qbin",
    "read_lens_tables",
    "read_mass_columns",
    "score_columns",
    "signal_names",
    "spearman",
    "trajectory_dirs",
]
