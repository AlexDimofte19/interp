"""`run_config.json` -- what produced the numbers in this folder, and how.

Every script in this module writes one into its output directory. It answers the questions
that a bare CSV of balanced accuracies cannot: which lens was the ruler, which vocabulary it
was baked against, which layer, how the bins were cut, whether balanced accuracy was pooled
from counts or averaged over rows, what the bootstrap resampled and with which seed, and how
many rows survived each filter.

IT IS ALSO A GUARD, not only a record. `build_loudness_x_reasoning_pos_heatmap.py` already
refused to write jlens-ruler figures into a folder holding logitlens-ruler ones, because the
two are not comparable and a folder that mixes them is silently wrong. `guard` generalises
that: name the fields that must not change within one folder, and a second run that disagrees
fails loudly instead of overwriting half the figures.

The aggregation method is recorded because the two balanced accuracies in `stats` are not the
same number -- see that module. A table that does not say which one it used cannot be
compared with one that used the other.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["RunConfig", "describe_input", "git_commit"]

FILENAME = "run_config.json"


def git_commit(repo: Path | None = None) -> str | None:
    """The commit the code was at, or None outside a repository."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo or Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def describe_input(path: str | Path) -> dict:
    """Identity of one input file: path, size and mtime.

    Deliberately not a content hash -- these are multi-hundred-MB per-token CSVs and hashing
    them would cost more than the analysis. Size and mtime are enough to tell two runs apart;
    the one thing that IS hashed is the signal vocabulary, which is small and whose content
    silently changes what every number means.
    """
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
    }


@dataclass
class RunConfig:
    """Accumulates provenance, then writes it beside the results.

    >>> cfg = RunConfig("analysis/loudness_distribution.py")
    >>> cfg.measurement(lens="jlens", signal="direction", layer=15)
    >>> cfg.to_dict()["measurement"]["loudness_column"]
    'jlens_direction_logmass_L15'
    >>> cfg.to_dict()["measurement"]["axis_label"]
    'J-lens direction logmass'
    """

    script: str
    inputs: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    method: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    _measurement: dict = field(default_factory=dict)

    def input(self, name: str, path: str | Path) -> None:
        """Record one input file under a caller-chosen name."""
        self.inputs[name] = describe_input(path)

    def measurement(
        self,
        *,
        lens: str,
        signal: str,
        layer: int | str,
        signal_json: str | Path | None = None,
    ) -> None:
        """The ruler: which lens, which vocabulary, which layer -- and the names they imply."""
        from . import columns as _columns
        from . import signals as _signals

        entry = {
            "lens": lens,
            "signal": signal,
            "layer": layer,
            "loudness_column": _columns.loudness_column(lens, signal, layer),
            "membership_column": _columns.membership_column(signal),
            "axis_label": _columns.axis_label(lens, signal),
        }
        if signal_json is not None:
            entry["signal_json"] = str(signal_json)
            path = Path(signal_json)
            if path.exists():
                sig = _signals.resolve(signal, path)
                entry["signal_fingerprint"] = sig.fingerprint(path)
                entry["signal_classes"] = list(sig.classes(path))
        self._measurement = entry

    def aggregation(self, **kwargs) -> None:
        """How rows were reduced: bin edges, the balanced-accuracy form, bootstrap settings.

        Pass `balanced_accuracy="rows"` or `"counts"` to name which of `stats`' two forms ran.
        """
        self.method.update(kwargs)

    def rows(self, stage: str, n: int) -> None:
        """Row count after one filter, so a surprising N is traceable to the step that caused it."""
        self.counts[stage] = n

    def to_dict(self) -> dict:
        return {
            "script": self.script,
            "argv": sys.argv[1:],
            "written_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "measurement": self._measurement,
            "inputs": self.inputs,
            "outputs": [str(o) for o in self.outputs],
            "parameters": self.params,
            "aggregation": self.method,
            "row_counts": self.counts,
        }

    def guard(self, out_dir: str | Path, *fields: str) -> None:
        """Refuse to write into a folder produced under different `fields`.

        Call before writing anything. `fields` are dotted paths into the measurement block,
        e.g. `"lens"`, `"signal"`, `"layer"`. A folder holding jlens-ruler figures must not
        also receive logitlens-ruler ones: the two are not comparable, and mixing them
        produces a directory that looks complete and is wrong.
        """
        path = Path(out_dir) / FILENAME
        if not path.exists():
            return
        try:
            previous = json.loads(path.read_text()).get("measurement", {})
        except (OSError, json.JSONDecodeError):
            return
        for f in fields:
            was, now = previous.get(f), self._measurement.get(f)
            if was is not None and now is not None and was != now:
                raise SystemExit(
                    f"{out_dir} already holds results built with {f}={was!r}; this run has "
                    f"{f}={now!r}. Refusing to mix them in one folder -- give this run its own --out."
                )

    def write(self, out_dir: str | Path) -> Path:
        """Write `run_config.json` into `out_dir`, creating it if needed."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n")
        return path
