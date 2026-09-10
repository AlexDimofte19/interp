"""What a probe reads, and what a row of its evaluation means -- the registry that lets one
evaluator serve `next_action` and `grid_tile`.

`eval_probe_per_token.py` and `eval_grid_probe_per_token.py` were the same script twice: five
byte-identical helpers, one CLI, and four real differences. Those four differences are the
interface below, and everything else is shared.

  1. FEATURES. next_action reads the raw `(D,)` activation. grid_tile appends the cell's
     coordinates, `(D + 2,)`, because one activation is asked about every cell of the grid.
  2. LABELS. next_action has one label per token -- the step's `agent_action`, shared by every
     token of that step. grid_tile has one per CELL, so a token carries many.
  3. ROW SHAPE. Both emit one row per (trajectory, step, token). next_action puts a single
     verdict on it; grid_tile puts per-class COUNTS on it (`n_true_{c}`,
     `{probe}_correct_{c}`), because the row summarises a whole step's cells. That is why
     `stats.bal_acc_from_counts` exists alongside `stats.bal_acc`.
  4. NAMING. Probe files are prefixed differently by their trainers.

A PROBE TYPE IS NOT A SIGNAL. The grid evaluator is deliberately pointed at the DIRECTION
vocabulary: the question it was built for is whether direction loudness predicts where the
GRID is decodable, and the answer is only meaningful if the ruler is unchanged. So the signal
and the probe type vary independently, and neither implies the other.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

__all__ = [
    "PROBE_TYPES",
    "GridTileProbeType",
    "NextActionProbeType",
    "ProbeType",
    "get_probe_type",
    "probe_type_names",
]


@dataclass
class StepPayload:
    """What one (trajectory, step) contributes, beyond its tokens.

    `labels` is the per-row label for a next_action probe, or the per-cell label vector for a
    grid_tile one; `cells` is empty for probe types that do not fan a token out.
    """

    labels: list[int]
    cells: list[list[int]]


class ProbeType(ABC):
    """One label a probe can carry. Subclasses own the four differences above."""

    name: str = "base"
    #: Class ids the balanced accuracy averages over. A class absent from a bucket is dropped.
    classes: tuple[int, ...] = ()
    #: Stripped off a probe filename to make its column key, so a key names the arm not the trainer.
    strip_prefixes: tuple[str, ...] = ()
    #: True when one token fans out into several predictions and rows carry per-class counts.
    per_cell: bool = False

    @abstractmethod
    def load_probe(self, path):
        """Load a trained probe from `path`."""

    @abstractmethod
    def step_payload(self, traj: dict, step: int, **kwargs) -> StepPayload:
        """The labels (and cells) this step is scored on, or empty when it is unusable."""

    @abstractmethod
    def result_columns(self, probe_names: list[str], full_probs: bool) -> list[str]:
        """Per-probe output columns, in the order `result_row` fills them."""

    def probe_key(self, path) -> str:
        """`"<parent dir>.<stem minus the trainer's prefix>"`.

        The parent directory is part of the key on purpose: it is what keeps two arms with the
        same filename apart, and assuming the prefix was NOT stripped once cost a round with a
        silently failed join.
        """
        key = f"{path.parent.name}.{path.stem}"
        for prefix in self.strip_prefixes:
            key = key.replace(prefix, "")
        return key


class NextActionProbeType(ProbeType):
    """One of four movement actions, one label per token."""

    name = "next_action"
    strip_prefixes = ("next_action_probe_",)
    per_cell = False
    #: NEXT_ACTION_TO_ID's order. `{probe}_p_LEFT` therefore means the same column for every
    #: arm, whatever that arm's own `label_to_idx` happens to be.
    action_cols = ("LEFT", "UP", "RIGHT", "DOWN")

    def __init__(self) -> None:
        from telos_interp.commands.prepare_activations_for_probing.prepare_activations_for_probing_fn import (
            NEXT_ACTION_TO_ID,
        )

        self._to_id = NEXT_ACTION_TO_ID
        self.classes = tuple(sorted(NEXT_ACTION_TO_ID.values()))

    def load_probe(self, path):
        from telos_interp.commands.train_next_action_probe.train_next_action_probe_fn import NextActionProbe

        return NextActionProbe.load(path)

    def step_payload(self, traj: dict, step: int, **kwargs) -> StepPayload:
        action = traj["steps"][step].get("agent_action")
        label = self._to_id.get(action)
        return StepPayload(labels=[] if label is None else [label], cells=[])

    def result_columns(self, probe_names: list[str], full_probs: bool) -> list[str]:
        cols = [f"{n}_pred" for n in probe_names]
        cols += [f"{n}_correct" for n in probe_names]
        cols += [f"{n}_p_true" for n in probe_names]
        if full_probs:
            cols += [f"{n}_p_{a}" for n in probe_names for a in self.action_cols]
        return cols


class GridTileProbeType(ProbeType):
    """The identity of a grid cell -- the "cognitive map" label. Many labels per token.

    `max_cells` caps the fan-out: padded to the widest grid every trajectory has 225 cells,
    and ~72k tokens x 225 is 16.2M predictions, so the default keeps a class-balanced sample.
    The draw is seeded per (trajectory, step), never from the global stream -- two runs that
    consume different numbers of draws would otherwise score different cells.
    """

    name = "grid_tile"
    strip_prefixes = ("grid_probe_", "cognitive_map_probe_")
    per_cell = True

    def __init__(self) -> None:
        from telos_interp.grid_utils import CELL_ID_TO_SYMBOL

        self.classes = tuple(sorted(CELL_ID_TO_SYMBOL))

    def load_probe(self, path):
        from telos_interp.commands.train_cognitive_map_probe.train_cognitive_map_probe_fn import (
            CognitiveMapProbe,
        )

        return CognitiveMapProbe.load(path)

    def step_payload(
        self,
        traj: dict,
        step: int,
        *,
        pad_to_size: int | None = None,
        max_cells: int | None = None,
        seed: int = 42,
        stem: str = "",
        **kwargs,
    ) -> StepPayload:
        from telos_interp.grid_utils import parse_grid_state

        grid_state = traj["steps"][step].get("grid_state")
        if not grid_state:
            return StepPayload(labels=[], cells=[])
        triples = parse_grid_state(grid_state, pad_to_size=pad_to_size)
        if max_cells is not None and len(triples) > max_cells:
            triples = random.Random(f"{stem}|{step}|{seed}").sample(triples, max_cells)
        return StepPayload(labels=[t[2] for t in triples], cells=triples)

    def result_columns(self, probe_names: list[str], full_probs: bool) -> list[str]:
        cols: list[str] = []
        for n in probe_names:
            cols += [f"{n}_n_correct", f"{n}_acc", f"{n}_mean_p_true"]
            cols += [f"{n}_correct_{c}" for c in self.classes]
        return cols


PROBE_TYPES: dict[str, type[ProbeType]] = {
    NextActionProbeType.name: NextActionProbeType,
    GridTileProbeType.name: GridTileProbeType,
}

DEFAULT_PROBE_TYPE = NextActionProbeType.name


def probe_type_names() -> list[str]:
    """Registered probe types, in registry order.

    >>> probe_type_names()
    ['next_action', 'grid_tile']
    """
    return list(PROBE_TYPES)


def get_probe_type(name: str) -> ProbeType:
    """Build the probe type called `name`.

    >>> get_probe_type("grid_tile").per_cell
    True
    >>> get_probe_type("next_action").per_cell
    False
    >>> get_probe_type("nope")
    Traceback (most recent call last):
        ...
    ValueError: Unknown probe type 'nope'; available: ['grid_tile', 'next_action']
    """
    try:
        cls = PROBE_TYPES[name]
    except KeyError:
        raise ValueError(f"Unknown probe type {name!r}; available: {sorted(PROBE_TYPES)}") from None
    return cls()
