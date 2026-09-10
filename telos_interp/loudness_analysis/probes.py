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
from collections import defaultdict

__all__ = [
    "PROBE_TYPES",
    "GridTileProbeType",
    "NextActionProbeType",
    "ProbeType",
    "get_probe_type",
    "probe_type_names",
]


class ProbeType(ABC):
    """One label a probe can carry. Subclasses own the four differences above."""

    name: str = "base"
    #: Class ids the balanced accuracy averages over. A class absent from a bucket is dropped.
    classes: tuple[int, ...] = ()
    #: The same classes as they appear in a per-token CSV's label column -- action NAMES for
    #: next_action, integer ids for grid_tile. NOT interchangeable with `classes`.
    analysis_classes: tuple = ()
    #: Which balanced accuracy applies: "rows" (one prediction per row) or "counts"
    #: (a row summarises many). See stats.py -- they are not the same number.
    aggregation: str = "rows"
    #: Stripped off a probe filename to make its column key, so a key names the arm not the trainer.
    strip_prefixes: tuple[str, ...] = ()
    #: True when one token fans out into several predictions and rows carry per-class counts.
    per_cell: bool = False

    def add_arguments(self, ap) -> None:  # noqa: B027
        """Probe-type-specific CLI flags. The shared ones live on the evaluator.

        Deliberately optional and non-abstract: next_action adds none, and forcing every
        future probe type to declare an empty override would be noise.
        """

    @abstractmethod
    def load_probe(self, path):
        """Load a trained probe from `path`."""

    @abstractmethod
    def prefix_columns(self) -> list[str]:
        """Leading columns, before the lens scores. `row_prefix` fills these, in order."""

    @abstractmethod
    def step_state(self, traj: dict, step: int, stem: str, args):
        """Per-step state shared by every token of that step, or None when unusable.

        Cached by the evaluator: cells are a property of the STEP, so a trajectory's ~240
        tokens must not re-parse and re-draw the same grid 240 times.
        """

    @abstractmethod
    def row_prefix(self, stem, traj, step, abs_pos, token_idx, token, state) -> list:
        """The row's leading cells, matching `prefix_columns`."""

    @abstractmethod
    def result_columns(self, probe_names: list[str], full_probs: bool) -> list[str]:
        """Per-probe output columns, in the order `score` fills them."""

    @abstractmethod
    def score(self, probes, acts, states, batch_size, device, full_probs) -> list[list]:
        """One list of result cells per input row, matching `result_columns`."""

    def probe_key(self, path) -> str:
        """`"<parent dir>.<stem minus the trainer's prefix>"`.

        The parent directory is part of the key on purpose: it is what keeps two arms with the
        same filename apart. The prefix IS stripped -- assuming otherwise once cost a round
        with a silently failed join.
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
        from telos_interp.commands.train_next_action_probe.train_next_action_probe_fn import (
            ACTION_ID_TO_NAME,
        )

        self._to_id = NEXT_ACTION_TO_ID
        self._id_to_name = ACTION_ID_TO_NAME
        self.classes = tuple(sorted(NEXT_ACTION_TO_ID.values()))
        # The per-token CSVs carry `label_name` and the rollout carries `model_action`,
        # both action names, so an analysis bins on names rather than ids.
        self.analysis_classes = self.action_cols

    def load_probe(self, path):
        from telos_interp.commands.train_next_action_probe.train_next_action_probe_fn import NextActionProbe

        return NextActionProbe.load(path)

    def prefix_columns(self) -> list[str]:
        return ["name", "size", "complexity", "step", "abs_pos", "token_idx", "token", "label", "label_name"]

    def step_state(self, traj: dict, step: int, stem: str, args):
        action = traj["steps"][step].get("agent_action", "")
        return self._to_id.get(action.upper())

    def row_prefix(self, stem, traj, step, abs_pos, token_idx, token, state) -> list:
        return [
            stem,
            traj["grid_params"].get("grid_width", ""),
            traj["grid_params"].get("grid_complexity", ""),
            step,
            abs_pos,
            token_idx,
            token,
            state,
            self._id_to_name.get(state, ""),
        ]

    def result_columns(self, probe_names: list[str], full_probs: bool) -> list[str]:
        cols = [f"{n}_pred" for n in probe_names]
        cols += [f"{n}_correct" for n in probe_names]
        cols += [f"{n}_p_true" for n in probe_names]
        if full_probs:
            cols += [f"{n}_p_{a}" for n in probe_names for a in self.action_cols]
        return cols

    def score(self, probes, acts, states, batch_size, device, full_probs) -> list[list]:
        import torch

        from telos_interp.commands.prepare_activations_for_probing.prepare_activations_for_probing_fn import (
            NEXT_ACTION_TO_ID,
        )

        batch = torch.stack(acts)
        labels = torch.tensor(states)
        preds, corrects, ptrue, pfull = {}, {}, {}, {}
        for name, probe in probes.items():
            out_pred, out_p, out_full = [], [], []
            order = [probe.label_to_idx[NEXT_ACTION_TO_ID[a]] for a in self.action_cols]
            for i in range(0, batch.shape[0], batch_size):
                chunk = batch[i : i + batch_size]
                probs = probe.predict_proba(chunk).cpu()
                idx = probs.argmax(dim=-1)
                out_pred.append(torch.tensor([probe.idx_to_label[j.item()] for j in idx]))
                cols = torch.tensor([probe.label_to_idx.get(int(v), 0) for v in labels[i : i + chunk.shape[0]]])
                out_p.append(probs.gather(1, cols[:, None]).squeeze(1))
                if full_probs:
                    out_full.append(probs[:, order])
            preds[name] = torch.cat(out_pred)
            ptrue[name] = torch.cat(out_p)
            corrects[name] = (preds[name] == labels).int()
            if full_probs:
                pfull[name] = torch.cat(out_full)

        names = list(probes)
        rows = []
        for i in range(len(acts)):
            cells = [int(preds[n][i]) for n in names]
            cells += [int(corrects[n][i]) for n in names]
            cells += [f"{float(ptrue[n][i]):.6f}" for n in names]
            if full_probs:
                cells += [f"{float(v):.6f}" for n in names for v in pfull[n][i]]
            rows.append(cells)
        return rows


class GridTileProbeType(ProbeType):
    """The identity of a grid cell -- the "cognitive map" label. Many labels per token.

    `--max-cells` decides affordability: padded to the widest grid every trajectory has 225
    cells, and ~72k tokens x 225 is 16.2M predictions, so the default keeps a class-balanced
    sample. The draw is seeded per (trajectory, step), never from the global stream -- two
    runs that consume different numbers of draws would otherwise score different cells.
    """

    name = "grid_tile"
    strip_prefixes = ("grid_probe_", "cognitive_map_probe_")
    per_cell = True
    aggregation = "counts"

    def __init__(self) -> None:
        from telos_interp.grid_utils import CELL_ID_TO_SYMBOL, CELL_SYMBOL_TO_ID

        self.classes = tuple(sorted(CELL_ID_TO_SYMBOL))
        # Class 7 is padding -- a cell that does not exist. It is excluded from any
        # balanced accuracy, because scoring a probe on cells outside the grid measures
        # nothing about the grid.
        self.analysis_classes = tuple(c for c in self.classes if c != CELL_SYMBOL_TO_ID["+"])

    def add_arguments(self, ap) -> None:
        ap.add_argument(
            "--pad-to-size",
            type=int,
            default=None,
            help="Pad every grid to this width before reading cells (default: native size, "
            "which excludes the padding class entirely).",
        )
        ap.add_argument(
            "--max-cells",
            type=int,
            default=None,
            help="Cap the cells scored per (trajectory, step). Padded to the widest grid a "
            "trajectory has 225 cells and ~72k tokens x 225 is 16.2M rows per pass.",
        )
        ap.add_argument("--seed", type=int, default=42, help="Seeds the per-step cell draw.")

    def load_probe(self, path):
        from telos_interp.commands.train_cognitive_map_probe.train_cognitive_map_probe_fn import (
            CognitiveMapProbe,
        )

        return CognitiveMapProbe.load(path)

    def prefix_columns(self) -> list[str]:
        return [
            "name",
            "size",
            "complexity",
            "step",
            "abs_pos",
            "token_idx",
            "token",
            "n_cells",
        ] + [f"n_true_{c}" for c in self.classes]

    def step_state(self, traj: dict, step: int, stem: str, args):
        import torch

        from telos_interp.grid_utils import parse_grid_state

        grid_state = traj["steps"][step].get("grid_state")
        if not grid_state:
            return None
        triples = parse_grid_state(grid_state, pad_to_size=getattr(args, "pad_to_size", None))
        max_cells = getattr(args, "max_cells", None)
        if max_cells is not None and len(triples) > max_cells:
            triples = random.Random(f"{stem}|{step}|{getattr(args, 'seed', 42)}").sample(triples, max_cells)
        if not triples:
            return None
        pos = torch.tensor([[t[0], t[1]] for t in triples], dtype=torch.float32)
        lab = torch.tensor([t[2] for t in triples], dtype=torch.long)
        return (pos, lab)

    def row_prefix(self, stem, traj, step, abs_pos, token_idx, token, state) -> list:
        lab = state[1]
        hist: dict[int, int] = defaultdict(int)
        for v in lab.tolist():
            hist[v] += 1
        return [
            stem,
            traj["grid_params"].get("grid_width", ""),
            traj["grid_params"].get("grid_complexity", ""),
            step,
            abs_pos,
            token_idx,
            token,
            int(lab.numel()),
        ] + [hist.get(c, 0) for c in self.classes]

    def result_columns(self, probe_names: list[str], full_probs: bool) -> list[str]:
        cols: list[str] = []
        for n in probe_names:
            cols += [f"{n}_n_correct", f"{n}_acc", f"{n}_mean_p_true"]
            cols += [f"{n}_correct_{c}" for c in self.classes]
        return cols

    def score(self, probes, acts, states, batch_size, device, full_probs) -> list[list]:
        """Tokens are batched together rather than scored one at a time: a token contributes C
        rows of width D+2, and one 8k-row forward over several tokens is far cheaper than ~240
        forwards of ~100 rows. The activation is broadcast across its own cells, so the
        (row, col) columns are the only part that varies within a token."""
        import torch

        per_probe: dict[str, list] = {n: [] for n in probes}
        n_tokens = len(acts)
        i = 0
        while i < n_tokens:
            # Grow the batch until one more token would exceed batch_size rows.
            j, rows_in_batch = i, 0
            while j < n_tokens:
                c = int(states[j][1].numel())
                if rows_in_batch and rows_in_batch + c > batch_size:
                    break
                rows_in_batch += c
                j += 1

            chunk_x, chunk_y, spans = [], [], []
            for k in range(i, j):
                pos, lab = states[k]
                a = acts[k].unsqueeze(0).expand(lab.numel(), -1)
                chunk_x.append(torch.cat([a, pos], dim=1))
                chunk_y.append(lab)
                spans.append(lab.numel())
            x = torch.cat(chunk_x).to(device)
            y = torch.cat(chunk_y)

            for name, probe in probes.items():
                probs = probe.predict_proba(x).cpu()
                pred = torch.tensor([probe.idx_to_label[t.item()] for t in probs.argmax(dim=-1)])
                # A label the probe never saw has no column; p_true is 0 there, which is the
                # honest reading -- the probe assigns it no mass at all.
                cols = torch.tensor([probe.label_to_idx.get(int(v), -1) for v in y])
                p_true = probs.gather(1, cols.clamp(min=0)[:, None]).squeeze(1)
                p_true[cols < 0] = 0.0
                correct = (pred == y).int()

                off = 0
                for span in spans:
                    sl = slice(off, off + span)
                    c_sl, y_sl = correct[sl], y[sl]
                    pc: dict[int, int] = defaultdict(int)
                    for v, ok in zip(y_sl.tolist(), c_sl.tolist(), strict=True):
                        pc[v] += ok
                    per_probe[name].append(
                        (int(c_sl.sum()), float(c_sl.float().mean()), float(p_true[sl].mean()), dict(pc))
                    )
                    off += span
            i = j

        names = list(probes)
        rows = []
        for idx in range(n_tokens):
            cells: list = []
            for n in names:
                n_correct, acc, mean_p, pc = per_probe[n][idx]
                cells += [n_correct, f"{acc:.6f}", f"{mean_p:.6f}"]
                cells += [pc.get(c, 0) for c in self.classes]
            rows.append(cells)
        return rows


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
