"""Pick the reasoning tokens and layers worth keeping on disk.

One forward pass over a long reasoning chain writes a `.pt` per (token, layer) — a
700-token chain over 17 layers is ~12k files per step — but a probe only ever trains on the
handful of tokens whose j-space is most direction-loaded. `jlens_top_filter` reads a
trajectory's jlens CSV and returns exactly that handful, so both

  * `scripts/jlens_reasoning_tokens.py`, which uses it to decide what to *write*, and
  * `scripts/delete_non_jlens_selected.py`, which uses it to decide what to *keep*

agree by construction: pruning an existing tree lands on the same files a filtered gather
would have produced.

Two arms come back, not one. The jlens arm is the top-scoring tokens; the **random** arm is
a seeded uniform draw over the same universe, kept because a `jlens_direction` result means
nothing without a matched control — and once the tree is pruned to the jlens arm, drawing a
uniform sample is no longer possible. The control has to be reserved *before* the deletion,
not after.

Ranking here is deliberately identical to
`prepare_activations_for_probing/jlens_token_selection.py`, which scores the same CSV for a
different purpose; `rank_layers_by_direction` is shared by both so they cannot drift.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path

from .jlens_csv import (
    TokenScore,
    csv_layers,
    load_direction_tokens,
    output_start,
    read_direction_counts,
    step_folder_index,
)

# Layer 15 is the project's standing comparison point (see general_probe_train.sh), so it is
# kept for every selected token regardless of how it scores. Without that, a layer-15
# baseline would need its own gather run.
DEFAULT_ALWAYS_LAYERS = (15,)


def rank_layers_by_direction(token_layers: dict[int, int], candidate_layers: list[int]) -> list[int]:
    """Candidate layers ordered by this token's direction count, best first.

    Ties break on ascending layer index. Shared with `_pick_layers` in
    `jlens_token_selection.py` so the filter and the probe-data preparer rank identically.

    >>> rank_layers_by_direction({7: 2, 15: 9, 23: 2}, [7, 15, 23])
    [15, 7, 23]
    """
    return sorted(candidate_layers, key=lambda layer: (-token_layers.get(layer, 0), layer))


@dataclass(frozen=True)
class TokenPick:
    """One selected reasoning token and the layers of it that are kept.

    `pos` is an absolute forward-pass position in CSV coordinates and an output-relative
    index (the `.pt` filename) after `to_disk_coords`.

    The scoring fields are `None` for the random arm. That absence is load-bearing
    downstream: `scripts/split_next_action_manifest.py` decides whether to *rank* a token's
    layers or *sample* them by whether a count is present, so a control that recorded counts
    would silently collapse onto its lowest-scoring layer.
    """

    step: int
    pos: int
    layers: tuple[int, ...]
    token: str = ""
    direction_count: int | None = None
    layer_direction_counts: dict[int, int] | None = None

    @property
    def key(self) -> tuple[int, int]:
        return (self.step, self.pos)


@dataclass(frozen=True)
class KeptTokens:
    """The two arms of a selection, each keyed by (step, pos)."""

    jlens: dict[tuple[int, int], TokenPick] = field(default_factory=dict)
    random: dict[tuple[int, int], TokenPick] = field(default_factory=dict)

    def merged(self) -> dict[tuple[int, int], tuple[int, ...]]:
        """Every kept (step, pos) mapped to the union of the layers either arm wants.

        The two arms are drawn from the same universe, so a token can land in both; the
        union is what has to survive on disk.
        """
        out: dict[tuple[int, int], set[int]] = {}
        for arm in (self.jlens, self.random):
            for key, pick in arm.items():
                out.setdefault(key, set()).update(pick.layers)
        return {key: tuple(sorted(layers)) for key, layers in sorted(out.items())}

    def activation_paths(self, model_folder: Path, category: str = "output") -> set[Path]:
        """The `.pt` files this selection keeps, in the gather_activations layout.

        Only meaningful after `to_disk_coords`: `pos` must be the output-relative token
        index that names the file.
        """
        return {
            model_folder / f"layer_{layer}" / f"step_{step}" / category / f"{pos}.pt"
            for (step, pos), layers in self.merged().items()
            for layer in layers
        }

    def num_files(self) -> int:
        return sum(len(layers) for layers in self.merged().values())


def _layers_for(
    score: TokenScore | None,
    candidate_layers: list[int],
    num_layers: int,
    always_layers: tuple[int, ...],
    rng: random.Random | None,
) -> tuple[int, ...]:
    """Top-N layers of one token (or a uniform draw when `rng` is given), plus `always_layers`.

    `always_layers` is unioned in rather than counted against the budget, so asking for 3
    layers with layer 15 always on gives 4 — unless 15 was already in the top 3.
    """
    if rng is not None:
        picked = rng.sample(candidate_layers, min(num_layers, len(candidate_layers)))
    else:
        per_layer = score.per_layer if score else {}
        picked = rank_layers_by_direction(per_layer, candidate_layers)[:num_layers]
    forced = [layer for layer in always_layers if layer in candidate_layers]
    return tuple(sorted(set(picked) | set(forced)))


def jlens_top_filter(
    signal_json: str | Path,
    jlens_csv: str | Path,
    *,
    num_tokens: int = 20,
    num_layers: int = 3,
    always_layers: tuple[int, ...] = DEFAULT_ALWAYS_LAYERS,
    random_tokens: int = 0,
    random_layers: int | None = None,
    seed: int = 42,
    seed_key: str = "",
    candidate_layers: list[int] | None = None,
    direction_classes: str = "all",
    top_k: int = 20,
) -> KeptTokens:
    """Select the reasoning tokens and layers worth keeping for one trajectory.

    Args:
        signal_json: JSON mapping UP/DOWN/LEFT/RIGHT to token strings — the signal being
            looked for in each token's top-k lens predictions (e.g.
            `data/jlens/direction_tokens_full.json`).
        jlens_csv: that trajectory's `{stem}_jlens_analysis.csv`.
        num_tokens: how many top-scoring tokens the jlens arm keeps.
        num_layers: how many top-scoring layers of each of those tokens to keep.
        always_layers: layers kept for every selected token of *both* arms regardless of
            score, so a fixed-layer baseline stays available after pruning.
        random_tokens: size of the matched control arm; 0 disables it.
        random_layers: layers per control token (defaults to `num_layers`).
        seed, seed_key: the control draw is seeded `f"{seed}-{seed_key}"`, matching
            `select_token_layer_pairs`, so a trajectory's sample is stable no matter what
            order trajectories are processed in.
        candidate_layers: the pool to choose from; defaults to every layer the CSV has rows
            for. Layers with no jlens matrix produce no rows and so are never selectable.
        direction_classes: "all" or a subset such as "UP,DOWN".
        top_k: how many `top_i` columns of the CSV to scan.

    Returns:
        A `KeptTokens` in **CSV coordinates** — `pos` is `abs_pos`. Call `to_disk_coords`
        to turn it into the `.pt` coordinates the activation tree uses.
    """
    jlens_csv = Path(jlens_csv)
    direction_tokens = load_direction_tokens(signal_json, direction_classes)
    scores = read_direction_counts(jlens_csv, direction_tokens, top_k=top_k)
    if candidate_layers is None:
        candidate_layers = csv_layers(jlens_csv)
    if not scores or not candidate_layers:
        return KeptTokens()

    # Same ordering as select_token_layer_pairs: score desc, then step, then position.
    ranked = sorted(scores, key=lambda key: (-scores[key].total(candidate_layers), key[0], key[1]))

    jlens_arm: dict[tuple[int, int], TokenPick] = {}
    for step, pos in ranked[:num_tokens]:
        score = scores[(step, pos)]
        layers = _layers_for(score, candidate_layers, num_layers, always_layers, rng=None)
        jlens_arm[(step, pos)] = TokenPick(
            step=step,
            pos=pos,
            layers=layers,
            token=score.token,
            direction_count=score.total(candidate_layers),
            layer_direction_counts={layer: score.per_layer.get(layer, 0) for layer in layers},
        )

    random_arm: dict[tuple[int, int], TokenPick] = {}
    if random_tokens > 0:
        rng = random.Random(f"{seed}-{seed_key}")
        universe = sorted(scores)
        # Drawn from the *whole* chain, not from what the jlens arm left over: overlap
        # between the arms is a legitimate outcome of a uniform draw, and excluding the
        # top-scoring tokens would make the control systematically low-scoring.
        chosen = sorted(rng.sample(universe, min(random_tokens, len(universe))))
        per_token = num_layers if random_layers is None else random_layers
        for step, pos in chosen:
            random_arm[(step, pos)] = TokenPick(
                step=step,
                pos=pos,
                layers=_layers_for(None, candidate_layers, per_token, always_layers, rng=rng),
            )

    return KeptTokens(jlens=jlens_arm, random=random_arm)


def to_disk_coords(kept: KeptTokens, trajectory_data: dict) -> KeptTokens:
    """Re-key a selection from CSV coordinates onto the ones the `.pt` tree uses.

    The CSV records `(step list index, abs_pos)`; a file is at
    `layer_{N}/step_{step_id}/output/{abs_pos - output_start}.pt`. Tokens whose step is not
    in the trajectory JSON are dropped rather than guessed at.
    """
    starts: dict[int, tuple[int, int]] = {}

    def resolve(step: int) -> tuple[int, int] | None:
        if step not in starts:
            try:
                starts[step] = (step_folder_index(trajectory_data, step), output_start(trajectory_data, step))
            except (IndexError, KeyError):
                return None
        return starts.get(step)

    def convert(arm: dict[tuple[int, int], TokenPick]) -> dict[tuple[int, int], TokenPick]:
        out: dict[tuple[int, int], TokenPick] = {}
        for pick in arm.values():
            resolved = resolve(pick.step)
            if resolved is None:
                continue
            folder, start = resolved
            moved = TokenPick(
                step=folder,
                pos=pick.pos - start,
                layers=pick.layers,
                token=pick.token,
                direction_count=pick.direction_count,
                layer_direction_counts=pick.layer_direction_counts,
            )
            out[moved.key] = moved
        return out

    return KeptTokens(jlens=convert(kept.jlens), random=convert(kept.random))


__all__ = [
    "DEFAULT_ALWAYS_LAYERS",
    "KeptTokens",
    "TokenPick",
    "jlens_top_filter",
    "rank_layers_by_direction",
    "to_disk_coords",
]
