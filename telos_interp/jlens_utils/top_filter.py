"""Pick the reasoning tokens and layers worth keeping on disk.

One forward pass over a long reasoning chain writes a `.pt` per (token, layer) — a
700-token chain over 17 layers is ~12k files per step — but a probe only ever trains on the
handful of tokens whose lens output is most direction-loaded. `top_filter` reads a
trajectory's analysis CSVs and returns exactly that handful, so both

  * `scripts/jlens_reasoning_tokens.py`, which uses it to decide what to *write*, and
  * `scripts/delete_non_jlens_selected.py`, which uses it to decide what to *keep*

agree by construction: pruning an existing tree lands on the same files a filtered gather
would have produced.

The result is **arm-keyed by method** (see `methods.py`). A scored arm — `jlens`,
`logitlens` — ranks by direction score, under whichever mode of `scoring.py` the run asks
for: the original `count`, or the direction probability mass the lens' own logprobs give
(`logprob_mass`). An unscored arm — `random` — is a seeded uniform
draw over the same universe, kept because a lens result means nothing without a matched
control, and because once the tree is pruned to the lens arm, drawing a uniform sample is
no longer possible. The control has to be reserved *before* the deletion, not after.

Ranking here is deliberately identical to
`prepare_activations_for_probing/jlens_token_selection.py`, which scores the same CSVs for
a different purpose; `rank_tokens` and `rank_layers_by_direction` are shared by both so
they cannot drift.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path

from .jlens_csv import (
    TokenScore,
    artifact_layers,
    load_direction_tokens,
    output_start,
    read_direction_scores,
    step_folder_index,
)
from .methods import DEFAULT_METHODS, get_method, score_artifact_path, scored_methods
from .scoring import DEFAULT_SCORE, get_score

# Layer 15 is the project's standing comparison point (see general_probe_train.sh), so it is
# kept for every selected token regardless of how it scores. Without that, a layer-15
# baseline would need its own gather run.
DEFAULT_ALWAYS_LAYERS = (15,)


def rank_layers_by_direction(
    token_layers: dict[int, float],
    candidate_layers: list[int],
    score_mode: str = DEFAULT_SCORE,
) -> list[int]:
    """Candidate layers ordered by this token's direction score, best first.

    Ties break on ascending layer index. Shared with `_pick_layers` in
    `jlens_token_selection.py` so the filter and the probe-data preparer rank identically.
    A layer the token has no score for takes the mode's `empty` value, which is 0 for a
    count but a large negative floor for a logprob — using 0 there would rank an unscored
    layer above every scored one.

    >>> rank_layers_by_direction({7: 2, 15: 9, 23: 2}, [7, 15, 23])
    [15, 7, 23]
    >>> rank_layers_by_direction({7: -2.5, 15: -9.0}, [7, 15, 23], "logprob_mass")
    [7, 15, 23]
    """
    empty = get_score(score_mode).empty
    return sorted(candidate_layers, key=lambda layer: (-token_layers.get(layer, empty), layer))


def rank_tokens(
    scores: dict[tuple[int, int], TokenScore],
    candidate_layers: list[int] | None = None,
) -> list[tuple[int, int]]:
    """Token keys ordered by total direction score over `candidate_layers`, best first.

    Each `TokenScore` carries its own `score_mode` and collapses itself accordingly, so this
    is one ordering for every mode — a count sums across layers, a probability mass
    logsumexps — and higher is better in all of them.

    Ties break on step then position, which makes the order total and therefore stable
    across runs and machines. Shared with `select_token_layer_pairs`, which ranks the same
    CSV to build a probe dataset — the two must agree or a prepared dataset and a pruned
    tree would disagree about which tokens matter.

    >>> scores = {(0, 5): TokenScore("a", {7: 1}), (0, 3): TokenScore("b", {7: 1}),
    ...           (0, 9): TokenScore("c", {7: 4})}
    >>> rank_tokens(scores, [7])
    [(0, 9), (0, 3), (0, 5)]
    """
    return sorted(scores, key=lambda key: (-scores[key].total(candidate_layers), key[0], key[1]))


@dataclass(frozen=True)
class TokenPick:
    """One selected reasoning token and the layers of it that are kept.

    `pos` is an absolute forward-pass position in CSV coordinates and an output-relative
    index (the `.pt` filename) after `to_disk_coords`.

    The scoring fields are `None` for an unscored arm. That absence is load-bearing
    downstream: `scripts/split_next_action_manifest.py` decides whether to *rank* a token's
    layers or *sample* them by whether a count is present, so a control that recorded counts
    would silently collapse onto its lowest-scoring layer.

    `direction_count` and `layer_direction_counts` hold the score under whichever
    `scoring.SCORES` mode the arm was built with — a count, or a (negative) logprob. The
    names are unchanged because every pruned tree on disk already has them; `score_mode`
    records which one the numbers are, and every ranker sorts by `-score` either way.
    """

    step: int
    pos: int
    layers: tuple[int, ...]
    token: str = ""
    direction_count: float | None = None
    layer_direction_counts: dict[int, float] | None = None
    score_mode: str = DEFAULT_SCORE

    @property
    def key(self) -> tuple[int, int]:
        return (self.step, self.pos)


Arm = dict[tuple[int, int], TokenPick]


@dataclass(frozen=True)
class KeptTokens:
    """A selection, one arm per method name, each keyed by (step, pos)."""

    arms: dict[str, Arm] = field(default_factory=dict)

    def __getitem__(self, method: str) -> Arm:
        """The named arm, or an empty one — an absent arm and an empty arm mean the same."""
        return self.arms.get(method, {})

    def __contains__(self, method: str) -> bool:
        return method in self.arms

    @property
    def names(self) -> list[str]:
        return list(self.arms)

    def merged(self, arms: list[str] | None = None) -> dict[tuple[int, int], tuple[int, ...]]:
        """Every kept (step, pos) mapped to the union of the layers any arm wants.

        The arms are drawn from the same universe, so a token can land in several; the
        union is what has to survive on disk. Pass `arms` to restrict to named ones — used
        to report which *particular* arm a missing file belongs to.
        """
        chosen = self.arms.values() if arms is None else [self[name] for name in arms]
        out: dict[tuple[int, int], set[int]] = {}
        for arm in chosen:
            for key, pick in arm.items():
                out.setdefault(key, set()).update(pick.layers)
        return {key: tuple(sorted(layers)) for key, layers in sorted(out.items())}

    def activation_paths(
        self, model_folder: Path, category: str = "output", arms: list[str] | None = None
    ) -> set[Path]:
        """The `.pt` files this selection keeps, in the gather_activations layout.

        Only meaningful after `to_disk_coords`: `pos` must be the output-relative token
        index that names the file.
        """
        return {
            model_folder / f"layer_{layer}" / f"step_{step}" / category / f"{pos}.pt"
            for (step, pos), layers in self.merged(arms).items()
            for layer in layers
        }

    def num_files(self) -> int:
        return sum(len(layers) for layers in self.merged().values())

    def with_arm(self, method: str, arm: Arm) -> "KeptTokens":
        """A copy with one arm added or replaced. Used to merge a new arm into a record."""
        return KeptTokens({**self.arms, method: arm})


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
        mode = score.score_mode if score else DEFAULT_SCORE
        picked = rank_layers_by_direction(per_layer, candidate_layers, mode)[:num_layers]
    forced = [layer for layer in always_layers if layer in candidate_layers]
    return tuple(sorted(set(picked) | set(forced)))


def _scored_arm(
    scores: dict[tuple[int, int], TokenScore],
    candidate_layers: list[int],
    num_tokens: int,
    num_layers: int,
    always_layers: tuple[int, ...],
) -> Arm:
    """The top `num_tokens` tokens by direction score, each with its top layers."""
    arm: Arm = {}
    for step, pos in rank_tokens(scores, candidate_layers)[:num_tokens]:
        score = scores[(step, pos)]
        empty = score.score.empty
        layers = _layers_for(score, candidate_layers, num_layers, always_layers, rng=None)
        arm[(step, pos)] = TokenPick(
            step=step,
            pos=pos,
            layers=layers,
            token=score.token,
            direction_count=score.total(candidate_layers),
            layer_direction_counts={layer: score.per_layer.get(layer, empty) for layer in layers},
            score_mode=score.score_mode,
        )
    return arm


def _sampled_arm(
    universe: list[tuple[int, int]],
    candidate_layers: list[int],
    num_tokens: int,
    num_layers: int,
    always_layers: tuple[int, ...],
    rng: random.Random,
) -> Arm:
    """A seeded uniform draw of tokens and of each one's layers.

    Drawn from the *whole* chain, not from what a scored arm left over: overlap between the
    arms is a legitimate outcome of a uniform draw, and excluding the top-scoring tokens
    would make the control systematically low-scoring.

    Carries no counts — see `TokenPick`.
    """
    chosen = sorted(rng.sample(universe, min(num_tokens, len(universe))))
    return {
        (step, pos): TokenPick(
            step=step,
            pos=pos,
            layers=_layers_for(None, candidate_layers, num_layers, always_layers, rng=rng),
        )
        for step, pos in chosen
    }


def arm_seed(seed: int, seed_key: str, method: str) -> str:
    """The seed string an unscored arm draws with.

    `random` keeps the bare `f"{seed}-{seed_key}"` form **permanently**: the activation tree
    has already been pruned to control arms drawn with it, and those draws are the only
    surviving record of a uniform sample over the full reasoning chain. Changing the formula
    would make the recorded control irreproducible. Any method added later gets its own
    stream, so two unscored arms of the same trajectory do not draw identically.

    >>> arm_seed(42, "traj_1", "random")
    '42-traj_1'
    >>> arm_seed(42, "traj_1", "shuffled")
    '42-traj_1-shuffled'
    """
    return f"{seed}-{seed_key}" if method == "random" else f"{seed}-{seed_key}-{method}"


def top_filter(
    signal_json: str | Path,
    trajectory_folder: str | Path,
    *,
    methods: tuple[str, ...] | list[str] = DEFAULT_METHODS,
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
    direction_score: str = DEFAULT_SCORE,
    csv_paths: dict[str, Path] | None = None,
) -> KeptTokens:
    """Select the reasoning tokens and layers worth keeping for one trajectory.

    Args:
        signal_json: JSON mapping UP/DOWN/LEFT/RIGHT to token strings — the signal being
            looked for in each token's top-k lens predictions (e.g.
            `data/jlens/direction_tokens_full.json`).
        trajectory_folder: the folder holding that trajectory's lens artifacts. Each scored
            method resolves its own inside it via `score_artifact_path` — which of the two
            (analysis CSV or direction-mass table) depends on `direction_score`.
        methods: which arms to build, by name (see `methods.METHODS`).
        num_tokens: how many top-scoring tokens each *scored* arm keeps.
        num_layers: how many layers of each selected token to keep.
        always_layers: layers kept for every selected token of *every* arm regardless of
            score, so a fixed-layer baseline stays available after pruning.
        random_tokens: size of an *unscored* arm. Separate from `num_tokens` so the control
            can be sized independently, and because the recorded configs on disk use both.
        random_layers: layers per unscored-arm token (defaults to `num_layers`).
        seed, seed_key: unscored arms draw with `arm_seed(...)`, so a trajectory's sample is
            stable no matter what order trajectories are processed in.
        candidate_layers: the layer pool to choose from. Defaults, *per scored arm*, to the
            layers that arm's own CSV has rows for — the two lenses legitimately cover
            different layer sets, and a layer with no rows can never be scored so must never
            be selected. An unscored arm draws from the union of those pools.
        direction_classes: "all" or a subset such as "UP,DOWN".
        top_k: how many `top_i` columns of the CSV to scan.
        direction_score: which `scoring.SCORES` mode turns a token's direction evidence into
            a number — `count` (the original), a top-k logprob mode that weights each hit by
            how much the lens believed it, or `logprob_mass_full`, which reads the
            direction-mass table instead and so sees the whole vocabulary rather than a
            top-20 window. A top-k logprob mode needs a CSV with `top_i_logprob` columns and
            raises on one written before they existed.
        csv_paths: override the artifact location per method. The gather script filters
            against its still-uncommitted `.tmp` files, which is not where
            `score_artifact_path` looks. Pass the artifact `direction_score` actually
            reads — the mass table for a `source="mass"` score, the analysis CSV otherwise.

    Returns:
        A `KeptTokens` in **CSV coordinates** — `pos` is `abs_pos`. Call `to_disk_coords`
        to turn it into the `.pt` coordinates the activation tree uses. An arm whose
        artifact is missing is absent from the result rather than raising: a trajectory that
        has not been analysed yet is a data state, not a bug.

    Raises:
        ValueError: on an unknown direction score, on a CSV too old to support one, on an
            unknown method name, or when an unscored arm is requested with no
            scored method alongside it. An unscored arm samples over the reasoning chain,
            and a lens CSV's rows are what enumerate that chain — there is nothing to draw
            from otherwise.
    """
    folder = Path(trajectory_folder)
    wanted = list(methods)
    for name in wanted:
        get_method(name)

    scored_wanted = [name for name in wanted if get_method(name).scored]
    unscored_wanted = [name for name in wanted if not get_method(name).scored]
    if unscored_wanted and not scored_wanted:
        raise ValueError(
            f"methods={wanted} asks for {unscored_wanted} with no scored method to enumerate "
            f"the reasoning chain; add one of {scored_methods()}"
        )

    get_score(direction_score)
    direction_tokens = load_direction_tokens(signal_json, direction_classes) if scored_wanted else set()

    # Read each requested lens' artifact once. A missing one leaves the arm out entirely.
    scores_by_method: dict[str, dict[tuple[int, int], TokenScore]] = {}
    pool_by_method: dict[str, list[int]] = {}
    for name in scored_wanted:
        path = (csv_paths or {}).get(name) or score_artifact_path(folder, name, direction_score)
        if path is None or not Path(path).exists():
            continue
        path = Path(path)
        scores = read_direction_scores(path, direction_tokens, top_k=top_k, score_mode=direction_score)
        pool = candidate_layers if candidate_layers is not None else artifact_layers(path, direction_score)
        if scores and pool:
            scores_by_method[name] = scores
            pool_by_method[name] = pool

    if not scores_by_method:
        return KeptTokens()

    arms: dict[str, Arm] = {}
    for name in wanted:
        if get_method(name).scored:
            if name not in scores_by_method:
                continue
            arms[name] = _scored_arm(
                scores_by_method[name], pool_by_method[name], num_tokens, num_layers, always_layers
            )
        else:
            if random_tokens <= 0:
                continue
            # The universe and the layer pool span every lens read, so a control drawn
            # alongside two lenses can reach anything either of them could have picked.
            universe = sorted({key for scores in scores_by_method.values() for key in scores})
            pool = (
                candidate_layers
                if candidate_layers is not None
                else sorted({layer for layers in pool_by_method.values() for layer in layers})
            )
            arms[name] = _sampled_arm(
                universe,
                pool,
                random_tokens,
                num_layers if random_layers is None else random_layers,
                always_layers,
                random.Random(arm_seed(seed, seed_key, name)),
            )

    return KeptTokens(arms)


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

    def convert(arm: Arm) -> Arm:
        out: Arm = {}
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
                score_mode=pick.score_mode,
            )
            out[moved.key] = moved
        return out

    return KeptTokens({name: convert(arm) for name, arm in kept.arms.items()})


__all__ = [
    "DEFAULT_ALWAYS_LAYERS",
    "Arm",
    "KeptTokens",
    "TokenPick",
    "arm_seed",
    "rank_layers_by_direction",
    "rank_tokens",
    "to_disk_coords",
    "top_filter",
]
