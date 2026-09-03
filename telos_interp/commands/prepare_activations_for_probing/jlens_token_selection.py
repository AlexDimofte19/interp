"""Pick which reasoning tokens and layers a `next_action` probe trains on.

`scripts/jlens_reasoning_tokens.py` writes, per trajectory, both the residual-stream
activations (gather_activations layout) and one `{stem}_{lens}_analysis.csv` per lens,
holding the top-20 predictions for every (reasoning token, layer). This module turns those
CSVs into a selection: the tokens whose lens output is most *direction-loaded* (the
per-trajectory version of notebooks/direction_token_location_analysis.ipynb), plus the
matched random controls.

The method registry, CSV reading, scoring and ranking all live in
`telos_interp.jlens_utils`, shared with the two scripts that prune the activation tree, so
a probe dataset and a pruned tree can never disagree about which tokens matter. What stays
here is the probe-side wrapping: the CLI's selection modes, and the `SampleRef` list a
manifest is built from.

The mode names are **generated from the registry** rather than listed, so a new lens gets
its `<lens>_direction` and `recorded_<lens>` modes for free.

Stdlib only — no torch — so the selection logic is testable on its own.
"""

import random
from dataclasses import dataclass
from pathlib import Path

from telos_interp.jlens_utils import (
    DEFAULT_JLENS_CSV_SUFFIX,
    DEFAULT_SCORE,
    DIRECTION_CLASSES,
    METHODS,
    TokenScore,
    get_score,
    jlens_csv_path,
    load_direction_tokens,
    output_start,
    rank_layers_by_direction,
    rank_tokens,
    read_direction_counts,
    read_direction_scores,
    read_selection_record,
    record_path,
    score_artifact_path,
    score_names,
    scored_methods,
    step_folder_index,
)

# Re-exported so prepare_activations_for_probing_fn and existing callers keep importing
# these from here.
__all__ = [
    "DEFAULT_JLENS_CSV_SUFFIX",
    "DEFAULT_SCORE",
    "DIRECTION_CLASSES",
    "RECORDED_SELECTIONS",
    "SCORED_SELECTIONS",
    "SampleRef",
    "SelectionConfig",
    "TokenScore",
    "TokenSelection",
    "jlens_csv_path",
    "load_direction_tokens",
    "output_start",
    "read_direction_counts",
    "score_names",
    "select_token_layer_pairs",
    "step_folder_index",
]

TokenSelection = str  # "all" | "<lens>_direction" | "random" | "recorded_<method>"
LayerSelection = str  # "spec" | "<lens>_direction" | "random"

# Scored modes that re-score a lens CSV, mapped to the method they score. `jlens_direction`
# is the pre-existing name and keeps working -- the recorded invocations in configs/**/*.conf
# depend on it.
SCORED_SELECTIONS = {f"{name}_direction": name for name in scored_methods()}

# Modes that read a pruned tree's {stem}_jlens_selection.json instead of re-scoring, mapped
# to the arm of the record they take.
RECORDED_SELECTIONS = {f"recorded_{name}": name for name in METHODS}

TOKEN_SELECTIONS = ("all", "random", *SCORED_SELECTIONS, *RECORDED_SELECTIONS)
LAYER_SELECTIONS = ("spec", "random", *SCORED_SELECTIONS)


@dataclass
class SelectionConfig:
    """How `next_action` picks its (token, layer) samples.

    Bundled so the seven knobs travel as one argument; the public
    `prepare_activations_for_probing` signature keeps them flat for the CLI.
    See that function's docstring for what each mode means.
    """

    token_selection: TokenSelection = "all"
    layer_selection: LayerSelection = "spec"
    num_tokens: int | None = None
    num_layers: int | None = None
    direction_tokens: set[str] | None = None
    jlens_top_k: int = 20
    seed: int | None = 42
    # Kept for the manifest's provenance record; `direction_tokens` is what is used.
    direction_tokens_path: str | None = None
    direction_classes: str = "all"
    # Which `jlens_utils.scoring` mode turns a row's direction hits into a number. Only
    # consulted by the modes that re-score a CSV: a `recorded_*` selection takes the score
    # the pruning run computed, which is fixed on disk and recorded in the arm's config.
    direction_score: str = DEFAULT_SCORE

    @property
    def is_default(self) -> bool:
        """True when selection reduces to the pre-existing behaviour (no CSV needed)."""
        return self.token_selection == "all" and self.layer_selection == "spec"

    @property
    def method(self) -> str | None:
        """The registry method this selection scores or reads, if any.

        >>> SelectionConfig(token_selection="logitlens_direction").method
        'logitlens'
        >>> SelectionConfig(token_selection="recorded_random").method
        'random'
        >>> SelectionConfig().method is None
        True
        """
        return SCORED_SELECTIONS.get(self.token_selection) or RECORDED_SELECTIONS.get(self.token_selection)

    def to_manifest(self) -> dict:
        """JSON-serializable record of how the samples were chosen."""
        return {
            "token_selection": self.token_selection,
            "layer_selection": self.layer_selection,
            "method": self.method,
            "num_tokens": self.num_tokens,
            "num_layers": self.num_layers,
            "direction_tokens_path": self.direction_tokens_path,
            "direction_classes": self.direction_classes,
            "direction_score": self.direction_score,
            "num_direction_tokens": len(self.direction_tokens) if self.direction_tokens else None,
            "jlens_top_k": self.jlens_top_k,
            "seed": self.seed,
        }


@dataclass
class SampleRef:
    """One (token, layer) activation file selected for training."""

    layer: int
    step: int  # step folder index (step_id), i.e. what layer_N/step_M uses
    token_idx: int  # output-relative index, i.e. the .pt filename
    path: Path
    token: str = ""
    # The token's direction score, under the manifest's `direction_score` mode -- a count,
    # or a (negative) logprob. Higher is better in every mode, which is what lets every
    # ranker downstream sort by `-score` without knowing which mode produced it.
    direction_count: float | None = None  # token score aggregated over candidate layers
    layer_direction_count: float | None = None  # score at this layer alone


def _pick_layers(
    token_layers: dict[int, float],
    candidate_layers: list[int],
    layer_selection: LayerSelection,
    num_layers: int | None,
    rng: random.Random,
    score_mode: str = DEFAULT_SCORE,
) -> list[int]:
    """Choose which layers of one token become samples."""
    if layer_selection == "spec" or num_layers is None:
        return list(candidate_layers)
    if layer_selection == "random":
        if num_layers >= len(candidate_layers):
            return list(candidate_layers)
        return sorted(rng.sample(candidate_layers, num_layers))
    if layer_selection in SCORED_SELECTIONS:
        return sorted(rank_layers_by_direction(token_layers, candidate_layers, score_mode)[:num_layers])
    raise ValueError(f"Unknown layer_selection: {layer_selection}")


def _select_from_record(
    *,
    trajectory_folder: Path,
    model_folder: Path,
    candidate_layers: list[int],
    candidate_steps: list[int],
    selection: SelectionConfig,
    verbose: bool = False,
) -> tuple[list["SampleRef"], int]:
    """Take the selection a pruning run already made, instead of re-scoring the CSV.

    On a tree that has been pruned to a selection, re-scoring works for a lens arm but
    *not* for the control: a uniform draw over the surviving tokens is no longer a uniform
    draw over the reasoning chain. The record is the only place the control survives, so
    every arm is read from it.

    `candidate_layers`/`candidate_steps` still narrow the result, which is what makes
    `--layers 15` a layer-15 dataset carved out of the same record.
    """
    path = record_path(trajectory_folder)
    if not path.exists():
        if verbose:
            print(f"  Skipped: no selection record at {path}")
        return [], 0

    method = RECORDED_SELECTIONS[selection.token_selection]
    kept, _ = read_selection_record(path)
    arm = kept[method]
    if not arm and verbose:
        # A record that predates this method -- e.g. asking for `logitlens` on a tree pruned
        # before that arm existed. Nothing to train on, and nothing that can be recovered
        # here: the arm has to be added by jlens_reasoning_tokens.py --extend.
        print(f"  Skipped: selection record at {path} has no '{method}' arm (has {kept.names})")
    step_set, layer_set = set(candidate_steps), set(candidate_layers)

    # An explicit num_tokens caps the arm to its top-K. The record is written in rank order
    # for the jlens arm, but sort defensively rather than trusting the file's order.
    # A pick with no score is a control arm's; it sorts to the bottom rather than to the top,
    # which `-(x or 0)` would do under a logprob score where 0 is the maximum.
    picks = sorted(
        arm.values(),
        key=lambda p: (
            -(p.direction_count if p.direction_count is not None else get_score(p.score_mode).empty),
            p.step,
            p.pos,
        ),
    )
    if selection.num_tokens is not None:
        picks = picks[: selection.num_tokens]

    samples: list[SampleRef] = []
    missing = 0
    for pick in sorted(picks, key=lambda p: (p.step, p.pos)):
        if pick.step not in step_set:
            continue
        for layer in pick.layers:
            if layer not in layer_set:
                continue
            file_path = model_folder / f"layer_{layer}" / f"step_{pick.step}" / "output" / f"{pick.pos}.pt"
            if not file_path.exists():
                missing += 1
                continue
            samples.append(
                SampleRef(
                    layer=layer,
                    step=pick.step,
                    token_idx=pick.pos,
                    path=file_path,
                    token=pick.token,
                    direction_count=pick.direction_count,
                    layer_direction_count=(pick.layer_direction_counts or {}).get(layer)
                    if pick.direction_count is not None
                    else None,
                )
            )
    return samples, missing


def select_token_layer_pairs(
    *,
    trajectory_folder: Path,
    model_folder: Path,
    trajectory_data: dict,
    candidate_layers: list[int],
    candidate_steps: list[int],
    available_tokens_by_step: dict[int, list[int]],
    selection: SelectionConfig,
    verbose: bool = False,
) -> tuple[list[SampleRef], int]:
    """Select the (token, layer) activation files to train on for one trajectory.

    Args:
        trajectory_folder: `<activations_dir>/<trajectory name>`, where the jlens CSV lives.
        model_folder: the `<model>` folder inside it (holds `layer_N/step_M/output`).
        trajectory_data: parsed trajectory JSON, used to map abs_pos -> output-relative index.
        candidate_layers: layers to choose from (the `layers` spec ∩ what is on disk).
        candidate_steps: step folder indices to choose from (the `steps` spec ∩ disk).
        available_tokens_by_step: {step folder index: sorted output token indices on disk}.
        selection: the modes, N, M, direction vocabulary and seed. The random draws are
            seeded per trajectory, so a trajectory's sample is stable regardless of
            iteration order or subsetting.

    Returns:
        (samples, missing) — the selected files, and how many selections pointed at a
        `.pt` that does not exist (skipped). Empty when the trajectory cannot be scored
        (e.g. the jlens CSV is absent).
    """
    if selection.token_selection in RECORDED_SELECTIONS:
        return _select_from_record(
            trajectory_folder=trajectory_folder,
            model_folder=model_folder,
            candidate_layers=candidate_layers,
            candidate_steps=candidate_steps,
            selection=selection,
            verbose=verbose,
        )

    rng = random.Random(f"{selection.seed}-{trajectory_folder.name}")
    num_tokens, num_layers = selection.num_tokens, selection.num_layers
    # Either axis can name a lens; they must name the same one, which the caller validates.
    scoring_method = SCORED_SELECTIONS.get(selection.token_selection) or SCORED_SELECTIONS.get(
        selection.layer_selection
    )

    scores: dict[tuple[int, int], TokenScore] = {}
    if scoring_method is not None:
        if selection.direction_tokens is None:
            raise ValueError(f"direction_tokens is required when a '{scoring_method}' mode is used")
        # Which artifact -- the analysis CSV, or the direction-mass table -- is the score's
        # to decide, not this function's.
        csv_path = score_artifact_path(trajectory_folder, scoring_method, selection.direction_score)
        if csv_path is None or not csv_path.exists():
            if verbose:
                print(f"  Skipped: no {scoring_method} {selection.direction_score} artifact at {csv_path}")
            return [], 0
        scores = read_direction_scores(
            csv_path,
            selection.direction_tokens,
            top_k=selection.jlens_top_k,
            score_mode=selection.direction_score,
        )

    # The CSV is keyed by (csv step, abs_pos); re-key it onto the on-disk
    # (step folder, output-relative index) coordinates the .pt files use.
    by_disk: dict[tuple[int, int], TokenScore] = {}
    for (csv_step, abs_pos), score in scores.items():
        try:
            key = (step_folder_index(trajectory_data, csv_step), abs_pos - output_start(trajectory_data, csv_step))
        except (IndexError, KeyError):
            continue
        by_disk[key] = score

    # Candidate tokens as (step folder index, output-relative index).
    step_set = set(candidate_steps)
    if selection.token_selection in SCORED_SELECTIONS:
        # rank_tokens is the same ordering jlens_utils.top_filter selects with, shared so a
        # prepared dataset and a pruned tree cannot disagree about which tokens rank highest.
        candidates = rank_tokens(
            {key: score for key, score in by_disk.items() if key[0] in step_set}, candidate_layers
        )
        if num_tokens is not None:
            if verbose and num_tokens > len(candidates):
                print(f"  Only {len(candidates)} scored tokens available (asked for {num_tokens})")
            candidates = candidates[:num_tokens]
    else:
        candidates = [
            (step_idx, token_idx)
            for step_idx in candidate_steps
            for token_idx in available_tokens_by_step.get(step_idx, [])
        ]
        if selection.token_selection == "random" and num_tokens is not None:
            if num_tokens < len(candidates):
                candidates = sorted(rng.sample(candidates, num_tokens))
            elif verbose:
                print(f"  Only {len(candidates)} tokens available (asked for {num_tokens})")

    samples: list[SampleRef] = []
    missing = 0
    for step_idx, token_idx in candidates:
        score = by_disk.get((step_idx, token_idx))
        token = score.token if score else ""
        total = score.total(candidate_layers) if score else None
        token_layers = score.per_layer if score else {}
        for layer in _pick_layers(
            token_layers,
            candidate_layers,
            selection.layer_selection,
            num_layers,
            rng,
            selection.direction_score,
        ):
            path = model_folder / f"layer_{layer}" / f"step_{step_idx}" / "output" / f"{token_idx}.pt"
            if not path.exists():
                missing += 1
                continue
            samples.append(
                SampleRef(
                    layer=layer,
                    step=step_idx,
                    token_idx=token_idx,
                    path=path,
                    token=token,
                    direction_count=total,
                    layer_direction_count=token_layers.get(layer) if score else None,
                )
            )
    return samples, missing
