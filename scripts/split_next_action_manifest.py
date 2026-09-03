#!/usr/bin/env python3
"""Narrow a token-major v3 manifest to the top tokens/layers, then split it by trajectory.

Token-major means one entry per (token, layer) rather than per trajectory: `next_action`
manifests, and `grid_tile` manifests prepared with a token selection. Both reference the
gathered tree in place through `activations_root` and copy nothing, so a split costs one
JSON file. The two differ only in the label -- one action per entry, or one grid -- and
everything below is about *which entries* survive, so both go through this script
unchanged.

Three reshapes that all have to happen before `train_next_action_probe` sees the data:

1. **Tokens.** `--tokens-per-trajectory K` keeps each trajectory's K best-scoring tokens,
   ranked exactly as `select_token_layer_pairs` ranks them, so it is identical to having
   prepared with `--num-tokens K`. That is what makes a top-1 / top-2 / top-3 sweep cheap:
   one prepared dataset, three splits, rather than three multi-hour prepares.

2. **Layers.** `--num-layers M` in prepare emits each token at its M best layers, and
   the trainer pools them into one (N, D) matrix fit by a single weight vector. Layers
   7 and 22 are not in a shared basis, so one probe cannot read them the same way, and
   the M rows are the same token with the same label anyway. `--layers-per-token 1`
   keeps the top-scoring layer, reproducing a `--num-layers 1` prepare exactly (same
   `(-count, layer)` tie-break as `_pick_layers`) without re-running that slow job.

   `--layers-per-token 1` still leaves the dataset spread over *many* layers: each token
   keeps its own best one. `--single-layer L` instead pins the whole dataset to one layer,
   which is the only way one weight vector reads one representation space. `--single-layer
   best` picks L for you, as the layer with the highest mean direction score across the
   dataset -- but read `best_layer`'s docstring before trusting that number: a manifest
   only holds the layers that were *selected*, so the mean is conditional on having won,
   except at the force-kept layer 15 where it is not. `scripts/jlens_layer_profile.py`
   computes the same thing over the CSVs, where every layer is scored for every token, and
   that is the number to pick L from.

3. **Split.** `train_next_action_probe` splits with `torch.randperm` over rows and
   documents the samples as i.i.d. That is false here: one trajectory owns every one of
   its selected tokens, all carrying that trajectory's `agent_action`. A row-level split
   puts near-duplicates of every eval row into training and inflates accuracy. (Same
   landmine as `_prepare_train_eval_v3` in the cognitive-map trainer, per CLAUDE.md --
   which now refuses an internal split on a token-major manifest and points here.)

   The seeded split is a function of the names *present*, so two datasets covering
   different trajectory sets get different eval sets at the same seed -- which silently
   scores two probes on different test data. `--eval-names FILE` pins the eval set to a
   list instead, so an arm prepared from a wider tree (e.g. the full reasoning-eos tree,
   36000 trajectories) is scored on exactly the trajectories the 3600-trajectory lens
   arms were scored on. Every listed name must be present, or it raises rather than
   quietly shrinking the shared test set.

Throughout, a `random` control arm is recognised by carrying **no** direction counts, and
is sampled uniformly wherever the jlens arm would be ranked. Ranking a control by a count
that is always absent would collapse every trajectory onto its lowest index and quietly
turn the comparison into a comparison of two jlens-shaped things.

Token-major manifests copy no activations -- `act_path` is resolved against the absolute
`activations_root` -- so each output is a lone manifest.json and costs nothing to write.
A grid_tile one also carries a `cells` map, its per-cell payload stored once per
(trajectory, step); each half keeps only the keys its own entries reference.

    python scripts/split_next_action_manifest.py PREPARED_DIR \
        --tokens-per-trajectory 3 --layers-per-token 1

writes PREPARED_DIR_train/manifest.json and PREPARED_DIR_eval/manifest.json.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telos_interp.jlens_utils import DEFAULT_SCORE  # noqa: E402

ENTRY_KEYS = ("samples", "trajectories")


def rank_key(score: float | None) -> tuple[bool, float]:
    """Sort key for a direction score, best first, whatever mode produced it.

    Not `-(score or 0)`: a count's minimum is 0, but a logprob score's *maximum* is 0, so a
    row whose score is missing would sort to the top of a scored group under a logprob mode
    and to the bottom under a count. Missing sorts last in both here, by ranking on
    "has a score" before the score itself.

    >>> sorted([3.0, None, 9.0], key=rank_key)
    [9.0, 3.0, None]
    >>> sorted([-9.0, None, -3.0], key=rank_key)
    [-3.0, -9.0, None]
    """
    return (score is None, -score if score is not None else 0.0)


def entries_key(manifest: dict) -> str:
    """Which manifest key holds the per-sample entries.

    `next_action` calls them `samples`; `grid_tile` keeps the `trajectories` key it has
    always used, token-major or not. The name is all that differs -- both are one entry per
    (token, layer) here.
    """
    for key in ENTRY_KEYS:
        if key in manifest:
            return key
    raise ValueError(f"manifest has none of {ENTRY_KEYS}; is it a v3 prepared manifest?")


def load_manifest(prepared_dir: Path) -> dict:
    """Read the manifest of a prepared dir and check it is a token-major v3 one."""
    manifest_path = prepared_dir / "manifest.json" if prepared_dir.is_dir() else prepared_dir
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    probe_type = manifest.get("probe_type")
    if probe_type not in ("next_action", "grid_tile"):
        raise ValueError(f"{manifest_path} has probe_type={probe_type!r}, expected 'next_action' or 'grid_tile'")
    # `activations_root` is exactly what marks a manifest as token-major. A grid_tile
    # dataset prepared without a selection has one entry per trajectory and copied its
    # activations; it carries no per-token fields for the thinning below to rank.
    if "activations_root" not in manifest:
        raise ValueError(
            f"{manifest_path} copied its activations, so it is one entry per trajectory. "
            "This script splits token-major manifests -- prepare with --token-selection."
        )
    entries_key(manifest)
    return manifest


def thin_layers(samples: list[dict], layers_per_token: int, seed: int) -> list[dict]:
    """Keep only the best `layers_per_token` rows of each distinct token.

    Mirrors `_pick_layers`: a jlens_direction arm ranks a token's layers by
    `layer_direction_count` descending, ties broken by ascending layer. A `random` arm
    records no counts, so it draws uniformly instead -- ranking those by a count that is
    always absent would silently collapse the control onto its lowest layer and destroy
    the comparison.

    The descending order is `rank_key`, which is score-mode agnostic: a count and a
    (negative) logprob are both "higher is better", and a missing score sorts last in both.
    """
    groups: dict[tuple, list[dict]] = {}
    for sample in samples:
        key = (sample.get("size"), sample["name"], sample["step"], sample["token_id"])
        groups.setdefault(key, []).append(sample)

    kept: list[dict] = []
    for key in sorted(groups, key=lambda k: tuple(str(part) for part in k)):
        rows = groups[key]
        if len(rows) <= layers_per_token:
            kept.extend(rows)
            continue
        if any(row.get("layer_direction_count") is not None for row in rows):
            rows = sorted(rows, key=lambda r: (rank_key(r.get("layer_direction_count")), r["layer"]))
        else:
            rows = random.Random(f"{seed}-{key}").sample(rows, layers_per_token)
        kept.extend(rows[:layers_per_token])
    return kept


def trajectory_strata(samples: list[dict]) -> dict[str, int]:
    """Each trajectory's stratum: its dominant action label, or its grid size.

    Computed from the **unthinned** samples on purpose. On a multi-step trajectory,
    thinning tokens can change which action dominates and so move the trajectory between
    strata -- which would give a different train/eval split at K=1 than at K=3 and make the
    top-K comparison meaningless. Single-step trajectories are unaffected either way.

    A grid_tile entry has no scalar label -- a trajectory carries one per cell -- so those
    stratify by grid size instead. That is the dimension whose mix has to match across the
    halves for a size-pooled probe to be readable: a 5x5 grid and a 15x15 grid padded to
    the same width have very different label distributions.
    """
    field = "label" if all("label" in sample for sample in samples) else "size"
    labels_by_name: dict[str, Counter] = {}
    for sample in samples:
        labels_by_name.setdefault(sample["name"], Counter())[sample.get(field, 0)] += 1
    return {name: counts.most_common(1)[0][0] for name, counts in labels_by_name.items()}


def thin_tokens(samples: list[dict], tokens_per_trajectory: int, seed: int) -> list[dict]:
    """Keep each trajectory's best `tokens_per_trajectory` tokens, all their layers.

    Ranked by `(rank_key(direction_count), step, token_id)` -- the same tie-break
    `select_token_layer_pairs` uses -- so this is identical to having prepared with
    `--num-tokens K`, without paying for another prepare. The score may be a count or a
    logprob; higher is better either way.

    A `random` control arm carries no `direction_count`, so it is sampled uniformly
    instead; ranking it by an always-absent count would take the lowest step/token index of
    every trajectory and stop being a control.
    """
    groups: dict[tuple, dict[tuple, list[dict]]] = {}
    for sample in samples:
        token_key = (sample.get("size"), sample["name"], sample["step"], sample["token_id"])
        groups.setdefault(sample["name"], {}).setdefault(token_key, []).append(sample)

    kept: list[dict] = []
    for name in sorted(groups):
        tokens = groups[name]
        if len(tokens) <= tokens_per_trajectory:
            keys = list(tokens)
        elif any(row.get("direction_count") is not None for rows in tokens.values() for row in rows):
            keys = sorted(
                tokens,
                key=lambda k: (rank_key(tokens[k][0].get("direction_count")), k[2], k[3]),
            )[:tokens_per_trajectory]
        else:
            keys = random.Random(f"{seed}-{name}").sample(sorted(tokens), tokens_per_trajectory)
        for key in sorted(keys, key=lambda k: (str(k[0]), k[2], k[3])):
            kept.extend(tokens[key])
    return kept


def mean_layer_scores(samples: list[dict]) -> dict[int, tuple[float, int]]:
    """{layer: (mean per-layer direction score, number of entries)} over the scored entries.

    Entries with no `layer_direction_count` -- a control arm's -- are skipped rather than
    counted as zero, which under a logprob score would be the maximum.
    """
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for sample in samples:
        value = sample.get("layer_direction_count")
        if value is None:
            continue
        layer = sample["layer"]
        totals[layer] = totals.get(layer, 0.0) + value
        counts[layer] = counts.get(layer, 0) + 1
    return {layer: (totals[layer] / counts[layer], counts[layer]) for layer in sorted(counts)}


def best_layer(samples: list[dict]) -> int | None:
    """The layer with the highest mean direction score in this manifest, or None if unscored.

    Ties break on the lower layer index, matching `rank_layers_by_direction`.

    **This mean is conditional.** A manifest holds a (token, layer) entry only when that
    layer was selected for that token, so a layer's mean here is its mean *given that it
    beat the other layers of that token* -- except layer 15, which is force-kept for every
    token (`top_filter.DEFAULT_ALWAYS_LAYERS`) and so carries an unconditional mean. The
    two are not comparable, and the bias runs against layer 15. The unconditional number
    comes from the CSVs, where every token is scored at every layer:
    `scripts/jlens_layer_profile.py`.

    >>> best_layer([{"layer": 7, "layer_direction_count": 1.0},
    ...             {"layer": 15, "layer_direction_count": 4.0}])
    15
    >>> best_layer([{"layer": 7}]) is None
    True
    """
    means = mean_layer_scores(samples)
    if not means:
        return None
    return min(means, key=lambda layer: (-means[layer][0], layer))


def keep_single_layer(samples: list[dict], layer: int) -> list[dict]:
    """Every entry at `layer`, dropping the rest.

    Unlike `thin_layers` this is not per token: a token with no entry at `layer` leaves the
    dataset entirely. Only layer 15 is guaranteed present for every selected token, so any
    other choice thins the token set too -- the caller reports by how much.
    """
    return [sample for sample in samples if sample["layer"] == layer]


def format_layer_means(samples: list[dict], chosen: int | None = None) -> str:
    """A per-layer mean/coverage table for the summary print."""
    means = mean_layer_scores(samples)
    tokens = len({(s.get("size"), s["name"], s["step"], s["token_id"]) for s in samples})
    lines = [f"{'layer':>5} {'mean':>9} {'entries':>8} {'coverage':>9}"]
    for layer, (mean, count) in means.items():
        mark = "  <- chosen" if layer == chosen else ""
        lines.append(f"{layer:>5} {mean:>9.4f} {count:>8} {count / tokens:>8.1%}{mark}")
    return "\n".join(lines)


def split_names(
    samples: list[dict], eval_split: float, seed: int, strata: dict[str, int] | None = None
) -> tuple[set[str], set[str]]:
    """Partition the unique trajectory names into (train, eval), stratified by label.

    Grouping alone can starve a class out of eval -- and `train_next_action_probe`
    silently drops eval samples whose label never appears in training. So trajectories
    are bucketed by their dominant action first and each bucket split separately.
    Names are sorted before shuffling so the split depends only on the seed, not on the
    order `prepare_activations_for_probing` happened to walk the folders in.

    `strata` should come from the unthinned samples (see `trajectory_strata`) so that every
    `--tokens-per-trajectory` value produces the same split and the results stay comparable.
    """
    present = {sample["name"] for sample in samples}
    stratum_of = (
        {name: label for name, label in strata.items() if name in present}
        if strata is not None
        else trajectory_strata(samples)
    )

    rng = random.Random(seed)
    train: set[str] = set()
    evaluation: set[str] = set()
    for label in sorted({stratum_of[name] for name in stratum_of}):
        names = sorted(name for name in stratum_of if stratum_of[name] == label)
        rng.shuffle(names)
        num_eval = min(len(names) - 1, max(1, round(len(names) * eval_split))) if eval_split > 0 else 0
        evaluation.update(names[:num_eval])
        train.update(names[num_eval:])

    if not train or (eval_split > 0 and not evaluation):
        raise ValueError(f"eval_split={eval_split} produced an empty split from {len(stratum_of)} trajectories")
    return train, evaluation


def eval_names_from_file(samples: list[dict], path: Path) -> tuple[set[str], set[str]]:
    """Use the trajectory names in `path` as the eval set verbatim; the rest train.

    `split_names` derives the partition from the names present plus the seed, so two
    datasets covering different trajectory sets get different eval sets even at the same
    seed. Comparing probes then compares them on different test data. This pins the eval
    set instead, so an arm prepared from a wider tree can be scored on exactly the
    trajectories the narrower arms were scored on.

    Every listed name must be present -- a silently-missing one would shrink the shared
    test set and reintroduce the mismatch this exists to prevent.
    """
    wanted = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not wanted:
        raise ValueError(f"{path} lists no trajectory names")

    present = {sample["name"] for sample in samples}
    missing = wanted - present
    if missing:
        example = ", ".join(sorted(missing)[:3])
        raise ValueError(
            f"{len(missing)} of {len(wanted)} names in {path} are absent from the manifest "
            f"(e.g. {example}). The eval set would not match the arms it is meant to share."
        )

    train = present - wanted
    if not train:
        raise ValueError(f"{path} covers every trajectory in the manifest; nothing left to train on")
    return train, set(wanted)


def label_histogram(samples: list[dict], id_to_action: dict[int, str]) -> str:
    """`UP=12 DOWN=9 ...` over a sample list, for the summary print.

    A grid_tile split falls back to `size5=120 size7=80 ...`: those entries carry a whole
    grid rather than one label, and the size mix is the thing worth eyeballing there.
    """
    if all("label" in sample for sample in samples):
        counts = Counter(sample["label"] for sample in samples)
        return " ".join(f"{id_to_action.get(k, k)}={counts[k]}" for k in sorted(counts))
    sizes = Counter(sample["size"] for sample in samples if "size" in sample)
    return " ".join(f"size{k}={sizes[k]}" for k in sorted(sizes))


def layer_histogram(samples: list[dict]) -> str:
    """`L15=340 L16=120 ...`, so the layers the direction signal picked are visible."""
    counts = Counter(sample["layer"] for sample in samples)
    return " ".join(f"L{layer}={counts[layer]}" for layer in sorted(counts))


def write_split(
    manifest: dict,
    samples: list[dict],
    out_dir: Path,
    layers_per_token: int | None,
    tokens_per_trajectory: int | None = None,
    key: str = "samples",
    single_layer: int | None = None,
) -> None:
    """Write a copy of `manifest` carrying only `samples`, plus a record of the split."""
    out = dict(manifest)
    out[key] = samples
    if "cells" in manifest:
        # Each half keeps only the per-cell payloads its own entries point at, so an eval
        # manifest does not carry the training grids around with it.
        referenced = {sample["cells_key"] for sample in samples}
        out["cells"] = {k: v for k, v in manifest["cells"].items() if k in referenced}
    out["split"] = {
        "source_manifest": str(manifest.get("_source", "")),
        "num_samples": len(samples),
        "num_trajectories": len({s["name"] for s in samples}),
        "tokens_per_trajectory": tokens_per_trajectory,
        "layers_per_token": layers_per_token,
        "single_layer": single_layer,
    }
    out.pop("_source", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(out, f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prepared_dir", type=Path, help="v3 token-major manifest directory (next_action or grid_tile)")
    parser.add_argument("--eval-split", type=float, default=0.2, help="fraction of trajectories held out")
    parser.add_argument(
        "--tokens-per-trajectory",
        type=int,
        default=None,
        help="keep only each trajectory's K top-ranked tokens (same ranking as --num-tokens K "
        "in prepare); a control arm is sampled uniformly instead",
    )
    parser.add_argument(
        "--layers-per-token",
        type=int,
        default=None,
        help="keep only each token's N best layers -- per token, so the dataset still spans "
        "many layers. Use --single-layer for one layer across the whole dataset",
    )
    parser.add_argument(
        "--single-layer",
        default=None,
        metavar="L|best",
        help="pin the WHOLE dataset to one layer, so a single weight vector reads a single "
        "representation space. 'best' picks the layer with the highest mean direction score "
        "in this manifest -- but that mean is conditional on selection (see best_layer's "
        "docstring); take L from scripts/jlens_layer_profile.py instead when it matters. "
        "Any layer but 15 also drops the tokens that never selected it. Pass the same "
        "explicit L to a control arm to keep the comparison matched",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-names",
        type=Path,
        default=None,
        help="file of trajectory names, one per line, to use as the eval set verbatim; "
        "everything else trains. Overrides --eval-split. Use it to hold a fixed test "
        "set across datasets that do not cover the same trajectories",
    )
    parser.add_argument("--train-out", type=Path, default=None, help="default: <prepared_dir>_train")
    parser.add_argument("--eval-out", type=Path, default=None, help="default: <prepared_dir>_eval")
    args = parser.parse_args()

    prepared_dir = args.prepared_dir
    manifest = load_manifest(prepared_dir)
    manifest["_source"] = str(prepared_dir.resolve())
    key = entries_key(manifest)
    samples = manifest[key]

    # Strata come from the full sample set, so every --tokens-per-trajectory value lands on
    # the same train/eval split and the top-K results stay comparable.
    strata = trajectory_strata(samples)

    if args.tokens_per_trajectory is not None:
        if args.tokens_per_trajectory < 1:
            raise ValueError(f"--tokens-per-trajectory must be >= 1, got {args.tokens_per_trajectory}")
        before = len(samples)
        samples = thin_tokens(samples, args.tokens_per_trajectory, args.seed)
        ranked = any(s.get("direction_count") is not None for s in samples)
        print(
            f"Tokens: kept {args.tokens_per_trajectory}/trajectory "
            f"({'top-ranked' if ranked else 'uniform draw -- control arm'}), "
            f"{before} -> {len(samples)} samples"
        )

    if args.layers_per_token is not None and args.single_layer is not None:
        raise ValueError("--layers-per-token and --single-layer both choose layers; pass one")

    if args.layers_per_token is not None:
        if args.layers_per_token < 1:
            raise ValueError(f"--layers-per-token must be >= 1, got {args.layers_per_token}")
        before = len(samples)
        samples = thin_layers(samples, args.layers_per_token, args.seed)
        print(f"Layers: kept {args.layers_per_token}/token, {before} -> {len(samples)} samples")
        print(f"  {layer_histogram(samples)}")

    if args.single_layer is not None:
        score_mode = manifest.get("selection", {}).get("direction_score", DEFAULT_SCORE)
        if args.single_layer == "best":
            chosen = best_layer(samples)
            if chosen is None:
                raise ValueError(
                    "--single-layer best needs per-layer direction scores, and this manifest "
                    "carries none (a control arm records none by design). Pass an explicit "
                    "layer -- the one the matching lens arm chose."
                )
        else:
            chosen = int(args.single_layer)
        print(f"Layer means over this manifest ({score_mode}, conditional on selection):")
        print(format_layer_means(samples, chosen))
        before_samples = len(samples)
        before_tokens = len({(s.get("size"), s["name"], s["step"], s["token_id"]) for s in samples})
        samples = keep_single_layer(samples, chosen)
        if not samples:
            raise ValueError(f"--single-layer {chosen} matched no entries in {prepared_dir}")
        after_tokens = len({(s.get("size"), s["name"], s["step"], s["token_id"]) for s in samples})
        print(
            f"Layers: pinned to L{chosen}, {before_samples} -> {len(samples)} samples, "
            f"{before_tokens} -> {after_tokens} tokens"
        )
        if after_tokens < before_tokens:
            print(
                f"  NOTE: {before_tokens - after_tokens} token(s) had no entry at L{chosen} and "
                "are gone; only L15 is kept for every selected token."
            )

    if args.eval_names is not None:
        train_names, eval_names = eval_names_from_file(samples, args.eval_names)
    else:
        train_names, eval_names = split_names(samples, args.eval_split, args.seed, strata)
    train_samples = [s for s in samples if s["name"] in train_names]
    eval_samples = [s for s in samples if s["name"] in eval_names]

    id_to_action = {v: k for k, v in manifest.get("action_to_id", {}).items()}
    train_out = args.train_out or prepared_dir.with_name(prepared_dir.name + "_train")
    eval_out = args.eval_out or prepared_dir.with_name(prepared_dir.name + "_eval")
    single = None if args.single_layer is None else chosen
    write_split(manifest, train_samples, train_out, args.layers_per_token, args.tokens_per_trajectory, key, single)
    write_split(manifest, eval_samples, eval_out, args.layers_per_token, args.tokens_per_trajectory, key, single)

    num_trajectories = len(train_names) + len(eval_names)
    print(f"Source: {prepared_dir}  ({len(samples)} samples, {num_trajectories} trajectories)")
    for split_name, split_samples, split_names_set in (
        ("train", train_samples, train_names),
        ("eval ", eval_samples, eval_names),
    ):
        histogram = label_histogram(split_samples, id_to_action)
        print(f"  {split_name} {len(split_samples):>7} samples / {len(split_names_set):>5} trajectories  {histogram}")
    print(f"Wrote {train_out}/manifest.json and {eval_out}/manifest.json")


if __name__ == "__main__":
    main()
