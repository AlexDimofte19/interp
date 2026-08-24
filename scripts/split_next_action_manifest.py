#!/usr/bin/env python3
"""Narrow a next_action v3 manifest to the top tokens/layers, then split it by trajectory.

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

3. **Split.** `train_next_action_probe` splits with `torch.randperm` over rows and
   documents the samples as i.i.d. That is false here: one trajectory owns every one of
   its selected tokens, all carrying that trajectory's `agent_action`. A row-level split
   puts near-duplicates of every eval row into training and inflates accuracy. (Same
   landmine as `_prepare_train_eval_v3` in the cognitive-map trainer, per CLAUDE.md.)

Throughout, a `random` control arm is recognised by carrying **no** direction counts, and
is sampled uniformly wherever the jlens arm would be ranked. Ranking a control by a count
that is always absent would collapse every trajectory onto its lowest index and quietly
turn the comparison into a comparison of two jlens-shaped things.

next_action manifests copy no activations -- `act_path` is resolved against the absolute
`activations_root` -- so each output is a lone manifest.json and costs nothing to write.

    python scripts/split_next_action_manifest.py PREPARED_DIR \
        --tokens-per-trajectory 3 --layers-per-token 1

writes PREPARED_DIR_train/manifest.json and PREPARED_DIR_eval/manifest.json.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_manifest(prepared_dir: Path) -> dict:
    """Read the manifest of a prepared dir and check it is a next_action v3 one."""
    manifest_path = prepared_dir / "manifest.json" if prepared_dir.is_dir() else prepared_dir
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("probe_type") != "next_action":
        raise ValueError(f"{manifest_path} has probe_type={manifest.get('probe_type')!r}, expected 'next_action'")
    if "samples" not in manifest:
        raise ValueError(f"{manifest_path} has no 'samples' key; is it a next_action manifest?")
    return manifest


def thin_layers(samples: list[dict], layers_per_token: int, seed: int) -> list[dict]:
    """Keep only the best `layers_per_token` rows of each distinct token.

    Mirrors `_pick_layers`: a jlens_direction arm ranks a token's layers by
    `layer_direction_count` descending, ties broken by ascending layer. A `random` arm
    records no counts, so it draws uniformly instead -- ranking those by a count that is
    always absent would silently collapse the control onto its lowest layer and destroy
    the comparison.
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
            rows = sorted(rows, key=lambda r: (-(r.get("layer_direction_count") or 0), r["layer"]))
        else:
            rows = random.Random(f"{seed}-{key}").sample(rows, layers_per_token)
        kept.extend(rows[:layers_per_token])
    return kept


def trajectory_strata(samples: list[dict]) -> dict[str, int]:
    """Each trajectory's dominant action label.

    Computed from the **unthinned** samples on purpose. On a multi-step trajectory,
    thinning tokens can change which action dominates and so move the trajectory between
    strata -- which would give a different train/eval split at K=1 than at K=3 and make the
    top-K comparison meaningless. Single-step trajectories are unaffected either way.
    """
    labels_by_name: dict[str, Counter] = {}
    for sample in samples:
        labels_by_name.setdefault(sample["name"], Counter())[sample["label"]] += 1
    return {name: counts.most_common(1)[0][0] for name, counts in labels_by_name.items()}


def thin_tokens(samples: list[dict], tokens_per_trajectory: int, seed: int) -> list[dict]:
    """Keep each trajectory's best `tokens_per_trajectory` tokens, all their layers.

    Ranked by `(-direction_count, step, token_id)` -- the same tie-break
    `select_token_layer_pairs` uses -- so this is identical to having prepared with
    `--num-tokens K`, without paying for another prepare.

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
                key=lambda k: (-(tokens[k][0].get("direction_count") or 0), k[2], k[3]),
            )[:tokens_per_trajectory]
        else:
            keys = random.Random(f"{seed}-{name}").sample(sorted(tokens), tokens_per_trajectory)
        for key in sorted(keys, key=lambda k: (str(k[0]), k[2], k[3])):
            kept.extend(tokens[key])
    return kept


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


def label_histogram(samples: list[dict], id_to_action: dict[int, str]) -> str:
    """`UP=12 DOWN=9 ...` over a sample list, for the summary print."""
    counts = Counter(sample["label"] for sample in samples)
    return " ".join(f"{id_to_action.get(k, k)}={counts[k]}" for k in sorted(counts))


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
) -> None:
    """Write a copy of `manifest` carrying only `samples`, plus a record of the split."""
    out = dict(manifest)
    out["samples"] = samples
    out["split"] = {
        "source_manifest": str(manifest.get("_source", "")),
        "num_samples": len(samples),
        "num_trajectories": len({s["name"] for s in samples}),
        "tokens_per_trajectory": tokens_per_trajectory,
        "layers_per_token": layers_per_token,
    }
    out.pop("_source", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(out, f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prepared_dir", type=Path, help="v3 next_action manifest directory")
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
        help="keep only each token's N best layers (1 = one probe, one representation space)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-out", type=Path, default=None, help="default: <prepared_dir>_train")
    parser.add_argument("--eval-out", type=Path, default=None, help="default: <prepared_dir>_eval")
    args = parser.parse_args()

    prepared_dir = args.prepared_dir
    manifest = load_manifest(prepared_dir)
    manifest["_source"] = str(prepared_dir.resolve())
    samples = manifest["samples"]

    # Strata come from the full sample set, so every --tokens-per-trajectory value lands on
    # the same train/eval split and the top-K results stay comparable.
    strata = trajectory_strata(samples)

    if args.tokens_per_trajectory is not None:
        if args.tokens_per_trajectory < 1:
            raise ValueError(f"--tokens-per-trajectory must be >= 1, got {args.tokens_per_trajectory}")
        before = len(samples)
        samples = thin_tokens(samples, args.tokens_per_trajectory, args.seed)
        ranked = any(s.get("direction_count") is not None for s in samples)
        print(f"Tokens: kept {args.tokens_per_trajectory}/trajectory "
              f"({'top-ranked' if ranked else 'uniform draw -- control arm'}), "
              f"{before} -> {len(samples)} samples")

    if args.layers_per_token is not None:
        if args.layers_per_token < 1:
            raise ValueError(f"--layers-per-token must be >= 1, got {args.layers_per_token}")
        before = len(samples)
        samples = thin_layers(samples, args.layers_per_token, args.seed)
        print(f"Layers: kept {args.layers_per_token}/token, {before} -> {len(samples)} samples")
        print(f"  {layer_histogram(samples)}")

    train_names, eval_names = split_names(samples, args.eval_split, args.seed, strata)
    train_samples = [s for s in samples if s["name"] in train_names]
    eval_samples = [s for s in samples if s["name"] in eval_names]

    id_to_action = {v: k for k, v in manifest.get("action_to_id", {}).items()}
    train_out = args.train_out or prepared_dir.with_name(prepared_dir.name + "_train")
    eval_out = args.eval_out or prepared_dir.with_name(prepared_dir.name + "_eval")
    write_split(manifest, train_samples, train_out, args.layers_per_token, args.tokens_per_trajectory)
    write_split(manifest, eval_samples, eval_out, args.layers_per_token, args.tokens_per_trajectory)

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
