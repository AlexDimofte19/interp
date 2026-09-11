#!/usr/bin/env python3
"""Per-token jlens *loudness* against a convinced / not-convinced label.

Training data for a classifier that asks one question: from how direction-loaded the residual
stream is at a single reasoning token, can you tell whether the model has already committed to
the right answer?

Two artifacts, one row per reasoning token, joined on ``(step, abs_pos)``:

  * ``{stem}_{lens}_direction_mass.csv`` -- the FEATURE. Each ``L{n}`` cell is
    ``log P(any direction word)`` at that (token, layer) over the whole 446-token
    ``direction_tokens_full.json``, computed on-device while the logits were still there. This
    is the full-vocabulary mass of log entry 42, not the top-20 count of entries 39/40.
  * ``telos_interp/loudness_analysis/rollouts/run_inference.py`` rollouts -- the LABEL, via
    ``convinced_sentence_idx``: the first sentence from which every later truncation answers
    correctly. See ``jlens_utils.commitment`` for the index-space trap between the mass table's
    ``reasoning_pos`` and the rollout's ``eos_token_pos``.

``--selection-arm`` restricts the rows to one arm of the ``{stem}_jlens_selection.json`` record
-- ``jlens`` for the top-20-by-``logprob_mass_full`` tokens, ``random`` for the matched uniform
control drawn with the same seed. Omit it (``none``) to keep every reasoning token, which is what
the held-out evaluation set wants. The arm is a row FILTER only: ``n_reasoning_tokens``,
``reasoning_frac`` and ``convinced_reasoning_frac`` are computed over the unfiltered chain, so a
token means the same thing in every dataset.

Nothing here loads a model or reads a ``.pt``: the feature is already a number in a CSV.

    python scripts/build_convinced_dataset.py \
        --exclude-names /workspace/prepared/next_action_mass_l15_eval_names.txt \
        --selection-arm random --out train_random20.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telos_interp.jlens_utils import (  # noqa: E402
    cohort,
    convinced_label,
    direction_mass_path,
    eos_positions,
    load_direction_tokens,
    mass_csv_layers,
    place_token,
    read_mass_meta,
    read_raw_record,
    read_rollout_steps,
    reasoning_offset,
    record_path,
    rollout_path,
    scored_methods,
    sentence_of_token,
)

IDENTITY = ["name", "size", "complexity", "run", "step", "reasoning_pos", "abs_pos", "token", "arm"]
LABEL = ["is_convinced", "cohort", "convinced_idx", "first_correct_idx", "rel_sentence"]
CONTEXT = [
    "n_reasoning_tokens",
    "reasoning_frac",
    "n_sentences",
    "sentence_idx",
    "pos_in_sentence",
    "sentence_len",
    "sentence_frac",
    "is_sentence_end",
    "x_sentence",
    "convinced_reasoning_frac",
    "n_switches",
]
ROLLOUT = ["sent_model_action", "sent_correct", "final_action", "ground_truth"]


def logmass_column(layer: int) -> str:
    """Feature column for one layer's loudness.

    >>> logmass_column(15)
    'dir_logmass_L15'
    """
    return f"dir_logmass_L{layer}"


def parse_layers(spec: str, available: list[int]) -> list[int]:
    """Which layers to emit as features. ``all`` is every layer the mass table covers.

    Raises rather than silently emitting a short row when a requested layer has no column --
    a missing feature would otherwise look like a legitimate value downstream.

    >>> parse_layers("all", [7, 15, 23])
    [7, 15, 23]
    >>> parse_layers("15,7", [7, 15, 23])
    [7, 15]
    """
    if spec.strip() == "all":
        return list(available)
    wanted = sorted({int(part) for part in spec.split(",") if part.strip()})
    missing = [layer for layer in wanted if layer not in available]
    if missing:
        raise SystemExit(f"--layers asks for {missing}, but the mass table only has {available}")
    return wanted


def read_names(path: Path | None) -> set[str]:
    """A whitespace-separated trajectory-name list, or an empty set."""
    if path is None or not str(path):
        return set()
    return set(Path(path).read_text().split())


def check_vocabulary(mass_path: Path, signal_json: Path) -> None:
    """Refuse to read a mass table whose sidecar names a different vocabulary.

    This project points two vocabularies (`direction_tokens_full.json`, `grid_tokens_full.json`)
    at the same activation trees, and a mass table's numbers are meaningless without knowing
    which produced them -- so a table with no sidecar at all is also refused.
    """
    meta = read_mass_meta(mass_path)
    if not meta:
        raise SystemExit(f"{mass_path} has no .meta.json sidecar; refusing to guess its vocabulary")
    got, want = Path(meta.get("signal_json", "")).name, Path(signal_json).name
    if got != want:
        raise SystemExit(f"{mass_path} was built against {got}, not {want}")
    if meta.get("direction_classes") != "all":
        raise SystemExit(f"{mass_path} covers direction_classes={meta.get('direction_classes')!r}, expected 'all'")


def selected_positions(folder: Path, arm: str) -> dict[tuple[int, int], float | str]:
    """`(step, abs_pos) -> pick score` for one arm of the trajectory's selection record.

    An unscored arm (`random`) records no `direction_count` -- deliberately, so downstream code
    can tell a sampled control from a ranked selection -- so its score is the empty string.
    """
    path = record_path(folder)
    if not path.exists():
        raise SystemExit(f"no selection record at {path}")
    record = read_raw_record(path)
    arms = record.get("arms", {})
    if arm not in arms:
        raise SystemExit(f"{path} has arms {sorted(arms)}, not {arm!r}")
    return {(p["step"], p["abs_pos"]): p.get("direction_count", "") for p in arms[arm]["picks"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens-root", type=Path, default=Path("/workspace/activations/jlens_mass_l15"))
    ap.add_argument(
        "--probs-root", type=Path, default=Path("/workspace/reasoning_theatre/trajectories_train_single_step_probs")
    )
    ap.add_argument("--lens", choices=scored_methods(), default="jlens")
    ap.add_argument("--names", type=Path, default=None, help="keep ONLY these trajectory names")
    ap.add_argument("--exclude-names", type=Path, default=None, help="drop these trajectory names")
    ap.add_argument(
        "--selection-arm",
        default="none",
        help="restrict rows to one arm of the selection record ('jlens', 'random'), or 'none' for every token",
    )
    ap.add_argument("--layers", default="all", help="'all', or a comma-separated layer list")
    ap.add_argument("--prob-layer", type=int, default=15, help="layer to also emit exponentiated, as dir_prob_L{n}")
    ap.add_argument("--signal-json", type=Path, default=Path("/workspace/jlens/direction_tokens_full.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    keep, drop = read_names(args.names), read_names(args.exclude_names)
    folders = {p.name: p for p in sorted(args.lens_root.glob("size*/*")) if p.is_dir()}
    if keep:
        missing = sorted(keep - set(folders))
        if missing:
            raise SystemExit(f"--names lists {len(missing)} trajectories absent from {args.lens_root}: {missing[:3]}")
        names = sorted(keep)
    else:
        names = sorted(set(folders) - drop)
    print(f"{len(folders)} trajectories in {args.lens_root} -> {len(names)} selected", flush=True)

    # Ġ is the tokenizer's leading space; the vocabulary JSON stores the decoded form.
    vocab = load_direction_tokens(args.signal_json, "all")
    arm = None if args.selection_arm == "none" else args.selection_arm

    layers: list[int] = []
    fields: list[str] = []
    writer = None
    n_rows = 0
    n_picks_expected = 0
    label_counts = {0: 0, 1: 0}
    cohort_counts: dict[str, int] = {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        for i, name in enumerate(names):
            folder = folders[name]
            mass_path = direction_mass_path(folder, args.lens)
            if mass_path is None or not mass_path.exists():
                raise SystemExit(f"no {args.lens} mass table at {mass_path}")
            check_vocabulary(mass_path, args.signal_json)

            if writer is None:
                layers = parse_layers(args.layers, mass_csv_layers(mass_path))
                features = [logmass_column(layer) for layer in layers]
                if args.prob_layer in layers:
                    features.append(f"dir_prob_L{args.prob_layer}")
                features += ["is_direction_token", "pick_score"]
                fields = IDENTITY + features + LABEL + CONTEXT + ROLLOUT
                writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()

            steps = read_rollout_steps(rollout_path(args.probs_root, name))
            picks = selected_positions(folder, arm) if arm else None
            if picks is not None:
                n_picks_expected += len(picks)

            # csv.DictReader, never pandas: decoded tokens include "NA", commas and newlines.
            with open(mass_path, encoding="utf-8", newline="") as f:
                mass_rows = list(csv.DictReader(f))
            by_step: dict[int, list[dict]] = {}
            for r in mass_rows:
                by_step.setdefault(int(r["step"]), []).append(r)

            for step_id, rows in by_step.items():
                rec = steps.get(step_id)
                if rec is None:
                    raise SystemExit(f"{name} step {step_id} has a mass table but no rollout")
                evals = rec["sentence_evals"]
                eos = eos_positions(rec)
                offset = reasoning_offset(eos)
                span = sentence_of_token(eos)
                actions = [e["model_action"] for e in evals]
                n_switches = sum(1 for a, b in zip(actions, actions[1:], strict=False) if a != b)
                conv = rec["convinced_sentence_idx"]
                # Chain-level stats come from the UNFILTERED chain, so an arm is a row filter
                # and never changes what a surviving row means.
                n_tok = len(rows)
                conv_frac = ""
                if conv is not None and conv >= 1:
                    conv_frac = (eos[conv] - offset) / (n_tok - 1) if n_tok > 1 else 1.0

                for r in rows:
                    key = (step_id, int(r["abs_pos"]))
                    if picks is not None and key not in picks:
                        continue
                    rp = int(r["reasoning_pos"])
                    place = place_token(rp + offset, eos, span, conv)
                    if place is None:
                        raise SystemExit(f"{name} step {step_id} reasoning_pos {rp} lands outside every sentence")
                    si = place.sentence_idx
                    label = convinced_label(conv, si)
                    row = {
                        "name": name,
                        "size": r["size"],
                        "complexity": r["complexity"],
                        "run": r["run"],
                        "step": step_id,
                        "reasoning_pos": rp,
                        "abs_pos": r["abs_pos"],
                        "token": r["token"],
                        "arm": args.selection_arm,
                        "is_direction_token": int(r["token"].replace("Ġ", " ") in vocab),
                        "pick_score": "" if picks is None else picks[key],
                        "is_convinced": label,
                        "cohort": cohort(conv),
                        "convinced_idx": "" if conv is None else conv,
                        "first_correct_idx": ""
                        if rec["first_correct_sentence_idx"] is None
                        else rec["first_correct_sentence_idx"],
                        "rel_sentence": "" if place.rel_sentence is None else place.rel_sentence,
                        "n_reasoning_tokens": n_tok,
                        "reasoning_frac": (rp / (n_tok - 1)) if n_tok > 1 else 1.0,
                        "n_sentences": len(evals) - 1,
                        "sentence_idx": si,
                        "pos_in_sentence": place.pos_in_sentence,
                        "sentence_len": place.sentence_len,
                        "sentence_frac": place.sentence_frac,
                        "is_sentence_end": place.is_sentence_end,
                        "x_sentence": "" if place.x_sentence is None else place.x_sentence,
                        "convinced_reasoning_frac": conv_frac,
                        "n_switches": n_switches,
                        "sent_model_action": evals[si]["model_action"],
                        "sent_correct": int(evals[si]["correct"]),
                        "final_action": actions[-1],
                        "ground_truth": rec["ground_truth"],
                    }
                    for layer in layers:
                        row[logmass_column(layer)] = f"{float(r[f'L{layer}']):.6f}"
                    if args.prob_layer in layers:
                        row[f"dir_prob_L{args.prob_layer}"] = f"{math.exp(float(r[f'L{args.prob_layer}'])):.9g}"
                    writer.writerow(row)
                    n_rows += 1
                    label_counts[label] += 1
                    cohort_counts[row["cohort"]] = cohort_counts.get(row["cohort"], 0) + 1
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(names)} trajectories, {n_rows} rows", flush=True)

    # Every pick must have found a mass row. A dropped pick would silently shrink an arm and
    # break the row-for-row matching between the jlens arm and its random control.
    if arm is not None and n_rows != n_picks_expected:
        raise SystemExit(f"{n_picks_expected} picks but {n_rows} rows written -- some pick had no mass-table row")
    if not n_rows:
        raise SystemExit("no rows written")

    pos = label_counts[1] / n_rows
    print(f"wrote {n_rows} rows over {len(names)} trajectories -> {args.out}", flush=True)
    print(f"  label: {label_counts[1]} convinced / {label_counts[0]} not ({pos:.1%} positive)", flush=True)
    print(f"  cohort: {dict(sorted(cohort_counts.items()))}", flush=True)
    print(f"  features: {len(layers)} layers {layers[0]}..{layers[-1]}", flush=True)

    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "lens_root": str(args.lens_root),
                "probs_root": str(args.probs_root),
                "lens": args.lens,
                "selection_arm": args.selection_arm,
                "names": str(args.names) if args.names else None,
                "exclude_names": str(args.exclude_names) if args.exclude_names else None,
                "signal_json": str(args.signal_json),
                "layers": layers,
                "n_trajectories": len(names),
                "n_rows": n_rows,
                "n_convinced": label_counts[1],
                "n_not_convinced": label_counts[0],
                "cohorts": dict(sorted(cohort_counts.items())),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
