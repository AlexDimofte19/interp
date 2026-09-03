#!/usr/bin/env python3
"""Per-token Jacobian-lens *loudness* placed inside its reasoning sentence.

LOUDNESS here is the full-vocabulary direction mass, not the top-20 count entries 39/40
used: the direction-mass table beside each analysis CSV holds ``log P(any direction word)``
at every (reasoning token, layer), computed on-device over the whole 446-token
``direction_tokens_full.json`` vocabulary. Layer 15's cell exponentiated is exactly
``sum(exp(logprob(t)) for t in direction_tokens)``.

Two keys join three artifacts, one row per reasoning token:

  * ``{stem}_jlens_direction_mass.csv`` -- the loudness. ``reasoning_pos`` indexes the
    analysis-tagged tokens of ``step["output_tokens"]``.
  * ``scripts/inference_oss/run_inference.py`` rollouts -- ``eos_token_pos`` indexes
    ``step["output_tokens"]`` itself, so the offset is ``eos[0] + 1`` (the analysis header
    ``<|channel|>analysis<|message|>``, always 3 here but read, not assumed).
  * the probe's own train/eval split -- only trajectories the probe TRAINED on are kept.

Sentence 0 of a rollout is the no-reasoning cutoff and owns no reasoning tokens, so every
reasoning token lands in a sentence with index >= 1.

Coordinates written for the plots:

  ``sentence_frac``  p / (L - 1) in [0, 1]: 0 is a sentence's first token, 1 its last.
  ``rel_sentence``   sentence_idx - convinced_idx. 0 IS the convinced sentence, so it
                     occupies x in [-1, 0] and the sentence after it [0, +1].
  ``x_sentence``     rel_sentence - 1 + sentence_frac: one sentence per unit, x = 0 is the
                     convinced eos token.
  ``reasoning_frac`` reasoning_pos / (n_reasoning_tokens - 1): position in the whole chain.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

FIELDS = [
    "name",
    "size",
    "complexity",
    "run",
    "step",
    "reasoning_pos",
    "abs_pos",
    "token",
    "n_reasoning_tokens",
    "reasoning_frac",
    "n_sentences",
    "sentence_idx",
    "pos_in_sentence",
    "sentence_len",
    "sentence_frac",
    "is_sentence_end",
    "convinced_idx",
    "first_correct_idx",
    "rel_sentence",
    "x_sentence",
    "convinced_reasoning_frac",
    "n_switches",
    "sent_model_action",
    "sent_correct",
    "final_action",
    "ground_truth",
    "dir_logmass_L15",
    "dir_prob_L15",
    "is_direction_token",
]


def sentence_of_token(eos: list[int]) -> dict[int, int]:
    """output-token index -> sentence index, for sentences 1.. (sentence 0 is the header)."""
    span: dict[int, int] = {}
    prev = eos[0]
    for si, e in enumerate(eos):
        if si == 0:
            continue
        for t in range(prev + 1, e + 1):
            span[t] = si
        prev = e
    return span


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens-root", type=Path, default=Path("/workspace/activations/jlens_mass_l15"))
    ap.add_argument(
        "--probs-root", type=Path, default=Path("/workspace/reasoning_theatre/trajectories_train_single_step_probs")
    )
    ap.add_argument(
        "--eval-names",
        type=Path,
        default=Path("/workspace/prepared/next_action_mass_l15_eval_names.txt"),
        help="names to EXCLUDE: the probe's held-out split. Empty string keeps everything.",
    )
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument(
        "--signal-json",
        type=Path,
        default=Path("/workspace/jlens/direction_tokens_full.json"),
        help="the vocabulary the mass table was built against; used only to flag whether the "
        "TOKEN ITSELF is a direction word, which is the standing confound for loudness.",
    )
    ap.add_argument("--out", type=Path, default=Path("/workspace/reasoning_theatre/loudness/per_token.csv"))
    args = ap.parse_args()

    held = set()
    if args.eval_names and str(args.eval_names):
        held = set(args.eval_names.read_text().split())
    names = sorted(p.name for p in args.lens_root.glob("size*/*") if p.is_dir())
    train = [n for n in names if n not in held]
    print(f"{len(names)} trajectories in {args.lens_root}, {len(held)} held out -> {len(train)} training", flush=True)

    # Ġ is the tokenizer's leading space; the vocabulary JSON stores the decoded form.
    vocab = {t for lst in json.loads(args.signal_json.read_text()).values() for t in lst}
    col = f"L{args.layer}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_traj = 0
    skipped: dict[str, int] = {}
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for i, name in enumerate(train):
            size = name.split("_size")[1].split("_")[0]
            mass_path = args.lens_root / f"size{size}" / name / f"{name}_jlens_direction_mass.csv"
            roll_path = args.probs_root / f"size{size}" / f"{name}.json"
            if not mass_path.exists() or not roll_path.exists():
                skipped["missing artifact"] = skipped.get("missing artifact", 0) + 1
                continue
            with open(roll_path, encoding="utf-8") as f:
                steps = {s["step_id"]: s for s in json.load(f)["steps"]}
            # csv.DictReader, never pandas: decoded tokens include "NA", commas and newlines.
            with open(mass_path, encoding="utf-8", newline="") as f:
                mass_rows = list(csv.DictReader(f))
            by_step: dict[int, list[dict]] = {}
            for r in mass_rows:
                by_step.setdefault(int(r["step"]), []).append(r)

            wrote = False
            for step_id, rows in by_step.items():
                rec = steps.get(step_id)
                if rec is None:
                    skipped["no rollout step"] = skipped.get("no rollout step", 0) + 1
                    continue
                evals = rec["sentence_evals"]
                eos = [e["eos_token_pos"] for e in evals]
                offset = eos[0] + 1  # first analysis-tagged token, i.e. reasoning_pos 0
                span = sentence_of_token(eos)
                acts = [e["model_action"] for e in evals]
                n_switch = sum(1 for a, b in zip(acts, acts[1:], strict=False) if a != b)
                conv = rec["convinced_sentence_idx"]
                n_tok = len(rows)
                conv_frac = ""
                if conv is not None and conv >= 1:
                    conv_frac = (eos[conv] - offset) / (n_tok - 1) if n_tok > 1 else 1.0
                for r in rows:
                    rp = int(r["reasoning_pos"])
                    tok_idx = rp + offset
                    si = span.get(tok_idx)
                    if si is None:
                        skipped["token outside every sentence"] = skipped.get("token outside every sentence", 0) + 1
                        continue
                    start = eos[si - 1] + 1
                    sent_len = eos[si] - start + 1
                    pos_in = tok_idx - start
                    lm = float(r[col])
                    frac = (pos_in / (sent_len - 1)) if sent_len > 1 else 1.0
                    rel = "" if conv is None else si - conv
                    w.writerow(
                        {
                            "name": name,
                            "size": r["size"],
                            "complexity": r["complexity"],
                            "run": r["run"],
                            "step": step_id,
                            "reasoning_pos": rp,
                            "abs_pos": r["abs_pos"],
                            "token": r["token"],
                            "n_reasoning_tokens": n_tok,
                            "reasoning_frac": (rp / (n_tok - 1)) if n_tok > 1 else 1.0,
                            "n_sentences": len(evals) - 1,
                            "sentence_idx": si,
                            "pos_in_sentence": pos_in,
                            "sentence_len": sent_len,
                            "sentence_frac": frac,
                            "is_sentence_end": int(tok_idx == eos[si]),
                            "convinced_idx": "" if conv is None else conv,
                            "first_correct_idx": ""
                            if rec["first_correct_sentence_idx"] is None
                            else rec["first_correct_sentence_idx"],
                            "rel_sentence": rel,
                            "x_sentence": "" if rel == "" else rel - 1 + frac,
                            "convinced_reasoning_frac": conv_frac,
                            "n_switches": n_switch,
                            "sent_model_action": evals[si]["model_action"],
                            "sent_correct": int(evals[si]["correct"]),
                            "final_action": acts[-1],
                            "ground_truth": rec["ground_truth"],
                            "dir_logmass_L15": f"{lm:.6f}",
                            "dir_prob_L15": f"{math.exp(lm):.9g}",
                            "is_direction_token": int(r["token"].replace("\u0120", " ") in vocab),
                        }
                    )
                    n_rows += 1
                    wrote = True
            n_traj += wrote
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(train)} trajectories, {n_rows} rows", flush=True)

    print(f"wrote {n_rows} rows over {n_traj} trajectories -> {args.out}", flush=True)
    for k, v in skipped.items():
        print(f"  skipped: {k}: {v}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
