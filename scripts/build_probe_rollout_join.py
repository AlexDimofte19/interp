#!/usr/bin/env python3
"""Join the reasoning-theatre sentence rollouts to the per-token probe/lens readouts.

Three artifacts, one key. For a held-out trajectory we have

  * `scripts/inference_oss/run_inference.py` output — for every reasoning sentence end, what
    the model ACTUALLY answers if reasoning is truncated there (`model_action`, `answer_prob`).
    Its `eos_token_pos` indexes `step["output_tokens"]`.
  * `/workspace/probes/heldout360_all_probes.csv` — every reasoning token of the same 360
    trajectories scored by all 26 next-action probes. Its `token_idx` indexes the same list.
  * `{stem}_jlens_analysis.csv` — the layer-15 Jacobian lens at every reasoning token: the
    rank/logprob of each literal action word and the top-20 j-space predictions.

So `token_idx == eos_token_pos` is the join, and every reasoning token can be placed inside
the sentence whose truncation eval it belongs to. One row per (trajectory, step, token).

Sentence 0 of a rollout is the *no-reasoning* cutoff (the analysis-header `<|message|>`); it
owns no reasoning tokens, so reasoning tokens carry sentence_idx >= 1.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ACTIONS = ("LEFT", "UP", "RIGHT", "DOWN")  # index == NEXT_ACTION_TO_ID
ACTION_ID = {a: i for i, a in enumerate(ACTIONS)}
CLASSES = ("UP", "DOWN", "LEFT", "RIGHT")
NO_MATCH = -40.0
TOP_K = 20
# A class absent from the top 20 gets a rank worse than any present one rather than a blank,
# so all four classes stay comparable at every token.
NO_RANK = TOP_K + 1


def logsumexp(xs: list[float]) -> float:
    """Stable logsumexp; empty -> NO_MATCH, the floor the jlens scoring module uses."""
    if not xs:
        return NO_MATCH
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def load_rollouts(probs_root: Path, names: list[str]) -> dict[str, dict]:
    """{stem: {step_id: step_record}} for `names` only.

    The rollout root holds all 36k training trajectories and sits on a FUSE mount, so it is
    addressed by name rather than globbed — the held-out set is 1% of it.
    """
    out: dict[str, dict] = {}
    for name in names:
        size = name.split("_size")[1].split("_")[0]
        path = probs_root / f"size{size}" / f"{name}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        out[name] = {s["step_id"]: s for s in doc["steps"]}
    return out


def sentence_index(eos_positions: list[int]) -> dict[int, int]:
    """Map every reasoning token_idx to the sentence whose truncation eval covers it.

    A sentence ends at its eos position, so token t belongs to the first sentence whose eos
    is >= t. Sentence 0's eos is the analysis header, which precedes every reasoning token.
    """
    span: dict[int, int] = {}
    prev = eos_positions[0]
    for si, eos in enumerate(eos_positions):
        if si == 0:
            continue
        for t in range(prev + 1, eos + 1):
            span[t] = si
        prev = eos
    return span


def load_lens(csv_path: Path, vocab: dict[str, set[str]]) -> dict[tuple[int, int], dict]:
    """{(step, reasoning_pos): lens readout} from a layer-15 `_jlens_analysis.csv`.

    Two independent directional readouts per token:
      * `act_*` — the logprob the lens puts on the literal answer word ("LEFT"/"UP"/...),
        which is what the model has to emit in the final channel.
      * `top20_*` — logsumexp of the top-20 predictions belonging to each direction class of
        the scoring vocabulary, i.e. how the token's *verbalized* direction mass splits.
    """
    rows: dict[tuple[int, int], dict] = {}
    # csv.DictReader, never pandas: decoded tokens include "NA", empty strings and commas.
    with open(csv_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["layer"] != "15":
                continue
            act = {a: float(r[f"{a}_logprob"]) for a in ACTIONS}
            rank = {a: int(r[f"{a}_rank"]) for a in ACTIONS}
            per_class: dict[str, list[float]] = {c: [] for c in CLASSES}
            best_rank: dict[str, int] = {}
            top1_class = ""
            n_dir = 0
            for i in range(1, TOP_K + 1):
                tok = r.get(f"top_{i}")
                lp = r.get(f"top_{i}_logprob")
                if tok is None or lp in (None, ""):
                    continue
                hit = False
                for c in CLASSES:
                    if tok in vocab[c]:
                        per_class[c].append(float(lp))
                        best_rank.setdefault(c, i)
                        if i == 1:
                            top1_class = c
                        hit = True
                n_dir += hit
            rows[(int(r["step"]), int(r["reasoning_pos"]))] = {
                "act_logprob": act,
                "act_rank": rank,
                # Three readouts of the same top-20 block, because they fail differently: mass
                # weights a hit by how much the lens believed it, count is the original score and
                # ignores belief, best rank is belief-free but position-sensitive.
                "top20_mass": {c: logsumexp(per_class[c]) for c in CLASSES},
                "top20_count": {c: len(per_class[c]) for c in CLASSES},
                "top20_bestrank": {c: best_rank.get(c, NO_RANK) for c in CLASSES},
                "top20_n_direction": n_dir,
                "top20_top1_class": top1_class,
            }
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-csv", type=Path, default=Path("/workspace/probes/heldout360_all_probes.csv"))
    ap.add_argument(
        "--probs-root", type=Path, default=Path("/workspace/reasoning_theatre/trajectories_train_single_step_probs")
    )
    ap.add_argument("--lens-root", type=Path, default=Path("/workspace/activations/heldout360_l15"))
    ap.add_argument("--direction-json", type=Path, default=Path("/workspace/jlens/direction_tokens_full.json"))
    ap.add_argument("--out", type=Path, default=Path("/workspace/reasoning_theatre/probe_vs_rollout/per_token.csv"))
    args = ap.parse_args()

    with open(args.direction_json, encoding="utf-8") as f:
        raw = json.load(f)
    vocab = {c: set(raw[c]) for c in CLASSES}

    print(f"reading probe rows from {args.probe_csv}", flush=True)
    with open(args.probe_csv, encoding="utf-8", newline="") as f:
        probe_rows = list(csv.DictReader(f))
    header = list(probe_rows[0].keys())
    pred_cols = [c for c in header if c.endswith("_pred")]
    ptrue_cols = [c for c in header if c.endswith("_p_true")]
    arms = [c[: -len("_pred")] for c in pred_cols]
    print(f"  {len(probe_rows)} rows, {len(arms)} probe arms", flush=True)

    names = sorted({r["name"] for r in probe_rows})

    print(f"reading {len(names)} rollouts from {args.probs_root}", flush=True)
    rollouts = load_rollouts(args.probs_root, names)
    print(f"  {len(rollouts)} rollout files", flush=True)

    lens_cache: dict[str, dict] = {}
    for name in names:
        size = name.split("_size")[1].split("_")[0]
        p = args.lens_root / f"size{size}" / name / f"{name}_jlens_analysis.csv"
        lens_cache[name] = load_lens(p, vocab) if p.exists() else {}
    missing_lens = [n for n in names if not lens_cache[n]]
    print(f"  lens CSVs loaded for {len(names) - len(missing_lens)}/{len(names)} trajectories", flush=True)

    out_header = [
        "name",
        "size",
        "complexity",
        "step",
        "abs_pos",
        "token_idx",
        "reasoning_pos",
        "token",
        "label",
        "label_name",
        "n_sentences",
        "sentence_idx",
        "sentence_frac",
        "pos_in_sentence",
        "sentence_len",
        "frac_in_sentence",
        "is_sentence_end",
        "sent_model_action",
        "sent_answer_prob",
        "sent_correct",
        "prev_model_action",
        "prev_answer_prob",
        "next_model_action",
        "first_correct_sentence_idx",
        "convinced_sentence_idx",
        "is_after_convinced",
        "jlens_mass_L15",
        "logitlens_mass_L15",
        "jlens_count",
        "logitlens_count",
        "lens_act_argmax",
        "lens_act_margin",
        "lens_act_rank_true",
        "lens_top20_argmax",
        "lens_top20_margin",
        "lens_top20_n_direction",
        "lens_top20_top1_class",
    ]
    out_header += [f"lens_act_logprob_{a}" for a in ACTIONS]
    out_header += [f"lens_top20_mass_{c}" for c in CLASSES]
    out_header += [f"lens_top20_count_{c}" for c in CLASSES]
    out_header += [f"lens_top20_bestrank_{c}" for c in CLASSES]
    out_header += pred_cols + ptrue_cols

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_no_sentence = 0
    n_no_lens = 0
    span_cache: dict[tuple[str, int], tuple[dict[int, int], list[int], dict]] = {}

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_header, extrasaction="ignore")
        w.writeheader()
        for r in probe_rows:
            name, step = r["name"], int(r["step"])
            steps = rollouts.get(name)
            if steps is None or step not in steps:
                n_no_sentence += 1
                continue
            key = (name, step)
            if key not in span_cache:
                rec = steps[step]
                eos = [e["eos_token_pos"] for e in rec["sentence_evals"]]
                span_cache[key] = (sentence_index(eos), eos, rec)
            span, eos, rec = span_cache[key]
            tok_idx = int(r["token_idx"])
            si = span.get(tok_idx)
            if si is None:
                n_no_sentence += 1
                continue
            evals = rec["sentence_evals"]
            n_sent = len(evals)
            ev = evals[si]
            prev = evals[si - 1]
            nxt = evals[si + 1] if si + 1 < n_sent else None
            start = eos[si - 1] + 1
            sent_len = eos[si] - start + 1
            pos_in = tok_idx - start

            row = dict(r)
            row["reasoning_pos"] = tok_idx - 3
            row["n_sentences"] = n_sent
            row["sentence_idx"] = si
            row["sentence_frac"] = (si / (n_sent - 1)) if n_sent > 1 else 0.0
            row["pos_in_sentence"] = pos_in
            row["sentence_len"] = sent_len
            row["frac_in_sentence"] = (pos_in / (sent_len - 1)) if sent_len > 1 else 1.0
            row["is_sentence_end"] = int(tok_idx == eos[si])
            row["sent_model_action"] = ev["model_action"]
            row["sent_answer_prob"] = ev["answer_prob"]
            row["sent_correct"] = int(bool(ev["correct"]))
            row["prev_model_action"] = prev["model_action"]
            row["prev_answer_prob"] = prev["answer_prob"]
            row["next_model_action"] = nxt["model_action"] if nxt else ""
            row["first_correct_sentence_idx"] = rec["first_correct_sentence_idx"]
            conv = rec["convinced_sentence_idx"]
            row["convinced_sentence_idx"] = conv
            row["is_after_convinced"] = "" if conv is None else int(si >= conv)

            lens = lens_cache[name].get((step, tok_idx - 3))
            if lens is None:
                n_no_lens += 1
                for a in ACTIONS:
                    row[f"lens_act_logprob_{a}"] = ""
                for c in CLASSES:
                    row[f"lens_top20_mass_{c}"] = ""
                    row[f"lens_top20_count_{c}"] = ""
                    row[f"lens_top20_bestrank_{c}"] = ""
                for k in (
                    "lens_act_argmax",
                    "lens_act_margin",
                    "lens_act_rank_true",
                    "lens_top20_argmax",
                    "lens_top20_margin",
                    "lens_top20_n_direction",
                    "lens_top20_top1_class",
                ):
                    row[k] = ""
            else:
                act = lens["act_logprob"]
                ordered = sorted(ACTIONS, key=lambda a: -act[a])
                row["lens_act_argmax"] = ordered[0]
                row["lens_act_margin"] = round(act[ordered[0]] - act[ordered[1]], 6)
                row["lens_act_rank_true"] = lens["act_rank"][r["label_name"]]
                mass = lens["top20_mass"]
                m_ord = sorted(CLASSES, key=lambda c: -mass[c])
                row["lens_top20_argmax"] = m_ord[0] if mass[m_ord[0]] > NO_MATCH else ""
                row["lens_top20_margin"] = round(mass[m_ord[0]] - mass[m_ord[1]], 6)
                row["lens_top20_n_direction"] = lens["top20_n_direction"]
                row["lens_top20_top1_class"] = lens["top20_top1_class"]
                for a in ACTIONS:
                    row[f"lens_act_logprob_{a}"] = round(act[a], 6)
                for c in CLASSES:
                    row[f"lens_top20_mass_{c}"] = round(mass[c], 6)
                    row[f"lens_top20_count_{c}"] = lens["top20_count"][c]
                    row[f"lens_top20_bestrank_{c}"] = lens["top20_bestrank"][c]
            w.writerow(row)
            n_written += 1

    print(f"wrote {n_written} rows -> {args.out}", flush=True)
    print(f"  dropped (no rollout / not in a sentence span): {n_no_sentence}", flush=True)
    print(f"  rows with no lens row: {n_no_lens}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
