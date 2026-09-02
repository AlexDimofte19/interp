#!/usr/bin/env python3
"""``build_probe_loudness.py`` over the HELD-OUT 360 and over EVERY reasoning token.

Entry 46 asked whether a louder token decodes better. It answered on the eval-720 split,
and -- the limitation this script removes -- it read each probe only on the tokens that
probe's own jlens selection had picked. Inside a fixed top-K arm the loudness axis is
truncated by construction, which is exactly why entry 46(b) needed a chain-length control.

Here every probe is read on the SAME 87,221 tokens: every reasoning token of the 360
held-out trajectories, a tree disjoint from every probe's training set
(``scripts/audit_trajectory_sets.py``). No selection sits between the loudness axis and
the label, so the axis spans the real distribution and no arm gets a token set tuned to it.

THE THREE ROWSETS ARE KEPT, and they now hold IDENTICAL rows. They are a presentational
device only: ``analyze_probe_loudness.py`` and ``plot_probe_loudness.py`` key off the
rowset name to decide which probes to score, so preserving the names lets both run
BYTE-UNCHANGED and every figure of entry 46 gets a direct counterpart. The consequence to
state in any write-up: ``loudness_distribution.png`` shows three coincident curves.

  p1_full   p1_lr, p1_mlp                   (probe 1: every sentence's loudest token)
  p1_top20  p1t20_lr, p1t20_mlp             (probe 1 thinned to top-20 per trajectory)
  p2        p2_lr, p2_mlp                   (probe 2: global top-20 by mass)
            base_lr, base_mlp               entry-38 FINAL-action baseline: the label contrast
            rand_lr, rand_mlp               matched random-selection control (entry 37(d): a floor)

This is a pure join of three artifacts -- no .pt, no model, no GPU -- so it is cheap to
re-run when any of them is rebuilt:

``--probe-csv``       ``eval_probe_per_token.py``'s one-pass scoring of all ten probes at
                      layer 15 over the held-out tree, with ``--full-probs``.
``--rollout-dir``     the ``every_token`` truncation arm (``truncation_strategies.py``):
                      the chain cut at EVERY token and the model asked for its action, so
                      ``label_local`` is a measured per-token belief rather than the
                      sentence-end answer standing in for one.
``--commitment-csv``  entries 39/40/41's join, already one row per reasoning token of these
                      same 360 trajectories. It supplies the loudness and the sentence
                      coordinates; the two GPU passes above only add the label and the
                      six local-belief probes.

COORDINATE TRAP, and it is silent: this script's ``sentence_frac`` is the WITHIN-sentence
position (0 at a sentence's first token, 1 at its last), which in the commitment-boundary
CSV is called ``frac_in_sentence``. That file's own ``sentence_frac`` is
``sentence_idx / n_sentences``, a different quantity. Reading the wrong one does not fail,
it quietly destroys the loudness-vs-position control -- entry 46's result (c).

``n_switches`` is counted over the DENSE per-token action sequence, so it is a strictly
larger number than entry 46's per-sentence count and the two are not comparable
digit-for-digit.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# csv.DictReader everywhere, never pandas: decoded tokens include "NA", empty strings,
# embedded commas and newlines, which pandas' NA handling silently corrupts.
csv.field_size_limit(10**9)

# id -> action, the fixed LEFT,UP,RIGHT,DOWN order eval_probe_per_token.py writes its
# --full-probs columns in and plot_commitment_probs.py already uses.
ID2A = {0: "LEFT", 1: "UP", 2: "RIGHT", 3: "DOWN"}
ACTIONS = ("LEFT", "UP", "RIGHT", "DOWN")

# entry-46 probe key -> the key eval_probe_per_token.py's probe_key() gives that same .pt
# ("<parent dir>.<stem>", with next_action_probe_ stripped).
PROBE_SOURCE = {
    "p1_lr": "probes.local_belief_p1_lr",
    "p1_mlp": "probes.local_belief_p1_mlp",
    "p1t20_lr": "probes.local_belief_p1_top20_lr",
    "p1t20_mlp": "probes.local_belief_p1_top20_mlp",
    "p2_lr": "probes.local_belief_p2_lr",
    "p2_mlp": "probes.local_belief_p2_mlp",
    "base_lr": "next_action_mass_l15.jlens_topall_lr",
    "base_mlp": "next_action_mass_l15.jlens_topall_mlp",
    "rand_lr": "next_action_mass_l15.random_topall_lr",
    "rand_mlp": "next_action_mass_l15.random_topall_mlp",
}

ROWSETS: dict[str, list[str]] = {
    "p1_full": ["p1_lr", "p1_mlp"],
    "p1_top20": ["p1t20_lr", "p1t20_mlp"],
    "p2": ["p2_lr", "p2_mlp", "base_lr", "base_mlp", "rand_lr", "rand_mlp"],
}

# Verbatim from build_probe_loudness.py: the header must diff clean against entry 46's.
BASE_FIELDS = [
    "rowset",
    "name",
    "size",
    "complexity",
    "step",
    "token_id",
    "reasoning_pos",
    "token",
    "is_direction_token",
    "cutoff_kind",
    "dir_logmass",
    "dir_prob",
    "mass_rank_in_traj",
    "mass_pct_in_traj",
    "mass_rank_in_sentence",
    "n_reasoning_tokens",
    "reasoning_frac",
    "n_sentences",
    "sentence_idx",
    "pos_in_sentence",
    "sentence_len",
    "sentence_frac",
    "is_sentence_end",
    "convinced_idx",
    "rel_sentence",
    "x_sentence",
    "n_switches",
    "label_local",
    "label_final",
    "ground_truth",
    "rollout_answer_prob",
    "rollout_correct",
]


def read_commitment(path: Path) -> dict[str, dict[int, dict[int, dict]]]:
    """``{name: {step: {token_id: coords}}}`` from the commitment-boundary CSV.

    Only the columns this script needs are kept. ``token_idx`` there indexes
    ``step["output_tokens"]`` -- the same coordinate as the rollout's ``eos_token_pos``
    and as entry 46's ``token_id`` -- which is what makes the join 1:1.
    """
    out: dict[str, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["name"]][int(r["step"])][int(r["token_idx"])] = {
                "abs_pos": int(r["abs_pos"]),
                "reasoning_pos": int(r["reasoning_pos"]),
                "token": r["token"],
                "size": r["size"],
                "complexity": r["complexity"],
                "label_final": r["label_name"],
                "dir_logmass": float(r["jlens_mass_L15"]),
                "n_sentences": r["n_sentences"],
                "sentence_idx": r["sentence_idx"],
                "pos_in_sentence": r["pos_in_sentence"],
                "sentence_len": r["sentence_len"],
                # NOT that file's `sentence_frac`, which is sentence_idx/n_sentences.
                "sentence_frac": r["frac_in_sentence"],
                "is_sentence_end": r["is_sentence_end"],
                "convinced_idx": r["convinced_sentence_idx"],
            }
    return out


def read_probe_csv(path: Path) -> tuple[dict[str, dict[int, dict[int, dict]]], list[str]]:
    """``{name: {step: {token_idx: {probe_key: (pred, {action: p})}}}}`` plus the mass column."""
    out: dict[str, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        have = set(reader.fieldnames or [])
        missing = [
            f"{src}_{suffix}"
            for src in PROBE_SOURCE.values()
            for suffix in ["pred", *[f"p_{a}" for a in ACTIONS]]
            if f"{src}_{suffix}" not in have
        ]
        if missing:
            raise SystemExit(
                f"{path} is missing {len(missing)} column(s), e.g. {missing[:3]}. "
                "Was eval_probe_per_token.py run with all ten --probe and --full-probs?"
            )
        for r in reader:
            cell = {
                key: (
                    ID2A[int(r[f"{src}_pred"])],
                    {a: float(r[f"{src}_p_{a}"]) for a in ACTIONS},
                )
                for key, src in PROBE_SOURCE.items()
            }
            cell["_mass"] = float(r["jlens_mass_L15"])
            cell["_label_final"] = r["label_name"]
            out[r["name"]][int(r["step"])][int(r["token_idx"])] = cell
    return out, sorted(have)


def read_rollouts(root: Path) -> dict[str, dict[int, dict]]:
    """``{name: {step: {"gt", "n_switches", "evals": {token_id: eval}}}}`` from the every_token arm.

    ``n_switches`` counts changes down the ordered eval list, the no-reasoning cutoff
    included, exactly as entry 46 counted them down the sentence list.
    """
    out: dict[str, dict[int, dict]] = {}
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text())
        if "steps" not in data:
            continue
        per_step: dict[int, dict] = {}
        for rec in data["steps"]:
            evals = rec["sentence_evals"]
            acts = [e["model_action"] for e in evals]
            per_step[rec["step_id"]] = {
                "gt": rec["ground_truth"],
                "n_switches": sum(1 for a, b in zip(acts, acts[1:], strict=False) if a != b),
                "evals": {e["eos_token_pos"]: e for e in evals},
            }
        out[path.stem] = per_step
    return out


def ranks(values: dict[int, float]) -> dict[int, int]:
    """token_id -> rank of its mass, 1 = loudest. Ties break toward the earlier token."""
    order = sorted(values, key=lambda t: (-values[t], t))
    return {t: i + 1 for i, t in enumerate(order)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    out_default = Path("/workspace/reasoning_theatre/probe_loudness_heldout360")
    ap.add_argument("--probe-csv", type=Path, default=out_default / "heldout360_10probes.csv")
    ap.add_argument(
        "--rollout-dir",
        type=Path,
        default=Path("/workspace/reasoning_theatre/rollout_strategies_heldout360/every_token"),
    )
    ap.add_argument(
        "--commitment-csv",
        type=Path,
        default=Path("/workspace/reasoning_theatre/probe_vs_rollout/per_token.csv"),
    )
    ap.add_argument(
        "--signal-json",
        type=Path,
        default=Path("/workspace/jlens/direction_tokens_full.json"),
        help="the vocabulary the mass table was built against; flags whether the TOKEN "
        "ITSELF is a direction word, which is the standing confound.",
    )
    ap.add_argument("--out", type=Path, default=out_default / "per_token.csv")
    ap.add_argument("--rowsets", default="all", help="comma-separated subset of ROWSETS, or 'all'.")
    ap.add_argument("--mass-tol", type=float, default=1e-6, help="max |probe CSV mass - commitment CSV mass|.")
    ap.add_argument("--limit", type=int, default=None, help="first N trajectories (smoke test).")
    args = ap.parse_args()

    wanted = list(ROWSETS) if args.rowsets == "all" else args.rowsets.split(",")
    unknown = [r for r in wanted if r not in ROWSETS]
    if unknown:
        raise SystemExit(f"unknown rowset(s) {unknown}; choose from {list(ROWSETS)}")
    vocab = {t for lst in json.loads(args.signal_json.read_text()).values() for t in lst}

    print(f"reading {args.commitment_csv}", flush=True)
    coords = read_commitment(args.commitment_csv)
    print(f"reading {args.probe_csv}", flush=True)
    probes, _ = read_probe_csv(args.probe_csv)
    print(f"reading {args.rollout_dir}", flush=True)
    rollouts = read_rollouts(args.rollout_dir)
    print(
        f"  {len(coords)} names in coords, {len(probes)} in probe CSV, {len(rollouts)} rollout file(s)",
        flush=True,
    )

    fields = list(BASE_FIELDS)
    for rs in wanted:
        for p in ROWSETS[rs]:
            fields += [f"{p}_pred", f"{p}_p_local", f"{p}_p_final", f"{p}_pmax"]

    names = sorted(coords)
    if args.limit:
        names = names[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    per_rowset: dict[str, int] = defaultdict(int)
    skipped: dict[str, int] = defaultdict(int)
    mass_mismatch = 0
    label_mismatch = 0
    no_action = 0

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()

        for ni, name in enumerate(names, 1):
            for step, toks in sorted(coords[name].items()):
                pstep = probes.get(name, {}).get(step, {})
                rstep = rollouts.get(name, {}).get(step)
                if rstep is None:
                    skipped["no rollout step"] += len(toks)
                    continue

                n_tok = len(toks)
                mass_rank = ranks({t: c["dir_logmass"] for t, c in toks.items()})
                by_sentence: dict[str, dict[int, float]] = defaultdict(dict)
                for t, c in toks.items():
                    by_sentence[c["sentence_idx"]][t] = c["dir_logmass"]
                sent_rank = {}
                for vals in by_sentence.values():
                    sent_rank.update(ranks(vals))

                for tok_id, c in sorted(toks.items()):
                    pcell = pstep.get(tok_id)
                    if pcell is None:
                        skipped["no probe row"] += 1
                        continue
                    ev = rstep["evals"].get(tok_id)
                    if ev is None:
                        skipped["no rollout eval"] += 1
                        continue
                    if abs(pcell["_mass"] - c["dir_logmass"]) > args.mass_tol:
                        mass_mismatch += 1
                    if pcell["_label_final"] != c["label_final"]:
                        label_mismatch += 1

                    lm = c["dir_logmass"]
                    rp = c["reasoning_pos"]
                    frac = float(c["sentence_frac"])
                    conv = c["convinced_idx"]
                    si = int(c["sentence_idx"])
                    rel = "" if conv in ("", None) else si - int(conv)
                    # The rollout answers with a single token and almost always emits one of
                    # the four actions; 1 eval in 87,581 emitted "NO" instead. Keep the row --
                    # dropping it would put a hole in an every-token grid for one degenerate
                    # generation -- but leave the local label and its probabilities blank.
                    # bal_acc() iterates over the four actions, so a blank truth is ignored
                    # rather than counted as a miss.
                    label_local = ev["model_action"]
                    if label_local not in ACTIONS:
                        no_action += 1
                        label_local = ""
                    label_final = c["label_final"]

                    row = {
                        "name": name,
                        "size": c["size"],
                        "complexity": c["complexity"],
                        "step": step,
                        "token_id": tok_id,
                        "reasoning_pos": rp,
                        "token": c["token"],
                        "is_direction_token": int(c["token"].replace("Ġ", " ") in vocab),
                        "cutoff_kind": ev["cutoff_kind"],
                        "dir_logmass": f"{lm:.6f}",
                        "dir_prob": f"{math.exp(lm):.9g}",
                        "mass_rank_in_traj": mass_rank[tok_id],
                        "mass_pct_in_traj": f"{mass_rank[tok_id] / n_tok:.6f}",
                        "mass_rank_in_sentence": sent_rank[tok_id],
                        "n_reasoning_tokens": n_tok,
                        "reasoning_frac": f"{(rp / (n_tok - 1)) if n_tok > 1 else 1.0:.6f}",
                        "n_sentences": c["n_sentences"],
                        "sentence_idx": si,
                        "pos_in_sentence": c["pos_in_sentence"],
                        "sentence_len": c["sentence_len"],
                        "sentence_frac": f"{frac:.6f}",
                        "is_sentence_end": c["is_sentence_end"],
                        "convinced_idx": conv,
                        "rel_sentence": rel,
                        "x_sentence": "" if rel == "" else f"{rel - 1 + frac:.6f}",
                        "n_switches": rstep["n_switches"],
                        "label_local": label_local,
                        "label_final": label_final,
                        "ground_truth": rstep["gt"],
                        "rollout_answer_prob": ev["answer_prob"],
                        "rollout_correct": int(bool(ev["correct"])),
                    }
                    for rs_name in wanted:
                        for pname in ROWSETS[rs_name]:
                            pred, probs4 = pcell[pname]
                            row[f"{pname}_pred"] = pred
                            row[f"{pname}_p_local"] = f"{probs4[label_local]:.6f}" if label_local else ""
                            row[f"{pname}_p_final"] = f"{probs4[label_final]:.6f}"
                            row[f"{pname}_pmax"] = f"{max(probs4.values()):.6f}"
                    # One identical row per rowset: the rowset now names WHICH PROBES are
                    # read, not which tokens, so every arm is scored on the same tokens.
                    for rs_name in wanted:
                        w.writerow({**row, "rowset": rs_name})
                        per_rowset[rs_name] += 1
                        n_rows += 1
            if ni % 100 == 0 or ni == len(names):
                print(f"    {ni}/{len(names)} trajectories, {n_rows} rows", flush=True)

    print(f"\nwrote {n_rows} rows -> {args.out}", flush=True)
    for rs_name in wanted:
        print(f"  {rs_name}: {per_rowset[rs_name]} rows", flush=True)
    print(f"  probe CSV vs commitment CSV mass mismatches (>{args.mass_tol}): {mass_mismatch}", flush=True)
    print(f"  final-label mismatches: {label_mismatch}", flush=True)
    print(f"  rows whose rollout emitted no valid action (label_local blank): {no_action}", flush=True)
    for k, v in sorted(skipped.items()):
        print(f"  skipped: {k}: {v}", flush=True)
    if mass_mismatch or label_mismatch or skipped:
        print("  *** non-zero guard; the join is not 1:1", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
