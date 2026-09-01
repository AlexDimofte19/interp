#!/usr/bin/env python3
"""One row per held-out token of the LOCAL-BELIEF probes, carrying its loudness, its place
in its sentence, and every probe's prediction on it.

This is the probe-side twin of ``build_sentence_loudness.py``. That script asked where
loudness LIVES over all 776k reasoning tokens of the training split; this one asks what the
probes trained ON the loud tokens (ICLR log entry 45) do as a function of it, on the 720
held-out trajectories those probes never saw.

LOUDNESS is entry 42's, unchanged: the layer-15 cell of ``{stem}_jlens_direction_mass.csv``,
i.e. ``log P(any direction word)`` over the whole 446-token ``direction_tokens_full.json``.
The eval manifests already carry it per sample as ``dir_logmass``; the mass CSV is re-read
anyway for the token text and for the within-trajectory RANK, and the two are cross-checked.

THREE ROWSETS, because the arms do not share a token set:

  p1_full   every sentence's loudest token          (cutoff_kind loudest_in_sentence)
  p1_top20  its top-20-per-trajectory thinning
  p2        the global top-20 by mass -- the SAME rows as the entry-38 baseline probe's
            eval split, so the final-action baseline and the random control are scored
            here too and every difference on those rows is the label or the training
            selection, never the tokens.

THE JOIN is ``(name, step, token_id)`` and every one of the three indexes
``step["output_tokens"]``: the manifest's ``token_id``, the rollout's ``eos_token_pos`` and
``reasoning_pos + eos[0] + 1`` in the mass table. Sentence 0 of a rollout is the
no-reasoning cutoff and owns no reasoning tokens, so every token lands in a sentence >= 1.

COORDINATES are ``build_sentence_loudness.py``'s, so the two CSVs can be read side by side:
``sentence_frac`` 0 at a sentence's first token and 1 at its last, ``rel_sentence`` =
sentence_idx - convinced_idx (0 IS the convinced sentence), ``x_sentence`` = rel - 1 + frac.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from telos_interp.commands.prepare_activations_for_probing.manifest_loader import load_next_action_compact
from telos_interp.probe_models import create_classification_model

PREPARED = Path("/workspace/prepared")
LB_PROBES = Path("/workspace/reasoning_theatre/local_belief_probes/probes")
BASELINE = Path("/workspace/probes/next_action_mass_l15")

# id -> action, the fixed LEFT,UP,RIGHT,DOWN order plot_commitment_probs.py already writes.
ID2A = {0: "LEFT", 1: "UP", 2: "RIGHT", 3: "DOWN"}

ROWSETS: dict[str, dict] = {
    "p1_full": {
        "eval_dir": PREPARED / "local_belief_p1_split_eval",
        "probes": {
            "p1_lr": LB_PROBES / "local_belief_p1_lr.pt",
            "p1_mlp": LB_PROBES / "local_belief_p1_mlp.pt",
        },
    },
    "p1_top20": {
        "eval_dir": PREPARED / "local_belief_p1_top20_split_eval",
        "probes": {
            "p1t20_lr": LB_PROBES / "local_belief_p1_top20_lr.pt",
            "p1t20_mlp": LB_PROBES / "local_belief_p1_top20_mlp.pt",
        },
    },
    "p2": {
        "eval_dir": PREPARED / "local_belief_p2_split_eval",
        "probes": {
            "p2_lr": LB_PROBES / "local_belief_p2_lr.pt",
            "p2_mlp": LB_PROBES / "local_belief_p2_mlp.pt",
            # the "previous probes": same tokens, same split, FINAL-action label.
            "base_lr": BASELINE / "next_action_probe_jlens_topall_lr.pt",
            "base_mlp": BASELINE / "next_action_probe_jlens_topall_mlp.pt",
            # the matched random-selection control, also final-action. Scored on tokens it
            # was not selected for -- entry 37(d)'s distribution shift applies, read it as
            # a floor, not as an arm.
            "rand_lr": BASELINE / "next_action_probe_random_topall_lr.pt",
            "rand_mlp": BASELINE / "next_action_probe_random_topall_mlp.pt",
        },
    },
}

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


def load_probe(path: Path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = create_classification_model(
        d["model_type"], d["input_dim"], d["num_classes"], d["hidden_dims"] or [], d["dropout"] or 0.0
    )
    m.load_state_dict(d["model_state_dict"])
    m.eval()
    return m, d.get("scaler_mean"), d.get("scaler_std")


def score(probe_path: Path, X: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (pred, probs) for one probe over the whole rowset."""
    m, mean, std = load_probe(probe_path)
    Xs = X if mean is None else (X - mean) / std
    m = m.to(device)
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xs), 8192):
            outs.append(torch.softmax(m(Xs[i : i + 8192].to(device)).float(), dim=-1).cpu())
    probs = torch.cat(outs)
    return probs.argmax(-1), probs


def mass_index(mass_path: Path, layer: int) -> dict[int, dict]:
    """{(step, reasoning_pos)} -> {token, logmass}, plus per-(step) ranks.

    Rank 1 is the LOUDEST token of that step's chain, which is the ordering
    ``jlens_top_k_global`` selected on.
    """
    col = f"L{layer}"
    # csv.DictReader, never pandas: decoded tokens include "NA", commas and newlines.
    with open(mass_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_step: dict[int, list[dict]] = {}
    for r in rows:
        by_step.setdefault(int(r["step"]), []).append(r)
    out: dict[tuple[int, int], dict] = {}
    for step, rs in by_step.items():
        rs.sort(key=lambda r: int(r["reasoning_pos"]))
        n = len(rs)
        order = sorted(range(n), key=lambda i: -float(rs[i][col]))
        rank = [0] * n
        for k, i in enumerate(order):
            rank[i] = k + 1
        for i, r in enumerate(rs):
            out[(step, int(r["reasoning_pos"]))] = {
                "token": r["token"],
                "logmass": float(r[col]),
                "rank": rank[i],
                "n": n,
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens-root", type=Path, default=Path("/workspace/activations/jlens_mass_l15"))
    ap.add_argument(
        "--probs-root", type=Path, default=Path("/workspace/reasoning_theatre/trajectories_train_single_step_probs")
    )
    ap.add_argument(
        "--signal-json",
        type=Path,
        default=Path("/workspace/jlens/direction_tokens_full.json"),
        help="the vocabulary the mass table was built against; flags whether the TOKEN "
        "ITSELF is a direction word, which is the standing confound.",
    )
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", type=Path, default=Path("/workspace/reasoning_theatre/probe_loudness/per_token.csv"))
    ap.add_argument("--rowsets", default="all", help="comma-separated subset of ROWSETS, or 'all'.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="first N trajectories of each rowset (smoke test).")
    args = ap.parse_args()

    wanted = list(ROWSETS) if args.rowsets == "all" else args.rowsets.split(",")
    vocab = {t for lst in json.loads(args.signal_json.read_text()).values() for t in lst}

    fields = list(BASE_FIELDS)
    for rs in wanted:
        for p in ROWSETS[rs]["probes"]:
            fields += [f"{p}_pred", f"{p}_p_local", f"{p}_p_final", f"{p}_pmax"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    skipped: dict[str, int] = {}
    mass_mismatch = 0
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()

        for rs_name in wanted:
            cfg = ROWSETS[rs_name]
            mpath = cfg["eval_dir"] / "manifest.json"
            manifest = json.loads(mpath.read_text())
            samples = manifest["samples"]
            print(f"[{rs_name}] {len(samples)} samples, {len(cfg['probes'])} probes", flush=True)

            data = load_next_action_compact(manifest, mpath)
            X = data["base_act"].float()
            y_local = data["labels"].long()
            y_final = torch.tensor([s.get("final_label", s["label"]) for s in samples], dtype=torch.long)

            preds: dict[str, torch.Tensor] = {}
            probs: dict[str, torch.Tensor] = {}
            for pname, ppath in cfg["probes"].items():
                preds[pname], probs[pname] = score(ppath, X, args.device)
                acc_l = (preds[pname] == y_local).float().mean().item()
                acc_f = (preds[pname] == y_final).float().mean().item()
                print(f"    {pname:10s} acc vs local {acc_l:.4f}  vs final {acc_f:.4f}", flush=True)

            # group rows by trajectory so each rollout JSON / mass CSV is read once
            by_name: dict[str, list[int]] = {}
            for i, s in enumerate(samples):
                by_name.setdefault(s["name"], []).append(i)
            names = sorted(by_name)
            if args.limit:
                names = names[: args.limit]

            for ni, name in enumerate(names):
                size = name.split("_size")[1].split("_")[0]
                mass_path = args.lens_root / f"size{size}" / name / f"{name}_jlens_direction_mass.csv"
                roll_path = args.probs_root / f"size{size}" / f"{name}.json"
                if not mass_path.exists() or not roll_path.exists():
                    skipped["missing artifact"] = skipped.get("missing artifact", 0) + len(by_name[name])
                    continue
                with open(roll_path, encoding="utf-8") as f:
                    steps = {s["step_id"]: s for s in json.load(f)["steps"]}
                mass = mass_index(mass_path, args.layer)
                spans: dict[int, dict[int, int]] = {}

                for i in by_name[name]:
                    s = samples[i]
                    rec = steps.get(s["step"])
                    if rec is None:
                        skipped["no rollout step"] = skipped.get("no rollout step", 0) + 1
                        continue
                    evals = rec["sentence_evals"]
                    eos = [e["eos_token_pos"] for e in evals]
                    offset = eos[0] + 1
                    tok_idx = int(s["token_id"])
                    if s["step"] not in spans:
                        spans[s["step"]] = sentence_of_token(eos)
                    si = spans[s["step"]].get(tok_idx)
                    if si is None:
                        skipped["token outside every sentence"] = skipped.get("token outside every sentence", 0) + 1
                        continue
                    rp = tok_idx - offset
                    mrow = mass.get((s["step"], rp))
                    if mrow is None:
                        skipped["no mass row"] = skipped.get("no mass row", 0) + 1
                        continue
                    lm = float(s.get("dir_logmass", mrow["logmass"]))
                    if abs(lm - mrow["logmass"]) > 1e-4:
                        mass_mismatch += 1

                    start = eos[si - 1] + 1
                    sent_len = eos[si] - start + 1
                    pos_in = tok_idx - start
                    frac = (pos_in / (sent_len - 1)) if sent_len > 1 else 1.0
                    conv = rec["convinced_sentence_idx"]
                    rel = "" if conv is None else si - conv
                    acts = [e["model_action"] for e in evals]
                    n_tok = mrow["n"]
                    # rank of this token's mass inside its own sentence, 1 = loudest.
                    sent_rank = 1 + sum(
                        1
                        for p in range(start - offset, eos[si] - offset + 1)
                        if (s["step"], p) in mass and mass[(s["step"], p)]["logmass"] > mrow["logmass"]
                    )

                    row = {
                        "rowset": rs_name,
                        "name": name,
                        "size": s.get("size", size),
                        "complexity": name.split("_comp")[1].split("_")[0],
                        "step": s["step"],
                        "token_id": tok_idx,
                        "reasoning_pos": rp,
                        "token": mrow["token"],
                        "is_direction_token": int(mrow["token"].replace("Ġ", " ") in vocab),
                        "cutoff_kind": s.get("cutoff_kind", ""),
                        "dir_logmass": f"{lm:.6f}",
                        "dir_prob": f"{math.exp(lm):.9g}",
                        "mass_rank_in_traj": mrow["rank"],
                        "mass_pct_in_traj": f"{mrow['rank'] / n_tok:.6f}",
                        "mass_rank_in_sentence": sent_rank,
                        "n_reasoning_tokens": n_tok,
                        "reasoning_frac": f"{(rp / (n_tok - 1)) if n_tok > 1 else 1.0:.6f}",
                        "n_sentences": len(evals) - 1,
                        "sentence_idx": si,
                        "pos_in_sentence": pos_in,
                        "sentence_len": sent_len,
                        "sentence_frac": f"{frac:.6f}",
                        "is_sentence_end": int(tok_idx == eos[si]),
                        "convinced_idx": "" if conv is None else conv,
                        "rel_sentence": rel,
                        "x_sentence": "" if rel == "" else f"{rel - 1 + frac:.6f}",
                        "n_switches": sum(1 for a, b in zip(acts, acts[1:], strict=False) if a != b),
                        "label_local": ID2A[int(y_local[i])],
                        "label_final": ID2A[int(y_final[i])],
                        "ground_truth": rec["ground_truth"],
                        "rollout_answer_prob": s.get("rollout_answer_prob", ""),
                        "rollout_correct": int(bool(s["rollout_correct"])) if "rollout_correct" in s else "",
                    }
                    for pname in cfg["probes"]:
                        pr = probs[pname][i]
                        row[f"{pname}_pred"] = ID2A[int(preds[pname][i])]
                        row[f"{pname}_p_local"] = f"{float(pr[int(y_local[i])]):.6f}"
                        row[f"{pname}_p_final"] = f"{float(pr[int(y_final[i])]):.6f}"
                        row[f"{pname}_pmax"] = f"{float(pr.max()):.6f}"
                    w.writerow(row)
                    n_rows += 1
                if (ni + 1) % 200 == 0:
                    print(f"    {ni + 1}/{len(names)} trajectories, {n_rows} rows", flush=True)

    print(f"wrote {n_rows} rows -> {args.out}", flush=True)
    print(f"  manifest dir_logmass vs mass table L{args.layer} mismatches: {mass_mismatch}", flush=True)
    for k, v in skipped.items():
        print(f"  skipped: {k}: {v}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
