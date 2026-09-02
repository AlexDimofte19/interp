"""Relabel a token-major next_action manifest with the model's LOCAL BELIEF.

Input: a v3 token-major `next_action` manifest (one entry per (token, layer),
each currently labeled with the trajectory's final `agent_action`) and the
directory of per-trajectory rollout JSONs from `run_inference.py` for one
truncation strategy.

For each sample we look up `(name, step, token_id)` in the rollout -- the
rollout's `eos_token_pos` indexes the same `step["output_tokens"]` list as the
manifest's `token_id` -- and replace `label` with the action the model actually
emitted when its reasoning was truncated at that token (`model_action`). Samples
with no matching cutoff, or a cutoff whose `model_action` is null, are dropped.

The original label is kept as `final_label`, plus `rollout_answer_prob`,
`rollout_correct`, `cutoff_kind` and `dir_logmass` for later analysis. Nothing
else about the manifest changes -- it stays token-major, `activations_root` is
untouched, no activations move -- so `split_next_action_manifest.py` and
`train_next_action_probe` read it unchanged.

    python relabel_manifest_from_rollout.py PREPARED_DIR ROLLOUT_DIR OUT_DIR
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

NEXT_ACTION_TO_ID = {"LEFT": 0, "UP": 1, "RIGHT": 2, "DOWN": 3}


def load_rollout_evals(rollout_dir: Path, names: set[str]) -> dict[tuple[str, int, int], dict]:
    """{(name, step_id, eos_token_pos): sentence_eval} for the wanted trajectory names."""
    out: dict[tuple[str, int, int], dict] = {}
    n_files = 0
    for name in sorted(names):
        size = name.split("_size")[1].split("_")[0]
        path = rollout_dir / f"{name}.json"
        if not path.exists():
            path = rollout_dir / f"size{size}" / f"{name}.json"
        if not path.exists():
            continue
        n_files += 1
        doc = json.loads(path.read_text())
        for step in doc["steps"]:
            sid = step["step_id"]
            for ev in step["sentence_evals"]:
                pos = ev.get("eos_token_pos")
                if pos is None:
                    continue
                # keep the first eval for a position (endpoint/interior dupes are rare and equal)
                out.setdefault((name, sid, pos), ev)
    print(f"  rollout: {n_files} files, {len(out)} (name, step, pos) cutoffs")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prepared_dir", type=Path, help="v3 token-major next_action manifest dir")
    ap.add_argument("rollout_dir", type=Path, help="dir of run_inference.py per-trajectory JSONs")
    ap.add_argument("out_dir", type=Path, help="output manifest dir")
    ap.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="also write a per-sample CSV (name, step, token_id, final, local, agree, prob, correct, kind).",
    )
    args = ap.parse_args()

    manifest = json.loads((args.prepared_dir / "manifest.json").read_text())
    key = "samples" if "samples" in manifest else "trajectories"
    samples = manifest[key]
    if manifest.get("probe_type") != "next_action":
        raise SystemExit(f"probe_type={manifest.get('probe_type')!r}, expected next_action")

    # canonical name per id (LEFT/UP/RIGHT/DOWN), ignoring aliases like TOP
    id_to_action = {v: k for k, v in NEXT_ACTION_TO_ID.items()}
    names = {s["name"] for s in samples}
    print(f"manifest: {len(samples)} samples over {len(names)} trajectories")

    evals = load_rollout_evals(args.rollout_dir, names)

    out_samples: list[dict] = []
    rows: list[dict] = []
    n_no_cutoff = 0
    n_null_action = 0
    n_flipped = 0
    agree_final = 0
    per_traj_kept: Counter = Counter()
    for s in samples:
        ev = evals.get((s["name"], s["step"], s["token_id"]))
        if ev is None:
            n_no_cutoff += 1
            continue
        ma = ev.get("model_action")
        if ma is None or ma not in NEXT_ACTION_TO_ID:
            n_null_action += 1
            continue
        local = NEXT_ACTION_TO_ID[ma]
        final = s.get("label")
        s2 = dict(s)
        s2["label"] = local
        s2["final_label"] = final
        s2["rollout_answer_prob"] = ev.get("answer_prob")
        s2["rollout_correct"] = bool(ev.get("correct"))
        s2["cutoff_kind"] = ev.get("cutoff_kind")
        s2["dir_logmass"] = ev.get("dir_logmass")
        s2["sentence_idx"] = ev.get("sentence_idx")
        s2["cut_sentence_idx"] = ev.get("cut_sentence_idx")
        # direction_count / layer_direction_count = the layer-15 loudness (log direction
        # mass) of the cut token, so split_next_action_manifest.py can rank tokens by
        # loudness (--tokens-per-trajectory K) instead of drawing uniformly, and analysis
        # can bin probe correctness by loudness. Absent on the endpoint cutoffs.
        lm = ev.get("dir_logmass")
        if lm is not None:
            s2["direction_count"] = lm
            s2["layer_direction_count"] = lm
        out_samples.append(s2)
        per_traj_kept[s["name"]] += 1
        if final is not None and local != final:
            n_flipped += 1
        if final is not None and local == final:
            agree_final += 1
        if args.report_csv is not None:
            rows.append(
                {
                    "name": s["name"],
                    "size": s.get("size"),
                    "step": s["step"],
                    "token_id": s["token_id"],
                    "token": s.get("token"),
                    "final_action": id_to_action.get(final, final),
                    "local_action": ma,
                    "agree": int(local == final) if final is not None else "",
                    "rollout_answer_prob": ev.get("answer_prob"),
                    "rollout_correct": int(bool(ev.get("correct"))),
                    "cutoff_kind": ev.get("cutoff_kind"),
                    "dir_logmass": ev.get("dir_logmass"),
                }
            )

    kept_names = set(per_traj_kept)
    print(f"kept {len(out_samples)}/{len(samples)} samples over {len(kept_names)} trajectories")
    print(f"  dropped: {n_no_cutoff} no matching cutoff, {n_null_action} null/invalid model_action")
    denom = n_flipped + agree_final
    if denom:
        print(
            f"  local vs final: {n_flipped} differ ({n_flipped / denom:.1%}), "
            f"{agree_final} agree ({agree_final / denom:.1%})"
        )
    lc = Counter(s["label"] for s in out_samples)
    fc = Counter(s["final_label"] for s in out_samples if s["final_label"] is not None)
    print("  local  label dist:", {id_to_action.get(k, k): v for k, v in sorted(lc.items())})
    print("  final  label dist:", {id_to_action.get(k, k): v for k, v in sorted(fc.items())})
    print(
        f"  per-traj kept: min {min(per_traj_kept.values())} mean "
        f"{sum(per_traj_kept.values()) / len(per_traj_kept):.1f} max {max(per_traj_kept.values())}"
    )

    out = dict(manifest)
    out[key] = out_samples
    out["label_source"] = {
        "kind": "local_belief",
        "rollout_dir": str(args.rollout_dir),
        "relabeled_from": str(args.prepared_dir.resolve()),
        "n_in": len(samples),
        "n_out": len(out_samples),
        "n_dropped_no_cutoff": n_no_cutoff,
        "n_dropped_null_action": n_null_action,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(out))
    print(f"wrote {args.out_dir}/manifest.json")

    if args.report_csv is not None:
        args.report_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.report_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
