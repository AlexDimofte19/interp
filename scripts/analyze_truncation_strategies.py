"""Compare the truncation-strategy rollout arms (ICLR log entry 43) on one table of cutoffs.

Three arms cut the same reasoning of the same 3,600 trajectories in three places -- every
sentence end (``eos``), each sentence's loudest token (``jlens_argmax_per_sentence``), and the
chain's 20 loudest tokens (``jlens_top_k_global``) -- and are then asked for the action. This
script pools all of them into one long table of cutoff-evals and reports the comparisons that
entry 43 says are the only admissible ones.

*** THE CONFOUND THIS SCRIPT EXISTS TO HANDLE. *** The loudest token in a sentence is very
often the direction word the model is writing down, so a raw accuracy gain for a loud arm is
NOT evidence that the commitment is readable earlier -- it may only mean the model was asked
just after it typed "UP". Entry 42(e) established the control and this script applies it: drop
cutoffs whose own token is in the direction vocabulary, then widen to a +-k window, because the
lens predicts the NEXT tokens and so the token just before " up" is loud without being a
direction word itself. ``verbalization_control.csv`` is that table, and the arm-vs-arm
comparison is only quotable from it, or from ``by_chain_position.csv`` at matched positions.

Two things make the arms comparable at all, and both are checked here rather than assumed:

* every arm keeps the same two ENDPOINT cutoffs (no reasoning, all reasoning), so those evals
  are literally the same prompt in every arm and their accuracies must agree --
  ``endpoint_agreement.csv`` is that check, and a disagreement means something in the rollout
  is not deterministic;
* the ``eos`` arm is read from the finished 36k rollout restricted to the same names, since it
  is byte-for-byte the same code path. Its evals carry no ``cutoff_kind``/``dir_logmass``, so
  those are reconstructed here: kind by position in the list, loudness by the same layer-15
  mass-table lookup the loud arms used.

Token text comes from the mass table, whose ``reasoning_pos`` is offset from the rollout's
``eos_token_pos`` by ``min(analysis_positions(output_tokens))`` -- read per trajectory, never
hardcoded to 3. Membership uses entry 42's own rule (``token.replace("\\u0120", " ") in vocab``)
so the numbers here and in ``loudness/`` mean the same thing.

    python scripts/analyze_truncation_strategies.py \\
        --out-root /workspace/reasoning_theatre/rollout_strategies \\
        --output-dir /workspace/reasoning_theatre/rollout_strategies/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inference_oss.truncation_strategies import (  # noqa: E402
    KIND_END_OF_REASONING,
    KIND_NO_REASONING,
    KIND_SENTENCE_END,
    LoudnessUnavailable,
    MassTableLoudness,
    analysis_positions,
)

DEFAULT_OUT_ROOT = Path("/workspace/reasoning_theatre/rollout_strategies")
DEFAULT_EOS_ROOT = Path("/workspace/reasoning_theatre/trajectories_train_single_step_probs")
DEFAULT_TRAJECTORIES = Path("/workspace/trajectories/reveng/trajectories_train_single_step")
DEFAULT_LENS_ROOT = Path("/workspace/activations/jlens_mass_l15")
DEFAULT_SIGNAL_JSON = Path("/workspace/jlens/direction_tokens_full.json")
LOUD_ARMS = ("jlens_argmax_per_sentence", "jlens_top_k_global")


def size_of(name: str) -> int:
    return int(name.split("_size")[1].split("_")[0])


def complexity_of(name: str) -> float:
    return float(name.split("_comp")[1].split("_")[0])


def load_vocab(path: Path) -> set[str]:
    """The direction vocabulary as one flat set of decoded strings (all four classes)."""
    raw = json.loads(path.read_text())
    return {t for cls in raw.values() for t in cls}


def read_tokens(lens: MassTableLoudness, name: str) -> dict[int, dict[int, str]]:
    """``{step_id: {reasoning_pos: token}}`` from the mass table, sidecar-checked.

    ``lens.load`` is called first purely so the ``.meta.json`` rule is enforced by the one
    place that owns it -- a mass table must never be read without knowing its vocabulary.
    """
    lens.load(name)
    path = lens.table_path(name)
    out: dict[int, dict[int, str]] = defaultdict(dict)
    with open(path, encoding="utf-8", newline="") as fh:  # DictReader, never pandas
        for row in csv.DictReader(fh):
            out[int(row["step"])][int(row["reasoning_pos"])] = row["token"]
    return out


def eos_arm_path(root: Path, name: str) -> Path:
    return root / f"size{size_of(name)}" / f"{name}.json"


def arm_result_path(root: Path, arm: str, name: str) -> Path:
    return root / arm / f"{name}.json"


def kind_for(idx: int, n: int) -> str:
    """Reconstruct the eos arm's cutoff kinds: first is the no-reasoning endpoint, last the full one."""
    if idx == 0:
        return KIND_NO_REASONING
    if idx == n - 1:
        return KIND_END_OF_REASONING
    return KIND_SENTENCE_END


def collect(args) -> pd.DataFrame:
    names = [ln.strip() for ln in open(args.names_file) if ln.strip()]
    vocab = load_vocab(args.signal_json)
    lens = MassTableLoudness(lens_root=args.lens_root, lens=args.lens, layer=args.layer)

    rows: list[dict] = []
    missing: dict[str, int] = defaultdict(int)
    for n_done, name in enumerate(names, 1):
        # token text + the offset that maps mass-table coordinates to output_tokens ones
        try:
            tokens_by_step = read_tokens(lens, name)
            mass_by_step = lens.load(name)
        except (LoudnessUnavailable, FileNotFoundError):
            missing["mass_table"] += 1
            continue
        traj = json.loads((DEFAULT_TRAJECTORIES / f"size{size_of(name)}" / f"{name}.json").read_text())
        offsets = {}
        for step in traj["steps"]:
            positions = analysis_positions(step.get("output_tokens") or [])
            offsets[step["step_id"]] = min(positions) if positions else 0

        for arm in args.arms:
            path = eos_arm_path(args.eos_root, name) if arm == "eos" else arm_result_path(args.out_root, arm, name)
            if not path.exists():
                missing[arm] += 1
                continue
            res = json.loads(path.read_text())
            for step in res["steps"]:
                sid = step["step_id"]
                off = offsets.get(sid, 0)
                toks = tokens_by_step.get(sid, {})
                mass = mass_by_step.get(sid, {})
                evals = step["sentence_evals"]
                for i, ev in enumerate(evals):
                    pos = ev["eos_token_pos"]
                    rpos = pos - off
                    tok = toks.get(rpos)
                    decoded = tok.replace("Ġ", " ") if tok is not None else None
                    window = {
                        k: any(
                            (toks.get(rpos + d, "").replace("Ġ", " ") in vocab)
                            for d in range(-k, k + 1)
                            if rpos + d in toks
                        )
                        for k in (1, 2, 3)
                    }
                    rows.append(
                        {
                            "arm": arm,
                            "name": name,
                            "size": size_of(name),
                            "complexity": complexity_of(name),
                            "step_id": sid,
                            "eval_idx": i,
                            "n_evals": len(evals),
                            "eos_token_pos": pos,
                            "reasoning_pos": rpos,
                            "cutoff_kind": ev.get("cutoff_kind") or kind_for(i, len(evals)),
                            "pos_in_sentence": ev.get("pos_in_sentence"),
                            "sentence_len": ev.get("sentence_len"),
                            "dir_logmass": ev.get("dir_logmass", mass.get(rpos)),
                            "token": tok,
                            "is_direction_token": int(decoded in vocab) if decoded is not None else None,
                            "dir_within_1": int(window[1]),
                            "dir_within_2": int(window[2]),
                            "dir_within_3": int(window[3]),
                            "correct": int(bool(ev["correct"])),
                            "model_action": ev.get("model_action"),
                            "ground_truth": step["ground_truth"],
                            "answer_prob": ev.get("answer_prob"),
                            "convinced_fraction": step.get("convinced_fraction"),
                            "first_correct_fraction": step.get("first_correct_fraction"),
                        }
                    )
        if n_done % 500 == 0:
            print(f"  {n_done}/{len(names)} names", flush=True)

    if missing:
        print(f"  missing results: {dict(missing)}")
    df = pd.DataFrame(rows)
    # chain fraction against a denominator shared by every arm: the end-of-reasoning position
    end = (
        df[df.cutoff_kind == KIND_END_OF_REASONING]
        .groupby(["name", "step_id"])["eos_token_pos"]
        .max()
        .rename("chain_len")
    )
    df = df.join(end, on=["name", "step_id"])
    df["chain_fraction"] = (df.eos_token_pos / df.chain_len).clip(0, 1)
    return df


def tables(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    mid = df[~df.cutoff_kind.isin([KIND_NO_REASONING, KIND_END_OF_REASONING])]

    overall = (
        df.groupby("arm")
        .apply(
            lambda g: pd.Series(
                {
                    # g["name"], never g.name: pandas binds the group key to the frame's .name
                    "n_trajectories": g["name"].nunique(),
                    "n_evals": len(g),
                    "cutoffs_per_step": len(g) / g.groupby(["name", "step_id"]).ngroups,
                    "cutoff_accuracy": g.correct.mean(),
                    "mean_answer_prob": g.answer_prob.mean(),
                }
            ),
            include_groups=False,
        )
        .join(
            mid.groupby("arm").apply(
                lambda g: pd.Series({"interior_accuracy": g.correct.mean(), "n_interior": len(g)}),
                include_groups=False,
            )
        )
        .join(
            df[df.cutoff_kind == KIND_NO_REASONING].groupby("arm").correct.mean().rename("no_reasoning_accuracy")
        )
        .join(
            df[df.cutoff_kind == KIND_END_OF_REASONING]
            .groupby("arm")
            .correct.mean()
            .rename("end_of_reasoning_accuracy")
        )
        .join(
            df.groupby("arm")[["convinced_fraction", "first_correct_fraction"]].mean(),
        )
    )
    overall.to_csv(out / "overall.csv")

    # the control: raw, then dropping direction-word cutoffs, then +-k windows
    ctrl = []
    for arm, g in mid.groupby("arm"):
        row = {"arm": arm, "n_interior": len(g), "accuracy_all": g.correct.mean()}
        row["share_direction_token"] = g.is_direction_token.mean()
        for col, label in (
            ("is_direction_token", "excl_token"),
            ("dir_within_1", "excl_pm1"),
            ("dir_within_2", "excl_pm2"),
            ("dir_within_3", "excl_pm3"),
        ):
            kept = g[g[col] == 0]
            row[f"{label}_kept_share"] = len(kept) / len(g)
            row[f"{label}_accuracy"] = kept.correct.mean() if len(kept) else float("nan")
        ctrl.append(row)
    pd.DataFrame(ctrl).set_index("arm").to_csv(out / "verbalization_control.csv")

    # matched position in the chain
    bins = pd.cut(mid.chain_fraction, [i / 10 for i in range(11)], include_lowest=True)
    by_pos = mid.groupby(["arm", bins], observed=True).agg(n=("correct", "size"), accuracy=("correct", "mean"))
    by_pos.to_csv(out / "by_chain_position.csv")

    excl = mid[mid.dir_within_1 == 0]
    by_pos_excl = excl.groupby(["arm", pd.cut(excl.chain_fraction, [i / 10 for i in range(11)], include_lowest=True)], observed=True).agg(
        n=("correct", "size"), accuracy=("correct", "mean")
    )
    by_pos_excl.to_csv(out / "by_chain_position_excl_pm1.csv")

    df.groupby(["arm", "size"]).agg(n=("correct", "size"), accuracy=("correct", "mean")).to_csv(
        out / "by_size.csv"
    )
    df.groupby(["arm", "complexity"]).agg(n=("correct", "size"), accuracy=("correct", "mean")).to_csv(
        out / "by_complexity.csv"
    )

    # ENDPOINT AGREEMENT IS THE NOISE FLOOR, and it is the first thing to read. Every arm
    # keeps the same two endpoint cutoffs, so those evals are the SAME PROMPT decoded greedily
    # -- identical cut position and identical prompt length, verified below. Any disagreement
    # is therefore the rollout's own reproducibility, not the strategy, and no arm-vs-arm
    # difference smaller than it means anything. It is not uniform: it concentrates entirely
    # where the model is unconfident (the no-reasoning endpoint, near chance), and vanishes
    # where it is sure (end of reasoning, p ~ 1). Broken out by kind for exactly that reason.
    endp = df[df.cutoff_kind.isin([KIND_NO_REASONING, KIND_END_OF_REASONING])]
    action = endp.pivot_table(
        index=["name", "step_id", "cutoff_kind"], columns="arm", values="model_action", aggfunc="first"
    ).dropna()
    prob = endp.pivot_table(index=["name", "step_id", "cutoff_kind"], columns="arm", values="answer_prob").dropna()
    pos = endp.pivot_table(index=["name", "step_id", "cutoff_kind"], columns="arm", values="eos_token_pos").dropna()
    arms = [a for a in action.columns]
    recs = []
    for kind, idx in action.groupby(level="cutoff_kind").groups.items():
        row = {"cutoff_kind": kind, "n": len(idx)}
        for a in arms:
            row[f"{a}_mean_answer_prob"] = prob.loc[idx, a].mean()
        for a in arms[1:]:
            row[f"{arms[0]}_vs_{a}_same_action"] = (action.loc[idx, arms[0]] == action.loc[idx, a]).mean()
            row[f"{arms[0]}_vs_{a}_same_cut_pos"] = (pos.loc[idx, arms[0]] == pos.loc[idx, a]).mean()
            row[f"{arms[0]}_vs_{a}_mean_abs_dprob"] = (prob.loc[idx, arms[0]] - prob.loc[idx, a]).abs().mean()
            row[f"{arms[0]}_vs_{a}_max_abs_dprob"] = (prob.loc[idx, arms[0]] - prob.loc[idx, a]).abs().max()
        recs.append(row)
    rec = {"n_endpoint_pairs": len(action)}
    for a in arms:
        rec[f"{a}_accuracy"] = df[(df.cutoff_kind.isin([KIND_NO_REASONING, KIND_END_OF_REASONING])) & (df.arm == a)].correct.mean()
    for r in recs:
        for a in arms[1:]:
            rec[f"{r['cutoff_kind']}_{arms[0]}_vs_{a}_same_action"] = r[f"{arms[0]}_vs_{a}_same_action"]
    pd.DataFrame(recs).to_csv(out / "endpoint_agreement.csv", index=False)

    # loudness actually achieved by each grid, as a description of what the arms did
    df.groupby("arm").dir_logmass.describe().to_csv(out / "loudness_by_arm.csv")

    summary = {
        "overall": json.loads(overall.to_json(orient="index")),
        "verbalization_control": json.loads(pd.DataFrame(ctrl).set_index("arm").to_json(orient="index")),
        "endpoint_agreement": rec,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--eos-root", type=Path, default=DEFAULT_EOS_ROOT)
    p.add_argument("--lens-root", type=Path, default=DEFAULT_LENS_ROOT)
    p.add_argument("--signal-json", type=Path, default=DEFAULT_SIGNAL_JSON)
    p.add_argument("--names-file", type=Path, default=None)
    p.add_argument("--lens", default="jlens")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--arms", nargs="+", default=["eos", *LOUD_ARMS])
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--per-token-csv", action="store_true", help="also write the full long table (large)")
    args = p.parse_args()
    args.names_file = args.names_file or args.out_root / "mass_l15_names.txt"
    args.output_dir = args.output_dir or args.out_root / "comparison"

    print(f"Collecting {args.arms} over {args.names_file} ...")
    df = collect(args)
    print(f"{len(df):,} cutoff-evals over {df['name'].nunique()} trajectories")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.per_token_csv:
        df.to_csv(args.output_dir / "per_cutoff.csv", index=False)
    summary = tables(df, args.output_dir)
    print(json.dumps(summary["overall"], indent=2))
    print(json.dumps(summary["endpoint_agreement"], indent=2))
    print(f"-> {args.output_dir}")


if __name__ == "__main__":
    main()
