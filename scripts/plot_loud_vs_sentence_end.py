"""Does the action at a LOUD token match its sentence's conclusion, or the one before it?

For every loud cutoff either jlens arm made, three rollout answers are already on disk:

* at the loud token itself   -- `jlens_argmax_per_sentence` (kind `loudest_in_sentence`, one per
                                sentence, at its argmax) or `jlens_top_k_global` (kind
                                `loud_top_k`, the chain's K loudest wherever they fall, so a
                                sentence may hold several cutoffs or none)
* at the END of its sentence -- the `eos` grid, eval k
* at the end of the PREVIOUS -- the `eos` grid, eval k-1

`cut_sentence_idx` is the sentence the loud cut lands in on the eos grid, and eos eval k is
exactly that sentence's end, so the join is by index and needs no re-derivation. Sentence 1's
"previous end" is the no-reasoning cutoff, i.e. the belief before any reasoning at all.

The question the two bars answer: at the loud token, has the model already arrived at what it
will say at the end of this sentence, or is it still carrying the previous sentence's answer?
Entry 41(c) put the commitment on a sentence's FIRST tokens and entry 42(a) showed a sentence
end is its quietest point; if both hold, the loud token should track the sentence it is IN.

*** READ THE SECOND GROUP, NOT THE FIRST. *** Where a sentence ends on the same action the
previous one did -- the large majority -- both bars are the same number by construction and
the comparison is empty. The right contrast is the subset where the two comparators DISAGREE,
which is the same restriction entry 41 applied ("the 2,074 sentence ends where those differ").

Two subsets are printed but deliberately kept OFF the plot, because they qualify the reading
rather than answer it: cutoffs that ARE the sentence end (~10% of sentences, trivially equal),
and cutoffs whose own token is a direction word (entry 42(e)'s confound -- the loudest token in
a sentence is very often the answer being typed, so agreement there is verbalization, not
anticipation).

*** THE CONTROL THIS CANNOT SUPPLY. *** A loud cutoff sits partway into its sentence and is
therefore simply CLOSER IN TIME to that sentence's end than to the previous one's. Proximity
alone could produce the entire lean, with loudness doing no work. Only a matched random-position
arm separates the two, and it does not exist yet -- see the TODO in claude_session_readme.md.

    python scripts/plot_loud_vs_sentence_end.py --arm jlens_argmax_per_sentence
    python scripts/plot_loud_vs_sentence_end.py --arm jlens_top_k_global
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inference_oss.truncation_strategies import (  # noqa: E402
    KIND_LOUD_TOP_K,
    KIND_LOUDEST_IN_SENTENCE,
    LoudnessUnavailable,
    MassTableLoudness,
    analysis_positions,
)

ROLLOUT_ROOT = Path("/workspace/reasoning_theatre/rollout_strategies")
# each arm's loud cutoffs carry their own kind; the endpoints are excluded by not listing theirs
LOUD_KINDS = {
    "jlens_argmax_per_sentence": KIND_LOUDEST_IN_SENTENCE,
    "jlens_top_k_global": KIND_LOUD_TOP_K,
}
EOS_ROOT = Path("/workspace/reasoning_theatre/trajectories_train_single_step_probs")
TRAJ_ROOT = Path("/workspace/trajectories/reveng/trajectories_train_single_step")
LENS_ROOT = Path("/workspace/activations/jlens_mass_l15")
SIGNAL_JSON = Path("/workspace/jlens/direction_tokens_full.json")
NAMES = Path("/workspace/reasoning_theatre/rollout_strategies/mass_l15_names.txt")

# Reference-palette categorical slots 1 and 2 (validated adjacent pair, light mode).
C_THIS, C_PREV = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Point estimate and 95% Wilson interval -- the normal approximation is wrong near 0 and 1."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p, z = k / n, 1.959964
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def collect(names: list[str], vocab: set[str], lens: MassTableLoudness, arm: str) -> list[dict]:
    loud_root, kind_wanted = ROLLOUT_ROOT / arm, LOUD_KINDS[arm]
    rows: list[dict] = []
    missing: dict[str, int] = defaultdict(int)
    for i, name in enumerate(names, 1):
        size = int(name.split("_size")[1].split("_")[0])
        loud_p, eos_p = loud_root / f"{name}.json", EOS_ROOT / f"size{size}" / f"{name}.json"
        if not (loud_p.exists() and eos_p.exists()):
            missing["result"] += 1
            continue
        loud, eos = json.loads(loud_p.read_text()), json.loads(eos_p.read_text())

        # token text for the direction-word control, in output_tokens coordinates
        toks: dict[int, dict[int, str]] = {}
        try:
            lens.load(name)
            with open(lens.table_path(name), encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):  # DictReader, never pandas
                    toks.setdefault(int(r["step"]), {})[int(r["reasoning_pos"])] = r["token"]
            traj = json.loads((TRAJ_ROOT / f"size{size}" / f"{name}.json").read_text())
            offs = {
                s["step_id"]: (min(analysis_positions(s.get("output_tokens") or [])) if s.get("output_tokens") else 0)
                for s in traj["steps"]
            }
        except (LoudnessUnavailable, FileNotFoundError, ValueError):
            missing["mass_table"] += 1
            offs = {}

        eos_by_step = {s["step_id"]: s["sentence_evals"] for s in eos["steps"]}
        for step in loud["steps"]:
            sid = step["step_id"]
            grid = eos_by_step.get(sid)
            if not grid:
                missing["eos_step"] += 1
                continue
            off = offs.get(sid)
            tk = toks.get(sid, {})
            for ev in step["sentence_evals"]:
                if ev.get("cutoff_kind") != kind_wanted:
                    continue
                k = ev.get("cut_sentence_idx")
                if k is None or k < 1 or k >= len(grid):
                    missing["sentence_idx"] += 1
                    continue
                tok = tk.get(ev["eos_token_pos"] - off) if off is not None else None
                rows.append(
                    {
                        "name": name,
                        "step_id": sid,
                        "sentence": k,
                        "loud_action": ev["model_action"],
                        "this_end_action": grid[k]["model_action"],
                        "prev_end_action": grid[k - 1]["model_action"],
                        "ground_truth": step["ground_truth"],
                        "coincides_with_end": int(ev["eos_token_pos"] == grid[k]["eos_token_pos"]),
                        "is_direction_token": (
                            int(tok.replace("Ġ", " ") in vocab) if tok is not None else None
                        ),
                        "pos_in_sentence": ev.get("pos_in_sentence"),
                        "sentence_len": ev.get("sentence_len"),
                        "dir_prob": ev.get("dir_prob"),
                    }
                )
        if i % 500 == 0:
            print(f"  {i}/{len(names)} names, {len(rows):,} sentences", flush=True)
    if missing:
        print(f"  skipped: {dict(missing)}")
    return rows


def rates(rows: list[dict]) -> dict:
    n = len(rows)
    this_k = sum(r["loud_action"] == r["this_end_action"] for r in rows)
    prev_k = sum(r["loud_action"] == r["prev_end_action"] for r in rows)
    return {
        "n": n,
        "this": wilson(this_k, n),
        "prev": wilson(prev_k, n),
        "acc": sum(r["loud_action"] == r["ground_truth"] for r in rows) / n if n else float("nan"),
    }


ARM_TITLE = {
    "jlens_argmax_per_sentence": "each sentence's loudest token",
    "jlens_top_k_global": "the chain's 20 loudest tokens",
}


def panel(ax, groups, *, title, show_legend, ylabel):
    """One facet: three subsets side by side, two bars each."""
    width, gap, xs = 0.34, 0.012, range(len(groups))
    for off, key, colour, label in ((-(width + gap) / 2, "this", C_THIS, "End of THIS sentence"),
                                    (+(width + gap) / 2, "prev", C_PREV, "End of PREVIOUS sentence")):
        vals = [g[1][key][0] for g in groups]
        lo = [g[1][key][0] - g[1][key][1] for g in groups]
        hi = [g[1][key][2] - g[1][key][0] for g in groups]
        bars = ax.bar([x + off for x in xs], vals, width, color=colour, label=label, zorder=3, linewidth=0)
        ax.errorbar([x + off for x in xs], vals, yerr=[lo, hi], fmt="none", ecolor=INK_MUTED,
                    elinewidth=1, capsize=2.5, zorder=4)
        for b, v, h in zip(bars, vals, hi):  # clear the error bar, not just the bar
            # a surface-coloured pad so a label near 0.25 is not struck through by the chance rule
            ax.text(b.get_x() + b.get_width() / 2, v + h + 0.02, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, color=INK, zorder=5,
                    bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=0.9))
    ax.axhline(0.25, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{g[0]}\nn = {g[1]['n']:,}" for g in groups], fontsize=8.5, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK, pad=8, loc="left")
    if ylabel:
        ax.set_ylabel("Share of loud cutoffs whose action matches", fontsize=9.5, color=INK_MUTED)
    if show_legend:
        ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK_MUTED)
    ax.set_ylim(0, 1.06)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="jlens_argmax_per_sentence", choices=sorted(LOUD_KINDS))
    ap.add_argument("--names-file", type=Path, default=NAMES)
    ap.add_argument("--output-dir", type=Path, default=ROLLOUT_ROOT / "comparison")
    ap.add_argument("--limit", type=int, default=0, help="first N names only (smoke test)")
    args = ap.parse_args()

    names = [ln.strip() for ln in open(args.names_file) if ln.strip()]
    if args.limit:
        names = names[: args.limit]
    vocab = {t for cls in json.loads(SIGNAL_JSON.read_text()).values() for t in cls}
    print(f"{args.arm}: {len(names)} trajectories, {len(vocab)} direction tokens")

    rows = collect(names, vocab, MassTableLoudness(lens_root=LENS_ROOT, lens="jlens", layer=15), args.arm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"loud_vs_sentence_end_{args.arm}"
    with open(args.output_dir / f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # The three subsets are the SAME comparison under two exclusions, so they are drawn side by
    # side rather than in three figures: the controls are the finding, not an appendix to it.
    def subsets(pool):
        return [
            ("no exclusion", rates(pool)),
            ("excl. cutoffs that ARE\nthe sentence end", rates([r for r in pool if not r["coincides_with_end"]])),
            ("excl. direction-word\ncutoffs", rates([r for r in pool if r["is_direction_token"] == 0])),
        ]

    differ = [r for r in rows if r["this_end_action"] != r["prev_end_action"]]
    print(f"\n{'subset':52s} {'n':>7s}  {'= this end':>10s}  {'= prev end':>10s}  {'acc':>6s}")
    for pool_name, pool in (("all cutoffs", rows), ("where this end != prev end", differ)):
        for label, st in subsets(pool):
            flat = label.replace("\n", " ")
            print(f"{pool_name + ' | ' + flat:52s} {st['n']:7,d}  {st['this'][0]:10.3f}  "
                  f"{st['prev'][0]:10.3f}  {st['acc']:6.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), dpi=200, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
    panel(axes[0], subsets(rows), title="All cutoffs — uninformative by construction",
          show_legend=False, ylabel=True)
    panel(axes[1], subsets(differ), title="Only where the two comparators DIFFER — the real test",
          show_legend=True, ylabel=False)
    axes[0].text(-0.44, 0.272, "chance (4 actions)", fontsize=8, color=INK_MUTED, ha="left")
    fig.suptitle(f"Action at {ARM_TITLE[args.arm]} vs. its sentence ends   ({args.arm})",
                 fontsize=12.5, color=INK, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.017,
             "Left: the two comparators are the same string in ~89% of cases, so both bars are the "
             "same number by construction. Right: the subset where they disagree.\n"
             "Bars are 95% Wilson intervals. Columns need not sum to 1 — the cutoff can emit a third "
             "action matching neither.",
             fontsize=7.6, color=INK_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 0.955))
    out = args.output_dir / f"{stem}.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
