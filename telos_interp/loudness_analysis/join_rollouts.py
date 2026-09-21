#!/usr/bin/env python3
"""Join a probe table, an every-token rollout and a commitment CSV into one per-token table.

Entry 46 asked whether a louder token decodes better. It answered on the eval-720 split,
and -- the limitation this script removes -- it read each probe only on the tokens that
probe's own jlens selection had picked. Inside a fixed top-K arm the loudness axis is
truncated by construction, which is exactly why entry 46(b) needed a chain-length control.

Here every probe is read on the SAME 87,221 tokens: every reasoning token of the 360
held-out trajectories, a tree disjoint from every probe's training set
(``scripts/audit_trajectory_sets.py``). No selection sits between the loudness axis and
the label, so the axis spans the real distribution and no arm gets a token set tuned to it.

THE PROBES ARE AN ARGUMENT. ``--probes`` names them as key=column_prefix pairs and
``--rowset`` names the set they form; nothing about which probes exist is baked into
this file any more. Every path, the lens, the signal and the layer are required too --
there is no default round.

This is a pure join of three artifacts -- no .pt, no model, no GPU -- so it is cheap to
re-run when any of them is rebuilt:

``--probe-csv``       ``score_probes_per_token.py``'s one-pass scoring of the probes named
                      by ``--probes``, over the held-out tree, with ``--full-probs``.
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

THE COMMITMENT BOUNDARY IS NOW PER-TOKEN. Every earlier script took the boundary from
``convinced_sentence_idx``, which ran "first cutoff from which every later answer is
correct" over SENTENCE ENDS only. The ``every_token`` arm evaluates the same rule at every
reasoning token, so the boundary no longer has to be a sentence end, and ``convinced_token()``
computes it here from the dense eval list. The sentence-end grid is wrong in both
directions and neither is rare: on the 264 held-out steps where both boundaries are inside
the chain it rounds a mid-sentence commitment UP to the next sentence end 70% of the time
(median 3 tokens), and 23% of the time it reports a boundary that is TOO EARLY because the
chain relapses between two sentence ends (median 29 tokens early). It also calls 96 of 360
steps "committed before writing anything" against the dense grid's 56.

Both coordinate families are written, so a figure can be rebuilt either way:

  convinced_idx / rel_sentence / x_sentence          LEGACY, sentence-end grid
  convinced_token_pos / rel_token / is_convinced     the per-token boundary

``convinced_token_pos`` is a position in ``step["output_tokens"]``; ``NO_REASONING_POS``
(-1) means the step was already committed at the no-reasoning cutoff -- the per-token twin
of ``convinced_sentence_idx == 0`` and the cohort entries 41/42 drop. ``rel_token`` is
``reasoning_pos - convinced_token_reasoning_pos``, signed, 0 AT the boundary token.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from telos_interp.loudness_analysis import columns as cols

# csv.DictReader everywhere, never pandas: decoded tokens include "NA", empty strings,
# embedded commas and newlines, which pandas' NA handling silently corrupts.
csv.field_size_limit(10**9)

# WHICH LENS DEFINES LOUDNESS. Both source CSVs carry jlens_mass_L15 and logitlens_mass_L15
# side by side, so this is a column choice and never a re-gather. It was hard-coded to the
# jlens, which silently binned the logitlens-selected probes by a ruler they were never
# selected by -- at layer 15 the two lenses' top-20 sets overlap only ~50%. Set by
# --mass-column; every downstream caption must name the lens it got.
DEFAULT_MASS_COLUMN = "jlens_mass_L15"

# id -> action, the fixed LEFT,UP,RIGHT,DOWN order eval_probe_per_token.py writes its
# --full-probs columns in and plot_commitment_probs.py already uses.
ID2A = {0: "LEFT", 1: "UP", 2: "RIGHT", 3: "DOWN"}
ACTIONS = ("LEFT", "UP", "RIGHT", "DOWN")


# entry-46 probe key -> the key eval_probe_per_token.py's probe_key() gives that same .pt
# ("<parent dir>.<stem>", with next_action_probe_ stripped).
def parse_probes(spec: str) -> dict[str, str]:
    """``key=column_prefix`` pairs -> ``{key: column_prefix}``, in the order given.

    The probes are an ARGUMENT, not a registry. This script used to carry ten entry-48
    probe keys as a module constant and exit on any table missing their columns, which
    made it unusable for any other round; ``--extra-probes`` could only add to that list,
    never replace it.

    ``key`` is the short name the output columns use; ``column_prefix`` is how the probe
    appears in the scorer's table, i.e. the part before ``_pred``.

    >>> parse_probes("p2_lr=probes.local_belief_p2_lr")
    {'p2_lr': 'probes.local_belief_p2_lr'}
    """
    out: dict[str, str] = {}
    for pair in (t.strip() for t in spec.split(",") if t.strip()):
        key, sep, src = pair.partition("=")
        if not sep or not key or not src:
            raise SystemExit(f"--probes entry {pair!r} is not key=column_prefix")
        if key in out:
            raise SystemExit(f"--probes key {key!r} is repeated; keys must be unique")
        out[key] = src
    if not out:
        raise SystemExit("--probes is empty")
    return out


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
    "convinced_token_pos",
    "convinced_token_reasoning_pos",
    "convinced_token_frac",
    "convinced_before_reasoning",
    "convinced_token_sentence_idx",
    "rel_token",
    "rel_sentence_token",
    "is_convinced",
    "n_switches",
    "label_local",
    "label_final",
    "ground_truth",
    "rollout_answer_prob",
    "rollout_correct",
]


def read_commitment(path: Path, mass_column: str = DEFAULT_MASS_COLUMN) -> dict[str, dict[int, dict[int, dict]]]:
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
                "dir_logmass": float(r[mass_column]),
                "n_sentences": r["n_sentences"],
                "sentence_idx": r["sentence_idx"],
                "pos_in_sentence": r["pos_in_sentence"],
                "sentence_len": r["sentence_len"],
                # NOT that file's `sentence_frac`, which is sentence_idx/n_sentences.
                "sentence_frac": r["frac_in_sentence"],
                "is_sentence_end": r["is_sentence_end"],
                # Optional: a row source built from a strided arm carries no sentence-end
                # boundary, and `--commitment off` is the run that does not want one.
                "convinced_idx": r.get("convinced_sentence_idx", ""),
            }
    return out


def read_probe_csv(
    path: Path, probe_source: dict[str, str], mass_column: str = DEFAULT_MASS_COLUMN
) -> tuple[dict[str, dict[int, dict[int, dict]]], list[str]]:
    """``{name: {step: {token_idx: {probe_key: (pred, {action: p})}}}}`` plus the mass column."""
    out: dict[str, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        have = set(reader.fieldnames or [])
        missing = [
            f"{src}_{suffix}"
            for src in probe_source.values()
            for suffix in ["pred", *[f"p_{a}" for a in ACTIONS]]
            if f"{src}_{suffix}" not in have
        ]
        if missing:
            raise SystemExit(
                f"{path} is missing {len(missing)} column(s), e.g. {missing[:3]}. "
                "Was score_probes_per_token.py run with every --probe and --full-probs?"
            )
        for r in reader:
            cell = {
                key: (
                    ID2A[int(r[f"{src}_pred"])],
                    {a: float(r[f"{src}_p_{a}"]) for a in ACTIONS},
                )
                for key, src in probe_source.items()
            }
            cell["_mass"] = float(r[mass_column])
            cell["_label_final"] = r["label_name"]
            out[r["name"]][int(r["step"])][int(r["token_idx"])] = cell
    return out, sorted(have)


KIND_NO_REASONING = "no_reasoning"
"""``cutoff_kind`` of the pre-reasoning cutoff, spelled here rather than imported: this
script is a stdlib-only join and ``truncation_strategies`` pulls in the model stack."""

NO_REASONING_POS = -1
"""``convinced_token_pos`` for a step already committed before it wrote a reasoning token.

A real boundary is an index into ``step["output_tokens"]`` and is therefore >= 0, so the
sentinel cannot collide with one. It is the per-token twin of ``convinced_sentence_idx == 0``
and marks the same cohort entries 41/42 drop.
"""


def convinced_index(corrects: list[bool]) -> int | None:
    """Smallest index from which every later eval is correct; None if the last one is wrong.

    Identical rule to ``run_inference.py::commitment_metrics``, restated here so the
    per-token boundary is computed from the dense eval list rather than inherited from a
    sentence-end grid.

    >>> convinced_index([False, True, False, True, True])
    3
    >>> convinced_index([True, True])
    0
    >>> convinced_index([True, False]) is None
    True
    """
    if not corrects or not corrects[-1]:
        return None
    for i in range(len(corrects) - 1, -1, -1):
        if not corrects[i]:
            return i + 1
    return 0


def convinced_token(evals: list[dict]) -> int | None:
    """The per-token commitment boundary, as a position in ``step["output_tokens"]``.

    ``evals`` is one step's whole cutoff list from the ``every_token`` arm: the
    no-reasoning cutoff plus one eval per reasoning token. The boundary is the first
    cutoff from which the truncated model answers correctly and never stops -- so
    ``NO_REASONING_POS`` when that is the no-reasoning cutoff, and ``None`` when the
    full chain itself answers wrong (no boundary exists).

    THIS IS THE DEFINITION CHANGE. The old boundary ran the same rule over sentence ENDS
    only, which is wrong in both directions: it rounds a mid-sentence commitment up to the
    next sentence end, and it misses a relapse that happens between two sentence ends.
    """
    ordered = sorted(
        (e for e in evals if e["cutoff_kind"] != KIND_NO_REASONING),
        key=lambda e: e["eos_token_pos"],
    )
    no_reasoning = [e for e in evals if e["cutoff_kind"] == KIND_NO_REASONING]
    positions = [NO_REASONING_POS] * len(no_reasoning) + [e["eos_token_pos"] for e in ordered]
    corrects = [bool(e["correct"]) for e in no_reasoning] + [bool(e["correct"]) for e in ordered]
    idx = convinced_index(corrects)
    return None if idx is None else positions[idx]


def ct_sentence(ct_pos: int, toks: dict[int, dict]) -> int | None:
    """Sentence index of the per-token boundary, or -1 for the no-reasoning sentinel.

    ``toks`` is the step's ``{token_id: coords}`` map, so the lookup is the boundary
    token's own ``sentence_idx``. The sentinel sits before sentence 0 and gets -1, which
    keeps ``rel_sentence_token`` a signed sentence offset for every row.

    ``None`` when the boundary token is absent from the row source, or is present but
    carries no sentence placement. Both are real rather than defensive: the cut at the
    first reasoning token has no sentence before it to be placed in, and it is the computed
    boundary on a large minority of steps -- 45 of the 70 Qwen held-out steps -- so raising
    here would fail an entire join over a boundary column the run may not even want.
    """
    if ct_pos == NO_REASONING_POS:
        return -1
    coords = toks.get(ct_pos)
    if coords is None or coords["sentence_idx"] == "":
        return None
    return int(coords["sentence_idx"])


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
                # The commitment boundary, recomputed from THIS dense list rather than
                # taken from the commitment CSV's sentence-end grid. See convinced_token().
                "convinced_token_pos": convinced_token(evals),
            }
        out[path.stem] = per_step
    return out


def ranks(values: dict[int, float]) -> dict[int, int]:
    """token_id -> rank of its mass, 1 = loudest. Ties break toward the earlier token."""
    order = sorted(values, key=lambda t: (-values[t], t))
    return {t: i + 1 for i, t in enumerate(order)}


def _resolve_loudness_column(path: Path, args) -> str:
    """The loudness column to read, checked against THAT table's OWN header.

    Resolved rather than assumed, so a table from either evaluator generation joins without a
    flag. An explicit --mass-column wins.

    Each input is resolved separately, because the two carry the same quantity under
    different legal spellings: ``score_probes_per_token.py`` writes the canonical
    ``{lens}_{signal}_logmass_L{layer}`` while ``join_rollout_answers.py`` writes
    ``{lens}_logmass_L{layer}``, and ``columns.py`` exists to accept both on read. Sharing
    one resolved name between them is what made a perfectly good row source look like it had
    no loudness at all.
    """
    if args.mass_column is not None:
        return args.mass_column
    with open(path, newline="", encoding="utf-8") as fh:
        fields = next(csv.reader(fh))
    try:
        return cols.resolve(fields, args.lens, args.signal_name, args.layer)
    except KeyError as exc:
        raise SystemExit(
            f"{path} carries no {args.lens}/{args.signal_name} loudness at layer "
            f"{args.layer}.\n{exc}\nPass --mass-column explicitly if the table uses a "
            "spelling this does not know."
        ) from None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", "--probe-csv", dest="probe_csv", type=Path, required=True)
    ap.add_argument("--rollout-dir", type=Path, required=True)
    ap.add_argument(
        "--commitment-csv",
        type=Path,
        required=True,
        help="Per-token commitment CSV. It is the ROW SOURCE: this join iterates its "
        "(name, step, token) keys and takes the sentence coordinates from it, so a token "
        "absent here is absent from the output whatever the probe table holds.",
    )
    ap.add_argument(
        "--signal-json",
        type=Path,
        required=True,
        help="the vocabulary the mass table was built against; flags whether the TOKEN "
        "at each cutoff is one of its words.",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--probes",
        required=True,
        help="Comma-separated key=column_prefix pairs naming the probes to join, e.g. "
        "'jlens_lr=qwen_p2_local_belief_jlens_l27_lr,random_lr=...'. key is the short name "
        "the output columns use; column_prefix is how the probe appears in --table, i.e. "
        "the part before _pred. Every named probe must have _pred and its four _p_{action} "
        "columns in the table, which means --full-probs on the scorer.",
    )
    ap.add_argument(
        "--rowset",
        required=True,
        help="Value written to the output's `rowset` column, which names WHICH PROBES a row "
        "carries. One rowset per run: the probes are given explicitly now, so a second one "
        "would only duplicate identical rows under another name.",
    )
    ap.add_argument("--mass-tol", type=float, default=1e-6, help="max |probe CSV mass - commitment CSV mass|.")
    ap.add_argument(
        "--commitment",
        choices=("auto", "on", "off"),
        default="auto",
        help="Whether to emit the commitment-boundary columns (convinced_*, rel_sentence, "
        "rel_token, is_convinced, x_sentence). 'auto' is the historical behaviour -- emit "
        "whatever the row source supports -- so nothing already on disk changes meaning. "
        "'off' blanks them all, for a row source built from a STRIDED arm, where the "
        "boundary is only resolved to +-stride and a relapse between two sampled cutoffs "
        "is invisible. 'on' requires the row source to carry convinced_sentence_idx and "
        "fails if it does not, rather than writing a silently empty column. Says it "
        "outright rather than inferring it from an absent column: inference from absence "
        "is what the --thin-mode round already cost (CLAUDE.md).",
    )
    ap.add_argument(
        "--lens",
        required=True,
        help="Which lens's loudness becomes the axis every downstream figure bins on. Say "
        "WHICH LENS in any caption built from the output -- the two lenses' top-20 sets "
        "overlap only about half, so an unqualified 'loudness' is not a quantity. Required: "
        "the scorer writes both lenses' columns and only this picks between them.",
    )
    ap.add_argument(
        "--signal-name",
        required=True,
        help="Which vocabulary the loudness was taken over. With --lens and --layer this "
        "builds the column read from the input table.",
    )
    ap.add_argument("--layer", type=int, required=True, help="Layer the loudness is read at.")
    ap.add_argument(
        "--mass-column",
        default=None,
        help="Read this column as the loudness axis instead of the one --lens/--signal-name/"
        "--layer imply. Needed only for a table whose columns follow none of the known "
        "spellings; the legacy ones (dir_logmass, dir_logmass_L15, {lens}_mass_L15) are "
        "resolved automatically.",
    )
    ap.add_argument("--limit", type=int, default=None, help="first N trajectories (smoke test).")
    args = ap.parse_args()

    # Both resolved before either is assigned back: _resolve_loudness_column treats a set
    # args.mass_column as the explicit override and returns it unread.
    probe_mass_column = _resolve_loudness_column(args.probe_csv, args)
    commitment_mass_column = _resolve_loudness_column(args.commitment_csv, args)
    args.mass_column = probe_mass_column

    print(
        f"loudness axis: {args.mass_column}  ({cols.axis_label(args.lens, args.signal_name, args.layer)})", flush=True
    )
    if commitment_mass_column != args.mass_column:
        print(f"  row source spells it {commitment_mass_column}; --mass-tol checks they agree", flush=True)
    probe_source = parse_probes(args.probes)
    print(f"{len(probe_source)} probe(s): {', '.join(probe_source)}", flush=True)
    vocab = {t for lst in json.loads(args.signal_json.read_text()).values() for t in lst}

    print(f"reading {args.commitment_csv}", flush=True)
    coords = read_commitment(args.commitment_csv, commitment_mass_column)
    has_boundary = any(
        c["convinced_idx"] not in ("", None)
        for steps in coords.values()
        for toks in steps.values()
        for c in toks.values()
    )
    if args.commitment == "on" and not has_boundary:
        raise SystemExit(
            f"--commitment on, but no row of {args.commitment_csv} carries a "
            "convinced_sentence_idx. A row source built from a strided arm has no "
            "sentence-end boundary; re-run with --commitment off, which blanks the "
            "boundary columns instead of writing an empty one that looks computed."
        )
    print(f"commitment columns: {args.commitment} (row source has a boundary: {has_boundary})", flush=True)
    print(f"reading {args.probe_csv}", flush=True)
    probes, _ = read_probe_csv(args.probe_csv, probe_source, args.mass_column)
    print(f"reading {args.rollout_dir}", flush=True)
    rollouts = read_rollouts(args.rollout_dir)
    print(
        f"  {len(coords)} names in coords, {len(probes)} in probe CSV, {len(rollouts)} rollout file(s)",
        flush=True,
    )

    fields = list(BASE_FIELDS)
    for p in probe_source:
        fields += [f"{p}_pred", f"{p}_p_local", f"{p}_p_final", f"{p}_pmax"]

    names = sorted(coords)
    if args.limit:
        names = names[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
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
                first_tok = min(toks)
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
                    # No placement on the sentence grid means no within-sentence position,
                    # which is the control every loudness figure pairs against loudness. The
                    # cut at the first reasoning token is the case that reaches here.
                    if c["sentence_idx"] == "" or c["sentence_frac"] == "":
                        skipped["no sentence placement"] += 1
                        continue
                    if abs(pcell["_mass"] - c["dir_logmass"]) > args.mass_tol:
                        mass_mismatch += 1
                    if pcell["_label_final"] != c["label_final"]:
                        label_mismatch += 1

                    lm = c["dir_logmass"]
                    rp = c["reasoning_pos"]
                    # Per-token boundary, in the three coordinates the figures bin on.
                    # reasoning_pos of the sentinel is -1, one step before the first
                    # reasoning token, so `rel_token` stays a signed token offset.
                    ct_pos = None if args.commitment == "off" else rstep["convinced_token_pos"]
                    ct_rp = None if ct_pos is None else (-1 if ct_pos == NO_REASONING_POS else ct_pos - first_tok)
                    ct_frac = None if ct_rp is None else (ct_rp / (n_tok - 1) if n_tok > 1 else 0.0)
                    # Which SENTENCE the per-token boundary falls in, so the entry-41 axis
                    # (+-6 sentences) can be redrawn around the corrected boundary instead
                    # of around the sentence end that happened to follow it.
                    ct_si = None if ct_pos is None else ct_sentence(ct_pos, toks)
                    frac = float(c["sentence_frac"])
                    conv = "" if args.commitment == "off" else c["convinced_idx"]
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
                        "convinced_token_pos": "" if ct_pos is None else ct_pos,
                        "convinced_token_reasoning_pos": "" if ct_rp is None else ct_rp,
                        "convinced_token_frac": "" if ct_frac is None else f"{ct_frac:.6f}",
                        "convinced_before_reasoning": "" if ct_pos is None else int(ct_pos == NO_REASONING_POS),
                        "convinced_token_sentence_idx": "" if ct_si is None else ct_si,
                        "rel_token": "" if ct_rp is None else rp - ct_rp,
                        "rel_sentence_token": "" if ct_si is None else si - ct_si,
                        "is_convinced": "" if ct_pos is None else int(tok_id >= ct_pos),
                        "n_switches": rstep["n_switches"],
                        "label_local": label_local,
                        "label_final": label_final,
                        "ground_truth": rstep["gt"],
                        "rollout_answer_prob": ev["answer_prob"],
                        "rollout_correct": int(bool(ev["correct"])),
                    }
                    for pname in probe_source:
                        pred, probs4 = pcell[pname]
                        row[f"{pname}_pred"] = pred
                        row[f"{pname}_p_local"] = f"{probs4[label_local]:.6f}" if label_local else ""
                        row[f"{pname}_p_final"] = f"{probs4[label_final]:.6f}"
                        row[f"{pname}_pmax"] = f"{max(probs4.values()):.6f}"
                    # One identical row per rowset: the rowset now names WHICH PROBES are
                    # read, not which tokens, so every arm is scored on the same tokens.
                    w.writerow({**row, "rowset": args.rowset})
                    n_rows += 1
            if ni % 100 == 0 or ni == len(names):
                print(f"    {ni}/{len(names)} trajectories, {n_rows} rows", flush=True)

    print(f"\nwrote {n_rows} rows -> {args.out}", flush=True)
    print(f"  rowset {args.rowset}: {n_rows} rows", flush=True)
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
