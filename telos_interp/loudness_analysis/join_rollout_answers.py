#!/usr/bin/env python3
"""Per-token LOUDNESS under both lenses, against the action the model gives if cut there.

One row per reasoning token of a trajectory set, carrying the two measurements that have
never sat in the same table:

  * **loudness** -- the layer-15 full-vocabulary direction mass, ``log P(any direction
    word)`` over the 446-token ``direction_tokens_full.json``, read from the direction-mass
    table beside each analysis CSV. Computed under BOTH lenses, so a row has
    ``jlens_logmass_L15`` and ``logitlens_logmass_L15`` side by side.
  * **the inferred action probability** -- truncate the model's own reasoning at exactly
    that token, append the fixed final-channel prefix (``<|end|>...{\\n  "action": "``), and
    read the action it then emits plus the probability it puts on that answer token. That is
    ``run_inference.py``'s ``every_token`` truncation arm, already on disk.

This is a PURE JOIN of artifacts -- no torch, no model, no GPU -- so it is cheap to re-run
when any of them is rebuilt. The three inputs and the key that ties them together:

  ``--rollout-dir``    the ``every_token`` arm. Each eval's ``eos_token_pos`` is an index
                       into ``step["output_tokens"]``.
  ``--lens-root``      ``{root}/size{N}/{name}/{name}_{lens}_direction_mass.csv``. Its
                       ``reasoning_pos`` indexes the ANALYSIS-TAGGED subset of those same
                       ``output_tokens``, so the two coordinates differ by
                       ``min(analysis_positions(...))`` -- never a hardcoded 3.
  ``--trajectories``   the trajectory JSONs, which supply that offset, the token text the
                       mapping is CHECKED against, and the ground-truth action.

Every mass table is read through ``MassTableLoudness``, which refuses one that has no
``.meta.json`` sidecar: this repo points two vocabularies at the same trees, so a table
whose vocabulary is unknown must not be read at all (CLAUDE.md).

Defaults are the held-out 360 -- 360 trajectories, 10 per size x complexity cell, drawn from
the pool the 3600-trajectory gather never touched, so overlap with every probe's training set
is zero -- which makes the no-argument invocation the intended run.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from telos_interp.jlens_utils import read_mass_meta
from telos_interp.loudness_analysis.rollouts.truncation_strategies import (
    KIND_NO_REASONING,
    LoudnessUnavailable,
    MassTableLoudness,
    analysis_positions,
)

# csv.DictReader everywhere, never pandas: decoded tokens include "NA", empty strings,
# embedded commas and newlines, which pandas' NA handling silently corrupts.
csv.field_size_limit(10**9)

DEFAULT_ROLLOUT_DIR = Path("/workspace/reasoning_theatre/rollout_strategies_heldout360/every_token")
DEFAULT_TRAJECTORIES = Path("/workspace/trajectories/heldout360")
DEFAULT_LENS_ROOT = Path("/workspace/activations/heldout360_lens")
DEFAULT_NAMES_FILE = Path("/workspace/trajectories/heldout360_names.txt")
DEFAULT_OUT = Path("/workspace/reasoning_theatre/loudness_vs_answer_prob/heldout360_per_token.csv")

# The strategy this join is written against. Any other arm cuts at a SUBSET of the reasoning
# tokens, so rows would be missing without anything looking wrong.
EXPECTED_STRATEGY = "every_token"

# Sidecar fields that must agree across the two lenses: they name the vocabulary the mass was
# computed over. Two lenses baked against different vocabularies are not comparable.
VOCAB_FIELDS = ("signal_json", "direction_classes", "num_direction_tokens")

# Identity and action columns; the per-lens loudness columns are inserted between them.
KEY_FIELDS = (
    "name",
    "size",
    "complexity",
    "run",
    "step",
    "step_id",
    "reasoning_pos",
    "token_idx",
    "abs_pos",
    "token",
)
ACTION_FIELDS = (
    "model_action",
    "answer_token",
    "answer_prob",
    "correct",
    "ground_truth",
    "cutoff_kind",
    "n_prompt_tokens",
)

REBUILD_HINT = """no rollout for {name}. The every_token arm is built with:

  NAMES_FILE={names} \\
  TRAJECTORIES={trajectories} \\
  LENS_ROOT={lens_root} \\
  OUT_ROOT={out_root} \\
    bash telos_interp/loudness_analysis/rollouts/run_inference_strategies.sh every_token"""


def lens_columns(lens: str, layer: int) -> tuple[str, str]:
    """``(logmass column, probability column)`` for one lens at one layer.

    The layer travels in the name, as in ``eval_probe_per_token.py``, so a table built at a
    different layer can never be mistaken for this one.

    >>> lens_columns("jlens", 15)
    ('jlens_logmass_L15', 'jlens_prob_L15')
    """
    return f"{lens}_logmass_L{layer}", f"{lens}_prob_L{layer}"


def fieldnames(lenses: list[str], layer: int) -> list[str]:
    """The CSV header: identity, then both lenses' loudness, then the truncated answer."""
    cols = list(KEY_FIELDS)
    for lens in lenses:
        cols.extend(lens_columns(lens, layer))
    return cols + list(ACTION_FIELDS)


def read_row_meta(path: Path) -> dict[int, dict[int, dict]]:
    """``{step: {reasoning_pos: {abs_pos, token, size, complexity, run}}}`` from a mass table.

    ``MassTableLoudness`` hands back only the layer's value, but the table's own ``token``
    column is what makes the ``reasoning_pos -> token_idx`` mapping checkable, and its
    ``abs_pos`` is the forward-pass coordinate the activation tree is named by. Both are read
    here rather than recomputed, so a row is described by the artifact that produced it.
    """
    out: dict[int, dict[int, dict]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(int(row["step"]), {})[int(row["reasoning_pos"])] = {
                "abs_pos": row["abs_pos"],
                "token": row["token"],
                "size": row["size"],
                "complexity": row["complexity"],
                "run": row["run"],
            }
    return out


class LensTable:
    """One lens's direction-mass table, with the row metadata ``MassTableLoudness`` drops.

    Delegates path resolution, the ``.meta.json`` sidecar rule and the layer-column check to
    ``MassTableLoudness`` -- the audited reader every other consumer of these tables uses --
    and adds a second pass over the same file for ``token`` / ``abs_pos``. Both are cached
    per trajectory, since the caller walks one file at a time.
    """

    def __init__(self, lens_root: Path, lens: str, layer: int) -> None:
        self.lens = lens
        self.loudness = MassTableLoudness(lens_root=lens_root, lens=lens, layer=layer)
        self._name: str | None = None
        self._meta_rows: dict[int, dict[int, dict]] = {}

    def sidecar(self, name: str) -> dict:
        """The table's ``.meta.json``, read before its values.

        A mass table is never read without its sidecar (CLAUDE.md): this repo points two
        vocabularies at the same trees, so a table whose vocabulary is unknown is unusable.
        Reading it first is what lets a layer the table does not cover fail as the
        configuration error it is, rather than as a trajectory whose artifact is missing.
        """
        path = self.loudness.table_path(name)
        meta = read_mass_meta(path)
        if not meta:
            raise LoudnessUnavailable(f"{path} has no .meta.json sidecar, so its vocabulary is unknown")
        return meta

    def load(self, name: str) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, dict]]]:
        """``({step: {reasoning_pos: logmass}}, {step: {reasoning_pos: row meta}})``."""
        scores = self.loudness.load(name)  # raises LoudnessUnavailable: no table, no sidecar, no column
        if name != self._name:
            self._meta_rows = read_row_meta(self.loudness.table_path(name))
            self._name = name
        return scores, self._meta_rows


def check_vocabularies(metas: dict[str, dict], layer: int, name: str) -> None:
    """Every lens must have scored the same vocabulary, and must cover ``layer``.

    Comparing a jlens mass to a logitlens mass is only meaningful if both are the total
    probability of the SAME set of direction words. The sidecars say which set that was.
    Both failures are configuration errors, not missing data, so both stop the run.
    """
    reference: tuple | None = None
    for lens, meta in metas.items():
        layers = meta.get("layers") or []
        if layer not in layers:
            raise SystemExit(f"{name}: {lens} mass table does not cover layer {layer} (has {layers})")
        vocab = tuple(meta.get(field) for field in VOCAB_FIELDS)
        if reference is None:
            reference = vocab
        elif vocab != reference:
            raise SystemExit(
                f"{name}: lenses were scored against different vocabularies -- "
                f"{dict(zip(VOCAB_FIELDS, reference, strict=True))} vs {dict(zip(VOCAB_FIELDS, vocab, strict=True))}"
            )


def check_strategy(config: dict, name: str) -> str | None:
    """Whether the rollout is the dense arm this join assumes; a message if it is not."""
    strategy = config.get("strategy")
    stride = config.get("stride", 1)
    if strategy != EXPECTED_STRATEGY:
        return f"{name}: rollout strategy is {strategy!r}, not {EXPECTED_STRATEGY!r} -- rows will be missing"
    if stride != 1:
        return f"{name}: rollout stride is {stride}, so only every {stride}th token was cut at"
    return None


def rollout_lens_column(config: dict, lenses: list[str], layer: int) -> str | None:
    """The joined column the rollout's own ``dir_logmass`` should reproduce, if any.

    The rollout recorded loudness under one lens/layer while it chose its cutoffs. Where that
    matches a lens we are joining, it is a free end-to-end check that the two artifacts are
    still indexed the same way -- the one thing this script cannot get wrong quietly.
    """
    lens = config.get("lens")
    if lens in lenses and config.get("layer") == layer:
        return lens_columns(lens, layer)[0]
    return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR, help="the every_token arm's results.")
    ap.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES, help="trajectory JSON root.")
    ap.add_argument("--lens-root", type=Path, default=DEFAULT_LENS_ROOT, help="root of the direction-mass tables.")
    ap.add_argument(
        "--names-file",
        type=Path,
        default=DEFAULT_NAMES_FILE,
        help="trajectory stems to join, one per line. Empty string takes every rollout in --rollout-dir.",
    )
    ap.add_argument("--lenses", default="jlens,logitlens", help="comma-separated; one loudness column pair each.")
    ap.add_argument("--layer", type=int, default=15, help="the layer whose mass is the loudness.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--include-no-reasoning",
        action="store_true",
        help="also emit the no_reasoning cutoff, which is not a token and carries no loudness.",
    )
    ap.add_argument("--mass-tol", type=float, default=1e-5, help="max |joined logmass - rollout's own|.")
    return ap


def resolve_names(args: argparse.Namespace) -> list[str]:
    """Trajectory stems to join, from the names file or from what the rollout directory holds."""
    if args.names_file and str(args.names_file):
        return sorted(set(args.names_file.read_text().split()))
    return sorted(p.stem for p in args.rollout_dir.glob("*.json"))


def trajectory_path(root: Path, name: str) -> Path | None:
    """``{root}/size{N}/{name}.json``, falling back to a search when the layout differs."""
    if "_size" in name:
        direct = root / f"size{name.split('_size')[1].split('_')[0]}" / f"{name}.json"
        if direct.exists():
            return direct
    matches = sorted(root.rglob(f"{name}.json"))
    return matches[0] if matches else None


class Tally:
    """Counters the join keeps, so a run reports what it dropped instead of losing it.

    ``token_mismatch`` and ``mass_mismatch`` are fatal at the end rather than at the first
    occurrence: knowing whether one row or every row is wrong is the difference between a
    corrupt trajectory and a broken coordinate mapping.
    """

    def __init__(self) -> None:
        self.skipped: dict[str, int] = {}
        self.token_mismatch = 0
        self.mass_mismatch = 0

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def report(self) -> None:
        for reason, count in sorted(self.skipped.items()):
            print(f"  skipped: {reason}: {count}", flush=True)


def add_loudness(row: dict, loaded: dict, lenses: list[str], layer: int, step_id: int, reasoning_pos: int) -> bool:
    """Fill in both lenses' columns for one token; ``False`` if a table disagrees on the token.

    The token check is the only thing standing between this join and a silent one-token
    shift, since ``reasoning_pos`` counts analysis-tagged tokens and ``token_idx`` counts all
    of them. A lens with no row for this token leaves its columns empty rather than failing:
    a partial table costs the covariate, not the run.
    """
    consistent = True
    for lens in lenses:
        logmass_col, prob_col = lens_columns(lens, layer)
        row[logmass_col] = ""
        row[prob_col] = ""
        if reasoning_pos < 0:
            continue
        scores, meta_rows = loaded[lens]
        value = scores.get(step_id, {}).get(reasoning_pos)
        if value is None:
            continue
        cell = meta_rows.get(step_id, {}).get(reasoning_pos, {})
        if cell.get("token", row["token"]) != row["token"]:
            consistent = False
        row["abs_pos"] = cell.get("abs_pos", row["abs_pos"])
        for key in ("size", "complexity", "run"):
            if not row.get(key):
                row[key] = cell.get(key, "")
        row[logmass_col] = f"{value:.6f}"
        row[prob_col] = f"{math.exp(value):.9g}"
    return consistent


def step_rows(
    *,
    name: str,
    step_index: int,
    step: dict,
    evals: list[dict],
    loaded: dict,
    lenses: list[str],
    layer: int,
    include_no_reasoning: bool,
    check_column: str | None,
    mass_tol: float,
    tally: Tally,
) -> list[dict]:
    """One row per reasoning token of one step, joined against that token's truncated answer."""
    step_id = step["step_id"]
    by_pos = {e["eos_token_pos"]: e for e in evals}
    output_tokens = step["output_tokens"]
    ana = analysis_positions(output_tokens)
    if not ana:
        tally.skip("no analysis tokens")
        return []
    offset = min(ana)

    positions = list(enumerate(ana))
    if include_no_reasoning:
        # reasoning_pos -1 keeps the sentinel one step before the first reasoning token, so
        # the column stays a signed offset from the start of reasoning.
        positions.insert(0, (-1, offset - 1))

    rows: list[dict] = []
    for reasoning_pos, token_idx in positions:
        ev = by_pos.get(token_idx)
        if ev is None:
            tally.skip("no rollout eval at token")
            continue
        is_sentinel = reasoning_pos < 0
        if is_sentinel and ev.get("cutoff_kind") != KIND_NO_REASONING:
            tally.skip("no_reasoning cutoff is not where expected")
            continue

        row = {
            "name": name,
            "size": "",
            "complexity": "",
            "run": "",
            "step": step_index,
            "step_id": step_id,
            "reasoning_pos": reasoning_pos,
            "token_idx": token_idx,
            "abs_pos": "",
            "token": "" if is_sentinel else output_tokens[token_idx].get("token", ""),
            "model_action": ev.get("model_action") or "",
            "answer_token": ev.get("answer_token", ""),
            "answer_prob": ev.get("answer_prob", ""),
            "correct": int(bool(ev.get("correct"))),
            "ground_truth": step["agent_action"],
            "cutoff_kind": ev.get("cutoff_kind", ""),
            "n_prompt_tokens": ev.get("n_prompt_tokens", ""),
        }
        if not add_loudness(row, loaded, lenses, layer, step_id, reasoning_pos):
            tally.token_mismatch += 1
            continue
        if check_column and not is_sentinel:
            recorded = ev.get("dir_logmass")
            joined = row[check_column]
            if recorded is not None and joined != "" and abs(float(joined) - recorded) > mass_tol:
                tally.mass_mismatch += 1
        rows.append(row)
    return rows


def load_pair(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    args = build_parser().parse_args()
    lenses = [lens.strip() for lens in args.lenses.split(",") if lens.strip()]
    if not lenses:
        raise SystemExit("--lenses named no lens")

    names = resolve_names(args)
    tables = {lens: LensTable(args.lens_root, lens, args.layer) for lens in lenses}
    header = fieldnames(lenses, args.layer)
    print(f"{len(names)} trajectory name(s); lenses {lenses} at layer {args.layer}", flush=True)
    print(f"rollouts {args.rollout_dir}\nmass     {args.lens_root}\ntrajectories {args.trajectories}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tally = Tally()
    warned: set[str] = set()
    n_rows = 0
    n_traj = 0
    check_column: str | None = None
    vocab_reported = False

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for i, name in enumerate(names):
            roll_path = args.rollout_dir / f"{name}.json"
            if not roll_path.exists():
                raise SystemExit(
                    REBUILD_HINT.format(
                        name=name,
                        names=args.names_file,
                        trajectories=args.trajectories,
                        lens_root=args.lens_root,
                        out_root=args.rollout_dir.parent,
                    )
                )
            traj_path = trajectory_path(args.trajectories, name)
            if traj_path is None:
                tally.skip("no trajectory JSON")
                continue
            rollout = load_pair(roll_path)
            trajectory = load_pair(traj_path)

            config = rollout.get("strategy", {})
            message = check_strategy(config, name)
            if message and message not in warned:
                print(f"  WARNING: {message}", flush=True)
                warned.add(message)
            check_column = rollout_lens_column(config, lenses, args.layer)

            try:
                metas = {lens: table.sidecar(name) for lens, table in tables.items()}
            except LoudnessUnavailable:
                tally.skip("no direction-mass table")
                continue
            check_vocabularies(metas, args.layer, name)
            # Past the checks a load failure is a surprise, not missing data: let it raise.
            loaded = {lens: table.load(name) for lens, table in tables.items()}
            if not vocab_reported:
                meta = metas[lenses[0]]
                print(
                    f"vocabulary: {meta.get('signal_json')} "
                    f"({meta.get('num_direction_tokens')} tokens, classes={meta.get('direction_classes')})",
                    flush=True,
                )
                vocab_reported = True

            evals_by_step = {step["step_id"]: step["sentence_evals"] for step in rollout["steps"]}
            wrote = False
            for step_index, step in enumerate(trajectory["steps"]):
                evals = evals_by_step.get(step["step_id"])
                if evals is None:
                    tally.skip("no rollout step")
                    continue
                rows = step_rows(
                    name=name,
                    step_index=step_index,
                    step=step,
                    evals=evals,
                    loaded=loaded,
                    lenses=lenses,
                    layer=args.layer,
                    include_no_reasoning=args.include_no_reasoning,
                    check_column=check_column,
                    mass_tol=args.mass_tol,
                    tally=tally,
                )
                writer.writerows(rows)
                n_rows += len(rows)
                wrote = wrote or bool(rows)
            n_traj += wrote
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(names)} trajectories, {n_rows} rows", flush=True)

    print(f"wrote {n_rows} rows over {n_traj} trajectories -> {args.out}", flush=True)
    tally.report()
    if check_column:
        print(
            f"  cross-check against the rollout's own {check_column}: {tally.mass_mismatch} mismatch(es)", flush=True
        )
    else:
        print("  cross-check against the rollout's own loudness: not applicable (lens/layer differ)", flush=True)
    if tally.token_mismatch:
        raise SystemExit(
            f"{tally.token_mismatch} row(s) whose mass-table token disagreed with the trajectory: the "
            "reasoning_pos -> token_idx mapping is off. Rows were dropped; the CSV is not usable."
        )
    if tally.mass_mismatch:
        raise SystemExit(
            f"{tally.mass_mismatch} row(s) where the joined loudness disagreed with the value the rollout "
            f"recorded at the same cutoff (tol {args.mass_tol}). The two artifacts are indexed differently."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
