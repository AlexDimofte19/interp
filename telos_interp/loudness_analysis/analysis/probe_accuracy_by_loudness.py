#!/usr/bin/env python3
"""Numbers behind the probe-loudness figures: does a LOUDER token decode better, and is it
loudness doing the work or the token's position in its sentence?

The probe-side twin of ``analyze_sentence_loudness.py``, and the local-belief re-run of
ICLR log entry 37's finding 1 ("the mass score predicts decodability, monotonically") --
which was measured with a FINAL-action label, on all 87k held-out tokens rather than on the
tokens a probe was actually trained for.

ACCURACY is BALANCED accuracy over the four actions, as everywhere else in this file: the
mean of the four per-class recalls, over the classes present in the bin. A plain accuracy
would move with the bin's class mix, and the loud bins are direction-word-heavy.

Bootstraps resample TRAJECTORY NAMES, not rows -- tokens inside a trajectory share a
sentence structure, a commitment boundary and a label.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from telos_interp.loudness_analysis import columns as _cols
from telos_interp.loudness_analysis import probes as _probes
from telos_interp.loudness_analysis import provenance as _prov
from telos_interp.loudness_analysis import stats as _stats

ACTIONS = ["LEFT", "UP", "RIGHT", "DOWN"]
PROBES = {
    "p1_full": ["p1_lr", "p1_mlp"],
    "p1_top20": ["p1t20_lr", "p1t20_mlp"],
    "p2": ["p2_lr", "p2_mlp", "base_lr", "base_mlp", "rand_lr", "rand_mlp"],
}
# which label a probe was TRAINED against -- base_/rand_ are the entry-38 final-action probes
FINAL_LABEL_PROBES = {"base_lr", "base_mlp", "rand_lr", "rand_mlp"}


# CLASSES is set from --probe-type in main(); the statistics themselves live in stats.py,
# which holds BOTH balanced accuracies (row-averaged and count-pooled) because they are not
# the same number and both are in use.
# A list, mutated in place from --probe-type rather than rebound, so the module-level
# helpers below close over the same object without a `global`.
CLASSES: list = list(ACTIONS)


def bal_acc(truth: np.ndarray, pred: np.ndarray) -> float:
    """Mean per-class recall over the classes present. nan on an empty bin."""
    return _stats.bal_acc(truth, pred, CLASSES)


def boot_bal_acc(df: pd.DataFrame, truth_col: str, pred_col: str, n: int, rng) -> tuple[float, float, float]:
    """Balanced accuracy with a 95% CI from resampling trajectory names."""
    return _stats.boot_bal_acc(df, truth_col, pred_col, n, rng, CLASSES)


# Bins for the signed TOKEN distance to the per-token commitment boundary. Token distances
# run to +-2000 and are heavily skewed, so these are log-spaced rather than a clip: bin 0 is
# the boundary token itself, negatives are before it, positives after.
REL_TOKEN_BINS = [-(10**6), -500, -200, -100, -50, -20, -5, -1, 0, 5, 20, 50, 100, 200, 500, 10**6]
REL_TOKEN_LABELS = [
    "<=-500",
    "-500..-200",
    "-200..-100",
    "-100..-50",
    "-50..-20",
    "-20..-5",
    "-5..-1",
    "0 (boundary)",
    "1..5",
    "6..20",
    "21..50",
    "51..100",
    "101..200",
    "201..500",
    ">500",
]


qbin = _stats.qbin


def by_bin(df: pd.DataFrame, bincol: str, probes: list[str], n_boot: int, rng) -> list[dict]:
    out = []
    for b, g in df.groupby(bincol, observed=True):
        row = {
            "bin": str(b),
            "n": int(len(g)),
            "n_traj": int(g["name"].nunique()),
            "mean_logmass": float(g["loudness"].mean()),
            "mean_prob": float(g["loudness_prob"].mean()),
            "mean_sentence_frac": float(g["sentence_frac"].mean()),
            "share_signal_word": float(g["is_signal_word"].mean()),
            "share_local_eq_final": float((g["label_local"] == g["label_final"]).mean()),
        }
        for p in probes:
            pt, lo, hi = boot_bal_acc(g, "label_local", f"{p}_pred", n_boot, rng)
            row[f"{p}_local"] = pt
            row[f"{p}_local_lo"] = lo
            row[f"{p}_local_hi"] = hi
            row[f"{p}_final"] = bal_acc(g["label_final"].to_numpy(), g[f"{p}_pred"].to_numpy())
        out.append(row)
    return out


def follows(df: pd.DataFrame, probes: list[str]) -> dict:
    """On the rows where the local belief and the final action DISAGREE, which does the
    probe land on? Entry 45(a), now as a function of whatever the caller grouped by."""
    d = df[df["label_local"] != df["label_final"]]
    out = {"n": int(len(d)), "n_traj": int(d["name"].nunique()) if len(d) else 0}
    for p in probes:
        if not len(d):
            out[p] = {}
            continue
        pred = d[f"{p}_pred"]
        out[p] = {
            "pred_eq_local": float((pred == d["label_local"]).mean()),
            "pred_eq_final": float((pred == d["label_final"]).mean()),
            "neither": float(((pred != d["label_local"]) & (pred != d["label_final"])).mean()),
        }
    return out


def _prepare(df: pd.DataFrame, args) -> tuple[pd.DataFrame, str, int, int]:
    """Map whatever the table calls its columns onto this module's internal names.

    Doing it once here is what lets every analysis below stay written in terms of `loudness`
    and `is_signal_word` regardless of which lens, vocabulary or layer produced the table --
    and lets a table written before the naming convention still load.
    """
    mass_col = args.loudness_column or _cols.resolve(df.columns, args.lens, args.signal_name, args.layer)
    if mass_col != "loudness":
        df["loudness"] = df[mass_col]
    if "loudness_prob" not in df.columns:
        df["loudness_prob"] = np.exp(df["loudness"])

    member = _cols.membership_column(args.signal_name)
    if "is_signal_word" not in df.columns:
        if member in df.columns:
            df["is_signal_word"] = df[member]
        elif args.signal_words is not None:
            vocab = {t for lst in json.loads(args.signal_words.read_text()).values() for t in lst}
            df["is_signal_word"] = df["token"].isin(vocab).astype(int)
        else:
            df["is_signal_word"] = 0

    n_before = len(df)
    if args.exclude_signal_words:
        if args.signal_words is not None:
            vocab = {t for lst in json.loads(args.signal_words.read_text()).values() for t in lst}
            flag = df["token"].isin(vocab).astype(int)
        else:
            flag = df["is_signal_word"].astype(int)
        drop = flag.astype(bool)
        if args.exclude_radius > 0:
            # A neighbour is a neighbour WITHIN its own trajectory and step -- shifting across
            # the whole frame would leak the last token of one chain onto the first of the next.
            grouped = flag.groupby([df["name"], df["step"]])
            for shift in range(1, args.exclude_radius + 1):
                drop |= grouped.shift(shift).fillna(0).astype(bool)
                drop |= grouped.shift(-shift).fillna(0).astype(bool)
        df = df[~drop].copy()
        print(
            f"--exclude-signal-words (radius {args.exclude_radius}): "
            f"{n_before} -> {len(df)} rows ({n_before - len(df)} dropped)",
            flush=True,
        )
    return df, mass_col, n_before, len(df)


# ---------------------------------------------------------------------------------------------
# Counts mode: a raw evaluator table, where one row summarises many predictions.
# ---------------------------------------------------------------------------------------------


def decile_table(df: pd.DataFrame, probe: str, score: str, n_bins: int) -> pd.DataFrame:
    """Balanced accuracy per loudness bin, rebuilt from the per-class count columns."""
    d = df.copy()
    d["bin"] = qbin(d[score], n_bins, labels=False)
    rows = []
    for b, g in d.groupby("bin", observed=True):
        ba, recalls = _stats.bal_acc_from_counts(g, probe, CLASSES)
        rows.append(
            {
                "bin": int(b),
                "n_tokens": int(len(g)),
                "n_traj": int(g["name"].nunique()),
                "mean_logmass": float(g[score].mean()),
                "balanced_acc": ba,
                "plain_acc": _stats.plain_accuracy(g, probe),
                "n_classes_with_support": len(recalls),
            }
        )
    return pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)


def gap_bootstrap(df: pd.DataFrame, probe: str, score: str, n_bins: int, n_boot: int, seed: int) -> dict:
    """Trajectory-clustered bootstrap of (top bin BA - bottom bin BA).

    The bins are recomputed inside each resample: the decile edges are themselves a function
    of the sample, and holding them fixed would understate the spread.
    """
    names = df["name"].unique()
    rng = np.random.default_rng(seed)
    by_name = dict(df.groupby("name").__iter__())
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(names), size=len(names), replace=True)
        rs = pd.concat([by_name[names[i]] for i in pick], ignore_index=True)
        t = decile_table(rs, probe, score, n_bins)
        if len(t) < 2:
            continue
        draws.append(t["balanced_acc"].iloc[-1] - t["balanced_acc"].iloc[0])
    if not draws:
        return {}
    base = decile_table(df, probe, score, n_bins)
    return {
        "gap": float(base["balanced_acc"].iloc[-1] - base["balanced_acc"].iloc[0]),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "n_boot": len(draws),
    }


def verdict(gap: float, lo: float, hi: float, reference: float) -> str:
    """Read the gap as an EFFECT SIZE against a reference, not as a significance test.

    At 87k tokens a 2pp gap is comfortably resolvable, so "the CI excludes zero" is nearly
    guaranteed and says almost nothing on its own -- reporting it alone would call a -0.02
    drift "correlated" and invite exactly the wrong conclusion. What a specificity question
    turns on is whether the gap approaches the matched same-token, same-layer reference.
    """
    if lo <= 0 <= hi:
        return f"straddles zero: no detectable trend (reference gap {reference:+.3f})"
    direction = "RISES with loudness" if gap > 0 else "FALLS slightly as loudness rises"
    frac = abs(gap) / abs(reference) if reference else float("inf")
    if gap < 0:
        return (
            f"{direction}; |gap| is {frac:.0%} of the reference {reference:+.3f} "
            "and OPPOSITE in sign -> loudness does not predict this label"
        )
    if frac < 0.25:
        return f"{direction} but |gap| is only {frac:.0%} of the reference {reference:+.3f} -> far weaker"
    return f"{direction}, {frac:.0%} of the reference {reference:+.3f} -> comparable to the reference effect"


def run_counts_mode(df: pd.DataFrame, args, ptype) -> dict:
    """The whole analysis for a per-cell probe type, on a raw evaluator table."""
    probes = sorted({c.rsplit("_n_correct", 1)[0] for c in df.columns if c.endswith("_n_correct")})
    if args.probes_filter:
        probes = [p for p in probes if p in args.probes_filter]
    if not probes:
        raise SystemExit(
            "no probe columns found: a counts-mode table needs {probe}_n_correct and "
            "{probe}_correct_{class} columns, as score_probes_per_token.py --probe-type "
            f"{args.probe_type} writes. Columns present: {sorted(df.columns)[:12]}..."
        )
    print(f"counts mode ({args.probe_type}): {len(probes)} probe(s) over {len(df)} rows", flush=True)

    out: dict = {"mode": "counts", "probe_type": args.probe_type, "probes": probes, "bins": {}}
    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    for probe in probes:
        table = decile_table(df, probe, "loudness", args.deciles)
        table.to_csv(args.out / "tables" / f"{probe}_by_loudness_decile.csv", index=False)
        gap = gap_bootstrap(df, probe, "loudness", args.deciles, args.boot, 0)
        entry = {
            "table": table.to_dict(orient="records"),
            "gap": gap,
            "spearman_loudness_vs_accuracy": (
                _stats.spearman(df, "loudness", f"{probe}_acc") if f"{probe}_acc" in df.columns else None
            ),
        }
        if gap:
            entry["verdict"] = verdict(gap["gap"], gap["lo"], gap["hi"], args.reference_gap)
            print(f"  {probe}: {entry['verdict']}", flush=True)
        out["bins"][probe] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--per-token", type=Path, default=Path("/workspace/reasoning_theatre/probe_loudness/per_token.csv")
    )
    ap.add_argument(
        "--all-token-loudness",
        type=Path,
        default=Path("/workspace/reasoning_theatre/loudness/per_token.csv"),
        help="entry 42's table over EVERY reasoning token, the reference the selected "
        "tokens are compared against. Skipped if absent.",
    )
    ap.add_argument("--out", type=Path, default=Path("/workspace/reasoning_theatre/probe_loudness"))
    # Extend by flag, never by editing the registry (entry 47): with no flag the output of a
    # re-run against entry 48's per_token.csv is byte-identical.
    ap.add_argument(
        "--extra-probes",
        default="",
        help="Comma-separated probe names to add to --extra-probes-rowset, e.g. 'randb_lr,randb_mlp'. "
        "They must already be columns of the per-token CSV.",
    )
    ap.add_argument("--extra-probes-rowset", default="p2", help="Rowset the --extra-probes join.")
    ap.add_argument(
        "--extra-final-label-probes",
        default="",
        help="Which of the --extra-probes were trained on the FINAL action rather than the local "
        "belief. Gets the trained_on field right; empty means all the new arms are belief-trained.",
    )
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument(
        "--probe-type",
        default=_probes.DEFAULT_PROBE_TYPE,
        choices=_probes.probe_type_names(),
        help="What the probes decode. Sets the classes balanced accuracy averages over (default: %(default)s).",
    )
    ap.add_argument(
        "--lens",
        default="jlens",
        help="Which lens's loudness is the binning axis (default: %(default)s). Every table "
        "and caption names it -- an unqualified 'loudness' is not a quantity.",
    )
    ap.add_argument("--signal-name", default="direction", help="Which vocabulary (default: %(default)s).")
    ap.add_argument("--layer", type=int, default=15, help="Layer the loudness is read at.")
    ap.add_argument(
        "--loudness-column",
        default=None,
        help="Read this column as the axis instead of the one --lens/--signal-name/--layer "
        "imply. Legacy spellings are resolved automatically.",
    )
    ap.add_argument(
        "--exclude-signal-words",
        action="store_true",
        help="Drop rows whose token is a signal word before doing anything, and produce the "
        "SAME tables from what is left. This is the verbalisation control: the tokens a lens "
        "calls loud are disproportionately the words the model has already typed, so a result "
        "that survives here is not 'the token says up'.",
    )
    ap.add_argument(
        "--exclude-radius",
        type=int,
        default=0,
        help="With --exclude-signal-words, also drop rows within N tokens of a signal word "
        "(0 = only the word itself). Loudness bleeds into neighbours, so N=1..3 asks whether "
        "the gradient survives away from the verbalisation entirely.",
    )
    ap.add_argument(
        "--signal-words",
        type=Path,
        default=None,
        help="JSON of {class: [tokens]} defining the signal words for --exclude-signal-words. "
        "Defaults to the table's own is_<signal>_token column when it has one.",
    )
    ap.add_argument(
        "--reference-gap",
        type=float,
        default=0.1559,
        help="Effect size the counts-mode gap is read against (default: the matched action "
        "gap from next_action_mass_l15.random_topall_mlp). A gap is reported as a fraction "
        "of this, not as a significance test.",
    )
    ap.add_argument(
        "--probes-filter",
        default="",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        help="Restrict counts mode to these probe keys (default: every probe in the table).",
    )
    args = ap.parse_args()
    extra = [t.strip() for t in args.extra_probes.split(",") if t.strip()]
    if extra:
        if args.extra_probes_rowset not in PROBES:
            raise SystemExit(f"unknown --extra-probes-rowset {args.extra_probes_rowset!r}")
        dupes = [e for e in extra if e in PROBES[args.extra_probes_rowset]]
        if dupes:
            raise SystemExit(f"--extra-probes already in the rowset: {dupes}")
        PROBES[args.extra_probes_rowset].extend(extra)
        FINAL_LABEL_PROBES.update(t.strip() for t in args.extra_final_label_probes.split(",") if t.strip())
        print(f"rowset {args.extra_probes_rowset}: {PROBES[args.extra_probes_rowset]}", flush=True)
    rng = np.random.default_rng(0)

    # csv.DictReader elsewhere, but this file is ours and quotes its token column; the
    # NA-corruption rule is why keep_default_na is off.
    df = pd.read_csv(args.per_token, keep_default_na=False, na_values=[""], low_memory=False)

    ptype = _probes.get_probe_type(args.probe_type)
    CLASSES[:] = ptype.analysis_classes
    df, mass_col, n_before, n_after = _prepare(df, args)
    args.out.mkdir(parents=True, exist_ok=True)

    if ptype.aggregation == "counts":
        # A raw evaluator table: one row per token summarising that step's cells. None of the
        # joined columns (label_local, {probe}_pred, sentence coordinates) exist here.
        summary = run_counts_mode(df, args, ptype)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        cfg = _prov.RunConfig("loudness_analysis/analysis/probe_accuracy_by_loudness.py")
        cfg.measurement(lens=args.lens, signal=args.signal_name, layer=args.layer)
        cfg.input("per_token", args.per_token)
        cfg.params.update({"probe_type": args.probe_type, "deciles": args.deciles})
        cfg.aggregation(
            balanced_accuracy="counts",
            classes=list(CLASSES),
            resolved_loudness_column=mass_col,
            bootstrap="trajectory-clustered, bin edges recomputed per resample",
            n_boot=args.boot,
            seed=0,
        )
        cfg.rows("input", n_before)
        cfg.rows("after_signal_word_filter", n_after)
        cfg.guard(args.out, "lens", "signal", "layer")
        cfg.write(args.out)
        print(f"\nwrote {args.out / 'summary.json'}")
        return 0

    (args.out / "tables").mkdir(exist_ok=True)
    summary: dict = {"per_token": str(args.per_token), "rowsets": {}}

    for rs, all_probes in PROBES.items():
        d = df[df["rowset"] == rs].copy()
        if not len(d):
            continue
        probes = [p for p in all_probes if f"{p}_pred" in d.columns]
        print(f"\n=== {rs}: {len(d)} rows, {d['name'].nunique()} trajectories, probes {probes}", flush=True)
        S: dict = {"n": int(len(d)), "n_traj": int(d["name"].nunique()), "probes": probes}

        # ---- overall, and the loudness of what these probes actually see
        S["loudness"] = {
            "mean_logmass": float(d["loudness"].mean()),
            "median_logmass": float(d["loudness"].median()),
            "mean_prob": float(d["loudness_prob"].mean()),
            "mean_mass_rank_in_traj": float(d["mass_rank_in_traj"].mean()),
            "median_mass_pct_in_traj": float(d["mass_pct_in_traj"].median()),
            "share_rank1_in_sentence": float((d["mass_rank_in_sentence"] == 1).mean()),
            "mean_sentence_frac": float(d["sentence_frac"].mean()),
            "share_sentence_end": float(d["is_sentence_end"].mean()),
            "share_signal_word": float(d["is_signal_word"].mean()),
        }
        S["overall"] = {}
        for p in probes:
            pt, lo, hi = boot_bal_acc(d, "label_local", f"{p}_pred", args.boot, rng)
            S["overall"][p] = {
                "bal_acc_local": pt,
                "lo": lo,
                "hi": hi,
                "bal_acc_final": bal_acc(d["label_final"].to_numpy(), d[f"{p}_pred"].to_numpy()),
                "trained_on": "final" if p in FINAL_LABEL_PROBES else "local",
            }
            print(
                f"  {p:10s} local {pt:.4f} [{lo:.4f},{hi:.4f}]  final {S['overall'][p]['bal_acc_final']:.4f}",
                flush=True,
            )

        # ---- Q1: accuracy by loudness decile
        d["mass_decile"] = qbin(d["loudness"], args.deciles, labels=False)
        S["by_mass_decile"] = by_bin(d, "mass_decile", probes, args.boot, rng)
        pd.DataFrame(S["by_mass_decile"]).to_csv(args.out / "tables" / f"{rs}_by_mass_decile.csv", index=False)

        # the monotonicity claim, stated as a number rather than eyeballed
        S["mass_monotonicity"] = {}
        for p in probes:
            v = [r[f"{p}_local"] for r in S["by_mass_decile"]]
            diffs = np.diff(v)
            S["mass_monotonicity"][p] = {
                "first": float(v[0]),
                "last": float(v[-1]),
                "delta": float(v[-1] - v[0]),
                "n_reversals": int((diffs < 0).sum()),
                "spearman": float(pd.Series(v).corr(pd.Series(range(len(v))), method="spearman")),
            }

        # ---- Q2: position in the sentence, and the 3x3 that separates it from loudness
        d["sentfrac_decile"] = qbin(d["sentence_frac"], args.deciles, labels=False)
        S["by_sentence_frac_decile"] = by_bin(d, "sentfrac_decile", probes, 0, rng)
        pd.DataFrame(S["by_sentence_frac_decile"]).to_csv(
            args.out / "tables" / f"{rs}_by_sentence_frac.csv", index=False
        )

        d["mass_t"] = qbin(d["loudness"], 3, labels=False)
        d["pos_t"] = qbin(d["sentence_frac"], 3, labels=False)
        grid = []
        for (mt, pt_), g in d.groupby(["mass_t", "pos_t"], observed=True):
            row = {"mass_tercile": int(mt), "pos_tercile": int(pt_), "n": int(len(g))}
            for p in probes:
                row[f"{p}_local"] = bal_acc(g["label_local"].to_numpy(), g[f"{p}_pred"].to_numpy())
            grid.append(row)
        S["mass_x_position"] = grid
        pd.DataFrame(grid).to_csv(args.out / "tables" / f"{rs}_mass_x_position.csv", index=False)
        # the marginal each way: spread ACROSS mass at fixed position, and vice versa
        gdf = pd.DataFrame(grid)
        S["mass_x_position_marginals"] = {}
        for p in probes:
            c = f"{p}_local"
            S["mass_x_position_marginals"][p] = {
                "mean_range_over_mass_at_fixed_position": float(
                    gdf.groupby("pos_tercile")[c].agg(lambda s: s.max() - s.min()).mean()
                ),
                "mean_range_over_position_at_fixed_mass": float(
                    gdf.groupby("mass_tercile")[c].agg(lambda s: s.max() - s.min()).mean()
                ),
            }

        # ---- Q2c: CHAIN LENGTH, which is the confound that actually bites here.
        # A fixed top-K per trajectory takes the K loudest of a 60-token chain and the K
        # loudest of a 460-token one, so within such an arm `dir_logmass` is correlated
        # with how long the chain is -- and long chains are harder (they change their mind
        # more, so local != final more often). The two effects run OPPOSITE ways, which is
        # what flattens p2's decile curve. Re-cut the mass terciles WITHIN chain-length
        # quartiles: if the gradient is loudness it survives, if it was length it dies.
        d["len_q"] = qbin(d["n_reasoning_tokens"], 4, labels=False)
        d["mass_t_in_len"] = d.groupby("len_q", observed=True)["loudness"].transform(
            lambda s: qbin(s, 3, labels=False)
        )
        S["chain_length"] = {
            "corr_logmass_n_reasoning": float(d["loudness"].corr(d["n_reasoning_tokens"])),
            "by_len_quartile": [],
        }
        for lq, g in d.groupby("len_q", observed=True):
            row = {
                "len_quartile": int(lq),
                "n": int(len(g)),
                "mean_n_reasoning": float(g["n_reasoning_tokens"].mean()),
                "mean_logmass": float(g["loudness"].mean()),
                "share_local_eq_final": float((g["label_local"] == g["label_final"]).mean()),
            }
            for p in probes:
                v = [
                    bal_acc(gg["label_local"].to_numpy(), gg[f"{p}_pred"].to_numpy())
                    for _, gg in g.groupby("mass_t_in_len", observed=True)
                ]
                row[f"{p}_by_mass_tercile"] = v
                row[f"{p}_delta"] = float(v[-1] - v[0]) if len(v) > 1 else float("nan")
            S["chain_length"]["by_len_quartile"].append(row)
        pd.DataFrame(S["chain_length"]["by_len_quartile"]).to_csv(
            args.out / "tables" / f"{rs}_chain_length_control.csv", index=False
        )

        # ---- Q3: the commitment boundary
        # LEGACY axis: the boundary located on the sentence-end grid.
        rel = d[d["rel_sentence"].notna()].copy()
        if len(rel):
            rel["rel_clipped"] = rel["rel_sentence"].clip(-6, 6).astype(int)
            S["by_rel_sentence"] = by_bin(rel, "rel_clipped", probes, 0, rng)
            pd.DataFrame(S["by_rel_sentence"]).to_csv(args.out / "tables" / f"{rs}_by_rel_sentence.csv", index=False)

        # PER-TOKEN boundary, two axes. The degenerate cohort is dropped from both: a step
        # already committed at the no-reasoning cutoff has no pre-boundary side, so keeping
        # it would fill the positive bins with rows that have no negative counterpart.
        live = d[(d["rel_token"].notna()) & (d["convinced_before_reasoning"] == 0)].copy()
        if len(live):
            # Same +-6 SENTENCE axis as by_rel_sentence, so the two tables are read side by
            # side and the only difference is where the boundary was placed.
            live["rel_sent_tok_clipped"] = live["rel_sentence_token"].clip(-6, 6).astype(int)
            S["by_rel_sentence_token"] = by_bin(live, "rel_sent_tok_clipped", probes, 0, rng)
            pd.DataFrame(S["by_rel_sentence_token"]).to_csv(
                args.out / "tables" / f"{rs}_by_rel_sentence_token.csv", index=False
            )
            # And the axis the redefinition actually buys: signed TOKEN distance. Token
            # distances run to +-2000, so the bins are log-spaced rather than clipped --
            # a +-6 clip would hold 4% of the rows.
            live["rel_token_bin"] = pd.cut(
                live["rel_token"], bins=REL_TOKEN_BINS, labels=REL_TOKEN_LABELS, include_lowest=True
            )
            S["by_rel_token"] = by_bin(live, "rel_token_bin", probes, 0, rng)
            pd.DataFrame(S["by_rel_token"]).to_csv(args.out / "tables" / f"{rs}_by_rel_token.csv", index=False)

        # ---- Q4: the verbalization control, crossed with loudness
        S["by_signal_word"] = {}
        for isdir, g in d.groupby("is_signal_word", observed=True):
            key = "is_signal_word" if isdir else "not_signal_word"
            S["by_signal_word"][key] = {
                "n": int(len(g)),
                "mean_logmass": float(g["loudness"].mean()),
                **{
                    p: {
                        "bal_acc_local": bal_acc(g["label_local"].to_numpy(), g[f"{p}_pred"].to_numpy()),
                        "bal_acc_final": bal_acc(g["label_final"].to_numpy(), g[f"{p}_pred"].to_numpy()),
                    }
                    for p in probes
                },
            }
        nd = d[d["is_signal_word"] == 0].copy()
        nd["mass_decile"] = qbin(nd["loudness"], args.deciles, labels=False)
        S["by_mass_decile_no_signal_words"] = by_bin(nd, "mass_decile", probes, 0, rng)
        pd.DataFrame(S["by_mass_decile_no_signal_words"]).to_csv(
            args.out / "tables" / f"{rs}_by_mass_decile_no_signal_words.csv", index=False
        )

        # ---- Q5: does the probe follow the belief MORE when the token is loud?
        S["follows_overall"] = follows(d, probes)
        S["follows_by_mass_decile"] = [
            {"bin": int(b), **follows(g, probes)} for b, g in d.groupby("mass_decile", observed=True)
        ]

        summary["rowsets"][rs] = S

    # ---- Q6: where the selected tokens sit in the whole-chain loudness distribution
    if args.all_token_loudness.exists():
        allrows = pd.read_csv(
            args.all_token_loudness, usecols=["all_token_loudness", "sentence_frac", "is_signal_word"]
        )
        q = [0.5, 0.75, 0.9, 0.95, 0.99]
        summary["all_token_reference"] = {
            "n": int(len(allrows)),
            "mean_logmass": float(allrows["all_token_loudness"].mean()),
            "quantiles": {str(x): float(allrows["all_token_loudness"].quantile(x)) for x in q},
            "share_signal_word": float(allrows["is_signal_word"].mean()),
            "note": "entry 42's training-split table over EVERY reasoning token",
        }
        cuts = allrows["all_token_loudness"].to_numpy()
        for rs in summary["rowsets"]:
            sel = df[df["rowset"] == rs]["loudness"].to_numpy()
            summary["rowsets"][rs]["loudness"]["mean_percentile_in_all_tokens"] = float(
                np.searchsorted(np.sort(cuts), sel).mean() / len(cuts)
            )

    cfg = _prov.RunConfig("loudness_analysis/analysis/probe_accuracy_by_loudness.py")

    cfg.measurement(lens=args.lens, signal=args.signal_name, layer=args.layer, signal_json=args.signal_words)

    cfg.input("per_token", args.per_token)

    if args.all_token_loudness:
        cfg.input("all_token_loudness", args.all_token_loudness)

    cfg.params.update(
        {
            "probe_type": args.probe_type,
            "deciles": args.deciles,
            "exclude_signal_words": args.exclude_signal_words,
            "exclude_radius": args.exclude_radius,
            "extra_probes": extra,
        }
    )

    cfg.aggregation(
        balanced_accuracy=ptype.aggregation,
        classes=list(CLASSES),
        resolved_loudness_column=mass_col,
        bootstrap="trajectory-clustered",
        n_boot=args.boot,
        seed=0,
    )

    cfg.rows("input", n_before)

    cfg.rows("after_signal_word_filter", n_after)

    cfg.guard(args.out, "lens", "signal", "layer")

    cfg.write(args.out)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\nwrote {args.out / 'summary.json'} and {len(list((args.out / 'tables').glob('*.csv')))} tables", flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
