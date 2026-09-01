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

ACTIONS = ["LEFT", "UP", "RIGHT", "DOWN"]
PROBES = {
    "p1_full": ["p1_lr", "p1_mlp"],
    "p1_top20": ["p1t20_lr", "p1t20_mlp"],
    "p2": ["p2_lr", "p2_mlp", "base_lr", "base_mlp", "rand_lr", "rand_mlp"],
}
# which label a probe was TRAINED against -- base_/rand_ are the entry-38 final-action probes
FINAL_LABEL_PROBES = {"base_lr", "base_mlp", "rand_lr", "rand_mlp"}


def bal_acc(truth: np.ndarray, pred: np.ndarray) -> float:
    """Mean per-class recall over the classes present. nan on an empty bin."""
    recalls = []
    for a in ACTIONS:
        m = truth == a
        if m.sum():
            recalls.append(float((pred[m] == truth[m]).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def boot_bal_acc(df: pd.DataFrame, truth_col: str, pred_col: str, n: int, rng) -> tuple[float, float, float]:
    """Balanced accuracy with a 95% CI from resampling trajectory names."""
    point = bal_acc(df[truth_col].to_numpy(), df[pred_col].to_numpy())
    names = df["name"].to_numpy()
    uniq = np.unique(names)
    if len(uniq) < 2 or n == 0:
        return point, float("nan"), float("nan")
    idx_of = {u: np.flatnonzero(names == u) for u in uniq}
    t, p = df[truth_col].to_numpy(), df[pred_col].to_numpy()
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_of[uniq[k]] for k in pick])
        draws.append(bal_acc(t[rows], p[rows]))
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def qbin(s: pd.Series, q: int, labels=None) -> pd.Series:
    """Quantile bins that survive ties (mass is heavy-tailed and has repeated floors)."""
    try:
        return pd.qcut(s, q, labels=labels, duplicates="drop")
    except ValueError:
        return pd.cut(s, q, labels=labels)


def by_bin(df: pd.DataFrame, bincol: str, probes: list[str], n_boot: int, rng) -> list[dict]:
    out = []
    for b, g in df.groupby(bincol, observed=True):
        row = {
            "bin": str(b),
            "n": int(len(g)),
            "n_traj": int(g["name"].nunique()),
            "mean_logmass": float(g["dir_logmass"].mean()),
            "mean_prob": float(g["dir_prob"].mean()),
            "mean_sentence_frac": float(g["sentence_frac"].mean()),
            "share_direction_token": float(g["is_direction_token"].mean()),
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
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--deciles", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    # csv.DictReader elsewhere, but this file is ours and quotes its token column; the
    # NA-corruption rule is why keep_default_na is off.
    df = pd.read_csv(args.per_token, keep_default_na=False, na_values=[""], low_memory=False)
    args.out.mkdir(parents=True, exist_ok=True)
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
            "mean_logmass": float(d["dir_logmass"].mean()),
            "median_logmass": float(d["dir_logmass"].median()),
            "mean_prob": float(d["dir_prob"].mean()),
            "mean_mass_rank_in_traj": float(d["mass_rank_in_traj"].mean()),
            "median_mass_pct_in_traj": float(d["mass_pct_in_traj"].median()),
            "share_rank1_in_sentence": float((d["mass_rank_in_sentence"] == 1).mean()),
            "mean_sentence_frac": float(d["sentence_frac"].mean()),
            "share_sentence_end": float(d["is_sentence_end"].mean()),
            "share_direction_token": float(d["is_direction_token"].mean()),
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
        d["mass_decile"] = qbin(d["dir_logmass"], args.deciles, labels=False)
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

        d["mass_t"] = qbin(d["dir_logmass"], 3, labels=False)
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
        d["mass_t_in_len"] = d.groupby("len_q", observed=True)["dir_logmass"].transform(
            lambda s: qbin(s, 3, labels=False)
        )
        S["chain_length"] = {
            "corr_logmass_n_reasoning": float(d["dir_logmass"].corr(d["n_reasoning_tokens"])),
            "by_len_quartile": [],
        }
        for lq, g in d.groupby("len_q", observed=True):
            row = {
                "len_quartile": int(lq),
                "n": int(len(g)),
                "mean_n_reasoning": float(g["n_reasoning_tokens"].mean()),
                "mean_logmass": float(g["dir_logmass"].mean()),
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
        rel = d[d["rel_sentence"].notna()].copy()
        if len(rel):
            rel["rel_clipped"] = rel["rel_sentence"].clip(-6, 6).astype(int)
            S["by_rel_sentence"] = by_bin(rel, "rel_clipped", probes, 0, rng)
            pd.DataFrame(S["by_rel_sentence"]).to_csv(args.out / "tables" / f"{rs}_by_rel_sentence.csv", index=False)

        # ---- Q4: the verbalization control, crossed with loudness
        S["by_direction_token"] = {}
        for isdir, g in d.groupby("is_direction_token", observed=True):
            key = "is_direction_word" if isdir else "not_direction_word"
            S["by_direction_token"][key] = {
                "n": int(len(g)),
                "mean_logmass": float(g["dir_logmass"].mean()),
                **{
                    p: {
                        "bal_acc_local": bal_acc(g["label_local"].to_numpy(), g[f"{p}_pred"].to_numpy()),
                        "bal_acc_final": bal_acc(g["label_final"].to_numpy(), g[f"{p}_pred"].to_numpy()),
                    }
                    for p in probes
                },
            }
        nd = d[d["is_direction_token"] == 0].copy()
        nd["mass_decile"] = qbin(nd["dir_logmass"], args.deciles, labels=False)
        S["by_mass_decile_no_direction_words"] = by_bin(nd, "mass_decile", probes, 0, rng)
        pd.DataFrame(S["by_mass_decile_no_direction_words"]).to_csv(
            args.out / "tables" / f"{rs}_by_mass_decile_nodir.csv", index=False
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
            args.all_token_loudness, usecols=["dir_logmass_L15", "sentence_frac", "is_direction_token"]
        )
        q = [0.5, 0.75, 0.9, 0.95, 0.99]
        summary["all_token_reference"] = {
            "n": int(len(allrows)),
            "mean_logmass": float(allrows["dir_logmass_L15"].mean()),
            "quantiles": {str(x): float(allrows["dir_logmass_L15"].quantile(x)) for x in q},
            "share_direction_token": float(allrows["is_direction_token"].mean()),
            "note": "entry 42's training-split table over EVERY reasoning token",
        }
        cuts = allrows["dir_logmass_L15"].to_numpy()
        for rs in summary["rowsets"]:
            sel = df[df["rowset"] == rs]["dir_logmass"].to_numpy()
            summary["rowsets"][rs]["loudness"]["mean_percentile_in_all_tokens"] = float(
                np.searchsorted(np.sort(cuts), sel).mean() / len(cuts)
            )

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\nwrote {args.out / 'summary.json'} and {len(list((args.out / 'tables').glob('*.csv')))} tables", flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
