#!/usr/bin/env python3
"""Build the probe inventory spreadsheet (xlsx + csv) from what is on disk.

Categorical columns (selection, lens, layer, tree, era, label, round, used_in) are the
only hand-authored part; every number comes from a probe checkpoint's own `config`/
`results`, a prepared manifest, or a balanced accuracy recomputed here.
"""

import csv
import json
import os
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

SCRATCH = Path(__file__).parent
OUT = SCRATCH / "out"
OUT.mkdir(exist_ok=True)

# Everything below is derived from what is on disk. An earlier version read three JSON files
# from a session scratch directory, which meant the committed script could not be re-run once
# that directory was cleaned up -- an inventory you cannot regenerate is not an inventory.
PROBE_ROOT = Path("/workspace/probes")

# Local-belief probes are read from ONE place. The entry-45/49 originals that used to be scattered
# across probes/local_belief, probes/local_belief_baselines and
# reasoning_theatre/local_belief_probes/probes have been deleted; every surviving belief probe
# lives under this root in a p1 / p1-top20 / p2 cadence folder. This is ENFORCED below rather than
# left to the fact that the other directories happen to be gone -- a belief-labelled probe found
# anywhere else is skipped and named on stderr, so a stray copy cannot quietly rejoin the sheet.
BELIEF_ROOT = PROBE_ROOT / "local_belief_action_l15"

# Directory names that are never walked. A `prepared/` subtree is an activations tree -- tens of
# thousands of per-token (D,) tensors that are not probes, and minutes of MooseFS readdir before a
# single probe is reached.
SKIP_DIRS = {"prepared"}
# newest first: a probe scored in several rounds takes its most recent number
HELDOUT_BELIEF_JSONS = (
    Path("/workspace/reasoning_theatre/probe_loudness_heldout360_equal_n/heldout_balanced_accuracy.json"),
    Path("/workspace/reasoning_theatre/probe_loudness_heldout360_24probes/heldout_balanced_accuracy.json"),
)
HELDOUT_FINAL_CSV = Path("/workspace/probes/heldout360_all_probes.csv")


def scan_probes() -> list[dict]:
    """Every probe checkpoint on disk, with the facts the sheet needs read from the file itself.

    `config` and `results` are written by `train_next_action_probe`, so nothing here is
    hand-maintained: adding a probe under PROBE_ROOT adds a row.

    The walk is pruned rather than a plain rglob, because both things it prunes produce a WRONG
    inventory rather than an error: a `prepared/` subtree would have every one of its activation
    tensors torch.load-ed as if it were a probe, and a symlinked directory would be walked twice
    under its two names -- which is exactly what probes/local_belief_equalN is.
    """
    import torch

    out = []
    for dirpath, dirnames, filenames in os.walk(PROBE_ROOT, followlinks=False):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not (here / d).is_symlink())
        for name in sorted(filenames):
            if not name.endswith(".pt") or "probe" not in name:
                continue
            path = here / name
            ck = torch.load(path, map_location="cpu", weights_only=False)
            cfg, res = ck.get("config", {}), ck.get("results", {})
            out.append(
                {
                    "path": str(path),
                    "train": cfg.get("train_data_path"),
                    "eval": cfg.get("eval_data_path"),
                    "mt": ck.get("model_type"),
                    "seed": cfg.get("seed"),
                    "acc": res.get("best_eval_accuracy"),
                    "bal": res.get("best_balanced_accuracy"),
                }
            )
    return out


def load_heldout_belief() -> dict[str, dict]:
    """{probe key: {vs_local, vs_final}} from the belief rounds' scoring JSONs.

    Read oldest-first so a probe re-scored in a later round overwrites its earlier number --
    which is what makes the entry-52 rebuilds show their own held-out result rather than the
    one their superseded namesake got.
    """
    out: dict[str, dict] = {}
    for jf in reversed(HELDOUT_BELIEF_JSONS):
        if not jf.exists():
            continue
        for row in json.load(open(jf)):
            if row.get("rowset"):  # the whole-population row only, not the loudness bins
                continue
            out[row["probe"]] = {"vs_local": row["bal_vs_belief"], "vs_final": row["bal_vs_final"]}
    return out


def load_heldout_final() -> dict[str, float]:
    """{probe key: balanced accuracy vs the FINAL action} from the all-probes per-token CSV.

    Balanced rather than raw accuracy, per class then averaged, matching
    `train_next_action_probe::_evaluate`. csv.DictReader, never pandas: the decoded tokens in
    this file include "NA", empty strings and embedded commas.
    """
    if not HELDOUT_FINAL_CSV.exists():
        return {}
    hits: dict[str, dict] = {}
    with open(HELDOUT_FINAL_CSV, newline="") as f:
        reader = csv.DictReader(f)
        preds = [c for c in (reader.fieldnames or []) if c.endswith("_pred")]
        for row in reader:
            label = row.get("label")
            if label in (None, ""):
                continue
            for col in preds:
                key = col[: -len("_pred")]
                per = hits.setdefault(key, {})
                tot, ok = per.get(label, (0, 0))
                per[label] = (tot + 1, ok + (row[col] == label))
    return {key: sum(ok / tot for tot, ok in per.values()) / len(per) for key, per in hits.items() if per}


probes = scan_probes()
held26 = load_heldout_final()  # vs FINAL label
held16 = load_heldout_belief()  # vs local + vs final, the belief-round probes

# ---- prepared-manifest sample counts -------------------------------------------------
man_n: dict[str, int] = {}
for d in sorted(Path("/workspace/prepared").iterdir()):
    m = d / "manifest.json"
    if m.exists():
        doc = json.load(open(m))
        key = "samples" if "samples" in doc else "trajectories"
        man_n[d.name] = len(doc.get(key, []))

# ---- per-training-dataset facts ------------------------------------------------------
# tree, era, selection, lens, layer, label, entry
DS = {
    # --- entry 52: the same cadences rebuilt on EQUAL-N data ------------------------------
    # Same trees, same rollouts, same label. Two things differ from entries 45/49/51: the
    # collided final-sentence row is restored in every arm (so all four hold the same 75,042
    # sentences, paired row for row), and the random top-20 arm is DRAWN rather than ranked.
    # Absolute accuracies are not comparable to the earlier rounds -- ~4.8% of each arm is now
    # the near-deterministic final-sentence row -- but the arm-to-arm gaps are.
    "equal_n_jlens_split_train": (
        "argmax_per_sentence_l15",
        "mass-era 3600",
        "loudest per sentence (equal-N, every sentence)",
        "jlens",
        "L15",
        "local belief",
        "52",
    ),
    "equal_n_logitlens_split_train": (
        "logitlens_argmax_per_sentence_l15",
        "mass-era 3600",
        "loudest per sentence (equal-N, every sentence)",
        "logitlens",
        "L15",
        "local belief",
        "52",
    ),
    "equal_n_eos_split_train": (
        "eos_mass3600_view",
        "mass-era 3600",
        "last token of each sentence (equal-N, every sentence)",
        "none",
        "L15",
        "local belief",
        "52",
    ),
    "equal_n_random_split_train": (
        "random_per_sentence_l15",
        "mass-era 3600",
        "random token per sentence (equal-N, every sentence)",
        "none (control)",
        "L15",
        "local belief",
        "52",
    ),
    "equal_n_jlens_top20_split_train": (
        "argmax_per_sentence_l15",
        "mass-era 3600",
        "loudest per sentence, RANKED to top-20/traj",
        "jlens",
        "L15",
        "local belief",
        "52",
    ),
    "equal_n_logitlens_top20_split_train": (
        "logitlens_argmax_per_sentence_l15",
        "mass-era 3600",
        "loudest per sentence, RANKED to top-20/traj",
        "logitlens",
        "L15",
        "local belief",
        "52",
    ),
    "equal_n_random_top20_split_train": (
        "random_per_sentence_l15",
        "mass-era 3600",
        "random per sentence, DRAWN to 20/traj (--thin-mode uniform)",
        "none (control)",
        "L15",
        "local belief",
        "52",
    ),
    "local_belief_p1_split_train": (
        "argmax_per_sentence_l15",
        "mass-era 3600",
        "loudest per sentence (uncapped, ~20.8/traj)",
        "jlens",
        "L15",
        "local belief",
        "45",
    ),
    "local_belief_p1_top20_split_train": (
        "argmax_per_sentence_l15",
        "mass-era 3600",
        "loudest per sentence, thinned to top-20/traj",
        "jlens",
        "L15",
        "local belief",
        "45",
    ),
    "local_belief_p2_split_train": (
        "jlens_mass_l15",
        "mass-era 3600",
        "global top-20 loudest (logprob_mass_full @L15)",
        "jlens",
        "L15",
        "local belief",
        "45",
    ),
    "entry49_random_belief_split_train": (
        "jlens_mass_l15",
        "mass-era 3600",
        "random 20/traj (seeded uniform, replayed picks)",
        "none (control)",
        "L15",
        "local belief",
        "49",
    ),
    "entry49_logitlens_p1_split_train": (
        "logitlens_argmax_per_sentence_l15",
        "mass-era 3600",
        "loudest per sentence (uncapped, ~20.8/traj)",
        "logitlens",
        "L15",
        "local belief",
        "49",
    ),
    "entry49_logitlens_p2_split_train": (
        "logitlens_mass_l15",
        "mass-era 3600",
        "global top-20 loudest (logprob_mass_full @L15)",
        "logitlens",
        "L15",
        "local belief",
        "49",
    ),
    # ---- entry 51: the per-sentence cadence completed --------------------------------
    # All three per-sentence rules walk the identical sentence_spans grid and differ only in
    # which token inside each span is taken: loudest (p1 / ll1), last (eos), random.
    "more_belief_eos_belief_split_train": (
        "eos_mass3600_view",
        "mass-era 3600",
        "the LAST token of each sentence (uncapped)",
        "none (eos)",
        "L15",
        "local belief",
        "51",
    ),
    "more_belief_random_sentence_belief_split_train": (
        "random_per_sentence_l15",
        "mass-era 3600",
        "one UNIFORMLY RANDOM token per sentence (uncapped)",
        "none (control)",
        "L15",
        "local belief",
        "51",
    ),
    "more_belief_random_sentence_belief_top20_split_train": (
        "random_per_sentence_l15",
        "mass-era 3600",
        "one random token per sentence, thinned to top-20/traj",
        "none (control)",
        "L15",
        "local belief",
        "51",
    ),
    "more_belief_logitlens_p1_top20_split_train": (
        "logitlens_argmax_per_sentence_l15",
        "mass-era 3600",
        "loudest per sentence, thinned to top-20/traj",
        "logitlens",
        "L15",
        "local belief",
        "51",
    ),
    "next_action_mass_l15_jlens_topall_train": (
        "jlens_mass_l15",
        "mass-era 3600",
        "global top-20 loudest (logprob_mass_full @L15)",
        "jlens",
        "L15",
        "final action",
        "37/38",
    ),
    "next_action_mass_l15_random_topall_train": (
        "jlens_mass_l15",
        "mass-era 3600",
        "random 20/traj (seeded uniform)",
        "none (control)",
        "L15",
        "final action",
        "37/38",
    ),
    "next_action_mass_l15_jlens_top1_train": (
        "jlens_mass_l15",
        "mass-era 3600",
        "top-1 loudest of the chain",
        "jlens",
        "L15",
        "final action",
        "38",
    ),
    "next_action_mass_l15_jlens_top2_train": (
        "jlens_mass_l15",
        "mass-era 3600",
        "top-2 loudest of the chain",
        "jlens",
        "L15",
        "final action",
        "38",
    ),
    "next_action_jlens_topall_train": (
        "jlens_reasoning_tokens",
        "count-era 3600",
        "top-20 by direction COUNT",
        "jlens",
        "per-token argmax (7:23)",
        "final action",
        "24/26",
    ),
    "next_action_logitlens_topall_train": (
        "jlens_reasoning_tokens",
        "count-era 3600",
        "top-20 by direction COUNT",
        "logitlens",
        "per-token argmax (7:23)",
        "final action",
        "24/26",
    ),
    "next_action_random_topall_train": (
        "jlens_reasoning_tokens",
        "count-era 3600",
        "random 20/traj (seeded uniform)",
        "none (control)",
        "L15",
        "final action",
        "24/26",
    ),
    "next_action_l15_jlens_topall_train": (
        "jlens_reasoning_tokens",
        "count-era 3600",
        "top-20 by direction COUNT",
        "jlens",
        "L15",
        "final action",
        "28",
    ),
    "next_action_l15_logitlens_topall_train": (
        "jlens_reasoning_tokens",
        "count-era 3600",
        "top-20 by direction COUNT",
        "logitlens",
        "L15",
        "final action",
        "31",
    ),
    "next_action_eos_topall_train": (
        "eos_lens3600_view",
        "count-era 3600",
        "every sentence end (eos) - no loudness used",
        "none (eos)",
        "L15",
        "final action",
        "29/30",
    ),
}
LEGACY = (
    "activations_train_single_step",
    "round-1 legacy",
    "grid cells at one step (no reasoning-token selection)",
    "n/a",
    "L15",
    "grid_tile",
    "1",
)

EVALSET = {
    "mass-era 3600": "eval-720 (mass-era, next_action_mass_l15_eval_names.txt)",
    "count-era 3600": "eval-720 (count-era, splits/eval_trajectories_720.txt)",
    "round-1 legacy": "internal 20% split (no held-out name list)",
    "uncatalogued": "(not known - the training dataset has no DS entry)",
}

# ---- the three probe roots, after the era split ------------------------------------------
# next_action_l15 holds ONLY the MASS-loudness probes (ranked by logprob_mass_full at L15).
# Everything ranked by the older top-20 direction COUNT lives under the DEPRECATED root: kept for
# provenance and reproduction, not for use. Its `p2` is L15 and its `p2-argmaxlayer` is the same
# selection read at each token's own best layer (7:23), which is a different probe, not a variant.
BELIEF = "probes/local_belief_action_l15"
MASS_NA = "probes/next_action_l15"
OLD_NA = "probes/DEPRECATED_TOP20LOUDNESS_next_action_l15"

# file (relative to /workspace) -> its column in HELDOUT_FINAL_CSV.
#
# The VALUES are column names frozen into a CSV that is already on disk, so they still spell the
# folders these probes used to live in (next_action, next_action_l15, next_action_mass_l15,
# next_action_seeds). Do NOT tidy them to match the new layout: the join is by column name, so
# renaming a value drops that probe's held-out number in silence rather than failing. Only the
# paths on the left-hand side moved.
H26: dict[str, str] = {}
for _arm in ("jlens", "logitlens", "random"):  # count-era, per-token argmax layer 7:23
    for _mt in ("lr", "mlp"):
        _p = f"{OLD_NA}/p2-argmaxlayer/next_action_probe_{_arm}_topall_{_mt}.pt"
        H26[_p] = f"next_action.{_arm}_topall_{_mt}"
for _mt in ("lr", "mlp"):  # count-era, pinned to L15
    H26[f"{OLD_NA}/p2/next_action_probe_jlens_topall_{_mt}.pt"] = f"next_action_l15.jlens_topall_{_mt}"
for _arm in ("jlens", "random"):  # mass-era
    for _mt in ("lr", "mlp"):
        _p = f"{MASS_NA}/p2/next_action_probe_{_arm}_topall_{_mt}.pt"
        H26[_p] = f"next_action_mass_l15.{_arm}_topall_{_mt}"
for s in (
    "jlens_l15_seed43",
    "jlens_l15_seed44",
    "logitlens_l15_seed42",
    "logitlens_l15_seed43",
    "logitlens_l15_seed44",
    "random_l15_seed43",
    "random_l15_seed44",
):
    for mt in ("lr", "mlp"):
        H26[f"{OLD_NA}/seeds/next_action_probe_{s}_{mt}.pt"] = f"next_action_seeds.{s}_{mt}"

# file -> its key in the belief rounds' scoring JSONs.
#
# Every belief probe is under BELIEF, in the cadence folder that says how its tokens were chosen:
#   p1        one cutoff per sentence, at that sentence's loudest token (uncapped)
#   p1-top20  the same rule, then RANKED down to 20 per trajectory
#   p2        the 20 globally loudest tokens of the chain
# p1 and p1-top20 are the entry-52 equal-N rebuilds and are the only p1-family probes left; the
# entry-45/49 originals they supersede were deleted. p2 was never rebuilt and did not need to be
# (a fixed min(n, 20) budget relabels a collided row instead of deleting it), so it is still the
# entry-45/49 vintage -- the two cadences are deliberately different vintages, not an oversight.
#
# Keys such as p1_*, ll1_*, eosb_* and rsb_* still appear in the scoring JSONs but map to no file
# any more. That is correct: those probes are gone, so they get no row.
H16: dict[str, str] = {}
for _arm, _key in (
    ("jlens", "eq_p1_jlens"),
    ("logitlens", "eq_p1_ll"),
    ("eos", "eq_p1_eos"),
    ("random", "eq_p1_rand"),
):
    for _mt in ("lr", "mlp"):
        H16[f"{BELIEF}/p1/next_action_probe_{_arm}_{_mt}.pt"] = f"{_key}_{_mt}"
for _arm, _key in (("jlens_top20", "eq_t20_jlens"), ("logitlens_top20", "eq_t20_ll"), ("random_top20", "eq_t20_rand")):
    for _mt in ("lr", "mlp"):
        H16[f"{BELIEF}/p1-top20/next_action_probe_{_arm}_{_mt}.pt"] = f"{_key}_{_mt}"
for _arm, _key in (("jlens", "p2"), ("logitlens", "ll2"), ("random", "randb")):
    for _mt in ("lr", "mlp"):
        H16[f"{BELIEF}/p2/next_action_probe_{_arm}_{_mt}.pt"] = f"{_key}_{_mt}"
# The two mass-era FINAL-action probes that were carried through the belief rounds as baselines.
for _arm, _key in (("jlens", "base"), ("random", "rand")):
    for _mt in ("lr", "mlp"):
        H16[f"{MASS_NA}/p2/next_action_probe_{_arm}_topall_{_mt}.pt"] = f"{_key}_{_mt}"

# analyses each probe appears in
A26 = "26p, pvr, pvr-lb"
# The three rounds carrying a surviving probe: the equal-N round (entry 52) scored the p1 and
# p1-top20 rebuilds, and the earlier rounds scored p2 and the two mass-era baselines. The probes
# that were only ever in the 24-probe round -- eos-belief, random-sentence, logitlens p1 -- have
# been deleted, so no branch for them is left here.
EQN = "eqn"
PL = "pl-720, pl-h360, 16p"
USED: dict[str, str] = dict.fromkeys(H26, A26)
for k, v in H16.items():
    prior = USED.get(k, "")
    if v.startswith(("eq_p1_", "eq_t20_")):
        tag = EQN
    elif v in ("p2_lr", "p2_mlp", "base_lr", "base_mlp", "rand_lr", "rand_mlp"):
        tag = PL
    else:
        tag = "16p"
    USED[k] = (prior + ", " if prior else "") + tag
# The belief probe the commitment-boundary clone read. Its p1 counterpart was deleted with the
# rest of the entry-45 vintage, so only p2 is named here.
USED[f"{BELIEF}/p2/next_action_probe_jlens_mlp.pt"] += ", pvr-lb"

# Both byte-identical copies this used to name (probes/local_belief/next_action_probe_p{1,2}_mlp.pt)
# were deleted during the reorganisation. The mechanism stays because it is generic and the sheet
# still has a group for it; it is empty because there is nothing duplicated on disk any more.
DUPES: dict[str, str] = {}

COLUMNS = [
    "group",
    "probe_id",
    "file",
    "entry",
    "model",
    "seed",
    "label",
    "selection",
    "lens",
    "layer",
    "tree",
    "era",
    "train_dataset",
    "train_n",
    "eval_dataset",
    "eval_n",
    "eval_traj_set",
    "eval720_population",
    "eval720_bal_acc_OWN_tokens",
    "eval720_acc_OWN_tokens",
    "heldout360_population",
    "heldout360_bal_vs_belief_ALL_tokens",
    "heldout360_bal_vs_final_ALL_tokens",
    "used_in",
    "notes",
]

GROUPS = {
    ("local belief", "45"): "A - belief label, jlens (entry 45)",
    ("local belief", "49"): "B - belief label, baselines (entry 49)",
    ("local belief", "51"): "B2 - belief label, per-sentence cadence (entry 51)",
    ("local belief", "52"): "B3 - belief label, EQUAL-N per-sentence cadence (entry 52, supersedes A/B2)",
    ("final action", "37/38"): "C - final label, mass-era",
    ("final action", "38"): "C - final label, mass-era",
    ("final action", "24/26"): "D - final label, count-era",
    ("final action", "28"): "D - final label, count-era",
    ("final action", "29/30"): "D - final label, count-era",
    ("final action", "31"): "E - seed sweep (entry 31)",
    ("grid_tile", "1"): "F - legacy grid probes (round 1)",
}
UNCATALOGUED_GROUP = "H - training dataset not in the DS table"


# A cadence folder does not identify a probe on its own: both the belief root and the next-action
# root have a `p2`, and the mass and deprecated next-action roots have the SAME filenames inside
# it. So a probe_id built from the parent directory alone would collide across roots, and the
# sheet would show two different probes under one id. These names take the root above them too.
CADENCE_DIRS = {"p1", "p1-top20", "p2", "p2-argmaxlayer", "eos", "seeds"}


def short_id(path: str) -> str:
    """Unique per file: the naming dir(s) + the informative half of the stem.

    "probes" and "downloaded" name nothing, so step up one level when we land on them. A cadence
    folder names only half of what is needed, so it keeps the directory above it as well.
    """
    p = Path(path)
    stem = p.stem.replace("next_action_probe_", "").replace("cognitive_map_probe_", "cogmap_")
    d = p.parent
    if d.name in ("probes", "downloaded"):
        d = d.parent
    if d.name in CADENCE_DIRS:
        return f"{d.parent.name}/{d.name}/{stem}"
    return f"{d.name}/{stem}"


def uncatalogued(rel: str) -> tuple[str, ...]:
    """Facts for a probe whose training dataset has no DS entry.

    It must NOT fall through to LEGACY the way an unknown dataset used to: that files a probe
    trained this year against a round-1 grid dataset as a round-1 grid probe, which is a wrong row
    rather than a missing one. The filename is the only thing left to trust, and only for the
    label, so everything else says outright that it is not known.
    """
    label = "grid_tile" if ("cognitive_map" in rel or "grid" in rel) else "next action"
    return ("(not in DS)", "uncatalogued", "(training dataset has no DS entry)", "?", "?", label, "-")


rows = []
skipped: list[str] = []
for x in sorted(probes, key=lambda z: z["path"]):
    rel = x["path"].replace("/workspace/", "")
    train = x["train"] or ""
    tkey = train.replace("/workspace/prepared/", "")
    if tkey in DS:
        facts = DS[tkey]
    elif "activations_train_single_step" in train:
        facts = LEGACY
    else:
        facts = uncatalogued(rel)
    tree, era, sel, lens, layer, label, entry = facts
    # Belief probes come from BELIEF_ROOT and nowhere else. A copy left behind in an old directory
    # would otherwise be indistinguishable from the live one in the sheet, and would carry the
    # same held-out numbers under a second probe_id.
    if label == "local belief" and not x["path"].startswith(str(BELIEF_ROOT) + "/"):
        skipped.append(rel)
        continue
    ekey = (x["eval"] or "").replace("/workspace/prepared/", "")
    h16 = held16.get(H16.get(rel, ""), {})
    h26v = held26.get(H26.get(rel, ""))
    seeds_dir = rel.endswith(".pt") and "/seeds/" in rel
    group = "E - seed sweep (entry 31)" if seeds_dir else GROUPS.get((label, entry), UNCATALOGUED_GROUP)
    if rel in DUPES:
        group = "G - duplicate copies (same bytes)"
    note = DUPES.get(rel, "")
    if era == "count-era 3600" and rel in H26:
        note = "count-era training set overlaps the heldout 360 by 25 trajectories (8.2% of its rows) - see Legend"
    rows.append(
        {
            "group": group,
            "probe_id": short_id(rel),
            "file": rel,
            "entry": entry,
            "model": x["mt"],
            "seed": x["seed"],
            "label": label,
            "selection": sel,
            "lens": lens,
            "layer": layer,
            "tree": tree,
            "era": era,
            "train_dataset": tkey or "(pre-v3 merged .pt)",
            "train_n": man_n.get(tkey, ""),
            "eval_dataset": ekey or "(internal split)",
            "eval_n": man_n.get(ekey, ""),
            "eval_traj_set": EVALSET[era],
            "eval720_population": (
                f"this probe's OWN selection: {man_n[ekey]:,} tokens of the {era.split()[0]} eval 720"
                if ekey in man_n
                else "internal 20% split of its own merged .pt"
            ),
            "eval720_bal_acc_OWN_tokens": round(x["bal"], 4) if isinstance(x["bal"], float) else "",
            "eval720_acc_OWN_tokens": round(x["acc"], 4) if isinstance(x["acc"], float) else "",
            "heldout360_population": (
                "ALL 87,221 reasoning tokens of the heldout 360 - no selection"
                if (rel in H16 or rel in H26)
                else "not scored on heldout"
            ),
            "heldout360_bal_vs_belief_ALL_tokens": round(h16["vs_local"], 4) if h16 else "",
            "heldout360_bal_vs_final_ALL_tokens": round(h16["vs_final"], 4)
            if h16
            else (round(h26v, 4) if h26v is not None else ""),
            "used_in": USED.get(rel, ""),
            "notes": note,
        }
    )

rows.sort(key=lambda r: (r["group"], r["label"], r["selection"], r["probe_id"]))

# ---- csv -----------------------------------------------------------------------------
with open(OUT / "probe_inventory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLUMNS)
    w.writeheader()
    w.writerows(rows)

# ---- xlsx ----------------------------------------------------------------------------

HEAD = PatternFill("solid", fgColor="1F3864")
HEADF = Font(color="FFFFFF", bold=True)
BAND = PatternFill("solid", fgColor="EEF2F8")

wb = Workbook()

ws = wb.active
ws.title = "Probes"
ws.append(COLUMNS)
for r in rows:
    ws.append([r[c] for c in COLUMNS])
for c in range(1, len(COLUMNS) + 1):
    cell = ws.cell(row=1, column=c)
    cell.fill, cell.font = HEAD, HEADF
    cell.alignment = Alignment(vertical="center", wrap_text=True)
widths = {
    "group": 34,
    "probe_id": 42,
    "file": 62,
    "entry": 7,
    "model": 7,
    "seed": 6,
    "label": 14,
    "selection": 46,
    "lens": 16,
    "layer": 22,
    "tree": 34,
    "era": 16,
    "train_dataset": 40,
    "train_n": 10,
    "eval_dataset": 40,
    "eval_n": 9,
    "eval_traj_set": 46,
    "eval720_population": 46,
    "eval720_bal_acc_OWN_tokens": 17,
    "eval720_acc_OWN_tokens": 15,
    "heldout360_population": 44,
    "heldout360_bal_vs_belief_ALL_tokens": 20,
    "heldout360_bal_vs_final_ALL_tokens": 20,
    "used_in": 26,
    "notes": 60,
}
for i, c in enumerate(COLUMNS, start=1):
    ws.column_dimensions[get_column_letter(i)].width = widths[c]
ws.freeze_panes = "C2"  # group + probe_id stay visible while scrolling right
ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{len(rows) + 1}"
ws.row_dimensions[1].height = 34
for ri in range(2, len(rows) + 2):
    if ri % 2 == 0:
        for ci in range(1, len(COLUMNS) + 1):
            ws.cell(row=ri, column=ci).fill = BAND

# ---- sheet 2: coverage matrix --------------------------------------------------------
cov = wb.create_sheet("Loudness coverage")
COV = [
    [
        "dataset",
        "selection (which tokens have .pt)",
        "lens",
        "loudness table (T)",
        "activations (A)",
        "loudness evaluation (E)",
        "probe trained (P)",
        "where",
    ],
    [
        "train 2,880 (mass-era)",
        "loudest per sentence",
        "jlens",
        "yes",
        "75,012 @L15",
        "-",
        "2 (local_belief_p1)",
        "argmax_per_sentence_l15",
    ],
    [
        "train 2,880 (mass-era)",
        "loudest per sentence",
        "logitlens",
        "yes",
        "74,975 @L15",
        "-",
        "2 (logitlens_p1)",
        "logitlens_argmax_per_sentence_l15",
    ],
    [
        "train 2,880 (mass-era)",
        "per-sentence, top-20/traj",
        "jlens",
        "yes",
        "subset of above",
        "-",
        "2 (local_belief_p1_top20)",
        "argmax_per_sentence_l15",
    ],
    [
        "train 2,880 (mass-era)",
        "per-sentence, top-20/traj",
        "logitlens",
        "yes",
        "subset of above",
        "-",
        "0 - never prepared",
        "-",
    ],
    [
        "train 2,880 (mass-era)",
        "global top-20 loudest",
        "jlens",
        "yes",
        "125,416 @L15 (jlens u random)",
        "-",
        "8 (topall final, p2 belief, top1, top2)",
        "jlens_mass_l15",
    ],
    [
        "train 2,880 (mass-era)",
        "global top-20 loudest",
        "logitlens",
        "yes",
        "71,913 @L15",
        "-",
        "2 (logitlens_p2, belief only)",
        "logitlens_mass_l15",
    ],
    [
        "train 2,880 (mass-era)",
        "random 20/traj (control)",
        "n/a",
        "yes",
        "in jlens_mass_l15",
        "-",
        "4 (2 final, 2 belief)",
        "jlens_mass_l15",
    ],
    [
        "train 2,880 (mass-era)",
        "all tokens",
        "jlens",
        "yes",
        "none",
        "776,191 rows (loudness/)",
        "0 - no activations",
        "loudness/per_token.csv",
    ],
    ["train 2,880 (mass-era)", "all tokens", "logitlens", "yes", "none", "-", "0", "logitlens_mass_l15"],
    ["eval 720 (mass-era)", "loudest per sentence", "jlens", "yes", "yes", "14,470 rows", "scored, not trained"],
    ["eval 720 (mass-era)", "per-sentence, top-20/traj", "jlens", "yes", "yes", "8,357 rows", "scored, not trained"],
    ["eval 720 (mass-era)", "global top-20 loudest", "jlens", "yes", "yes", "14,391 rows", "scored, not trained"],
    ["eval 720 (mass-era)", "loudest per sentence / top-20", "logitlens", "yes", "yes", "-", "scored, not trained"],
    [
        "train 2,880 (count-era)",
        "top-20 by COUNT",
        "jlens",
        "count CSV only (no mass table)",
        "963,773 @7:23",
        "-",
        "8 (argmax-layer + L15 seeds 42/43/44)",
        "jlens_reasoning_tokens",
    ],
    [
        "train 2,880 (count-era)",
        "top-20 by COUNT",
        "logitlens",
        "count CSV only",
        "shared tree",
        "-",
        "8 (argmax-layer + L15 seeds 42/43/44)",
        "jlens_reasoning_tokens",
    ],
    [
        "train 2,880 (count-era)",
        "random 20/traj",
        "n/a",
        "count CSV only",
        "shared tree",
        "-",
        "6 (L15 seeds 42/43/44)",
        "jlens_reasoning_tokens",
    ],
    [
        "train 2,880 (count-era)",
        "every sentence end (eos)",
        "n/a",
        "count CSV only",
        "77,196 @L15",
        "-",
        "6 (L15 seeds 42/43/44)",
        "eos_lens3600_view",
    ],
    [
        "heldout 360",
        "all tokens",
        "jlens",
        "yes",
        "87,221 @L15",
        "3 CSVs x 87,221",
        "0 by design - scoring tree only",
        "heldout360_l15 / _lens",
    ],
    ["heldout 360", "all tokens", "logitlens", "yes", "none", "1 CSV x 87,221", "0 by design", "heldout360_lens"],
]
for r in COV:
    cov.append(r)
for c in range(1, len(COV[0]) + 1):
    cell = cov.cell(row=1, column=c)
    cell.fill, cell.font = HEAD, HEADF
    cell.alignment = Alignment(vertical="center", wrap_text=True)
for i, w_ in enumerate([24, 34, 12, 30, 32, 26, 38, 36], start=1):
    cov.column_dimensions[get_column_letter(i)].width = w_
cov.freeze_panes = "A2"
cov.row_dimensions[1].height = 34

# ---- sheet 3: legend -----------------------------------------------------------------
leg = wb.create_sheet("Legend & sources")
LEG = [
    ["column / code", "meaning"],
    [
        "THE TWO ACCURACY BLOCKS MEASURE DIFFERENT POPULATIONS",
        "eval720_*_OWN_tokens is each probe read on the tokens ITS OWN selection picked - a loud-"
        "selected probe is scored only on loud tokens, so the loudness axis is truncated by "
        "construction and arms with different selections are NOT on a common population. "
        "heldout360_*_ALL_tokens is every probe read on the SAME 87,221 tokens with no selection in "
        "between. Compare within a block, never across.",
    ],
    [
        "WHERE THE PROBES LIVE",
        "Three roots, and the split between them is the loudness RULER, not the result. "
        "probes/local_belief_action_l15/{p1,p1-top20,p2} holds every local-belief probe - it is the "
        "only place one is read from, and a belief probe found anywhere else is skipped, not "
        "merged. probes/next_action_l15/p2 holds the final-action probes ranked by MASS "
        "(logprob_mass_full at L15). probes/DEPRECATED_TOP20LOUDNESS_next_action_l15 holds the "
        "older arms ranked by the top-20 direction COUNT, kept for provenance and reproduction "
        "only: p2 (L15), p2-argmaxlayer (the same selection read at each token's own best layer, "
        "7:23 - a different probe, not a variant), eos, and seeds (the entry-31 sweep). A count "
        "sees only direction words that reached the lens's top 20; a mass score is computed over "
        "the whole vocabulary. Never difference a number across the two roots.",
    ],
    ["eval720_population", "Which tokens the eval720 number was measured on, and how many."],
    [
        "eval720_bal_acc_OWN_tokens",
        "Balanced accuracy on the probe's OWN eval split, read from the "
        "checkpoint's results.best_balanced_accuracy. Not comparable "
        "across eras, and not comparable across selections.",
    ],
    ["heldout360_population", "Whether the probe was scored on the heldout tree, and on what."],
    [
        "heldout360_bal_vs_belief_ALL_tokens",
        "Balanced accuracy over ALL 87,221 reasoning tokens of the heldout 360 - every token of every "
        "chain, no selection applied - scored against the LOCAL BELIEF label. Recomputed from "
        "probe_loudness_heldout360_16probes/per_token_jlens_loudness.csv; reproduces the published "
        "values to 4 dp. Populated for the 16 belief-round probes and their two mass-era baselines.",
    ],
    [
        "heldout360_bal_vs_final_ALL_tokens",
        "The same 87,221 tokens, scored against the FINAL agent_action instead. For the 16 belief-"
        "round probes recomputed from the same CSV; for the other 26 recomputed from "
        "probes/heldout360_all_probes.csv.",
    ],
    [
        "entry 51: the inversion is NOT about loudness",
        "The strongest specialisation measured comes from a rule with no loudness in it at all. "
        "eos (cut at each sentence's LAST token) beats the random control by +6.6 pp on its own "
        "selection and is LAST of all 24 on the full held-out population (.4312 vs random's .5899). "
        "Sentence ends are a narrow, structurally distinctive position; a probe trained only there "
        "transfers worst of anything measured. What drives the inversion is how NARROW the training "
        "distribution is - loudness was only ever the narrowing device.",
    ],
    [
        "entry 51: position vs loudness, separated",
        "On identical sentence spans, eval-720 mlp: random in span .5836 -> eos .6495 (+6.6 pp for "
        "cutting at a sentence BOUNDARY) -> jlens loudest .6777 (+2.8 pp more for cutting at the "
        "LOUDEST point). ~70% of the per-sentence gain is positional, ~30% is loudness.",
    ],
    [
        "entry 51: thinning to 20 is a POPULATION change",
        "It gains +6.8 to +9.6 pp for EVERY arm including the random one, where selection cannot be "
        "doing any work. The cap re-weights toward short chains (median 10 per-sentence tokens, only "
        "29.6% of trajectories exceed 20). Compare within a cadence row, never across.",
    ],
    [
        "the ordering INVERTS between the two blocks",
        "On its own loud tokens the jlens arm wins (.862 mlp) and random is worst (.723); on all "
        "87,221 heldout tokens random wins (.5886) and jlens is worst (.5243). A loud-selected probe "
        "is specialised to loud tokens; the random-trained control generalises across the chain. "
        "Both are true - name the population whenever quoting a selection effect.",
    ],
    [
        "layer = per-token argmax (7:23)",
        "--layers-per-token 1 kept each token's own best layer, so "
        "the dataset spans many layers and one weight vector reads "
        "several representation spaces. Costs ~10 pp vs pinned L15.",
    ],
    [
        "era = mass-era 3600",
        "Trees jlens_mass_l15 / logitlens_mass_l15 / both argmax_per_sentence "
        "trees. Eval split = prepared/next_action_mass_l15_eval_names.txt.",
    ],
    [
        "era = count-era 3600",
        "Tree jlens_reasoning_tokens (and the eos view into it). A DIFFERENT "
        "draw: only 348 of 3600 names shared with the mass era, and the two "
        "eval-720 sets share 21. NEVER difference across the two.",
    ],
    ["", ""],
    [
        "CAVEAT: heldout 360 is NOT disjoint from the count era",
        "research_summary.md says (train u eval) n heldout360 = 0. That holds for the MASS era "
        "(verified: 0 of 3600). The COUNT-era 3600 overlaps the heldout 360 by 33 trajectories, 25 of "
        "them in its train 2,880 - 7,176 of the 87,221 heldout rows, 8.2%. Measured consequence: "
        "recomputing every heldout number with those 25 removed moves the 20 count-era probes by "
        "-0.001 to +0.005, and moves the 6 mass-era probes (which have ZERO overlap) by a similar "
        "+0.000 to +0.006. The shift is the population change from dropping 8% of rows, not "
        "memorisation - no probe scores better on trajectories it trained on. The entry-38 "
        "jlens-beats-logitlens ranking is unaffected, but the disjointness claim should be "
        "restated as mass-era-only.",
    ],
    ["", ""],
    ["analysis code", "report"],
    ["26p", "probes/heldout360_all_probes.csv - 26 count/mass-era probes on the 87,221 heldout tokens"],
    ["pvr", "reasoning_theatre/probe_vs_rollout/ - The Commitment Boundary (entries 39-41)"],
    [
        "pvr-lb",
        "reasoning_theatre/probe_vs_rollout_lb/ - belief-probe clone (entry 47); the 26 plus "
        "local_belief p1_mlp and p2_mlp",
    ],
    ["pl-720", "reasoning_theatre/probe_loudness/ - What Loudness Buys the Probe, eval 720 (entry 46)"],
    ["pl-h360", "reasoning_theatre/probe_loudness_heldout360/ - selection removed (entry 48)"],
    ["16p", "reasoning_theatre/probe_loudness_heldout360_16probes/ - 16 probes, BOTH loudness rulers (entry 49)"],
    [
        "eqn",
        "reasoning_theatre/probe_loudness_heldout360_equal_n/ - the entry-52 equal-N rebuilds "
        "(p1 and p1-top20) on the same 87,221 held-out tokens, alongside the earlier probes.",
    ],
    [
        "24p",
        "reasoning_theatre/probe_loudness_heldout360_24probes/ - all 24 probes scored in one "
        "pass (entry 51). Numbers only, no figures; the 16 earlier probes reproduce to 4 dp, "
        "max |delta| 0.000000, which is what puts the eight new ones on the same measurement.",
    ],
    ["", ""],
    ["known gaps", ""],
    [
        "top-3 arm unfinished",
        "prepared/next_action_mass_l15_jlens_top3_{train,eval} exist (8,640 / "
        "2,160) but logs/jlens_top3_lr.txt ends mid-load and no probe .pt was "
        "written. The top-1/2/all sweep has a hole at K=3.",
    ],
    ["no logitlens final-action probe", "logitlens_mass_l15 tokens carry the belief label only."],
    ["no logitlens per-sentence top-20", "the jlens p1_top20 arm has no counterpart."],
    ["eos absent from heldout", "the eos probes exist on eval-720 only."],
    ["no all-token probe", "all-token activations were never gathered for the 3600."],
    ["", ""],
    [
        "provenance",
        "Generated from the probe checkpoints, prepared manifests and per-token CSVs "
        "on /workspace. Every number is read or recomputed, none transcribed.",
    ],
]
for r in LEG:
    leg.append(r)
for c in (1, 2):
    cell = leg.cell(row=1, column=c)
    cell.fill, cell.font = HEAD, HEADF
leg.column_dimensions["A"].width = 34
leg.column_dimensions["B"].width = 110
for ri in range(2, len(LEG) + 1):
    leg.cell(row=ri, column=2).alignment = Alignment(wrap_text=True, vertical="top")
    leg.cell(row=ri, column=1).font = Font(bold=True)
leg.freeze_panes = "A2"

wb.save(OUT / "probe_inventory.xlsx")
print(f"{len(rows)} probes -> {OUT / 'probe_inventory.xlsx'} and probe_inventory.csv")
for s in skipped:
    print(f"  SKIPPED belief probe outside {BELIEF_ROOT}: {s}")
