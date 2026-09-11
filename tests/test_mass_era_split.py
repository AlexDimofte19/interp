"""Tests for scripts/build_mass_era_split.sh and scripts/verify_mass_era_split.py.

The split materialises a partition that already existed as a flag. Its whole value is that it
cannot drift from that partition, so the tests are about the ways it could drift silently:

  * the train half is the COMPLEMENT of the pinned eval list, never a fresh draw -- a
    re-drawn eval set would invalidate every probe number on disk,
  * the two views are closed: each holds exactly its own names, and a name belonging to the
    other half is not reachable through it,
  * a view carries the gather's per-trajectory artifacts (analysis CSV, direction-mass table,
    its `.meta.json`) alongside the tensors, because a view that lost the mass table would
    score against a different vocabulary without saying so,
  * the verifier FAILS when any of that is untrue -- a checker that always passes is worse
    than no checker.

Everything runs against a miniature tree built in `tmp_path`; nothing here touches /workspace.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "scripts/build_mass_era_split.sh"
VERIFY = REPO / "scripts/verify_mass_era_split.py"

SIZES = ("size5", "size11")
LENS_ARTIFACTS = (
    "_jlens_analysis.csv",
    "_jlens_direction_mass.csv",
    "_jlens_direction_mass.csv.meta.json",
)


def name(size: str, i: int) -> str:
    return f"together_ai_openai_gpt-oss-20b_{size}_comp0.0_{i}"


def build_fixture(root: Path, n_per_size: int = 10, n_eval_per_size: int = 2) -> dict:
    """A miniature stand-in for the mass-era world: a source tree, trajectories, name lists."""
    tree = root / "activations/jlens_mass_l15"
    traj = root / "trajectories/reveng/trajectories_train_single_step"
    names: list[str] = []
    for size in SIZES:
        for i in range(n_per_size):
            nm = name(size, i)
            names.append(nm)
            d = tree / size / nm
            (d / "openai__gpt-oss-20b/layer_15/step_0/output").mkdir(parents=True)
            (d / "openai__gpt-oss-20b/layer_15/step_0/output/0.pt").write_bytes(b"tensor")
            for suffix in LENS_ARTIFACTS:
                (d / f"{nm}{suffix}").write_text("{}" if suffix.endswith(".json") else "token,layer\n")
            (traj / size).mkdir(parents=True, exist_ok=True)
            (traj / size / f"{nm}.json").write_text("{}")

    ev = [name(size, i) for size in SIZES for i in range(n_eval_per_size)]
    splits, prepared, rt = root / "splits", root / "prepared", root / "rt"
    for d in (splits, prepared, rt / "rollout_strategies"):
        d.mkdir(parents=True, exist_ok=True)
    (rt / "rollout_strategies/mass_l15_names.txt").write_text("\n".join(names) + "\n")
    (prepared / "next_action_mass_l15_eval_names.txt").write_text("\n".join(ev) + "\n")
    return {
        "root": root,
        "tree": tree,
        "traj": traj,
        "all": set(names),
        "eval": set(ev),
        "train": set(names) - set(ev),
    }


def env_for(fx: dict) -> dict:
    root = fx["root"]
    return {
        **os.environ,
        "WS": str(root),
        "ACT": str(root / "activations"),
        "SPLITS": str(root / "splits"),
        "PREPARED": str(root / "prepared"),
        "RT": str(root / "rt"),
        "TRAJ": str(fx["traj"]),
        "SRC_TREE": str(fx["tree"]),
        "MASS_NAMES": str(root / "rt/rollout_strategies/mass_l15_names.txt"),
        "MASS_EVAL_NAMES": str(root / "prepared/next_action_mass_l15_eval_names.txt"),
        "TRAIN_NAMES": str(root / "splits/mass_train_2880.txt"),
        "EVAL_NAMES": str(root / "splits/mass_eval_720.txt"),
        "TRAIN_VIEW": str(root / "activations/mass_train2880_view"),
        "EVAL_VIEW": str(root / "activations/mass_eval720_view"),
        "HELDOUT_NAMES": str(root / "does-not-exist.txt"),
    }


def run_build(fx: dict, **extra) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(BUILD)], env={**env_for(fx), **extra}, capture_output=True, text=True, check=False
    )


def run_verify(fx: dict, **extra) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VERIFY)], env={**env_for(fx), **extra}, capture_output=True, text=True, check=False
    )


def read_names(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


@pytest.fixture
def built(tmp_path):
    fx = build_fixture(tmp_path)
    proc = run_build(fx)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    fx["build"] = proc
    return fx


def test_dry_run_writes_nothing(tmp_path):
    fx = build_fixture(tmp_path)
    proc = run_build(fx, DRY_RUN="1")
    assert proc.returncode == 0, proc.stderr
    assert "dry run" in proc.stdout
    assert not (fx["root"] / "splits/mass_train_2880.txt").exists()
    assert not (fx["root"] / "activations/mass_train2880_view").exists()


def test_train_is_the_complement_of_the_pinned_eval_list(built):
    root = built["root"]
    train = read_names(root / "splits/mass_train_2880.txt")
    ev = read_names(root / "splits/mass_eval_720.txt")
    assert ev == built["eval"], "the eval half must be the pinned list, unchanged"
    assert train == built["train"]
    assert not (train & ev)
    assert train | ev == built["all"]


def test_views_hold_exactly_their_own_half(built):
    root = built["root"]
    for view, want in (("mass_train2880_view", built["train"]), ("mass_eval720_view", built["eval"])):
        acts = {p.name for p in (root / "activations" / view / "activations").glob("size*/*")}
        trajs = {p.stem for p in (root / "activations" / view / "trajectories").glob("size*/*.json")}
        assert acts == want
        assert trajs == want


def test_a_view_cannot_reach_the_other_half(built):
    """The point of the split: no path from one view to a trajectory of the other."""
    root = built["root"]
    train_view = root / "activations/mass_train2880_view/activations"
    reachable = {p.resolve().name for p in train_view.glob("size*/*")}
    assert not (reachable & built["eval"])


def test_views_carry_the_lens_artifacts_and_the_tensors(built):
    root = built["root"]
    for p in (root / "activations/mass_eval720_view/activations").glob("size*/*"):
        for suffix in LENS_ARTIFACTS:
            assert (p / f"{p.name}{suffix}").exists(), f"{p.name}{suffix} not reachable through the view"
        assert (p / "openai__gpt-oss-20b/layer_15/step_0/output/0.pt").exists()


def test_rerun_is_idempotent(built):
    root = root_of = built["root"]
    before = sorted(str(p) for p in (root / "activations").rglob("*"))
    train_before = (root_of / "splits/mass_train_2880.txt").read_text()
    proc = run_build(built)
    assert proc.returncode == 0, proc.stderr
    assert "already complete" in proc.stdout
    assert sorted(str(p) for p in (root / "activations").rglob("*")) == before
    assert (root_of / "splits/mass_train_2880.txt").read_text() == train_before


def test_verifier_passes_on_a_consistent_split(built):
    proc = run_verify(built)
    assert proc.returncode == 0, proc.stdout
    out = proc.stdout
    assert "all checks passed" in out
    assert "train and eval are disjoint" in out
    assert "no dangling symlinks" in out
    # 20 trajectories is not the canonical 3,600, so that claim is skipped rather than failed.
    assert "skipping the 2880/720 check" in out


def test_verifier_fails_on_a_dangling_symlink(built):
    """A dangling link is an EMPTY directory to the prepare step, not an error -- so the
    verifier is the only thing standing between that and a silently short dataset."""
    root = built["root"]
    victim = next((root / "activations/mass_eval720_view/activations").glob("size*/*"))
    victim.resolve().rename(victim.resolve().parent / "moved-away")
    proc = run_verify(built)
    assert proc.returncode == 1
    assert "dangling" in proc.stdout


def test_verifier_fails_when_the_eval_list_drifts(built):
    """Re-drawing the eval half is the one change that would invalidate every published
    number, and it would otherwise leave no trace."""
    root = built["root"]
    ev = sorted(read_names(root / "splits/mass_eval_720.txt"))
    swapped = sorted(read_names(root / "splits/mass_train_2880.txt"))[0]
    (root / "splits/mass_eval_720.txt").write_text("\n".join(ev[1:] + [swapped]) + "\n")
    proc = run_verify(built)
    assert proc.returncode == 1
    assert "SAME 720" in proc.stdout


def test_verifier_fails_when_the_halves_overlap(built):
    root = built["root"]
    train = sorted(read_names(root / "splits/mass_train_2880.txt"))
    leaked = sorted(read_names(root / "splits/mass_eval_720.txt"))[0]
    (root / "splits/mass_train_2880.txt").write_text("\n".join(train + [leaked]) + "\n")
    proc = run_verify(built)
    assert proc.returncode == 1
    assert "disjoint" in proc.stdout


def test_verifier_reports_the_historical_train_definition(tmp_path):
    """`audit_trajectory_sets.py` used to recover "train 2880" from one arm's prepared
    manifest. Materialising the set must not redefine it, so the verifier checks against that
    manifest whenever it is present."""
    fx = build_fixture(tmp_path)
    manifest_dir = fx["root"] / "prepared/local_belief_p2_split_train"
    manifest_dir.mkdir(parents=True)
    # Deliberately wrong by one name: the check has to notice.
    wrong = sorted(fx["train"])[1:]
    (manifest_dir / "manifest.json").write_text(json.dumps({"samples": [{"name": n} for n in wrong]}))
    run_build(fx)
    proc = run_verify(fx)
    assert proc.returncode == 1
    assert "SAME 2880" in proc.stdout
