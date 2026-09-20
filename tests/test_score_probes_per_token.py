"""The merged evaluator must reproduce, value for value, the two scripts it replaces.

`score_probes_per_token.py --probe-type next_action` replaces `scripts/eval_probe_per_token.py`
and `--probe-type grid_tile` replaces `grid_cell_analysis/eval_grid_probe_per_token.py`. The
golden CSVs under `tests/data/score_probes_per_token/` were written BY those two scripts on
this exact fixture, before they were deleted -- so the comparison is against what the
originals actually produced, not against a remembered schema.

The HEADER is expected to differ, and only in the lens score columns: those were
`{lens}_mass_L15`, which names neither the vocabulary the mass was taken over nor -- for the
grid evaluator, deliberately pointed at the DIRECTION vocabulary -- the fact that the ruler
and the label disagree. They are now `{lens}_{signal}_logmass_L15`. Every VALUE must be
identical, and that is what is asserted here.

Everything is built under `tmp_path`; nothing reads /workspace.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
# Produced by the two originals on this fixture, before they were deleted. Not derived
# from the merged script, which is what makes the comparison meaningful.
GOLDEN = Path(__file__).resolve().parent / "data" / "score_probes_per_token"
MODEL = "openai__gpt-oss-20b"
LAYER = 15
DIM = 8
N_TOKENS = 6
PROMPT_LEN = 3

# The vocabulary. Two of the six decoded tokens are signal words, so the count score varies
# across tokens rather than being constant (which would hide a mis-join).
SIGNAL = {"UP": [" up"], "DOWN": [" down"], "LEFT": [" left"], "RIGHT": [" right"]}
TOKENS = [" the", " up", " agent", " left", " moves", " now"]


def _trajectory() -> dict:
    """One trajectory, one step, six output tokens, a 3x3 grid."""
    return {
        "grid_params": {"grid_width": 3, "grid_complexity": 1.0},
        "model_params": {"model": MODEL},
        # output_start() reads the prefix/suffix off the TRAJECTORY prompt and the
        # grid_state_tokens off the STEP: 2 + 1 + 0 = 3 = PROMPT_LEN.
        "prompt": {"prompt_prefix_tokens": [1, 2], "prompt_suffix_tokens": []},
        "steps": [
            {
                "step_id": 0,
                # grid_state[0] is the COLUMN HEADER; later lines are "row_id cell cell ...".
                # 9 cells over 4 classes, so per-class counts are non-trivial.
                "grid_state": ["  0 1 2 ", "0 A _ # ", "1 _ _ _ ", "2 # _ G "],
                "grid_state_tokens": [3],
                "output_tokens": list(range(10, 10 + N_TOKENS)),
                "agent_action": "UP",
            }
        ],
    }


def _write_lens_tree(root: Path, stem: str) -> None:
    folder = root / "size3" / stem
    folder.mkdir(parents=True)

    # Analysis CSV: one row per (token, layer). Only layer 15 and 7, enough for a "best layer".
    with open(folder / f"{stem}_jlens_analysis.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "abs_pos", "token", "layer"] + [f"top_{i}" for i in range(1, 21)])
        for i, tok in enumerate(TOKENS):
            for layer in (7, LAYER):
                top = [tok] + [" x"] * 19
                w.writerow([0, PROMPT_LEN + i, tok, layer] + top)

    # Mass table: wide (token x layer).
    with open(folder / f"{stem}_jlens_direction_mass.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "abs_pos", "token", "L7", f"L{LAYER}"])
        for i, tok in enumerate(TOKENS):
            w.writerow([0, PROMPT_LEN + i, tok, -3.0 - i, -1.0 - i * 0.5])
    (folder / f"{stem}_jlens_direction_mass.csv.meta.json").write_text(
        json.dumps({"signal_json": "direction_tokens_full.json", "signal_name": "direction"})
    )


def _write_activations(root: Path, stem: str) -> None:
    out = root / "size3" / stem / MODEL / f"layer_{LAYER}" / "step_0" / "output"
    out.mkdir(parents=True)
    for i in range(N_TOKENS):
        torch.save(torch.arange(DIM, dtype=torch.float32) + i, out / f"{i}.pt")


def _save_next_action_probe(path: Path) -> None:
    from telos_interp.commands.train_next_action_probe.train_next_action_probe_fn import NextActionProbe
    from telos_interp.probe_models import LogisticRegressionProbe

    torch.manual_seed(0)
    model = LogisticRegressionProbe(DIM, 4)
    probe = NextActionProbe(
        model=model,
        model_type="lr",
        input_dim=DIM,
        label_to_idx={0: 0, 1: 1, 2: 2, 3: 3},
        idx_to_label={0: 0, 1: 1, 2: 2, 3: 3},
        device="cpu",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(path)


def _save_grid_probe(path: Path) -> None:
    from telos_interp.commands.train_cognitive_map_probe.train_cognitive_map_probe_fn import CognitiveMapProbe
    from telos_interp.probe_models import LogisticRegressionProbe

    torch.manual_seed(0)
    model = LogisticRegressionProbe(DIM + 2, 8)
    probe = CognitiveMapProbe(
        model=model,
        model_type="lr",
        input_dim=DIM + 2,
        label_to_idx={c: c for c in range(8)},
        idx_to_label={c: c for c in range(8)},
        device="cpu",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(path)


@pytest.fixture
def tree(tmp_path):
    stem = "together_ai_openai_gpt-oss-20b_size3_comp1.0_1"
    lens = tmp_path / "lens"
    acts = tmp_path / "acts"
    trajs = tmp_path / "trajs"
    _write_lens_tree(lens, stem)
    _write_activations(acts, stem)
    (trajs / "size3").mkdir(parents=True)
    (trajs / "size3" / f"{stem}.json").write_text(json.dumps(_trajectory()))
    signal = tmp_path / "direction_tokens_full.json"
    signal.write_text(json.dumps(SIGNAL))
    return {"stem": stem, "lens": lens, "acts": acts, "trajs": trajs, "signal": signal, "tmp": tmp_path}


def _run(script: str, out: Path, tree, *extra) -> Path:
    cmd = [
        sys.executable,
        str(REPO / script),
        "--activations-dir",
        str(tree["acts"]),
        "--lens-dir",
        str(tree["lens"]),
        "--trajectories-dir",
        str(tree["trajs"]),
        "--signal-json",
        str(tree["signal"]),
        "--layer",
        str(LAYER),
        "--out",
        str(out),
        *extra,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout}\n{proc.stderr}"
    assert out.exists(), f"{script} wrote no CSV:\n{proc.stdout}"
    return out


def _read(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _assert_same_values(old_path: Path, new_path: Path, renamed: dict[str, str]) -> None:
    """Every column of `old` appears in `new` (under its new name) with identical values."""
    old_fields, old_rows = _read(old_path)
    new_fields, new_rows = _read(new_path)

    assert old_rows, "the fixture produced no rows, so this proves nothing"
    assert len(old_rows) == len(new_rows), f"{len(old_rows)} old rows vs {len(new_rows)} new"

    for field in old_fields:
        target = renamed.get(field, field)
        assert target in new_fields, f"column {field!r} -> {target!r} missing from the merged output"
        old_vals = [r[field] for r in old_rows]
        new_vals = [r[target] for r in new_rows]
        assert old_vals == new_vals, f"column {field!r} -> {target!r} differs:\n  old={old_vals}\n  new={new_vals}"


def test_next_action_matches_eval_probe_per_token(tree):
    probe = tree["tmp"] / "probes" / "next_action_probe_arm_lr.pt"
    _save_next_action_probe(probe)

    old = GOLDEN / "next_action.csv"
    new = _run(
        "telos_interp/loudness_analysis/score_probes_per_token.py",
        tree["tmp"] / "new.csv",
        tree,
        "--probe",
        str(probe),
        "--full-probs",
        "--probe-type",
        "next_action",
    )

    renamed = {
        "jlens_count": "jlens_direction_count",
        f"jlens_mass_L{LAYER}": f"jlens_direction_logmass_L{LAYER}",
        "jlens_mass_best_layer": "jlens_direction_logmass_best_layer",
        "jlens_mass_best": "jlens_direction_logmass_best",
        "logitlens_count": "logitlens_direction_count",
        f"logitlens_mass_L{LAYER}": f"logitlens_direction_logmass_L{LAYER}",
        "logitlens_mass_best_layer": "logitlens_direction_logmass_best_layer",
        "logitlens_mass_best": "logitlens_direction_logmass_best",
    }
    _assert_same_values(old, new, renamed)


def test_grid_tile_matches_eval_grid_probe_per_token(tree):
    probe = tree["tmp"] / "probes" / "grid_probe_arm_lr.pt"
    _save_grid_probe(probe)

    old = GOLDEN / "grid_tile.csv"
    new = _run(
        "telos_interp/loudness_analysis/score_probes_per_token.py",
        tree["tmp"] / "new_grid.csv",
        tree,
        "--probe",
        str(probe),
        "--probe-type",
        "grid_tile",
    )

    renamed = {
        "jlens_count": "jlens_direction_count",
        f"jlens_mass_L{LAYER}": f"jlens_direction_logmass_L{LAYER}",
        "jlens_mass_best_layer": "jlens_direction_logmass_best_layer",
        "jlens_mass_best": "jlens_direction_logmass_best",
        "logitlens_count": "logitlens_direction_count",
        f"logitlens_mass_L{LAYER}": f"logitlens_direction_logmass_L{LAYER}",
        "logitlens_mass_best_layer": "logitlens_direction_logmass_best_layer",
        "logitlens_mass_best": "logitlens_direction_logmass_best",
    }
    _assert_same_values(old, new, renamed)


def test_run_config_records_the_ruler_and_the_vocabulary(tree):
    probe = tree["tmp"] / "probes" / "next_action_probe_arm_lr.pt"
    _save_next_action_probe(probe)
    out = tree["tmp"] / "cfg" / "per_token.csv"
    _run(
        "telos_interp/loudness_analysis/score_probes_per_token.py",
        out,
        tree,
        "--probe",
        str(probe),
        "--probe-type",
        "next_action",
    )
    cfg = json.loads((out.parent / "run_config.json").read_text())
    assert cfg["measurement"]["signal"] == "direction"
    assert cfg["measurement"]["loudness_column"] == f"jlens_direction_logmass_L{LAYER}"
    # The fingerprint is what actually decides comparability between two tables.
    assert len(cfg["measurement"]["signal_fingerprint"]) == 12
    assert cfg["row_counts"]["token_rows"] == N_TOKENS


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_cache_and_threads_do_not_change_the_rows(tree):
    """The fetch strategy is not allowed to be visible in the output.

    Three runs over one fixture: the serial read, the threaded read, and a threaded read that
    packs a cache -- then a fourth that must be served BY that cache. All four CSVs must be
    byte-identical. This is the whole safety argument for --cache-activations: it changes how
    tensors are fetched and nothing else, so a cached table and an uncached one are the same
    measurement and can sit beside each other.
    """
    probe = tree["tmp"] / "probes" / "next_action_probe_arm_lr.pt"
    _save_next_action_probe(probe)
    common = ("--probe", str(probe), "--full-probs", "--probe-type", "next_action")
    cache = tree["tmp"] / "cache"

    serial = _run("telos_interp/loudness_analysis/score_probes_per_token.py",
                  tree["tmp"] / "serial.csv", tree, *common, "--read-threads", "1")
    threaded = _run("telos_interp/loudness_analysis/score_probes_per_token.py",
                    tree["tmp"] / "threaded.csv", tree, *common, "--read-threads", "4")
    fill = _run("telos_interp/loudness_analysis/score_probes_per_token.py",
                tree["tmp"] / "fill.csv", tree, *common, "--cache-activations",
                "--cache-dir", str(cache))
    assert list(cache.glob("*.pt")), "the run did not write a cache"
    hit = _run("telos_interp/loudness_analysis/score_probes_per_token.py",
               tree["tmp"] / "hit.csv", tree, *common, "--cache-activations",
               "--cache-dir", str(cache))

    baseline = serial.read_text()
    for other in (threaded, fill, hit):
        assert other.read_text() == baseline, f"{other.name} differs from the serial read"


def test_a_stale_cache_is_rebuilt_not_trusted(tree):
    """A cache built for a different key list must not be served.

    The tree it was packed from can be pruned or extended afterwards, and the lens tables can
    select a different universe of tokens. Either way the packed file no longer answers the
    question being asked, so it is rebuilt rather than returned.
    """
    from telos_interp.loudness_analysis.score_probes_per_token import _cache_file, load_activations

    act_folder = tree["acts"] / "size3" / tree["stem"] / MODEL
    cache = tree["tmp"] / "stale_cache"
    keys = [(0, i) for i in range(N_TOKENS)]

    full = load_activations(act_folder, LAYER, keys, tree["stem"], cache, threads=2)
    assert len(full) == N_TOKENS
    assert _cache_file(cache, tree["stem"], LAYER).exists()

    # A different request over the same trajectory: the cache holds the wrong key list.
    fewer = load_activations(act_folder, LAYER, keys[:2], tree["stem"], cache, threads=1)
    assert list(fewer) == keys[:2]
    for k in keys[:2]:
        assert torch.equal(fewer[k], full[k])

    # A key that is not on disk is reported missing rather than invented.
    missing = load_activations(act_folder, LAYER, [(0, 999)], tree["stem"], None, threads=1)
    assert missing == {}
