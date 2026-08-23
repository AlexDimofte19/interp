"""End-to-end tests for scripts/jlens_reasoning_tokens.py against a stub model.

The stub model, the `env` / `signal_json` fixtures and the `_run` / `_select_args` helpers
live in tests/conftest.py, shared with tests/test_delete_non_jlens_selected.py.

Two things are pinned here. First, that packing steps into one padded forward pass never
moves a value -- the stub's activations depend only on (layer, token id, position), so a
batched run and an unbatched run must agree *exactly* and any disagreement is an offset
bug rather than float noise. Second, that --signal-json writes exactly the selection the
filter chose, and nothing else.
"""

import csv
import importlib
import json

import pytest
import torch
from tests.conftest import (
    DIM,
    SCORABLE_LAYERS,
    _run,
    _select_args,
    _StubModel,
    activation_value,
    jrt,
)


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_batched_and_unbatched_runs_agree_exactly(env):
    """Packing steps into one padded forward must not move a single value."""
    single = _run(env, "single", "--forward-batch-size", "1")
    batched = _run(env, "batched", "--forward-batch-size", "4")

    stem = env["stem"]
    assert (single / f"{stem}_jlens_analysis.csv").read_bytes() == (
        batched / f"{stem}_jlens_analysis.csv"
    ).read_bytes()

    single_pts = sorted(p.relative_to(single) for p in single.rglob("*.pt"))
    batched_pts = sorted(p.relative_to(batched) for p in batched.rglob("*.pt"))
    assert single_pts == batched_pts
    assert single_pts
    for rel in single_pts:
        a = torch.load(single / rel, map_location="cpu", weights_only=True)
        b = torch.load(batched / rel, map_location="cpu", weights_only=True)
        assert torch.equal(a, b), rel


def test_saved_activations_are_the_right_token(env):
    """Each .pt must hold its own token's residual stream, not a neighbour's.

    Also pins the layout: layer_{N}/step_{M}/output/{output-relative index}.pt.
    """
    out = _run(env, "acts", "--forward-batch-size", "4")
    model_dir = out / "stub__model"
    trajectory = json.loads(env["traj_path"].read_text())
    n_prefix, n_suffix = 2, 1

    for step_id, step in enumerate(trajectory["steps"]):
        n_analysis = sum(1 for t in step["output_tokens"] if "analysis" in t["token_groups"])
        output_start = n_prefix + len(step["grid_state_tokens"]) + n_suffix
        for layer in range(19, 24):
            folder = model_dir / f"layer_{layer}" / f"step_{step_id}" / "output"
            saved = sorted(int(p.stem) for p in folder.glob("*.pt"))
            assert saved == list(range(n_analysis)), (layer, step_id, saved)
            for out_idx in saved:
                tensor = torch.load(folder / f"{out_idx}.pt", weights_only=True)
                token_id = step["output_tokens"][out_idx]["token_id"]
                expected = activation_value(layer, token_id, output_start + out_idx)
                assert torch.equal(tensor, expected), (layer, step_id, out_idx)
                # a view saved by mistake would drag its whole storage along
                assert tensor.numel() == DIM


def test_csv_covers_lens_layers_only_and_keeps_row_order(env):
    """Layer 19 has no jlens matrix: activations yes, CSV rows no."""
    out = _run(env, "rows", "--forward-batch-size", "4")
    rows = _read_csv(out / f"{env['stem']}_jlens_analysis.csv")

    assert {r["layer"] for r in rows} == {"20", "21", "22", "23"}
    assert (out / "stub__model" / "layer_19" / "step_0" / "output" / "0.pt").exists()

    # rows are grouped by step (ascending), then layer-major within a step, then by
    # position in the reasoning chain - the order the pre-batching script produced
    seen = []
    for row in rows:
        key = (int(row["step"]), int(row["layer"]))
        if not seen or seen[-1] != key:
            assert key not in seen, f"rows for {key} are not contiguous"
            seen.append(key)
    assert seen == sorted(seen)

    step0 = [r for r in rows if r["step"] == "0" and r["layer"] == "20"]
    assert [int(r["reasoning_pos"]) for r in step0] == [0, 1, 2, 3]
    assert {r["agent_action"] for r in step0} == {"UP"}
    assert all(len(r[f"top_{i}"]) > 0 for r in step0 for i in range(1, 21))


def test_lens_chunking_does_not_change_the_csv(env):
    """--batch-size only bounds the logits; it must not alter a single row."""
    whole = _run(env, "chunk_off", "--batch-size", "0")
    chunked = _run(env, "chunk_on", "--batch-size", "2")
    stem = env["stem"]
    assert (whole / f"{stem}_jlens_analysis.csv").read_bytes() == (
        chunked / f"{stem}_jlens_analysis.csv"
    ).read_bytes()


def test_no_save_activations_writes_only_the_csv(env):
    out = _run(env, "csv_only", "--no-save-activations")
    assert (out / f"{env['stem']}_jlens_analysis.csv").exists()
    assert list(out.rglob("*.pt")) == []


def _reference_run(env, out_name):
    """The pre-batching inner loop, kept verbatim as a golden reference.

    Transcribed from the commit before the speedups (one forward per step, one lens
    matmul per layer, one `tok.decode` per top-k id, serial `.pt` writes). Everything the
    optimisation touched is supposed to be semantics-preserving, and this is what
    "preserved" is measured against.
    """
    from telos_interp.commands.gather_activations.gather_activations_utils import (
        extract_activations_single_pass,
        save_activations_to_files,
    )

    sampled = importlib.import_module("scripts.jlens_action_ranks_sampled")
    assets = sampled.ensure_unembed_assets(env["jlens_dir"])
    ids, tok = sampled.action_token_ids()
    id_cols = [ids[a] for a in jrt.ACTIONS]
    lens = torch.load(env["jlens_dir"] / "gpt-oss-20b_jacobian_lens.pt", map_location="cpu")
    lm_head, norm_w, eps = assets["lm_head"], assets["norm_weight"].float(), assets["rms_eps"]

    model = _StubModel()
    model.eval()
    layer_indices = list(range(19, 24))
    trajectory = json.loads(env["traj_path"].read_text())
    name = jrt.parse_name(env["stem"])
    n_prefix = len(trajectory["prompt"]["prompt_prefix_tokens"])
    n_suffix = len(trajectory["prompt"]["prompt_suffix_tokens"])

    traj_dir = env["tmp"] / out_name / "size5" / env["stem"]
    traj_dir.mkdir(parents=True)
    output_base = traj_dir / "stub__model"
    header = (
        ["size", "complexity", "run", "step", "reasoning_pos", "abs_pos", "token", "layer",
         "agent_action"]
        + [f"{a}_rank" for a in jrt.ACTIONS]
        + [f"{a}_logprob" for a in jrt.ACTIONS]
        + [f"top_{i}" for i in range(1, jrt.TOP_K + 1)]
    )

    with open(traj_dir / f"{env['stem']}_jlens_analysis.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for si, step in enumerate(trajectory["steps"]):
            positions = jrt.reasoning_token_positions(trajectory, step)
            abs_positions = [p[1] for p in positions]
            all_ids = (
                [t["token_id"] for t in trajectory["prompt"]["prompt_prefix_tokens"]]
                + [t["token_id"] for t in step["grid_state_tokens"]]
                + [t["token_id"] for t in trajectory["prompt"]["prompt_suffix_tokens"]]
                + [t["token_id"] for t in step["output_tokens"]]
            )
            input_ids = torch.tensor([all_ids[: max(abs_positions) + 1]])
            acts = extract_activations_single_pass(model, input_ids, abs_positions, layer_indices)
            output_start = n_prefix + len(step["grid_state_tokens"]) + n_suffix
            remapped = {
                layer: {ap - output_start: acts[layer][ap] for ap in abs_positions if ap in acts[layer]}
                for layer in layer_indices
            }
            save_activations_to_files(remapped, output_base, step_idx=step["step_id"], category="output")

            agent_action = step.get("agent_action", "")
            for layer in layer_indices:
                if layer == jrt.TARGET_LAYER:
                    J = None
                elif layer in lens["J"]:
                    J = lens["J"][layer].float()
                else:
                    continue
                h = torch.stack([acts[layer][ap] for _, ap, _ in positions]).float()
                with torch.no_grad():
                    if J is not None:
                        h = h @ J.T
                    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w
                    logits = (h.to(lm_head.dtype) @ lm_head.T).float()
                    ranks = torch.stack([(logits > logits[:, t : t + 1]).sum(1) for t in id_cols], dim=1)
                    own = torch.stack([logits[:, t] for t in id_cols], dim=1)
                    logprobs = own - logits.logsumexp(-1, keepdim=True)
                    topk = logits.topk(jrt.TOP_K, dim=1).indices
                ranks, logprobs, topk = ranks.tolist(), logprobs.tolist(), topk.tolist()
                for (rp, ap, token), r, lp, tk in zip(positions, ranks, logprobs, topk, strict=True):
                    writer.writerow(
                        [name["size"], name["comp"], name["run"], si, rp, ap, token, layer, agent_action]
                        + r
                        + [round(x, 4) for x in lp]
                        + [tok.decode([i]) for i in tk]
                    )
    return traj_dir


def test_matches_the_pre_batching_implementation(env):
    """CSV byte-for-byte and every .pt tensor, against the loop this replaced."""
    reference = _reference_run(env, "reference")
    current = _run(env, "current", "--forward-batch-size", "4", "--batch-size", "2")

    stem = env["stem"]
    assert (current / f"{stem}_jlens_analysis.csv").read_bytes() == (
        reference / f"{stem}_jlens_analysis.csv"
    ).read_bytes()

    ref_pts = sorted(p.relative_to(reference) for p in reference.rglob("*.pt"))
    cur_pts = sorted(p.relative_to(current) for p in current.rglob("*.pt"))
    assert ref_pts == cur_pts
    assert len(ref_pts) == 55
    for rel in ref_pts:
        a = torch.load(reference / rel, map_location="cpu", weights_only=True)
        b = torch.load(current / rel, map_location="cpu", weights_only=True)
        assert torch.equal(a, b), rel


def test_crashed_run_leaves_no_csv(env, monkeypatch):
    """The done-marker contract: a failure must not leave a CSV that looks complete."""
    def boom(*_a, **_k):
        raise RuntimeError("forward exploded")

    monkeypatch.setattr(jrt, "group_consecutive", boom)
    with pytest.raises(RuntimeError):
        _run(env, "crashed")
    assert list((env["tmp"] / "crashed").rglob("*.csv")) == []


# --- selective gathering ----------------------------------------------------------------
#
# The point of --signal-json is that a filtered gather and a full gather followed by
# pruning must land on the same files. These tests pin the first half of that; the
# equivalence itself is tests/test_delete_non_jlens_selected.py.



def _expected_selection(env, full_run_dir, signal_json, **overrides):
    """What the filter selects, computed from an unfiltered run's CSV."""
    from telos_interp.jlens_utils import jlens_top_filter, to_disk_coords

    kwargs = {
        "num_tokens": 2,
        "num_layers": 2,
        "always_layers": (),
        "random_tokens": 1,
        "seed": 42,
        "seed_key": env["stem"],
        "candidate_layers": SCORABLE_LAYERS,
        "top_k": jrt.TOP_K,
    }
    kwargs.update(overrides)
    kept = jlens_top_filter(signal_json, full_run_dir / f"{env['stem']}_jlens_analysis.csv", **kwargs)
    return to_disk_coords(kept, json.loads(env["traj_path"].read_text()))


def test_selective_run_saves_exactly_the_selection(env, signal_json):
    """Only the filter's picks survive, and each holds the value a full run would have."""
    full = _run(env, "full")
    picked = _run(env, "picked", *_select_args(signal_json))

    assert (picked / f"{env['stem']}_jlens_analysis.csv").read_bytes() == (
        full / f"{env['stem']}_jlens_analysis.csv"
    ).read_bytes(), "the CSV is the full sweep either way; only the .pt tree is filtered"

    expected = _expected_selection(env, full, signal_json)
    model_dir = picked / "stub__model"
    assert {p.resolve() for p in picked.rglob("*.pt")} == {
        p.resolve() for p in expected.activation_paths(model_dir)
    }

    # a second forward pass must reproduce the first one exactly, not merely closely
    for rel in sorted(p.relative_to(picked) for p in picked.rglob("*.pt")):
        assert torch.equal(
            torch.load(picked / rel, weights_only=True),
            torch.load(full / rel, weights_only=True),
        ), rel


def test_selective_run_is_a_large_saving(env, signal_json):
    full = _run(env, "full")
    picked = _run(env, "picked", *_select_args(signal_json))
    assert len(list(picked.rglob("*.pt"))) < len(list(full.rglob("*.pt"))) / 3


def test_selection_record_is_written_and_reloadable(env, signal_json):
    from telos_interp.jlens_utils import read_selection_record, record_path

    full = _run(env, "full")
    picked = _run(env, "picked", *_select_args(signal_json))

    kept, config = read_selection_record(record_path(picked))
    assert kept == _expected_selection(env, full, signal_json)
    assert config["num_tokens"] == 2 and config["candidate_layers"] == SCORABLE_LAYERS
    assert len(kept.jlens) == 2 and len(kept.random) == 1


def test_control_arm_carries_no_counts(env, signal_json):
    """What lets split_next_action_manifest tell a control apart from a ranked selection."""
    from telos_interp.jlens_utils import read_selection_record, record_path

    picked = _run(env, "picked", *_select_args(signal_json, **{"random-tokens": 2}))
    kept, _ = read_selection_record(record_path(picked))
    assert all(p.direction_count is not None for p in kept.jlens.values())
    assert all(p.direction_count is None for p in kept.random.values())


def test_always_layers_are_forced_into_every_pick(env, signal_json):
    from telos_interp.jlens_utils import read_selection_record, record_path

    picked = _run(env, "picked", *_select_args(signal_json, **{"always-layers": "20", "num-layers": 1}))
    kept, _ = read_selection_record(record_path(picked))
    for pick in list(kept.jlens.values()) + list(kept.random.values()):
        assert 20 in pick.layers, pick


def test_unscorable_layers_are_never_selected(env, signal_json):
    """Layer 19 has no jlens matrix, so forcing it is a no-op rather than an error."""
    from telos_interp.jlens_utils import read_selection_record, record_path

    picked = _run(env, "picked", *_select_args(signal_json, **{"always-layers": "19", "num-layers": 1}))
    kept, config = read_selection_record(record_path(picked))
    assert 19 not in config["candidate_layers"]
    assert all(19 not in pick.layers for pick in kept.jlens.values())
    assert not list(picked.glob("**/layer_19/**/*.pt"))


def test_selective_batching_does_not_move_a_value(env, signal_json):
    """Pass 2 packs steps too, so it needs the same padding guarantee pass 1 has."""
    single = _run(env, "single", *_select_args(signal_json), "--forward-batch-size", "1")
    batched = _run(env, "batched", *_select_args(signal_json), "--forward-batch-size", "4")

    rels = sorted(p.relative_to(single) for p in single.rglob("*.pt"))
    assert rels == sorted(p.relative_to(batched) for p in batched.rglob("*.pt"))
    assert rels
    for rel in rels:
        assert torch.equal(
            torch.load(single / rel, weights_only=True),
            torch.load(batched / rel, weights_only=True),
        ), rel


def test_selective_crash_leaves_no_csv(env, signal_json, monkeypatch):
    """The done-marker must not appear if the second pass never ran."""
    def boom(*_a, **_k):
        raise RuntimeError("second pass exploded")

    monkeypatch.setattr(jrt, "save_selected_activations", boom)
    with pytest.raises(RuntimeError):
        _run(env, "crashed", *_select_args(signal_json))
    out = env["tmp"] / "crashed"
    assert list(out.rglob("*_jlens_analysis.csv")) == []
