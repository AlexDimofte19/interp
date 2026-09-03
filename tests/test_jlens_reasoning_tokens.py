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
from telos_interp.jlens_utils import NO_MATCH_LOGPROB
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


# Columns whose value is a float the lens computed. Everything else -- ids, decoded token
# strings, layer indices, ranks -- is exact and must match character for character.
_FLOAT_COLS = tuple([f"{a}_logprob" for a in jrt.ACTIONS] + [f"top_{i}_logprob" for i in range(1, jrt.TOP_K + 1)])


def assert_csvs_agree(path_a, path_b, tol=1.5e-4):
    """The two CSVs are the same table, floats compared numerically.

    Not `read_bytes()`: `--forward-batch-size` and `--batch-size` change the shape of the
    lens GEMMs, which reassociates their reductions and moves the last bits of a logit. At
    4 printed decimals that is invisible ~always, but the row now carries a logprob for
    every one of the TOP_K predictions rather than only the four actions, so ~400 floats per
    row get a chance to sit on a rounding boundary and one eventually does. The claim worth
    testing is that the batching changes no *content*, which is this.
    """
    rows_a, rows_b = _read_csv(path_a), _read_csv(path_b)
    assert len(rows_a) == len(rows_b)
    for i, (ra, rb) in enumerate(zip(rows_a, rows_b, strict=True)):
        assert list(ra) == list(rb), i
        for col, va in ra.items():
            vb = rb[col]
            if col in _FLOAT_COLS:
                assert abs(float(va) - float(vb)) <= tol, (i, col, va, vb)
            else:
                assert va == vb, (i, col, va, vb)


def test_batched_and_unbatched_runs_agree_exactly(env):
    """Packing steps into one padded forward must not move a single value."""
    single = _run(env, "single", "--forward-batch-size", "1")
    batched = _run(env, "batched", "--forward-batch-size", "4")

    stem = env["stem"]
    assert_csvs_agree(single / f"{stem}_jlens_analysis.csv", batched / f"{stem}_jlens_analysis.csv")

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
    # every top_i carries the lens' own logprob for it, and they descend with the rank
    for row in step0:
        lps = [float(row[f"top_{i}_logprob"]) for i in range(1, 21)]
        assert lps == sorted(lps, reverse=True)
        assert all(lp <= 0.0 for lp in lps)


def test_lens_chunking_does_not_change_the_csv(env):
    """--batch-size only bounds the logits; it must not alter a single value."""
    whole = _run(env, "chunk_off", "--batch-size", "0")
    chunked = _run(env, "chunk_on", "--batch-size", "2")
    stem = env["stem"]
    assert_csvs_agree(whole / f"{stem}_jlens_analysis.csv", chunked / f"{stem}_jlens_analysis.csv")


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
        ["size", "complexity", "run", "step", "reasoning_pos", "abs_pos", "token", "layer", "agent_action"]
        + [f"{a}_rank" for a in jrt.ACTIONS]
        + [f"{a}_logprob" for a in jrt.ACTIONS]
        + [f"top_{i}" for i in range(1, jrt.TOP_K + 1)]
        + [f"top_{i}_logprob" for i in range(1, jrt.TOP_K + 1)]
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
                    top = logits.topk(jrt.TOP_K, dim=1)
                    topk = top.indices
                    # the same normaliser as the action logprobs, applied naively rather
                    # than reusing the one lens_predictions computes once
                    toplp = top.values - logits.logsumexp(-1, keepdim=True)
                ranks, logprobs = ranks.tolist(), logprobs.tolist()
                topk, toplp = topk.tolist(), toplp.tolist()
                for (rp, ap, token), r, lp, tk, tlp in zip(positions, ranks, logprobs, topk, toplp, strict=True):
                    writer.writerow(
                        [name["size"], name["comp"], name["run"], si, rp, ap, token, layer, agent_action]
                        + r
                        + [round(x, 4) for x in lp]
                        + [tok.decode([i]) for i in tk]
                        + [round(x, 4) for x in tlp]
                    )
    return traj_dir


def test_matches_the_pre_batching_implementation(env):
    """Every CSV cell and every .pt tensor, against the loop this replaced."""
    reference = _reference_run(env, "reference")
    current = _run(env, "current", "--forward-batch-size", "4", "--batch-size", "2")

    stem = env["stem"]
    assert_csvs_agree(current / f"{stem}_jlens_analysis.csv", reference / f"{stem}_jlens_analysis.csv")

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
    from telos_interp.jlens_utils import to_disk_coords, top_filter

    kwargs = {
        "methods": ["jlens", "random"],
        "num_tokens": 2,
        "num_layers": 2,
        "always_layers": (),
        "random_tokens": 1,
        "seed": 42,
        "seed_key": env["stem"],
        "top_k": jrt.TOP_K,
    }
    kwargs.update(overrides)
    kept = top_filter(signal_json, full_run_dir, **kwargs)
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
    assert {p.resolve() for p in picked.rglob("*.pt")} == {p.resolve() for p in expected.activation_paths(model_dir)}

    # a second forward pass must reproduce the first one exactly, not merely closely
    for rel in sorted(p.relative_to(picked) for p in picked.rglob("*.pt")):
        assert torch.equal(
            torch.load(picked / rel, weights_only=True),
            torch.load(full / rel, weights_only=True),
        ), rel


# --- the direction-mass table -------------------------------------------------------------
#
# The second artifact: wide (one row per reasoning token, one column per layer) where the
# analysis CSV is long, and computed over the WHOLE direction vocabulary rather than over the
# direction words that reached the top-k. The property worth pinning is exactly that
# difference -- when the top-k window is wide enough to hold every direction word, the two
# must agree; the table exists for when it is not.


def test_direction_mass_table_is_written_beside_the_csv(env, signal_json):
    out = _run(env, "mass", "--direction-mass-json", str(signal_json))
    stem = env["stem"]
    mass_path = out / f"{stem}_jlens_direction_mass.csv"
    assert mass_path.exists()

    rows = _read_csv(mass_path)
    layer_cols = [c for c in rows[0] if c.startswith("L")]
    assert [int(c[1:]) for c in layer_cols] == sorted(SCORABLE_LAYERS), "one column per lensed layer"

    # one row per (step, reasoning token) -- not per (token, layer) as the analysis CSV is
    analysis = _read_csv(out / f"{stem}_jlens_analysis.csv")
    assert len(rows) == len({(r["step"], r["abs_pos"]) for r in analysis})
    for row in rows:
        values = [float(row[c]) for c in layer_cols if row[c] != ""]
        assert values and all(v <= 0.0 for v in values), "a log probability"


def test_direction_mass_carries_the_vocabulary_that_made_it(env, signal_json):
    """Two vocabularies get pointed at these trees; a table without provenance is unreadable."""
    out = _run(env, "mass_meta", "--direction-mass-json", str(signal_json))
    meta = json.loads((out / f"{env['stem']}_jlens_direction_mass.csv.meta.json").read_text())
    assert meta["signal_json"] == str(signal_json)
    assert meta["num_direction_tokens"] == 32  # the stub vocabulary, all of it round-tripping
    assert meta["lens"] == "jlens"
    assert meta["layers"] == sorted(SCORABLE_LAYERS)


def test_direction_mass_equals_the_topk_mass_when_the_window_holds_everything(env, tmp_path):
    """The two agree exactly where the top-k can see the whole vocabulary -- so the only
    difference between them is truncation, which is the reason the table exists."""
    from telos_interp.jlens_utils import read_direction_mass, read_direction_scores

    # A vocabulary of ONE token: it either makes the top 20 or carries negligible mass, and
    # in the stub's random lens it comfortably does for some (token, layer) pairs.
    signal = tmp_path / "one.json"
    signal.write_text(json.dumps({"UP": ["<7>"], "DOWN": [], "LEFT": [], "RIGHT": []}))
    out = _run(env, "mass_agree", "--direction-mass-json", str(signal))
    stem = env["stem"]

    topk = read_direction_scores(
        out / f"{stem}_jlens_analysis.csv", {"<7>"}, top_k=jrt.TOP_K, score_mode="logprob_mass"
    )
    full = read_direction_mass(out / f"{stem}_jlens_direction_mass.csv")
    assert set(topk) == set(full)

    compared = 0
    for key, score in topk.items():
        for layer, value in score.per_layer.items():
            if value == NO_MATCH_LOGPROB:
                continue  # the token missed the top-k here; only the table can see it
            assert value == pytest.approx(full[key].per_layer[layer], abs=1e-3), (key, layer)
            compared += 1
    assert compared, "the fixture never put the direction token in the top-k; nothing was compared"
    # and the table sees mass the top-k window missed, which is the point
    assert any(
        topk[key].per_layer[layer] == NO_MATCH_LOGPROB for key, score in full.items() for layer in score.per_layer
    ), "the top-k window held everything; this fixture cannot show the difference"


def test_no_direction_mass_suppresses_the_table(env, signal_json):
    out = _run(env, "no_mass", *_select_args(signal_json), "--no-direction-mass")
    assert list(out.glob("*_direction_mass.csv")) == []
    assert (out / f"{env['stem']}_jlens_analysis.csv").exists()


def test_no_top_logprobs_drops_the_columns_but_not_the_mass(env, signal_json):
    out = _run(env, "no_toplp", "--direction-mass-json", str(signal_json), "--no-top-logprobs")
    stem = env["stem"]
    fields = list(_read_csv(out / f"{stem}_jlens_analysis.csv")[0])
    assert "top_1" in fields and "top_1_logprob" not in fields
    assert (out / f"{stem}_jlens_direction_mass.csv").exists()


def test_selecting_on_the_mass_table_needs_one(env, signal_json):
    with pytest.raises(SystemExit, match="writes none"):
        _run(
            env,
            "mass_missing",
            *_select_args(signal_json),
            "--direction-score",
            "logprob_mass_full",
            "--no-direction-mass",
        )


def test_selection_can_rank_on_the_mass_table(env, signal_json):
    """The selection reads the table it was told to, while it is still a .tmp."""
    from telos_interp.jlens_utils import read_selection_record, record_path

    out = _run(env, "mass_select", *_select_args(signal_json), "--direction-score", "logprob_mass_full")
    kept, configs = read_selection_record(record_path(out))
    assert configs["jlens"]["direction_score"] == "logprob_mass_full"
    picks = list(kept["jlens"].values())
    assert picks and all(p.score_mode == "logprob_mass_full" for p in picks)
    # Per layer the score is a log probability and so <= 0. The token's own score aggregates
    # ACROSS layers, which adds probabilities that belong to different distributions -- it is
    # a ranking statistic, not a probability, and may exceed 0. Pinned so nobody "fixes" it.
    assert all(isinstance(p.direction_count, float) for p in picks)
    assert all(v <= 0.0 for p in picks for v in (p.layer_direction_counts or {}).values())
    assert list(out.rglob("*.pt")), "the arm still gathered activations"


def test_selective_run_is_a_large_saving(env, signal_json):
    full = _run(env, "full")
    picked = _run(env, "picked", *_select_args(signal_json))
    assert len(list(picked.rglob("*.pt"))) < len(list(full.rglob("*.pt"))) / 3


def test_selection_record_is_written_and_reloadable(env, signal_json):
    from telos_interp.jlens_utils import read_selection_record, record_path

    full = _run(env, "full")
    picked = _run(env, "picked", *_select_args(signal_json))

    kept, configs = read_selection_record(record_path(picked))
    assert kept == _expected_selection(env, full, signal_json)
    assert configs["jlens"]["num_tokens"] == 2
    assert len(kept["jlens"]) == 2 and len(kept["random"]) == 1


def test_control_arm_carries_no_counts(env, signal_json):
    """What lets split_next_action_manifest tell a control apart from a ranked selection."""
    from telos_interp.jlens_utils import read_selection_record, record_path

    picked = _run(env, "picked", *_select_args(signal_json, **{"random-tokens": 2}))
    kept, _ = read_selection_record(record_path(picked))
    assert all(p.direction_count is not None for p in kept["jlens"].values())
    assert all(p.direction_count is None for p in kept["random"].values())


def test_always_layers_are_forced_into_every_pick(env, signal_json):
    from telos_interp.jlens_utils import read_selection_record, record_path

    picked = _run(env, "picked", *_select_args(signal_json, **{"always-layers": "20", "num-layers": 1}))
    kept, _ = read_selection_record(record_path(picked))
    for pick in list(kept["jlens"].values()) + list(kept["random"].values()):
        assert 20 in pick.layers, pick


def test_unscorable_layers_are_never_selected(env, signal_json):
    """Layer 19 has no jlens matrix, so forcing it is a no-op rather than an error."""
    from telos_interp.jlens_utils import read_selection_record, record_path

    picked = _run(env, "picked", *_select_args(signal_json, **{"always-layers": "19", "num-layers": 1}))
    kept, _ = read_selection_record(record_path(picked))
    assert all(19 not in pick.layers for pick in kept["jlens"].values())
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


# --- the logit lens ----------------------------------------------------------------------
#
# The logit lens is the same code path with nothing transported (see
# jrt.build_lens_transports), so what these pin is not the arithmetic but the two things
# that genuinely differ: which layers each lens can score, and that one forward pass really
# does feed both.


def _csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_lens_both_writes_two_csvs_from_one_pass(env):
    out = _run(env, "both", "--lens", "both")
    jlens = out / f"{env['stem']}_jlens_analysis.csv"
    logit = out / f"{env['stem']}_logitlens_analysis.csv"
    assert jlens.exists() and logit.exists()

    # the jlens can only score layers it has a matrix for; the logit lens needs none
    assert sorted({int(r["layer"]) for r in _csv_rows(jlens)}) == SCORABLE_LAYERS
    assert sorted({int(r["layer"]) for r in _csv_rows(logit)}) == [19, *SCORABLE_LAYERS]


def test_the_lenses_agree_at_the_target_layer(env):
    """At TARGET_LAYER the jlens is the identity, so it *is* the logit lens there.

    A free end-to-end check that --lens both did not cross its wires: the two CSVs are
    produced by different transports but must coincide on the one layer where the transports
    are the same function.
    """
    out = _run(env, "both", "--lens", "both")

    def at_target(name):
        rows = _csv_rows(out / f"{env['stem']}_{name}_analysis.csv")
        return [r for r in rows if int(r["layer"]) == jrt.TARGET_LAYER]

    jlens_rows, logit_rows = at_target("jlens"), at_target("logitlens")
    assert jlens_rows and jlens_rows == logit_rows


def test_logitlens_alone_never_loads_the_jacobian(env):
    """--lens logitlens must not need gpt-oss-20b_jacobian_lens.pt at all."""
    (env["jlens_dir"] / "gpt-oss-20b_jacobian_lens.pt").unlink()
    out = _run(env, "logit", "--lens", "logitlens")
    assert (out / f"{env['stem']}_logitlens_analysis.csv").exists()
    assert not (out / f"{env['stem']}_jlens_analysis.csv").exists()


def test_a_missing_lens_csv_means_unfinished(env):
    """A trajectory analysed by one lens is not done under --lens both."""
    _run(env, "tree", "--lens", "jlens")
    out = _run(env, "tree", "--lens", "both")
    assert (out / f"{env['stem']}_logitlens_analysis.csv").exists()


def test_selecting_a_lens_whose_csv_is_not_written_is_refused(env, signal_json):
    with pytest.raises(SystemExit, match="writes no CSV"):
        _run(env, "bad", "--lens", "jlens", *_select_args(signal_json, methods="jlens,logitlens"))


# --- extending an already-selected tree --------------------------------------------------
#
# The real tree has been pruned to jlens+random, so the logitlens arm cannot be recovered by
# re-filtering -- the tokens it would pick are gone. `--extend` is the only path that adds
# one, and it is only safe if it leaves the existing arms completely alone.


def _extend(env, out_name, signal_json, *extra):
    return _run(
        env,
        out_name,
        "--lens",
        "both",
        "--extend",
        *_select_args(signal_json, methods="jlens,logitlens,random"),
        *extra,
    )


def test_extend_adds_an_arm_without_disturbing_the_others(env, signal_json):
    from telos_interp.jlens_utils import read_selection_record, record_path

    picked = _run(env, "tree", *_select_args(signal_json, methods="jlens,random"))
    before_kept, before_configs = read_selection_record(record_path(picked))
    before_files = {p: p.stat().st_mtime_ns for p in picked.rglob("*.pt")}
    assert before_kept.names == ["jlens", "random"]

    _extend(env, "tree", signal_json)

    after_kept, after_configs = read_selection_record(record_path(picked))
    assert sorted(after_kept.names) == ["jlens", "logitlens", "random"]
    # the arms that were already there are untouched, picks and config alike
    assert after_kept["jlens"] == before_kept["jlens"]
    assert after_kept["random"] == before_kept["random"]
    assert after_configs["random"] == before_configs["random"]
    assert after_kept["logitlens"], "the new arm has to have selected something"

    # files that already existed were not rewritten -- the arms overlap heavily
    after_files = {p: p.stat().st_mtime_ns for p in picked.rglob("*.pt")}
    assert set(before_files) <= set(after_files), "extending must never delete"
    assert all(after_files[p] == mtime for p, mtime in before_files.items())

    # ...and the new arm's files are now on disk
    model_dir = picked / "stub__model"
    assert after_kept.activation_paths(model_dir, arms=["logitlens"]) <= set(after_files)


def test_extend_writes_a_v2_record_with_per_arm_config(env, signal_json):
    from telos_interp.jlens_utils import RECORD_FORMAT_VERSION, read_raw_record, record_path

    picked = _run(env, "tree", *_select_args(signal_json, methods="jlens,random"))
    _extend(env, "tree", signal_json)

    record = read_raw_record(record_path(picked))
    assert record["format_version"] == RECORD_FORMAT_VERSION
    assert sorted(record["arms"]) == ["jlens", "logitlens", "random"]
    assert record["arms"]["logitlens"]["config"]["lens"] == "both"
    assert record["arms"]["jlens"]["config"]["lens"] == "jlens"


def test_extend_is_idempotent(env, signal_json):
    from telos_interp.jlens_utils import read_selection_record, record_path

    picked = _run(env, "tree", *_select_args(signal_json, methods="jlens,random"))
    _extend(env, "tree", signal_json)
    once, _ = read_selection_record(record_path(picked))
    files = sorted(p.relative_to(picked) for p in picked.rglob("*.pt"))

    _extend(env, "tree", signal_json)
    twice, _ = read_selection_record(record_path(picked))
    assert twice == once
    assert sorted(p.relative_to(picked) for p in picked.rglob("*.pt")) == files


def test_extend_dry_run_writes_nothing(env, signal_json, capsys):
    from telos_interp.jlens_utils import read_selection_record, record_path

    picked = _run(env, "tree", *_select_args(signal_json, methods="jlens,random"))
    before_kept, _ = read_selection_record(record_path(picked))
    before = sorted(p.relative_to(picked) for p in picked.rglob("*.pt"))

    _extend(env, "tree", signal_json, "--dry-run")

    after_kept, _ = read_selection_record(record_path(picked))
    assert after_kept == before_kept, "a dry run must not touch the record"
    assert sorted(p.relative_to(picked) for p in picked.rglob("*.pt")) == before
    assert not (picked / f"{env['stem']}_logitlens_analysis.csv").exists()
    assert "DRY RUN" in capsys.readouterr().out


def test_extend_without_a_record_is_skipped(env, signal_json, capsys):
    """A never-selected trajectory is not silently given a full gather instead."""
    _extend(env, "fresh", signal_json)
    assert not list((env["tmp"] / "fresh").rglob("*.pt"))
    assert "no selection record" in capsys.readouterr().out


def test_dropping_an_arm_is_refused_without_extend(env, signal_json, capsys):
    """The guard that protects the control arm: it cannot be redrawn once the tree is cut."""
    from telos_interp.jlens_utils import read_selection_record, record_path

    picked = _run(env, "tree", *_select_args(signal_json, methods="jlens,random"))
    before, _ = read_selection_record(record_path(picked))

    _run(env, "tree", "--lens", "logitlens", "--overwrite", *_select_args(signal_json, methods="logitlens"))

    after, _ = read_selection_record(record_path(picked))
    assert after == before, "the record must survive a run that would have dropped an arm"
    assert "REFUSED" in capsys.readouterr().out


def test_overwrite_record_permits_it_explicitly(env, signal_json):
    from telos_interp.jlens_utils import read_selection_record, record_path

    picked = _run(env, "tree", *_select_args(signal_json, methods="jlens,random"))
    _run(
        env,
        "tree",
        "--lens",
        "logitlens",
        "--overwrite",
        "--overwrite-record",
        *_select_args(signal_json, methods="logitlens"),
    )
    after, _ = read_selection_record(record_path(picked))
    assert after.names == ["logitlens"]


def test_the_extend_wrapper_invocation_works(env, signal_json):
    """Exactly what scripts/jlens_extend_logitlens.sh runs, on an already-selected tree.

    --lens logitlens (the jlens CSV is not recomputed), --select-methods logitlens (only the
    new arm), --select-random-tokens 0 (the control is inherited from the record, never
    redrawn -- a fresh draw could only sample the survivors).
    """
    from telos_interp.jlens_utils import read_selection_record, record_path

    picked = _run(env, "tree", *_select_args(signal_json, methods="jlens,random"))
    before, _ = read_selection_record(record_path(picked))

    _run(
        env,
        "tree",
        "--lens",
        "logitlens",
        "--extend",
        *_select_args(signal_json, methods="logitlens", **{"random-tokens": 0}),
    )

    after, _ = read_selection_record(record_path(picked))
    assert sorted(after.names) == ["jlens", "logitlens", "random"]
    assert after["random"] == before["random"], "the inherited control must be untouched"
    assert after["jlens"] == before["jlens"]
    assert after["logitlens"]
    # the jlens CSV was never rewritten, and the logitlens one now exists beside it
    assert (picked / f"{env['stem']}_jlens_analysis.csv").exists()
    assert (picked / f"{env['stem']}_logitlens_analysis.csv").exists()


# --------------------------------------------------------------------------------------
# --names-file: pinning a second tree to an existing tree's trajectory set.
# --------------------------------------------------------------------------------------


def test_names_file_keeps_only_the_listed_trajectories(env, tmp_path):
    """The listed stem is processed; an unlisted one leaves the run with nothing to do.

    This is the only way to make a second gather cover the same trajectories as an
    existing one -- ICLR entry 36's correction records that --per-combo/--seed does NOT
    reproduce a previous draw (two runs with identical flags overlapped by 348 of 3600).
    """
    names = tmp_path / "names.txt"

    names.write_text(env["stem"] + "\n")
    kept = _run(env, "kept", "--names-file", str(names))
    assert (kept / f"{env['stem']}_jlens_analysis.csv").exists()

    names.write_text("some_other_trajectory\n")
    with pytest.raises(ValueError, match="no trajectories left to process"):
        _run(env, "missed", "--names-file", str(names))
    assert not (env["tmp"] / "missed").exists()


def test_names_file_is_applied_before_the_size_filter(env, tmp_path):
    """--names-file narrows first, then --sizes narrows what is left."""
    names = tmp_path / "names.txt"
    names.write_text(env["stem"] + "\n")
    with pytest.raises(ValueError, match="no trajectories left to process"):
        _run(env, "sized", "--names-file", str(names), "--sizes", "11")


def test_an_empty_selection_is_an_error_not_an_empty_success(env, tmp_path):
    """Filtering everything away must raise rather than exit 0 with no CSV written.

    The repo has been bitten by a script that "succeeded" while writing nothing
    (analyze_probe_rollout.py, entry 47); a mistyped --names-file is the same trap.
    """
    names = tmp_path / "names.txt"
    names.write_text("nothing_matches_this\n")
    with pytest.raises(ValueError, match="no trajectories left to process"):
        _run(env, "empty", "--names-file", str(names))
