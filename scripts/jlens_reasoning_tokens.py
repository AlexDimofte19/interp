"""Full j-space sweep over reasoning tokens: one row per (reasoning token, layer).

Combines gather_activations' single-pass activation extraction with
jlens_action_ranks' lens application: for every reasoning ("analysis") token of
every step, capture the residual stream at every requested layer in one forward
pass, apply the Jacobian lens, and record the top-20 predicted tokens plus each
token's position in the reasoning chain.

Each forward pass now produces two artefacts, so downstream analyses never have
to re-run gather_activations over the same trajectories:

  {activations_dir}/size{S}/{stem}/{stem}_jlens_analysis.csv
  {activations_dir}/size{S}/{stem}/{model}/layer_{N}/step_{M}/output/{idx}.pt

The .pt layout is exactly gather_activations' (reasoning tokens are output
tokens, hence the `output` category, keyed by output-relative index), so every
existing activation loader works unchanged.

Both the activation hooks and the jlens fit capture decoder-block *outputs*, so
layer indices line up directly; layer 23 is the jlens target (lens is identity
there: just norm + unembed).

Needs a GPU that fits gpt-oss-20b — run on the server, not the laptop.
First run downloads one 4.2 GB shard to cache lm_head + final norm (see
ensure_unembed_assets), reused from jlens_action_ranks_sampled.py.

Usage:
  python scripts/jlens_reasoning_tokens.py \
    --trajectory-paths /workspace/trajectories/trajectories_test_full \
    --jlens_dir /workspace/jlens \
    --layers 7:23 \
    --activations-dir /workspace/activations/jlens_reasoning_tokens
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# Running this file directly puts scripts/ on sys.path[0], not the repo root, so the
# sibling `scripts.*` imports below would miss. Put the repo root first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Heavy deps (torch/transformers) and sibling-script imports are done lazily inside
# main() so --self-test runs on any machine with only the stdlib.

TARGET_LAYER = 23  # jlens target: final decoder block, lens is identity here
ACTIONS = ["RIGHT", "LEFT", "UP", "DOWN"]
TOP_K = 20
# together_ai_openai_gpt-oss-20b_size11_comp1.0_987.json -> size / comp / run
NAME_RE = re.compile(r"size(?P<size>\d+)_comp(?P<comp>[\d.]+)_(?P<run>\d+)")


def reasoning_token_positions(trajectory: dict, step: dict) -> list[tuple[int, int, str]]:
    """(reasoning_pos, abs_pos, token_text) for each analysis token of a step.

    reasoning_pos is the 0-based index within the reasoning chain; abs_pos is the
    token's absolute position in prefix+grid+suffix+output (what the forward pass
    and the lens see). Falls back to all output tokens if none are tagged analysis.
    """
    n_prefix = len(trajectory["prompt"]["prompt_prefix_tokens"])
    n_grid = len(step["grid_state_tokens"])
    n_suffix = len(trajectory["prompt"]["prompt_suffix_tokens"])
    output_start = n_prefix + n_grid + n_suffix

    out = step["output_tokens"]
    idxs = [i for i, t in enumerate(out) if "analysis" in t.get("token_groups", [])] or list(range(len(out)))
    return [(rp, output_start + oi, out[oi].get("token", "")) for rp, oi in enumerate(idxs)]


def parse_name(stem: str) -> dict:
    m = NAME_RE.search(stem)
    return m.groupdict() if m else {"size": "", "comp": "", "run": ""}


def trajectory_activation_dir(activations_dir: Path, stem: str) -> Path:
    """<activations_dir>/size{S}/<stem>, mirroring the gather_activations layout.

    Falls back to <activations_dir>/<stem> for trajectories whose filename carries
    no size{N} (rather than creating a bare `size/` folder).
    """
    size = parse_name(stem)["size"]
    return activations_dir / f"size{size}" / stem if size else activations_dir / stem


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM
    from telos_interp.commands.gather_activations.gather_activations_fn import _resolve_torch_dtype
    from telos_interp.commands.gather_activations.gather_activations_utils import (
        extract_activations_single_pass,
        parse_index_specification,
        sanitize_model_id,
        save_activations_to_files,
    )
    from scripts.inference_oss.run_inference import expand_paths
    from scripts.jlens_action_ranks_sampled import action_token_ids, ensure_unembed_assets
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory-paths", nargs="+", required=True,
                    help="Trajectory JSON file(s), directory, or glob(s).")
    ap.add_argument("--jlens_dir", type=Path, required=True)
    ap.add_argument("--activations-dir", type=Path, required=True,
                    help="Base dir for the gather_activations-style tree: per-trajectory "
                         "activations under size{S}/{stem}/{model}/layer_N/step_M/output/ and the "
                         "trajectory's jlens CSV at size{S}/{stem}/{stem}_jlens_analysis.csv.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Reprocess trajectories whose jlens CSV already exists (default: skip).")
    ap.add_argument("--sizes", default=None,
                    help="Comma-separated grid sizes to keep (matched against size{S} in the "
                         "filename), e.g. '11,15'. Default: all sizes.")
    ap.add_argument("--complexities", default=None,
                    help="Comma-separated complexities to keep (matched against comp{C}), "
                         "e.g. '0.0,1.0'. Default: all complexities.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Reasoning tokens per lens matmul (caps the [B, vocab] logits). "
                         "Default: a whole step's reasoning chain at once.")
    ap.add_argument("--max-trajectories", type=int, default=None,
                    help="Process at most N trajectory files (default: all).")
    ap.add_argument("--seed", type=int, default=None,
                    help="With --max-trajectories, randomly sample N using this seed "
                         "(default: take the first N in sorted order).")
    ap.add_argument("--layers", default="all",
                    help="Comma/range spec of layer indices, or 'all' (default).")
    ap.add_argument("--steps", default="all",
                    help="Comma/range spec of step indices, or 'all' (default).")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--torch-dtype", default="auto")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device for the lens matmuls (defaults to cuda if available).")
    args = ap.parse_args()
    paths = expand_paths(args.trajectory_paths)
    if args.sizes or args.complexities:
        sizes = {s.strip() for s in args.sizes.split(",")} if args.sizes else None
        comps = {c.strip() for c in args.complexities.split(",")} if args.complexities else None
        paths = [
            p for p in paths
            if (sizes is None or parse_name(Path(p).stem)["size"] in sizes)
            and (comps is None or parse_name(Path(p).stem)["comp"] in comps)
        ]
        print(f"{len(paths)} after size/complexity filter", flush=True)
    if args.max_trajectories is not None and args.max_trajectories < len(paths):
        if args.seed is not None:
            import random
            paths = sorted(random.Random(args.seed).sample(paths, args.max_trajectories))
        else:
            paths = paths[: args.max_trajectories]
    print(f"{len(paths)} trajectory file(s)", flush=True)

    with open(paths[0]) as f:
        model_id = json.load(f)["model_params"]["model_id"]

    print(f"loading model: {model_id}", flush=True)
    resolved = _resolve_torch_dtype(args.torch_dtype, model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=args.device_map, dtype=resolved if resolved is not None else "auto"
    )
    model.eval()
    sanitized_model = sanitize_model_id(model_id)
    num_layers = model.config.num_hidden_layers
    layer_indices = parse_index_specification(args.layers, num_layers)
    print(f"{num_layers} layers; extracting {layer_indices}", flush=True)

    print("loading jlens + unembed assets...", flush=True)
    lens = torch.load(args.jlens_dir / "gpt-oss-20b_jacobian_lens.pt", map_location="cpu")
    assets = ensure_unembed_assets(args.jlens_dir)
    ids, tok = action_token_ids()
    id_cols = [ids[a] for a in ACTIONS]

    dev = torch.device(args.device)
    lm_head = assets["lm_head"].to(dev)
    norm_w = assets["norm_weight"].float().to(dev)
    eps = assets["rms_eps"]

    header = (
        ["size", "complexity", "run", "step", "reasoning_pos", "abs_pos", "token", "layer",
         "agent_action"]
        + [f"{a}_rank" for a in ACTIONS]
        + [f"{a}_logprob" for a in ACTIONS]
        + [f"top_{i}" for i in range(1, TOP_K + 1)]
    )
    written = 0

    for traj_path in paths:
        stem = Path(traj_path).stem
        traj_dir = trajectory_activation_dir(args.activations_dir, stem)
        csv_path = traj_dir / f"{stem}_jlens_analysis.csv"
        if csv_path.exists() and not args.overwrite:
            print(f"{stem}: CSV exists, skipping (use --overwrite to redo)", flush=True)
            continue

        with open(traj_path) as f:
            trajectory = json.load(f)
        name = parse_name(stem)
        n_steps = len(trajectory["steps"])
        step_idxs = parse_index_specification(args.steps, n_steps, clamp=True)
        print(f"{stem}: {len(step_idxs)}/{n_steps} steps", flush=True)

        n_prefix = len(trajectory["prompt"]["prompt_prefix_tokens"])
        n_suffix = len(trajectory["prompt"]["prompt_suffix_tokens"])
        output_base = traj_dir / sanitized_model
        traj_dir.mkdir(parents=True, exist_ok=True)
        # write to a temp file and rename at the end, so a crashed run leaves no CSV
        # and is redone on the next invocation instead of being skipped as done
        tmp_path = csv_path.with_suffix(".csv.tmp")
        nan_count = 0
        total_count = 0

        with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)

            for si in step_idxs:
                step = trajectory["steps"][si]
                positions = reasoning_token_positions(trajectory, step)
                if not positions:
                    continue
                abs_positions = [p[1] for p in positions]

                # full prompt = prefix + grid + suffix + output; truncate at the last needed pos
                all_ids = (
                    [t["token_id"] for t in trajectory["prompt"]["prompt_prefix_tokens"]]
                    + [t["token_id"] for t in step["grid_state_tokens"]]
                    + [t["token_id"] for t in trajectory["prompt"]["prompt_suffix_tokens"]]
                    + [t["token_id"] for t in step["output_tokens"]]
                )
                input_ids = torch.tensor([all_ids[: max(abs_positions) + 1]])
                acts = extract_activations_single_pass(model, input_ids, abs_positions, layer_indices)

                # Save the raw residual streams in gather_activations' layout before the lens
                # math, so layers without a jlens matrix (skipped below) still get written.
                # Reasoning tokens are output tokens, keyed by their output-relative index.
                output_start = n_prefix + len(step["grid_state_tokens"]) + n_suffix
                remapped = {
                    layer: {ap - output_start: acts[layer][ap] for ap in abs_positions if ap in acts[layer]}
                    for layer in layer_indices
                }
                nan_count += save_activations_to_files(
                    remapped, output_base, step_idx=step["step_id"], category="output"
                )
                total_count += sum(len(v) for v in remapped.values())

                agent_action = step.get("agent_action", "")
                # chunk reasoning tokens so [B, vocab] logits stay bounded (bs=None -> whole step)
                bs = args.batch_size or len(positions)
                for layer in layer_indices:
                    if layer == TARGET_LAYER:
                        J = None
                    elif layer in lens["J"]:
                        J = lens["J"][layer].float().to(dev)
                    else:
                        continue
                    for i in range(0, len(positions), bs):
                        chunk = positions[i : i + bs]
                        h = torch.stack([acts[layer][ap] for _, ap, _ in chunk]).float().to(dev)  # [B, d]
                        with torch.no_grad():
                            if J is not None:
                                h = h @ J.T
                            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w
                            logits = (h.to(lm_head.dtype) @ lm_head.T).float()  # [B, vocab]
                            ranks = torch.stack(
                                [(logits > logits[:, t : t + 1]).sum(1) for t in id_cols], dim=1
                            )  # [B, 4], rank 0 = argmax
                            own = torch.stack([logits[:, t] for t in id_cols], dim=1)
                            logprobs = own - logits.logsumexp(-1, keepdim=True)
                            topk = logits.topk(TOP_K, dim=1).indices  # [B, TOP_K]
                        ranks, logprobs, topk = ranks.cpu().tolist(), logprobs.cpu().tolist(), topk.cpu().tolist()
                        for (rp, ap, token), r, lp, tk in zip(chunk, ranks, logprobs, topk):
                            writer.writerow(
                                [name["size"], name["comp"], name["run"], si, rp, ap, token, layer, agent_action]
                                + r
                                + [round(x, 4) for x in lp]
                                + [tok.decode([i]) for i in tk]
                            )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        os.replace(tmp_path, csv_path)
        written += 1
        print(f"  wrote {csv_path} ({total_count} activations)", flush=True)
        if nan_count > 0:
            print(f"  WARNING: {nan_count}/{total_count} activations contain NaN values!")
            print("  This is often caused by multi-GPU setups. Try using CUDA_VISIBLE_DEVICES=0")

    print(f"done: {written} trajectory folder(s) under {args.activations_dir}", flush=True)


def _self_test() -> None:
    """Position math only — no model, no torch load. Run: python ... --self-test."""
    traj = {"prompt": {"prompt_prefix_tokens": [{}] * 3, "prompt_suffix_tokens": [{}] * 2}}
    step = {
        "grid_state_tokens": [{}] * 4,  # output_start = 3 + 4 + 2 = 9
        "output_tokens": [
            {"token": "a", "token_groups": ["analysis"]},
            {"token": "b", "token_groups": ["final", "action"]},
            {"token": "c", "token_groups": ["analysis"]},
        ],
    }
    pos = reasoning_token_positions(traj, step)
    assert pos == [(0, 9, "a"), (1, 11, "c")], pos
    # no analysis tags -> all output tokens
    step2 = {"grid_state_tokens": [], "output_tokens": [{"token": "x", "token_groups": []}]}
    traj2 = {"prompt": {"prompt_prefix_tokens": [], "prompt_suffix_tokens": []}}
    assert reasoning_token_positions(traj2, step2) == [(0, 0, "x")]
    assert parse_name("foo_size11_comp1.0_987") == {"size": "11", "comp": "1.0", "run": "987"}
    # output layout mirrors gather_activations: <root>/size{S}/<stem>/...
    stem = "together_ai_openai_gpt-oss-20b_size11_comp1.0_987"
    assert trajectory_activation_dir(Path("/acts"), stem) == Path("/acts/size11") / stem
    assert trajectory_activation_dir(Path("/acts"), "nosize") == Path("/acts/nosize")
    print("self-test ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
