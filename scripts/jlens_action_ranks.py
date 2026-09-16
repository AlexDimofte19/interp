"""Apply the Jacobian lens (jlens) to saved gpt-oss-20b activations and rank action tokens.

For every activation file under --activations_root
  (size{S}/..._comp{C}_{run}/openai__gpt-oss-20b/layer_{L}/step_0/prompt_suffix/{T}.pt)
computes lens logits = lm_head(rms_norm(h @ J[L].T)) and records the rank
(0 = argmax) of the LEFT / RIGHT / UP / DOWN tokens. Writes one CSV row per file.

Both these activations and the jlens fit capture decoder-block *outputs* via
forward hooks, so layer indices line up directly. layer_23 is the jlens target
layer itself: no transport, just norm + unembed.

First run downloads one 4.2 GB safetensors shard from openai/gpt-oss-20b to
extract lm_head + final norm (cached as gpt-oss-20b_unembed.pt in --jlens_dir).

Two optional flags widen it, and both used to be a separate `_sampled` fork that duplicated
this file (including `ensure_unembed_assets` verbatim):

  --runs_per_combo N   keep only the first N runs per (size, complexity, layer). For a first
                       pass on a tree too big to sweep whole.
  --trajectories_root  join each row to the step's `agent_action` and widen the CSV with the
                       four action logprobs and the top/bottom 5 decoded tokens. WITHOUT it the
                       narrow schema above is written, unchanged -- so an existing consumer
                       (`jlens_rank_analysis.py`) reads the same columns it always did.

Usage:
  python scripts/jlens_action_ranks.py \
    --activations_root C:/Uni/Thesis/data/activations_train_single_step \
    --jlens_dir C:/Uni/Thesis/data/jlens \
    --out jlens_action_ranks.csv
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path

import torch
from telos_interp.jlens_utils import DEFAULT_MODEL_ID, ModelSpec, get_model, validate_against_index

ACTIONS = ["RIGHT", "LEFT", "UP", "DOWN"]
MODEL_ID = DEFAULT_MODEL_ID  # this script's own default; the helpers below take a spec
TARGET_LAYER = 23  # jlens target: final decoder block, lens is identity here

PATH_RE = re.compile(
    r"size(?P<size>\d+)_comp(?P<comp>[\d.]+)_(?P<run>\d+).*?"
    r"layer_(?P<layer>\d+).*?prompt_suffix"
)


def ensure_unembed_assets(jlens_dir: Path, spec: ModelSpec | None = None) -> dict:
    """Extract lm_head.weight + the final norm from the HF checkpoint (once).

    Everything model-specific -- the repo to download from, the cache filename, the two
    tensor keys and where `rms_norm_eps` sits in `config.json` -- comes from `spec`
    (`telos_interp/jlens_utils/models.py`). `None` keeps this script's own gpt-oss default,
    so the cache it has always written keeps its name and is still read.

    The keys are validated against the index before the shard is fetched: on a wrapped
    model the near misses are the right rank and a plausible shape, so a wrong key is
    silently wrong logits rather than an error, and finding that out *after* a multi-GB
    download is the expensive order to find it out in.
    """
    spec = spec or get_model(MODEL_ID)
    cache = jlens_dir / spec.unembed_file
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=True)

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    cfg = json.load(open(hf_hub_download(spec.model_id, "config.json")))
    index = json.load(open(hf_hub_download(spec.model_id, "model.safetensors.index.json")))
    validate_against_index(spec, index["weight_map"])
    shard = index["weight_map"][spec.lm_head_key]
    assert index["weight_map"][spec.norm_key] == shard, (
        f"{spec.lm_head_key} is in {shard} but {spec.norm_key} is in "
        f"{index['weight_map'][spec.norm_key]}; this path fetches one shard"
    )
    print(f"downloading {shard} (one-time; extracted tensors are cached)...")
    shard_path = hf_hub_download(spec.model_id, shard)
    with safe_open(shard_path, framework="pt") as f:
        assets = {
            "lm_head": f.get_tensor(spec.lm_head_key),
            "norm_weight": f.get_tensor(spec.norm_key),
            # A VLM wrapper puts this under text_config; a bare lookup raises KeyError.
            "rms_eps": spec.text_config(cfg)["rms_norm_eps"],
        }
    torch.save(assets, cache)
    print(f"cached {cache} — you may delete the shard from the HF cache to reclaim disk")
    return assets


@cache
def trajectory_actions(traj_file: Path) -> dict:
    """{step_id: agent_action} for one trajectory json."""
    steps = json.load(open(traj_file, encoding="utf-8"))["steps"]
    return {s["step_id"]: s["agent_action"] for s in steps}


def agent_action_for(f: Path, traj_root: Path) -> str:
    # f = .../size{S}/{run_folder}/openai__gpt-oss-20b/layer_L/step_{n}/prompt_suffix/{T}.pt
    run_folder = f.parents[4].name
    size_dir = f.parents[5].name
    step_id = int(f.parents[1].name.split("_")[1])
    traj_file = traj_root / size_dir / f"{run_folder}.json"
    if not traj_file.exists():
        return ""
    return trajectory_actions(traj_file).get(step_id, "")


def action_token_ids(spec: ModelSpec | None = None) -> tuple[dict, object]:
    """The four action token ids and the tokenizer that produced them.

    Takes a spec because the tokenizer has to be the one that wrote the trajectory's
    token ids: the CSV's decoded `top_i` strings and the direction vocabulary's id
    resolution both go through it, and a mismatched tokenizer names different strings
    for the same ids without erroring anywhere.
    """
    from transformers import AutoTokenizer

    spec = spec or get_model(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(spec.model_id)
    ids = {}
    for a in ACTIONS:
        enc = tok.encode(a, add_special_tokens=False)
        assert len(enc) == 1, f"{a!r} is not a single token: {enc}"
        ids[a] = enc[0]
    return ids, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations_root", type=Path, required=True)
    ap.add_argument("--jlens_dir", type=Path, required=True)
    ap.add_argument(
        "--trajectories_root",
        type=Path,
        default=None,
        help="join agent_action and widen the CSV with logprobs and top/bottom tokens",
    )
    ap.add_argument(
        "--runs_per_combo",
        type=int,
        default=None,
        help="keep the first N runs (lowest run index) per size/complexity/layer; default: all",
    )
    ap.add_argument("--out", type=Path, default=Path("jlens_action_ranks.csv"))
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    files = sorted(args.activations_root.glob("size*/*/*/layer_*/step_0/prompt_suffix/*.pt"))
    print(f"{len(files)} activation files")
    if not files:
        sys.exit("no activation files found")

    # group by layer so each J matrix is loaded onto the GPU once
    by_layer = defaultdict(list)
    skipped = 0
    parsed = []
    for f in files:
        m = PATH_RE.search(f.as_posix())
        if not m:
            skipped += 1
            continue
        parsed.append((f, m))
    if skipped:
        print(f"warning: {skipped} files did not match the expected path pattern")

    # keep only the first N runs per (size, complexity, layer)
    if args.runs_per_combo is not None:
        runs_per_combo = defaultdict(set)
        for _, m in parsed:
            runs_per_combo[(m["size"], m["comp"], m["layer"])].add(int(m["run"]))
        keep = {k: set(sorted(v)[: args.runs_per_combo]) for k, v in runs_per_combo.items()}
        parsed = [(f, m) for f, m in parsed if int(m["run"]) in keep[(m["size"], m["comp"], m["layer"])]]
        print(f"{len(parsed)} files after sampling {args.runs_per_combo} runs per combo")

    for f, m in parsed:
        by_layer[int(m["layer"])].append((f, m))

    lens = torch.load(args.jlens_dir / "gpt-oss-20b_jacobian_lens.pt", map_location="cpu")
    assets = ensure_unembed_assets(args.jlens_dir)
    ids, tok = action_token_ids()
    print(f"action token ids: {ids}")

    dev = torch.device(args.device)
    lm_head = assets["lm_head"].to(dev)  # [vocab, d_model] bf16
    norm_w = assets["norm_weight"].float().to(dev)
    eps = assets["rms_eps"]
    id_cols = [ids[a] for a in ACTIONS]

    wide = args.trajectories_root is not None
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["size", "complexity", "layer", "run", "token"]
        if wide:
            header += ["agent_action"]
        header += [f"{a}_position" for a in ACTIONS]
        if wide:
            header += (
                [f"{a}_logprob" for a in ACTIONS]
                + [f"top_{i}" for i in range(1, 6)]
                + [f"bottom_{i}" for i in range(1, 6)]
            )
        writer.writerow(header)
        for layer, items in sorted(by_layer.items()):
            if layer == TARGET_LAYER:
                J = None
            elif layer in lens["J"]:
                J = lens["J"][layer].float().to(dev)
            else:
                print(f"warning: no jlens matrix for layer {layer}, skipping {len(items)} files")
                continue
            for i in range(0, len(items), args.batch_size):
                chunk = items[i : i + args.batch_size]
                h = (
                    torch.stack([torch.load(f, map_location="cpu", weights_only=True) for f, _ in chunk])
                    .float()
                    .to(dev)
                )
                with torch.no_grad():
                    if J is not None:
                        h = h @ J.T
                    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w
                    logits = (h.to(lm_head.dtype) @ lm_head.T).float()  # [B, vocab]
                    ranks = torch.stack(
                        [(logits > logits[:, tid : tid + 1]).sum(1) for tid in id_cols],
                        dim=1,
                    )  # [B, 4], rank 0 = argmax
                    if wide:
                        own = torch.stack([logits[:, tid] for tid in id_cols], dim=1)
                        logprobs = own - logits.logsumexp(-1, keepdim=True)  # [B, 4]
                        top5 = logits.topk(5, dim=1).indices
                        bot5 = logits.topk(5, dim=1, largest=False).indices  # bottom_1 = worst
                ranks = ranks.cpu().tolist()
                if wide:
                    logprobs = logprobs.cpu().tolist()
                    top5, bot5 = top5.cpu().tolist(), bot5.cpu().tolist()
                    for (f, m), r, lp, t5, b5 in zip(chunk, ranks, logprobs, top5, bot5):
                        writer.writerow(
                            [
                                m["size"],
                                m["comp"],
                                layer,
                                m["run"],
                                f.stem,
                                agent_action_for(f, args.trajectories_root),
                            ]
                            + r
                            + [round(x, 4) for x in lp]
                            + [tok.decode([i]) for i in t5 + b5]
                        )
                else:
                    for (f, m), r in zip(chunk, ranks):
                        writer.writerow([m["size"], m["comp"], layer, m["run"], f.stem] + r)
                print(f"layer {layer}: {min(i + args.batch_size, len(items))}/{len(items)}", end="\r", flush=True)
            print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
