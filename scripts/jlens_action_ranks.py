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
from pathlib import Path

import torch

ACTIONS = ["RIGHT", "LEFT", "UP", "DOWN"]
MODEL_ID = "openai/gpt-oss-20b"
TARGET_LAYER = 23  # jlens target: final decoder block, lens is identity here

PATH_RE = re.compile(
    r"size(?P<size>\d+)_comp(?P<comp>[\d.]+)_(?P<run>\d+).*?"
    r"layer_(?P<layer>\d+).*?prompt_suffix"
)


def ensure_unembed_assets(jlens_dir: Path) -> dict:
    """Extract lm_head.weight + final norm from the HF checkpoint (once)."""
    cache = jlens_dir / "gpt-oss-20b_unembed.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=True)

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    index = json.load(open(hf_hub_download(MODEL_ID, "model.safetensors.index.json")))
    shard = index["weight_map"]["lm_head.weight"]
    assert index["weight_map"]["model.norm.weight"] == shard
    print(f"downloading {shard} (~4.2 GB, one-time; extracted tensors are cached)...")
    shard_path = hf_hub_download(MODEL_ID, shard)
    with safe_open(shard_path, framework="pt") as f:
        assets = {
            "lm_head": f.get_tensor("lm_head.weight"),
            "norm_weight": f.get_tensor("model.norm.weight"),
            "rms_eps": cfg["rms_norm_eps"],
        }
    torch.save(assets, cache)
    print(f"cached {cache} — you may delete the shard from the HF cache to reclaim disk")
    return assets


def action_token_ids() -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = {}
    for a in ACTIONS:
        enc = tok.encode(a, add_special_tokens=False)
        assert len(enc) == 1, f"{a!r} is not a single token: {enc}"
        ids[a] = enc[0]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations_root", type=Path, required=True)
    ap.add_argument("--jlens_dir", type=Path, required=True)
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
    for f in files:
        m = PATH_RE.search(f.as_posix())
        if not m:
            skipped += 1
            continue
        by_layer[int(m["layer"])].append((f, m))
    if skipped:
        print(f"warning: {skipped} files did not match the expected path pattern")

    lens = torch.load(args.jlens_dir / "gpt-oss-20b_jacobian_lens.pt", map_location="cpu")
    assets = ensure_unembed_assets(args.jlens_dir)
    ids = action_token_ids()
    print(f"action token ids: {ids}")

    dev = torch.device(args.device)
    lm_head = assets["lm_head"].to(dev)  # [vocab, d_model] bf16
    norm_w = assets["norm_weight"].float().to(dev)
    eps = assets["rms_eps"]
    id_cols = [ids[a] for a in ACTIONS]

    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["size", "complexity", "layer", "run", "token"]
                        + [f"{a}_position" for a in ACTIONS])
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
                h = torch.stack(
                    [torch.load(f, map_location="cpu", weights_only=True) for f, _ in chunk]
                ).float().to(dev)
                with torch.no_grad():
                    if J is not None:
                        h = h @ J.T
                    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w
                    logits = (h.to(lm_head.dtype) @ lm_head.T).float()  # [B, vocab]
                    ranks = torch.stack(
                        [(logits > logits[:, tid : tid + 1]).sum(1) for tid in id_cols],
                        dim=1,
                    )  # [B, 4], rank 0 = argmax
                ranks = ranks.cpu().tolist()
                for (f, m), r in zip(chunk, ranks):
                    writer.writerow(
                        [m["size"], m["comp"], layer, m["run"], f.stem] + r
                    )
                print(f"layer {layer}: {min(i + args.batch_size, len(items))}/{len(items)}",
                      end="\r")
            print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
