"""Render an anthropics/jacobian-lens slice page for one trajectory step,
with LEFT/RIGHT/UP/DOWN pinned: rank-tracking charts + rank heatmap over
every (position, layer).

Needs a GPU that fits gpt-oss-20b — run on the server, not the laptop.
Requires `pip install -e jacobian-lens` (github.com/anthropics/jacobian-lens).

Usage:
  python scripts/jlens_slice_page.py \
    --trajectory /workspace/trajectories/.../together_..._size11_comp0.0_0.json \
    --step 0 \
    --lens /workspace/jlens/gpt-oss-20b_jacobian_lens.pt \
    --out slice_size11_comp0.0_run0_step0.html
"""

import argparse
import json
from pathlib import Path

import jlens
import torch
import transformers
from jlens.vis import build_page, compute_slice

ACTIONS = ["RIGHT", "LEFT", "UP", "DOWN"]
MODEL_ID = "openai/gpt-oss-20b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("slice.html"))
    ap.add_argument("--layer_stride", type=int, default=1)
    ap.add_argument(
        "--last_n_tokens",
        type=int,
        default=None,
        help="render only the last N positions (forward pass still uses all)",
    )
    ap.add_argument("--include_output", action="store_true", help="also include the step's generated output tokens")
    args = ap.parse_args()

    traj = json.load(open(args.trajectory, encoding="utf-8"))
    step = traj["steps"][args.step]
    ids = (
        [t["token_id"] for t in traj["prompt"]["prompt_prefix_tokens"]]
        + [t["token_id"] for t in step["grid_state_tokens"]]
        + [t["token_id"] for t in traj["prompt"]["prompt_suffix_tokens"]]
    )
    if args.include_output:
        ids += [t["token_id"] for t in step["output_tokens"]]
    print(f"{len(ids)} input tokens, agent_action={step['agent_action']}", flush=True)

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(str(args.lens))
    print(lens, flush=True)

    pinned = {tok.encode(a, add_special_tokens=False)[0] for a in ACTIONS}
    # feed the trajectory's exact token ids instead of re-tokenizing text
    input_ids = torch.tensor([ids])
    model.encode = lambda text, max_length=0: input_ids.to(model.input_device)

    slice_data = compute_slice(
        model,
        lens,
        prompt="",
        pinned_token_ids=pinned,
        layer_stride=args.layer_stride,
        last_n_tokens=args.last_n_tokens,
        max_seq_len=len(ids),
    )

    name = args.trajectory.stem
    page, _, payload = build_page(
        slice_data,
        prompt=tok.decode(ids),
        title=f"{name} step {args.step}",
        description=(
            f"gpt-oss-20b jlens slice; taken action: {step['agent_action']}. "
            "LEFT/RIGHT/UP/DOWN pinned - see their rank charts and heatmap."
        ),
        mode="embed",
    )
    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out} ({payload / 1e6:.1f} MB payload)", flush=True)


if __name__ == "__main__":
    main()
