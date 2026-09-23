#!/usr/bin/env python3
"""Does the lens's final norm reproduce the model's own logits? A 3-token pass/fail check.

At the last decoder block the logit lens IS the model's output head: RMS-norm the block's
output with the model's final-norm weight, multiply by lm_head. So on three consecutive
reasoning tokens of one trajectory, the lens's logits must match the logits the model itself
returns, to bf16 precision. A lens with the wrong norm does not -- which is what
claude_session_readme.md (2026-09-23) says every Qwen lens output did: `x * w` where
Qwen3.5/3.6's zero-centred RMSNorm is `x * (1 + w)`.

Three candidate norms are scored against the model's logits:

  * `w`            -- what every Qwen lens output on disk used
  * `1 + w`        -- the zero-centred norm the handoff says Qwen needs
  * `gather`       -- whatever build_loudness_tables.py now applies, i.e. the norm weight plus
                      the model spec's `norm_offset`. This is the one that decides PASS: the
                      check is on the code path the sweep will run, not on a formula.

Residuals come from the gather's own hook (`extract_activations_batched`, decoder-block
outputs, pre-norm) and the lens from its own `apply_lens_transport`, so nothing here
re-implements what it is checking. Needs the model: an 80 GB card for Qwen.

    uv run --extra gpu python scripts/check_lens_final_norm.py \\
        --trajectory /workspace/trajectories/qwen3.6-35b/replayed_single_step/heldout_72/size5/<stem>.json \\
        --jlens_dir /workspace/jlens/qwen3_6_35b
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telos_interp.commands.gather_activations.gather_activations_utils import (  # noqa: E402
    extract_activations_batched,
)
from telos_interp.loudness_analysis.build_loudness_tables import (  # noqa: E402
    apply_lens_transport,
    reasoning_token_positions,
    resolve_spec_or_exit,
)

# KL(model || lens) in nats, averaged over the three tokens. bf16 rounding in the model's own
# head leaves ~1e-3; a wrong norm is orders of magnitude off.
KL_PASS = 1e-2


def pick_tokens(trajectory: dict, offset: int, n: int = 3) -> tuple[list[int], list[int]]:
    """The token ids up to and including `n` consecutive reasoning tokens, and their positions."""
    step = trajectory["steps"][0]
    positions = reasoning_token_positions(trajectory, step)
    chosen = [p[1] for p in positions[offset : offset + n]]
    if len(chosen) < n or chosen != list(range(chosen[0], chosen[0] + n)):
        raise SystemExit(f"no {n} consecutive reasoning tokens at offset {offset} of step 0")
    ids = (
        [t["token_id"] for t in trajectory["prompt"]["prompt_prefix_tokens"]]
        + [t["token_id"] for t in step["grid_state_tokens"]]
        + [t["token_id"] for t in trajectory["prompt"]["prompt_suffix_tokens"]]
        + [t["token_id"] for t in step["output_tokens"]]
    )
    return ids[: chosen[-1] + 1], chosen


def compare(lens_logits: torch.Tensor, model_logits: torch.Tensor) -> dict:
    """KL(model || lens), max |logit difference| and top-1 agreement, over the tokens."""
    lp_model = torch.log_softmax(model_logits, -1)
    lp_lens = torch.log_softmax(lens_logits, -1)
    kl = (lp_model.exp() * (lp_model - lp_lens)).sum(-1)
    return {
        "kl": kl.mean().item(),
        "max_abs_dlogit": (lens_logits - model_logits).abs().max().item(),
        "top1_agree": int((lens_logits.argmax(-1) == model_logits.argmax(-1)).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", type=Path, required=True, help="one trajectory JSON")
    ap.add_argument("--jlens_dir", type=Path, required=True, help="holds the model's *_unembed.pt cache")
    ap.add_argument("--model-id", default=None, help="HF repo id, if the trajectory's serving id is unknown")
    ap.add_argument("--offset", type=int, default=10, help="first of the 3 reasoning tokens (default 10)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from scripts.jlens_action_ranks import ensure_unembed_assets
    from transformers import AutoModelForCausalLM

    trajectory = json.loads(args.trajectory.read_text())
    spec = resolve_spec_or_exit(trajectory["model_params"]["model_id"], args.model_id)
    ids, positions = pick_tokens(trajectory, args.offset)
    print(f"{spec.model_id}: {args.trajectory.name}, reasoning tokens at abs_pos {positions}")

    model = AutoModelForCausalLM.from_pretrained(spec.model_id, device_map=args.device, dtype="auto")
    model.eval()
    last = spec.resolve_target_layer(spec.num_hidden_layers(model.config))
    input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)

    # The residual the lens reads: the last block's output, pre-norm, from the gather's own hook.
    h = extract_activations_batched(model, input_ids, None, [positions], [last], keep_on_device=True)[last]
    with torch.no_grad():
        # The positions are the last three of the sequence, so logits_to_keep=3 is exactly them.
        model_logits = model(input_ids, use_cache=False, logits_to_keep=len(positions)).logits[0].float()

    assets = ensure_unembed_assets(args.jlens_dir, spec)
    lm_head = assets["lm_head"].to(args.device).float()
    w = assets["norm_weight"].float().to(args.device)
    eps = assets["rms_eps"]
    offset = getattr(spec, "norm_offset", None)

    candidates = {"w": w, "1 + w": 1.0 + w}
    if offset is not None:
        candidates[f"gather (w + {offset:g})"] = w + offset
    no_rows = torch.empty(0, dtype=torch.long, device=args.device)

    results = {}
    for name, norm_w in candidates.items():
        normed = apply_lens_transport(h.float().clone().unsqueeze(0), None, no_rows, norm_w, eps)[0]
        results[name] = compare(normed @ lm_head.T, model_logits)

    print(f"\n{'norm':<20}{'KL(model||lens)':>18}{'max |dlogit|':>15}{'top-1 agree':>13}")
    for name, r in results.items():
        print(f"{name:<20}{r['kl']:>18.3e}{r['max_abs_dlogit']:>15.3f}{r['top1_agree']:>10}/{len(positions)}")

    if offset is None:
        print("\n!! the model spec has no `norm_offset`: the fix is not in this checkout. The two rows above")
        print("   still say which norm is right; the gather cannot be run until the fix is in.")
        return 1
    gather = results[f"gather (w + {offset:g})"]
    passed = gather["kl"] < KL_PASS and gather["top1_agree"] == len(positions)
    print(
        f"\n{'PASS' if passed else 'FAIL'}: the gather's norm {'reproduces' if passed else 'does NOT reproduce'} "
        f"the model's logits (KL {gather['kl']:.2e}, threshold {KL_PASS:g})"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
