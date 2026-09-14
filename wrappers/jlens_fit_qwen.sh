#!/usr/bin/env bash
# The recorded invocation of the Qwen3.6-35B-A3B Jacobian-lens fit.
#
# The twin of scripts/reproduce_all.sh's stage 2 (jlens/jlens_fit_gpt_oss.py, 500 prompts,
# skip_first=414, max_seq_len=1024), against the replayed Qwen trajectories instead. Two
# things are not the same, and both are forced by the model:
#
#   1. LAYERS. Qwen3.6-35B-A3B is a hybrid stack -- 30 gated-DeltaNet `linear_attention`
#      layers and 10 `full_attention` layers, at indices 3, 7, ... 39. We fit the nine
#      full-attention layers below the top; 39 is itself full-attention and is the last
#      block, so it is the TARGET, where the lens is the identity by construction. That is
#      the same arrangement gpt-oss has at its layer 23.
#
#   2. THE WINDOW. Qwen's chains run 12k-30k tokens against gpt-oss's few hundred, so
#      "the first 1024 tokens" is the instruction prefix plus the opening of the reasoning,
#      not most of a trajectory. WINDOW_FRAC puts the averaged positions HALFWAY INTO THE
#      CHAIN while the forward still runs on everything before them, so those activations
#      carry their real context. The prefix is prefilled under no_grad and only the window
#      is in the autograd graph -- identical J, ~9x less backward work and graph memory.
#      tests/test_jlens_fit_qwen.py asserts that identity against a full-graph reference.
#
# SMOKE=1 runs two prompts and stops. Run it FIRST on any new host: it is what fixes
# DIM_BATCH from a measured peak instead of a guess, and it is where a missing backward in
# the DeltaNet path would surface. 71.9 GB of bf16 weights on an 80 GB card leave ~7 GB, so
# DIM_BATCH starts at 1; if that still OOMs, drop the lowest source layers (SOURCE_LAYERS
# "7 11 15 19 23 27 31 35" narrows the retained graph to 7..39, since jlens's
# start_graph_at keeps everything below min(source_layers) out of it).
#
# REQUIREMENTS. transformers >= 5.17 (4.57.x has no qwen3_5_moe, and Qwen ships no
# trust_remote_code fallback) -- that is now the pyproject pin. The `jlens` package itself
# is NOT a declared dependency and `uv sync` will not install it:
#
#   uv pip install -e /workspace/jlens/repo/jacobian-lens
#
# The run downloads 71.9 GB of weights on first use; only the tokenizer is cached.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml

TRAJECTORIES=${TRAJECTORIES:-/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576}
OUT_DIR=${OUT_DIR:-/workspace/jlens/qwen3_6_35b}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}

# 144 = 4 per (size x complexity) stratum, of the 549 replayed trajectories. The fit is
# resumable from OUT_DIR/ckpt.pt on the SAME prompt list, so extending to 288 or to all 549
# later is a continuation rather than a restart -- keep SEED and the window flags fixed, or
# the checkpoint refuses to resume (deliberately: it would be averaging two quantities).
N_PROMPTS=${N_PROMPTS:-144}
SEED=${SEED:-0}
SIZES=${SIZES:-}

# The window. WINDOW_FRAC=0 anchors it at the first reasoning token, which is the gpt-oss
# behaviour, and is the ablation rather than the run. MAX_WINDOW_END caps how far the
# no-grad prefill has to run on a 30k-token chain; empty = uncapped.
WINDOW_FRAC=${WINDOW_FRAC:-0.5}
WINDOW_SIZE=${WINDOW_SIZE:-1024}
MAX_WINDOW_END=${MAX_WINDOW_END:-}

SOURCE_LAYERS=${SOURCE_LAYERS:-"3 7 11 15 19 23 27 31 35"}
TARGET_LAYER=${TARGET_LAYER:-}      # empty = the last block, 39
DIM_BATCH=${DIM_BATCH:-1}
DTYPE=${DTYPE:-bfloat16}
# Single device on purpose. CLAUDE.md records device_map="auto" producing NaNs for the
# gpt-oss MoE; the fit checks the first prompt's J for non-finite values either way.
DEVICE_MAP=${DEVICE_MAP:-cuda}

EVAL_EVERY=${EVAL_EVERY:-8}
STOP_AT_DELTA=${STOP_AT_DELTA:-0.002}
MIN_PROMPTS=${MIN_PROMPTS:-48}

# The gated delta rule and the causal conv load kernels from the hub when they are present,
# and those are inference kernels -- a missing backward is what this fit would hit. Forcing
# the reference PyTorch path keeps the backward available; the log line
# "falling back to its reference PyTorch implementation" is the confirmation, not a warning
# to fix.
export DISABLE_KERNEL_MAPPING=${DISABLE_KERNEL_MAPPING:-1}

# Loading the model needs the `gpu` extra (accelerate, and the pinned kernels==0.12.0),
# which is not in the default dependencies. A `uv run` WITHOUT it also syncs the extra back
# out of the venv, so leaving it off can break the next run that assumed accelerate.
UV_EXTRAS=${UV_EXTRAS:---extra gpu}
EXTRA=${EXTRA:-}           # e.g. --dry-run --check-roundtrip, --keep-vision-tower

if [ -n "${SMOKE:-}" ]; then
    N_PROMPTS=2
    EVAL_EVERY=1
    MIN_PROMPTS=999999     # never stop early on a smoke run
    OUT_DIR=${SMOKE_OUT_DIR:-${OUT_DIR}_smoke}
    echo "== SMOKE: 2 prompts -> $OUT_DIR (delete it before the real run)"
fi

# `jlens` is not a declared dependency, and `uv run` syncs before it runs -- a sync whose extras
# differ from the last one PRUNES it. Checked through the same `uv run` so the check sees the
# post-sync venv, and failed fast: the alternative is discovering it hours in.
uv run --project "$REPO" $UV_EXTRAS python -c "import jlens" 2>/dev/null || {
    echo "!! jlens is not installed in the venv (uv sync prunes it -- it is not a declared dependency)." >&2
    echo "   uv pip install -e /workspace/jlens/repo/jacobian-lens" >&2
    exit 1
}

set -x
uv run --project "$REPO" $UV_EXTRAS python "$REPO/jlens/jlens_fit_qwen.py" \
    --trajectories-dir "$TRAJECTORIES" \
    --out-dir "$OUT_DIR" \
    --model-id "$MODEL_ID" \
    --n-prompts "$N_PROMPTS" \
    --seed "$SEED" \
    ${SIZES:+--sizes $SIZES} \
    --window-frac "$WINDOW_FRAC" \
    --window-size "$WINDOW_SIZE" \
    ${MAX_WINDOW_END:+--max-window-end "$MAX_WINDOW_END"} \
    --source-layers $SOURCE_LAYERS \
    ${TARGET_LAYER:+--target-layer "$TARGET_LAYER"} \
    --dim-batch "$DIM_BATCH" \
    --dtype "$DTYPE" \
    --device-map "$DEVICE_MAP" \
    --eval-every "$EVAL_EVERY" \
    --stop-at-delta "$STOP_AT_DELTA" \
    --min-prompts "$MIN_PROMPTS" \
    $EXTRA
