#!/usr/bin/env bash
# Qwen P2, step 7c: add the local-belief label to the held-out per-token table.
#
# score_probes_heldout_per_token.sh labels each token with its step's agent_action -- the
# FINAL action. probe_accuracy_by_loudness.py bins on `label_local`, which only the
# every_token rollout knows. This joins the two on (name, step, token_idx).
#
# --lens picks which lens's loudness becomes the axis the figures bin on; unlike the scorer,
# this one takes a single lens. Run it twice with different --lens and --out to get both.
#
# NOT RUNNABLE YET: join_rollouts.py hard-codes PROBE_SOURCE, ten gpt-oss probe columns it
# raises SystemExit without, and reads --commitment-csv unconditionally (an entry-39/40/41
# artefact with no Qwen equivalent). --extra-probes only ADDS to that registry. The
# invocation below is the one that should work once PROBE_SOURCE is replaceable and the
# commitment CSV optional.
set -euo pipefail

REPO=/workspace/repo/interp

TABLE=/workspace/results/qwen_p2_local_belief/heldout/per_token_scores.csv
ROLLOUTS=/workspace/rollouts/qwen_p2_heldout_every_token
OUT=/workspace/results/qwen_p2_local_belief/heldout/per_token.csv

SIGNAL_JSON=/workspace/repo/interp/data/jlens/qwen/direction_tokens_full_qwen3-6-35b-a3b.json
SIGNAL_NAME=direction

LAYER=27
LENS=jlens

mkdir -p "$(dirname "$OUT")"
cd "$REPO"

uv run python telos_interp/loudness_analysis/join_rollouts.py \
    --table "$TABLE" \
    --rollout-dir "$ROLLOUTS" \
    --signal-json "$SIGNAL_JSON" \
    --signal-name "$SIGNAL_NAME" \
    --lens "$LENS" \
    --layer "$LAYER" \
    --out "$OUT"
