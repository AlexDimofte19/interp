#!/usr/bin/env bash
# Qwen P2, loudness evaluation step 2: add the local-belief label to the held-out per-token table.
#
# score_probes_heldout_per_token.sh labels each token with its step's agent_action -- the
# FINAL action. probe_accuracy_by_loudness.py bins on `label_local`, which only the
# every_token rollout knows. This joins the two on (name, step, token_idx).
#
# --probes names which probes to carry, as key=column_prefix: the key is the short name the
# output columns use, the prefix is how the probe appears in the scorer's table.
#
# --lens picks which lens's loudness becomes the axis; the scorer writes both, so run this
# twice with different LENS and OUT to get both axes.
#
# STILL NEEDS A COMMITMENT CSV. --commitment-csv is the row source -- the join iterates its
# (name, step, token) keys and takes the sentence coordinates from it -- and no Qwen one
# exists yet.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # repo root, as the stage-1/2 wrappers do

RESULTS=/workspace/results/qwen_p2_local_belief/heldout
TABLE=$RESULTS/per_token_scores.csv
COMMITMENT=$RESULTS/commitment_per_token.csv
ROLLOUTS=/workspace/rollouts/qwen_p2_heldout_every_token
OUT=$RESULTS/per_token.csv

SIGNAL_JSON=/workspace/repo/interp/data/jlens/qwen/direction_tokens_full_qwen3-6-35b-a3b.json
SIGNAL_NAME=direction

LAYER=27
LENS=jlens
ROWSET=qwen_p2

P=qwen_p2_local_belief
PROBES="jlens_lr=${P}_jlens_l${LAYER}_lr,jlens_mlp=${P}_jlens_l${LAYER}_mlp"
PROBES="$PROBES,logitlens_lr=${P}_logitlens_l${LAYER}_lr,logitlens_mlp=${P}_logitlens_l${LAYER}_mlp"
PROBES="$PROBES,random_lr=${P}_random_l${LAYER}_lr,random_mlp=${P}_random_l${LAYER}_mlp"

mkdir -p "$RESULTS"
cd "$REPO"

uv run python telos_interp/loudness_analysis/join_rollouts.py \
    --table "$TABLE" \
    --rollout-dir "$ROLLOUTS" \
    --commitment-csv "$COMMITMENT" \
    --signal-json "$SIGNAL_JSON" \
    --signal-name "$SIGNAL_NAME" \
    --lens "$LENS" \
    --layer "$LAYER" \
    --probes "$PROBES" \
    --rowset "$ROWSET" \
    --out "$OUT"
