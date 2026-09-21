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
# --commitment-csv is the ROW SOURCE: the join iterates its (name, step, token) keys and
# takes the sentence coordinates from it, so a token absent there is absent from the output
# whatever the probe table holds. Build it first with build_commitment_per_token.sh.
#
# --commitment off, AND THAT IS A RESULT, NOT A SHORTCUT. The boundary columns (convinced_*,
# rel_sentence, rel_token, is_convinced, x_sentence) come out blank because this rollout is
# strided at 64: the boundary is resolved only to +-64 tokens and a relapse between two
# sampled cutoffs is invisible, so a number there would be precision the arm does not have.
# probe_accuracy_by_loudness.py reads none of them -- it bins on loudness and pairs against
# frac_in_sentence, both of which ARE filled -- so nothing downstream is lost. A commitment
# analysis on Qwen needs a dense (stride 1) or `eos` arm first; then run --commitment on,
# which fails loudly rather than writing an empty column that looks computed.
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

# The column prefix is score_probes_per_token.py's probe_key(): "<parent dir>.<stem>", so
# the probe directory name leads and the full stem follows it -- NOT the stem alone. Getting
# it wrong fails loudly ("missing 30 column(s)"), which is the only reason it is safe to
# write the doubled name out like this.
P=qwen_p2_local_belief
K="${P}.${P}"
PROBES="jlens_lr=${K}_jlens_l${LAYER}_lr,jlens_mlp=${K}_jlens_l${LAYER}_mlp"
PROBES="$PROBES,logitlens_lr=${K}_logitlens_l${LAYER}_lr,logitlens_mlp=${K}_logitlens_l${LAYER}_mlp"
PROBES="$PROBES,random_lr=${K}_random_l${LAYER}_lr,random_mlp=${K}_random_l${LAYER}_mlp"

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
    --commitment off \
    --out "$OUT"
