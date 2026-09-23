#!/usr/bin/env bash
# gpt-oss P2, loudness evaluation step 2: add the local-belief label to the held-out per-token table.
#
# The gpt-oss twin of wrappers/qwen_analysis/4_loudness_evaluation/join_heldout_rollouts.sh.
# score_probes_heldout_per_token.sh labels each token with the FINAL action. The decile
# notebook bins on `label_local`, which only the every_token rollout knows. This joins the two
# on (name, step, token_idx).
#
# --probes names which probes to carry, as key=column_prefix. --lens picks which lens's loudness
# becomes the axis; run twice with different LENS and OUT to get both axes.
#
# --commitment-csv is the ROW SOURCE: build it first with build_commitment_per_token.sh.
#
# --commitment off. The rollout here is dense, so the boundary is not limited by resolution as
# on Qwen, but the row source carries no boundary (see build_commitment_per_token.sh). The
# decile analysis reads none of the boundary columns, so nothing downstream is lost.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # repo root

RESULTS=/workspace/results/gptoss_p2_local_belief/heldout
TABLE=$RESULTS/per_token_scores.csv
COMMITMENT=$RESULTS/commitment_per_token.csv
ROLLOUTS=/workspace/reasoning_theatre/rollout_strategies_heldout360/every_token
OUT=$RESULTS/per_token.csv

SIGNAL_JSON=/workspace/jlens/direction_tokens_full.json
SIGNAL_NAME=direction

LAYER=15
LENS=jlens
ROWSET=gptoss_p2

# The column prefix is score_probes_per_token.py's probe_key(): "<parent dir>.<stem>". Getting
# it wrong fails loudly ("missing N column(s)").
P=gptoss_p2_local_belief
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
