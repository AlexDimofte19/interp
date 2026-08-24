#!/usr/bin/env bash
# The comp-0.0/0.2/0.4 next_action ("direction") probe run -- the recorded invocation behind
# that result. It is a thin wrapper: the actual sweep lives in train_next_action_arms.sh, so
# there is one implementation of "train every arm across a top-K sweep", not two.
#
# What this file pins is the *parameters* of the published run: which prepared datasets, where
# the probes and logs go. Everything else is train_next_action_arms.sh's default, and any of
# it can still be overridden here on the command line.
#
#   ./scripts/train_next_action_direction_probe.sh
#   ARMS="jlens random" ./scripts/train_next_action_direction_probe.sh
#
# The datasets come from scripts/prepare_next_action_jlens_by_complexity.sh (or
# prepare_next_action_arms.sh with COMPLEXITIES="0.0 0.2 0.4" and the same OUT).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export PREPARED=${PREPARED:-/workspace/prepared/next_action_comp0.0-0.2-0.4}
export PROBES=${PROBES:-/workspace/probes/next_action_comp0.0-0.2-0.4}

exec "$REPO/scripts/train_next_action_arms.sh" "$@"
