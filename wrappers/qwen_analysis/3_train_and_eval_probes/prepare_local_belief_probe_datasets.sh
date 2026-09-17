#!/usr/bin/env bash
# Qwen P2, step 4: build the LOCAL-BELIEF probe datasets for all three arms.
#
# A master over three arm scripts, run in order:
#
#   prepare_local_belief_jlens.sh       the Jacobian lens' loudest 60 tokens per trajectory
#   prepare_local_belief_logitlens.sh   the logit lens' loudest 60, ~half the same tokens
#   prepare_local_belief_random.sh      the matched control, 60 drawn uniformly
#
# Each is self-contained -- run one alone to rebuild just that arm -- and each does the same
# three stages for train and val. The stages are below; the per-arm notes are in the files.
#
#   1. prepare   (CPU, minutes)  tree + selection record -> token-major manifest
#   2. rollout   (GPU, hours)    truncate at each selected token, force one action token
#   3. relabel   (CPU, minutes)  join model_action onto the manifest as `label`
#
# WHY THE ROLLOUT IS HERE AT ALL. prepare labels every sample with the trajectory's final
# agent_action -- where the model ENDED UP. The probe is supposed to read where it WAS at
# that token. --strategy recorded_selection replays the arm's own token_idx picks, so the
# cut points are exactly the tokens that arm's probe trains on and nothing else is rolled
# out; the other four strategies choose their own cut points and would label the wrong ones.
#
# WHAT THE ROLLOUT DOES. output_tokens[:pos+1] is kept, the trajectory's own final-channel
# prefix is appended verbatim (lifted from the data, not re-tokenized), and exactly one
# token is generated -- which, because the prefix primes `{ "action": "`, IS the action.
# The model gets no opportunity to say anything else.
#
# LOCAL BELIEF vs FINAL ACTION. The new `label` is where the model was at that token;
# relabel keeps the old one as `final_label`, alongside rollout_answer_prob (its confidence
# in the answer it gave), rollout_correct (label == final_label, i.e. already committed),
# cutoff_kind and dir_logmass. The two coming apart before the model commits is the entire
# reason this pipeline exists.
#
# --keep-kinds recorded DROPS THE BOOKENDS. Every strategy appends a no_reasoning and an
# end_of_reasoning cutoff so its first and last eval matches every other arm's. They are the
# same two prompts in all three arms and end_of_reasoning is near-deterministic, so keeping
# them would inflate every arm equally and dilute the contrast. They remain in the rollout
# JSONs if an endpoint analysis ever wants them.
#
# SAMPLES ARE DROPPED. A cutoff that produced no parseable action disappears from the
# manifest, so the arms end up with slightly different row counts and none has the full 60
# per trajectory. Read each stage-3 drop count rather than assuming.
#
# THE OUTPUT IS A NEW PREFIX, /workspace/prepared/qwen_p2_local_<arm>_<half>. The
# final-action manifests under /workspace/prepared/qwen_p2 are left intact, since nothing
# else on disk would distinguish the two labels. Point train_next_action_probes_all_
# selections.sh at the local one.
#
# SERIAL ON PURPOSE. Stage 2 loads 66 GiB of weights onto one GPU (device_map=auto would
# spread this MoE over several and produce NaNs), so two arms cannot share a card. Every
# stage is resumable -- --skip-existing on the rollout, and prepare/relabel rewrite their
# manifest -- so a killed run is restarted by running the master again.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$HERE/prepare_local_belief_jlens.sh"
bash "$HERE/prepare_local_belief_logitlens.sh"
bash "$HERE/prepare_local_belief_random.sh"
