# `wrappers/` — recorded invocations

**These are not machinery. They are the record of how a published result was produced.**

Every file here does one thing: pin a set of *parameter values* onto one script in `scripts/`.
Strip the env block out of `jlens_mass_l15.sh` and what is left is the same `CMD=(...)` array as
`jlens_reasoning_tokens_filtered.sh` — the two differ in `--direction-score`,
`--select-candidate-layers` and `--select-num-layers`, and in nothing else. That is the whole
point of them, and it is why they live apart from the code they call: `scripts/` changes when
behaviour should change, `wrappers/` changes only when a *new run* is made.

This is the same role `configs/**/*.conf` plays for `interp-cli`, which is why the `.ps1` files
are here too.

**Do not "clean up" a wrapper.** Changing a default here rewrites the description of a run that
already happened. If a new run needs different values, add a file; if a wrapper is wrong about
what was run, fix it and say so in `ICLR log.txt`.

Each still resolves the repo from its own location (`REPO=.../..`), so they run from anywhere.

## The gather arms

All four drive `telos_interp/loudness_analysis/build_loudness_tables.py`. They differ in how a token is *scored* and
how much of the tree is *kept*.

| Wrapper | Tree it built | The parameters that matter |
|---|---|---|
| `jlens_reasoning_tokens.sh` | `activations/jlens_reasoning_tokens` (superseded) | full sweep, saves **every** (token, layer) — this is what filled the volume |
| `jlens_reasoning_tokens_filtered.sh` | `activations/jlens_reasoning_tokens` | the count-era tree: `count` scoring, top-20 tokens × 3 layers, ~75× less disk |
| `jlens_mass_l15.sh` | `activations/jlens_mass_l15`, `logitlens_mass_l15` | the mass-era tree: `logprob_mass_full`, `--select-candidate-layers 15`, 1 layer/token |
| `jlens_extend_logitlens.sh` | adds an arm to an existing tree | CSV-only pass + merge; the only way to add a lens to a **pruned** tree |

## The rollout arms

Both drive `scripts/inference_oss/run_inference_strategies.sh`, which drives `run_inference.py`.

| Wrapper | Arms it produced |
|---|---|
| `rollout_belief_baseline_arms.sh` | `recorded_selection` (random replay) + the two logit-lens loud arms |
| `rollout_more_belief_arms.sh` | `eos` and `random_per_sentence` — the per-sentence controls |

## Prepare / train / analyse

| Wrapper | Drives | Pins |
|---|---|---|
| `prepare_next_action_jlens_by_complexity.sh` | `scripts/prepare_next_action_arms.sh` | the comp-0.0/0.2/0.4 restriction |
| `train_next_action_direction_probe.sh` | `scripts/train_next_action_arms.sh` | the published comp-0.0/0.2/0.4 probe run |
| `run_next_action_arms.sh` | prepare + train, in order | the end-to-end convenience path |
| `build_convinced_datasets.sh` | `scripts/build_convinced_dataset.py` | the three convinced datasets: train 2880 / val 720 / eval 360 |
| `delete_non_jlens_selected.sh` | `scripts/delete_non_jlens_selected.py` | the prune that was actually applied — **dry-run by default** |
| `run_analysis.sh` | `scripts/inference_oss/analysis.py` | the sentence-end rollout's 13 figures |

## Older rounds

`script.sh`, `general_probe_train.sh`, `reasoning_theatre.ps1` and `run_commands.ps1` are the
round-1 `interp-cli` invocations, kept for the same reason as everything else here.
