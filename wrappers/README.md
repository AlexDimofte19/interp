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

Both drive `telos_interp/loudness_analysis/rollouts/run_inference_strategies.sh`, which drives `run_inference.py`.

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
| `run_analysis.sh` | `telos_interp/loudness_analysis/rollouts/analysis.py` | the sentence-end rollout's 13 figures |
| `loudness_report.sh` | `loudness_analysis/analysis/*` + `plotting/figures.py` | one ruler per run: `LENS`, `SIGNAL`, `PROBE_TYPE`, and `EXCLUDE_SIGNAL_WORDS` for the verbalisation control |

`loudness_report.sh` is the one wrapper here that does **not** record a published run: it
is the recorded invocation for the analysis and figure CLIs that replaced four analysis
scripts and three plotters. Run it twice, once per `LENS` -- at layer 15 the two lenses'
top-20 sets overlap only about half, so a folder holding both rulers is a folder of figures
that cannot be compared with each other. `provenance.RunConfig.guard` refuses the second
run into the same `--out` rather than overwriting half of them.

## The Qwen lens

| Wrapper | Drives | Pins |
|---|---|---|
| `jlens_fit_qwen.sh` | `jlens/jlens_fit_qwen.py` | the Qwen3.6-35B-A3B fit: 144 replayed trajectories, the nine full-attention source layers `3 7 ... 35` against target 39, and a mid-chain window (`WINDOW_FRAC 0.5`, `WINDOW_SIZE 1024`) |

The only wrapper here whose script is not under `scripts/`: `jlens/` holds both fits, and the
gpt-oss one is stage 2 of `reproduce_all.sh` rather than a wrapper. `SMOKE=1` runs two prompts
into a separate `--out-dir` and is what fixes `DIM_BATCH` from a measured peak -- run it first
on any new host. It has no published result yet.

## Older rounds

`script.sh`, `general_probe_train.sh`, `reasoning_theatre.ps1` and `run_commands.ps1` are the
round-1 `interp-cli` invocations, kept for the same reason as everything else here.

## The boundary round

| Wrapper | Drives | Pins |
|---|---|---|
| `boundary_cognitive_maps.sh` | `scripts/boundary_cognitive_maps.py` + `interp-cli train_cognitive_map_probe` | the four boundary probes: layer 15, `{lr, mlp}` x `{pre_reasoning, post_reasoning}`, trained on the 2,880 and selected on the 720, with the 360 reserved |

The one wrapper here that drives a whole pipeline rather than a single script, because the
experiment's eight stages have to agree about paths: the tree the gather writes is the tree
the manifests point into, and the manifests' partitions are the evaluation's partitions.
Every stage tests for its own output first, so it is resumable and a second run is a no-op
check. `STAGES=` selects, `FORCE=1` redoes, `DRY_RUN=1` prints. `LIMIT` is for smoke runs
only -- it truncates every membership list, which makes the numbers meaningless.
