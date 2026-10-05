# Model Internals Speak Volumes: J-Lens Loudness Predicts Decodability

We represent a signal by a small subset of the model's vocabulary and define **J-Lens loudness**
as the log probability mass the Jacobian lens (J-Lens) assigns to that subset at a given
activation. We compute loudness over the reasoning chains of GPT-OSS-20B and Qwen3.6-35B-A3B
agents navigating 2D grid worlds. We use it to pick the loudest layer and the loudest token
positions in each chain, then train linear and MLP probes there and test whether the signal is
more decodable. The main signal is **direction**: next-action probes, labelled with the model's
local belief. The control signal is **grid**: cell-content probes. Logit-Lens loudness and random
positions are the baselines.

## Setup

```bash
uv sync --extra gpu        # model loading needs the gpu extra
uv sync --extra notebook   # for the notebooks
```

Everything that loads a model needs a GPU. Run it on **one** GPU, because spreading these MoE
models over several produces NaNs. Qwen3.6-35B-A3B needs an 80 GB card. Every path is written
out at the top of each wrapper (trajectories under `/workspace/trajectories`, outputs under
`/workspace/{activations,prepared,rollouts,probes,results}`). Edit those variables to match your
machine.

The signal vocabularies are committed in `data/jlens/`. To rebuild them, run
`notebooks/direction_tokens.ipynb` and `notebooks/grid_tokens.ipynb` (or their `_qwen` versions),
then prune the grid vocabulary:

```bash
uv run python scripts/prune_grid_vocabulary.py --write
uv run python scripts/prune_grid_vocabulary.py --in data/jlens/qwen/grid_tokens_full_qwen3-6-35b-a3b.json --write
```

## Reproducing the experiments

There is one wrapper tree per model and signal:

```
wrappers/gptoss_analysis/direction/
wrappers/gptoss_analysis/grid/
wrappers/qwen_analysis/direction/
wrappers/qwen_analysis/grid/
```

**All four trees contain the same files, running the same code.** Only the values at the top of
each file (model, paths, layer) differ. Each step below names a file relative to a tree. To
reproduce it for any model and signal, run the same command from the respective folder. Run the
steps in order, and run all scripts with `bash` from anywhere: they `cd` to the repo root
themselves.

### 0. Fit the J-Lens (once per model)

```bash
bash wrappers/gptoss_analysis/fit_jlens.sh
bash wrappers/qwen_analysis/fit_jlens.sh
```

### 1. Find the loudest layer (Figure 2, Figure 5)

```bash
bash 1_loudest_layer/sample_loudness_profile.sh
bash 1_loudest_layer/join_loudness_profile.sh
```

Then run the notebook in `1_loudest_layer/`. It plots mean J-Lens and Logit-Lens loudness per layer
and reports the argmax. The layers used are GPT-OSS-20B direction 15 and grid 14, and
Qwen3.6-35B-A3B 27 for both.

### 2. Select tokens and build the datasets

```bash
bash 2_dataset_creation/1_p2_selection/p2_training_selection.sh   # J-Lens, Logit-Lens and random top-N, train split
bash 2_dataset_creation/1_p2_selection/p2_eval_selection.sh       # the same, eval split
bash 2_dataset_creation/1_p2_selection/heldout_sample.sh          # every reasoning token of the held-out set

bash 2_dataset_creation/2_preparations/prepare_jlens.sh
bash 2_dataset_creation/2_preparations/prepare_logitlens.sh
bash 2_dataset_creation/2_preparations/prepare_random.sh
```

Direction trees only: label every selected token with the model's local belief.

```bash
bash 2_dataset_creation/3_rollouts/rollout_jlens.sh
bash 2_dataset_creation/3_rollouts/rollout_logitlens.sh
bash 2_dataset_creation/3_rollouts/rollout_random.sh
```

### 3. Train and cross-evaluate the probes (Table 1, Table 4)

```bash
bash 3_train_and_eval_probes/train_*_all_selections.sh        # one file per tree
bash 3_train_and_eval_probes/cross_eval_*_all_selections.sh   # one file per tree
```

This trains a linear and an MLP probe on each selection, then evaluates every probe on every
selection's eval subset. The direction trees give Table 1 and the grid trees give Table 4.

### 4. Score the probes on every held-out token

Grid trees first build and evaluate the held-out cell dataset:

```bash
bash 4_loudness_evaluation/prepare_heldout_grid.sh
bash 4_loudness_evaluation/eval_heldout_grid.sh
```

Then, in every tree:

```bash
bash 4_loudness_evaluation/score_heldout_per_token.sh
```

### 5. Sentence-level selection (Table 2, GPT-OSS-20B direction only)

One cut per sentence: the J-Lens- or Logit-Lens-loudest token, a random token, or the sentence-final
punctuation (EOS). Run from `wrappers/gptoss_analysis/direction/5_sentence_level/`:

```bash
bash 2_activations/gather_eos.sh

for arm in jlens logitlens random eos; do 
  bash 1_rollouts/rollout_$arm.sh; 
done

for arm in jlens logitlens random; do
  bash 2_activations/gather_$arm.sh; 
done

bash 2_activations/link_end_of_reasoning.sh

for arm in jlens logitlens random eos; do     
  bash 3_preparations/prepare_$arm.sh; 
done

bash 3_preparations/intersect_arms.sh   # keep only the sentences all four arms hold

bash 4_train_and_eval_probes/train_sentence_probes_all_selections.sh

bash 4_train_and_eval_probes/cross_eval_sentence_probes_all_selections.sh
```

### 6. Figures

Once steps 1-4 have run for all four trees:

```bash
bash wrappers/loudness_analysis/build_random_eval_tables.sh   # per-token tables on the eval subsets
bash wrappers/loudness_analysis/join_opposite_signal.sh       # adds the other signal's loudness to the held-out tables
```

- `wrappers/loudness_analysis/probe_performance_by_loudness.ipynb` draws balanced accuracy per
  loudness decile for both models, both signals and both lenses (Figures 3, 4, 8, 9, 10). It also
  draws the signal-word-excluded variants and the opposite-signal controls. Set `EVAL` in its
  first cell to `"random_eval"` or `"heldout"`.
- `wrappers/lens_agreement/jlens_vs_logitlens_loudness.ipynb` draws the token-level agreement
  between J-Lens and Logit-Lens loudness, and between direction and grid loudness (Figures 6, 7).
