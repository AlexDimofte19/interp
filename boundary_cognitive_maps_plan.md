# Boundary grid-probe experiment

Status: agreed plan; implementation and training have not started.

## Outputs and isolation

Train four **general**, layer-15 probes: linear and MLP, each trained separately on pre-reasoning and post-reasoning boundary tokens. No size-specific probes in this pilot.

Store checkpoints under:

```text
/workspace/probes/2880_trajectory_trained_cogn_maps/
  general_l15_lr/
    pre_reasoning/
    post_reasoning/
  general_l15_mlp/
    pre_reasoning/
    post_reasoning/
```

Each boundary folder contains its checkpoint, training log, and configuration. Store prepared manifests and newly extracted boundary activations under a shared `prepared/` subdirectory of this experiment root.

Store evaluations under:

```text
/workspace/loudness_evaluation/2880_trajectory-trained_cogn_maps/
  eval_720_data/
    general_l15_lr/{pre_reasoning,post_reasoning,comparison}/
    general_l15_mlp/{pre_reasoning,post_reasoning,comparison}/
  heldout_360_data/
    general_l15_lr/{pre_reasoning,post_reasoning,comparison}/
    general_l15_mlp/{pre_reasoning,post_reasoning,comparison}/
  provenance/
```

Comparison folders contain paired results and trajectory plots. Preserve existing artifacts, use configuration-checked resumability, and avoid Claude's modified files. Run GPU stages serially when the device is available. Do not modify shared source datasets or overwrite another agent's artifacts.

## Minimal repository additions

Add only two executable scripts:

- `wrappers/boundary_cognitive_maps.sh`: explicit paths and resumable stage orchestration.
- `scripts/boundary_cognitive_maps.py`: boundary-specific adapter using existing preparation structures, gathering utilities, trainer, J-lens routines, metrics, and plotting styles.

Add one focused test module. This plan file is documentation, not another script. No changes to public APIs or existing default behavior are planned.

The adapter is needed because existing token-major preparation and loudness scoring target output/reasoning tokens rather than both boundary triplets. Reuse existing functionality rather than duplicating training, gathering, lens calculations, or statistical routines. Do not edit Claude's actively modified plotting implementation.

## Data and training

Use the exact existing membership lists; do not resplit:

- Training: `/workspace/splits/mass_train_2880.txt` (2,880 trajectories).
- Evaluation: `/workspace/splits/mass_eval_720.txt` (720 trajectories).
- Held out: `/workspace/trajectories/heldout360_names.txt` (360 trajectories).

Inspection verified these lists are pairwise disjoint and that the evaluation list matches `/workspace/prepared/next_action_mass_l15_eval_names.txt`.

Resolve source trajectories through the existing mass-dataset views and held-out directory. Validate cached activations against source trajectory contents; matching filenames alone are insufficient.

Locate `<|end|>`, `<|start|>`, and `assistant` at both boundaries using trajectory metadata and the analysis-to-final transition, not fixed output offsets. Extract missing layer-15 decoder-block output activations using the original token sequences. Record boundary category, token identity, category-relative index, absolute position, trajectory, and step.

Each triplet contributes three independent samples. Never concatenate the three activation vectors. Use each sample's corresponding step grid labels and preserve complete pre/post triplets. Keep every trajectory's samples in its prescribed partition.

Train linear and MLP probes separately for each boundary using the existing cognitive-map trainer and explicit train/eval manifests. Defaults:

- Seed: 42.
- Epochs: 50.
- Learning rate: `3e-4`.
- Weight decay: `0.001`.
- Batch size: 2048.
- Normalize activations using training statistics.
- Balanced class weights, without per-trajectory class downsampling.
- MLP hidden width: 1024; dropout: 0.
- Pad training grids to size 15 for compact-dataset compatibility.

Select checkpoints using the 720 set. Reserve the 360 set for final results. Evaluate suffix-trained probes only on suffix boundary tokens and output-trained probes only on post-reasoning boundary tokens.

## Evaluation and plots

Evaluate both the 720 and held-out 360 sets. Report native-cell balanced accuracy excluding padding, per-class recalls, and breakdowns by boundary token identity, grid size, and complexity.

Calculate J-lens grid log-probability mass at the same six boundary positions using the exact `data/jlens/grid_tokens_full.json`. Do not regenerate or silently substitute the vocabulary. Reuse existing J-lens transport, normalization, unembedding, and full-vocabulary mass routines directly on the boundary activations. Record vocabulary and lens hashes.

Write one table row per trajectory, step, boundary, and token, containing loudness and per-class probe correctness counts.

For each architecture and evaluation trajectory, produce one PNG showing aligned accuracy/loudness measurements and correlation scatter panels, distinguishing pre/post boundaries and the three token identities. Reuse existing plotting styles and include an HTML index.

Aggregate results include:

- Paired post-minus-pre accuracy differences, separately for linear and MLP probes.
- Balanced accuracy by loudness decile.
- Correlations within each boundary/token group.
- Correlations between paired loudness changes and accuracy changes.
- Trajectory-clustered bootstrap confidence intervals for aggregate comparisons.

Pool per-class counts before calculating aggregate balanced accuracy. Mark constant or insufficient samples as undefined correlations. Keep held-out findings separate from checkpoint-selection results. Test the expected post-reasoning performance drop without presuming its direction, and report associations without treating them as causal explanations.

Omit cognitive-map application, rollouts, and dense reasoning-token scoring.

## Validation and delivery

Test exact boundary selection, independent token sample dimensions, partition membership, step-specific labels, padding exclusion, loudness alignment, and count-based balanced accuracy against a hand-calculated example.

Smoke-test extraction, preparation, training, scoring, and plotting before the full run. Report incomplete or ambiguous boundaries explicitly; never silently replace trajectories or alter split membership.

Deliver four checkpoints, both evaluation suites, trajectory plots and indexes, correlation results, and reproducibility metadata. Preserve commands, source identities, split hashes, model/lens/vocabulary identities, parameters, and coverage in the run manifest.
