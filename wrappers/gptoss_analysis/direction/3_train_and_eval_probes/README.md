# gpt-oss P2, stage 3: the local-belief probes and the cross-selection matrix

The gpt-oss twin of `wrappers/qwen_analysis/3_train_and_eval_probes/`. Two wrappers:
`train_next_action_probes_all_selections.sh` trains one `next_action` probe per arm per model
type on the **local belief**. `cross_eval_local_belief_all_selections.sh` scores every one of
those probes on every arm's val slice. **Not yet run.**

Artifacts, once run:

```
/workspace/probes/gptoss_p2_local_belief/gptoss_p2_local_belief_{arm}_l15_{lr,mlp}.pt   6 probes
/workspace/probes/gptoss_p2_local_belief/logs/{arm}_l15_{type}.txt                       train logs
/workspace/results/gptoss_p2_local_belief/cross_selection_eval/{slice}/{arm}_{type}.txt  18 cells
```

## How this differs from the Qwen run, and why

Every difference comes from reusing the existing gpt-oss trees instead of gathering new ones:

| | Qwen P2 | gpt-oss P2 |
|---|---|---|
| trees | `qwen_p2_selection{,_eval}`, gathered for P2 | `jlens_mass_l15` (jlens + random), `logitlens_mass_l15` (logitlens), **reused** |
| layer | 27, the direction profile's jlens argmax | 15, the gpt-oss convention |
| tokens per arm | 60 | 20 |
| selection pool | a 20% sample of each chain | the whole chain (`--data_sample_p` never passed) |
| train / val | 549 / 52 trajectories | 2,880 / 720 (the mass-era partition) |
| held-out set | 72, stride-64 rollout | heldout360, **dense** (stride 1) rollout |

So a Qwen-vs-gpt-oss difference in these tables is the model **plus** N, layer and pool.
Compare within a model first.

## How to read it

Same rules as the Qwen README: **down a column, never across a row**; report
`final_balanced_accuracy` for the diagonal check, never `best_balanced_accuracy`. The trainer
never checkpoints the best weights.

## Reproducing

```bash
bash wrappers/gptoss_analysis/direction/3_train_and_eval_probes/train_next_action_probes_all_selections.sh
bash wrappers/gptoss_analysis/direction/3_train_and_eval_probes/cross_eval_local_belief_all_selections.sh
```
