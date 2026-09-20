# Qwen P2, stage 3 — the local-belief probes and the cross-selection matrix

Two wrappers. `train_next_action_probes_all_selections.sh` trains one `next_action` probe per
arm per model type on the **local belief**; `cross_eval_local_belief_all_selections.sh` reads
every one of those probes on every arm's val slice. Run 2026-09-20 on layer 27, gpt-oss's
`reasoning_theatre` line reproduced on Qwen3.6-35B-A3B.

Artifacts:

```
/workspace/probes/qwen_p2_local_belief/qwen_p2_local_belief_{arm}_l27_{lr,mlp}.pt   6 probes
/workspace/probes/qwen_p2_local_belief/logs/{arm}_l27_{type}.txt                    train logs
/workspace/results/qwen_p2_local_belief/cross_selection_eval/{slice}/{arm}_{type}.txt   18 cells
```

## How to read it

**Down a column, never across a row.** Within one slice every probe is scored on identical
rows, so a difference is the probe weights and nothing else. Across rows the populations
differ — loud tokens are not uniformly drawn tokens — and no comparison is available.

Balanced accuracy vs the LOCAL belief. Chance is 0.25. `N`=3,120 rows / 52 trajectories per
slice. Diagonal cells (probe scored on its own selection) in **bold**.

### lr

| probe ↓ &nbsp; slice → | jlens | logitlens | random |
|---|---|---|---|
| **jlens**     | **0.5025** | 0.4297 | 0.3200 |
| **logitlens** | 0.4764 | **0.4496** | 0.3416 |
| **random**    | 0.4321 | 0.4205 | **0.3471** |

### mlp

| probe ↓ &nbsp; slice → | jlens | logitlens | random |
|---|---|---|---|
| **jlens**     | **0.6209** | 0.5001 | 0.3464 |
| **logitlens** | 0.5981 | **0.5467** | 0.3723 |
| **random**    | 0.5280 | 0.4688 | **0.3958** |

## What it says

**Every slice is won by its own probe, under both model types.** That is specialisation, and it
is the thing the matrix exists to detect — a scoreboard of diagonals alone cannot see it.

**On the random slice the jlens probe comes last**, 4.9 points *below* the control (0.3464 vs
0.3958, mlp). The random slice is the only population here that a lens did not choose, so it is
the only one that speaks to tokens at large: the jlens probe does not generalise to tokens it
did not pick. This is the ICLR entry 49(c) order inversion reproducing on a second model.

**Most of the diagonal spread is the tokens, not the probe.** jlens-diagonal over
random-diagonal is +22.5 points (0.6209 vs 0.3958), but the *control-trained* probe scores
0.5280 on jlens-selected tokens against 0.3958 on its own — loud tokens are easier for any
probe. Holding the population fixed, the jlens probe's advantage over the control is +9.3
points inside the jlens slice and **−4.9** inside the random slice.

## The diagonal check, and the statistic trap

Each diagonal must reproduce what the trainer reported for that probe. All six do, exactly —
**against `final_balanced_accuracy`, not `best_balanced_accuracy`.**

| probe | final_bal (= diagonal) | best_bal | gap |
|---|---|---|---|
| jlens lr | 0.5025 | 0.5137 | +0.0112 |
| jlens mlp | 0.6209 | 0.6338 | +0.0128 |
| logitlens lr | 0.4496 | 0.4617 | +0.0122 |
| logitlens mlp | 0.5467 | 0.5586 | +0.0119 |
| random lr | 0.3471 | 0.3506 | +0.0036 |
| random mlp | 0.3958 | 0.4070 | +0.0111 |

`best_balanced_accuracy` is a running `max` over the 50 epochs and **the trainer never
checkpoints the best weights** — the saved probe is the last epoch. So the high-water mark
belongs to weights that were discarded, and quoting it overstates every arm by ~1.2 points.
Report `final_balanced_accuracy`. A diagonal that misses by about that much is this, not a
moved split; a diagonal that misses by more is a moved split and invalidates the whole matrix.

## Local belief vs final action

The label is what the model answered when its reasoning was cut at that token, not where the
trajectory ended up. The two agree on 92–94% of rows, so most cells barely separate them
(`bal acc vs FINAL action` tracks `vs LOCAL` to within ~0.005 everywhere).

The 190–239 rows per slice where they differ are where the question lives, and there the
probes lean **local**: jlens mlp on the jlens slice predicts the local belief on 38.4% of those
rows against the final action on 29.5%, with 32.1% neither. Every arm shows the same sign
except the two logitlens-probe cells on lens slices. These are 200-odd rows — directional, not
yet a result.

## The verbalization split, and where it is missing

Each cell splits on whether the cut token **is itself a direction word** the model had already
typed (ICLR 42(e)/43). The gap is large: jlens mlp reads 0.6752 on the 592 direction-word rows
of its own slice against 0.6078 on the other 2,528, and on the logitlens slice 0.7059 (N=188)
against 0.4866 (N=2,932).

**The control arm has no such split.** Its manifest samples carry no `token` field at all, so
`eval_local_belief.py` prints `[direction-word split unavailable]` for all six random-slice
cells — including the control-on-control diagonal, which is exactly the baseline the confound
most needs. The lens arms record the token because the selection ranked on it; the control's
uniform draw was recorded without it. Fixing this means re-emitting the control's manifest with
the token string carried through, not re-gathering.

## Known limits

- **The eval set is 36% of its name.** `mass_eval_144` holds 55 trajectory JSONs and 52
  gathered, so every number above rests on 3,120 rows over 52 trajectories.
- **No error bars.** Nothing here is bootstrapped, and the natural cluster is the trajectory,
  not the row — 60 tokens of one chain share a prompt, a grid and a train of thought.
- **One layer.** Layer 27 came from a pooled row-level argmax with no clustered bar; see log
  entry 59.

## Reproducing

```bash
# both model types, all three arms (MODEL_TYPES=mlp to resume half a sweep)
bash wrappers/qwen_analysis/3_train_and_eval_probes/train_next_action_probes_all_selections.sh
bash wrappers/qwen_analysis/3_train_and_eval_probes/cross_eval_local_belief_all_selections.sh
```

The first pass over a dataset opens one `.pt` per sample over MooseFS at ~50/s (~36 min for all
six); `--cache-activations` packs them beside each manifest so every later pass is seconds. The
cross-eval does **not** use that cache — its 18 cells each re-read their slice's 3,120 tensors,
six at a time per slice.
