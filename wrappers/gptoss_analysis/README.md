# gpt-oss P2: the Qwen P2 pipeline, on gpt-oss-20b

Twins of `wrappers/qwen_analysis/` (direction) and `wrappers/qwen_analysis/grid/` (grid), with
the same four stages under the same file names, pointed at gpt-oss-20b's data. **Not yet run.**

```
direction/   direction vocabulary, next_action probes on the local belief
grid/        pruned grid vocabulary, binary grid_tile probes (empty / wall / agent / goal)
```

## Reuse, not re-gather

gpt-oss already has the trees this pipeline needs, and they are **pruned**, so the wrappers
read them in place:

| tree | holds | used by |
|---|---|---|
| `jlens_mass_l15` | jlens + random arms, 20 tokens, layer 15, whole chain, 3,600 trajectories | direction (jlens, random), grid (random) |
| `logitlens_mass_l15` | logitlens arm, same 3,600 names | direction (logitlens) |
| `mass_{train2880,eval720}_view` | the mass-era partition, as trajectory folders | every train / val split |
| `heldout360_l15` | a layer-15 `.pt` for every reasoning token of the 360 | both held-out stages |
| `heldout360_lens` | both lenses' direction mass, 7:23 | direction held-out loudness |
| `reasoning_theatre/rollout_strategies_heldout360/every_token` | dense (stride 1) rollout | direction held-out local belief |
| `prepared/grid_binary_l15_heldout360` | grid_tile held-out dataset, 25 cells | grid held-out eval |

In `direction/`, the wrappers that sit where Qwen's gathers sit (`sample_loudness_profile.sh`,
`p2_*_selection.sh`, `heldout_sample.sh`) do not gather. They run
`direction/check_reused_tree.py`, which confirms every trajectory is present, unsampled, on the
right vocabulary, and has the recorded arms at candidate layer 15. They write nothing.

`grid/` has to gather where no tree exists: the grid-ranked lens arms (the direction tree never
saved those tokens), and mass tables against the **pruned** vocabulary. The existing
`heldout360_lens_grid` uses the full 1258-token vocabulary. It still reuses the random control
and the held-out tensors.

## Differences from Qwen P2

| | Qwen P2 | gpt-oss P2 |
|---|---|---|
| layer | 27 (direction jlens argmax) | 15 (convention; force-kept in every tree) |
| tokens per arm | 60 | 20 |
| selection pool | 20% sample | **whole chain** (`SAMPLE_PERCENT=1.0`) |
| train / val | 549 / 52 | 2,880 / 720 |
| held-out | 72, rollout stride 64 | 360, rollout **stride 1** |

## Open items

- **`--jlens_dir`** is `/workspace/jlens/gridenv`, per CLAUDE.md. On the host these files were
  written on, `gridenv/` holds `ckpt.pt` and the gpt-oss lens + unembed sit one level up in
  `/workspace/jlens/`. Check before the grid gathers run, or the first one re-downloads a
  4.2 GB shard to rebuild the unembed.
- **Grid stage 4 has no loudness binning** until a binary grid probe type exists in
  `loudness_analysis/probes.py`. See `wrappers/qwen_analysis/grid/README.md`.
- **Grid datasets are padded to 15** (prepare auto-pads grid_tile across sizes), as the
  existing `grid_binary_l15_*` datasets already are.
- **Commitment boundary.** The held-out rollout is dense, so the boundary is resolvable. But
  `join_rollout_answers.py` never fills `convinced_sentence_idx`, so the join still runs
  `--commitment off`.
