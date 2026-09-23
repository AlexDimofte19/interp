# Qwen final-norm fix

**The bug.** `apply_lens_transport` (`telos_interp/loudness_analysis/build_loudness_tables.py`)
applies the model's final RMSNorm as `x · w`. Qwen3.5/3.6 uses a zero-centred RMSNorm, `x · (1 + w)`.
The stored weights average 1.63 (range −0.23 to 2.48), so the lens used a gain of ~1.6 where the model
uses ~2.6, with the dimensions' relative weights shifted too. **Every Qwen J-lens and logit-lens number
on disk is affected.** gpt-oss (`x · w`) is not. The fitted J matrices are fine: the fit never touches
the final norm.

**What is affected, and what is not.**

| | affected? |
|---|---|
| Loudness tables (analysis CSVs, mass tables), and everything with loudness on an axis: layer profiles, decile plots, lens-agreement plots | **yes** |
| Which tokens the jlens / logitlens arms selected, and so what those probes trained on | **yes** (the picks were ranked with the wrong lens) |
| Random arm, all labels, residual activations (`.pt`), held-out probe accuracy overall | no |

**Hardware.** Everything that loads Qwen needs **one 80 GB card** (A100/H100 80 GB): the checkpoint is
67 GiB and all 256 experts stay resident. One GPU only: `device_map="auto"` across several NaNs this MoE.
Steps 0, 2, 3 and 4 are CPU.

## Step 0: the fix (here, before the A100 session)

1. `telos_interp/jlens_utils/models.py`: add `norm_offset: float = 0.0` to `ModelSpec`; Qwen's row
   sets `norm_offset=1.0`. A property of the model, declared in the registry like the norm key, never
   inferred from the weights.
2. `build_loudness_tables.py`: `norm_w = assets["norm_weight"].float().to(dev) + spec.norm_offset`,
   the one place the gather builds the norm, so both lenses get it.
3. Record `final_norm_offset` in every mass table's `.meta.json` (`annotate_mass_meta`), so a corrected
   table can be told from an old one without knowing which tree it sits in.
4. Tests: gpt-oss's lens output is bit-for-bit unchanged; Qwen's uses `1 + w`; the sidecar field is
   written. Full suite green.

`run_on_a100.sh` refuses to start without `norm_offset` in the checkout.

## Step 1: the GPU runs (A100): `run_on_a100.sh`

`bash wrappers/qwen_analysis/norm_fix/run_on_a100.sh`, about 3.2 h in all. Resumable, and `STAGES=` runs a subset.

| stage | what | tokens | time |
|---|---|---|---|
| `check` | `scripts/check_lens_final_norm.py`: 3 consecutive reasoning tokens of 1 held-out trajectory. At the last block the logit lens *is* the output head, so the gather's norm must reproduce the model's own logits (KL < 1e-2, 3/3 top-1). Also prints `w` and `1 + w` side by side. **Stops the script on FAIL.** | 3 | ~5 min |
| `eval_dir`, `eval_grid` | eval 52, L27, both lenses | same 20% sample (seed 42) as the old tables | ~11 min each |
| `ho_dir`, `ho_grid` | held-out 70, L27, both lenses | every reasoning token | ~20 min each |
| `prof_dir`, `prof_grid` | layer profiles, all 40 layers, both lenses | 5% of train 549 (~460k tokens) | ~61 min each |

All CSV-only (`--no-save-activations`), into new trees under `/workspace/activations/qwen_fixnorm/`.
Nothing on disk is overwritten. The one population change: the old direction profile was a 20% sample.
It is redone at 5%, as the grid one was (~1 h instead of ~4 h).

Order is cheapest-first, so the eval and held-out plots are unblocked after ~1 h even if the profiles
are left to run unattended.

## Step 2: redraw the loudness plots (CPU)

Every figure goes to a new folder (`*_fixnorm`); `provenance.py`'s `run_config.json` refuses to mix
two rulers in one folder, and the old figures stay for comparison.

- **Layer profiles.** `join_mass_tables.py` over `profile_p05_{direction,grid}`, then the two stage-1
  notebooks (`1_loudest_layer/qwen_p2_loudest_layer.ipynb`, `grid/1_loudest_layer/...`) pointed at
  the new joined CSVs. Does the direction argmax still say **27**? Everything downstream assumed it.
- **Held-out deciles.** `join_signal_loudness.py` widens the existing held-out per-token tables with
  the corrected columns from `heldout70_{direction,grid}` (no probe is re-scored). Then
  `summarise_probe_accuracy.py` and the two decile notebooks (`4_loudness_evaluation/`,
  `grid/4_loudness_evaluation/`).
- **Lens agreement.** `wrappers/lens_agreement/jlens_vs_logitlens_loudness.ipynb`, with the four Qwen
  tree constants pointed at `eval52_{direction,grid}`. gpt-oss cells unchanged.

## Step 3: record it

Append a log entry (`ICLR log.txt`): the check's three numbers, old vs new argmax layer, old vs new
decile slopes. Update `claude_session_readme.md`'s header, which currently says not to build on the
Qwen numbers.

## Step 4: decide about the probes (CPU, minutes)

Per eval trajectory, the overlap between the **recorded** picks (the old trees' `*_jlens_selection.json`,
what the probes trained on) and the top 60 under the **corrected** ranking from `eval52_*`.

- **High overlap** (≳ 80%): the lens-arm probes stand. Only their loudness axes changed, and step 2
  fixed those.
- **Low overlap:** the jlens/logitlens selections need re-gathering (≈ 1.8 h per signal for train
  549 + eval 52, from the original status files), then prepare and retrain those arms' probes. That
  is a separate decision, and hours of GPU.

The random arm needs nothing in either case.
