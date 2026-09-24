# Qwen norm fix: what was run, 2026-09-23/24 (A100 80 GB)

The Qwen lens applied the final RMSNorm as `x · w`; Qwen3.6 is `x · (1 + w)`. The fix (`99faead`,
`norm_offset=1.0`) was already committed. This run re-gathered the Qwen lens tables with it and redrew
every plot that has loudness on an axis. Nothing on disk was overwritten. Every output is in a new
`*_fixnorm` / `qwen_fixnorm` location, and the old ones stay for comparison. **Nothing is committed**:
the new files are untracked in the working tree on `reasoning_theatre`.

## 1. GPU: `run_on_a100.sh`, all 7 stages rc=0 (23:44 → 02:29 UTC)

| stage | result |
|---|---|
| `check` | PASS. KL(model‖lens) with the old `w`: 5.46e-02, max \|Δlogit\| 12.7. With `1 + w`: 1.25e-04, max \|Δlogit\| 0.055. Top-1 agrees 3/3 for both. |
| `eval_dir`, `eval_grid` | eval 52, 20% sample, L27 → `/workspace/activations/qwen_fixnorm/eval52_{direction,grid}` |
| `ho_dir`, `ho_grid` | held-out 70, every token, L27 → `/workspace/activations/qwen_fixnorm/heldout70_{direction,grid}` |
| `prof_dir`, `prof_grid` | train 549, 5%, all 40 layers → `/workspace/activations/qwen_fixnorm/profile_p05_{direction,grid}` |

These are lens CSVs and mass tables only (`--no-save-activations`), 20 GB in total. Each mass table's
`.meta.json` records `final_norm_offset: 1.0`. Logs and `status.txt` are in `/workspace/logs/qwen_fixnorm/`.

## 2. Plots (all sent to phone)

Each notebook is an executed `_fixnorm` copy **next to its original in `wrappers/`**. The original is left
as the record of the old run. `wrappers/README.md` § "The Qwen final-norm fix" lists them all.

**Layer profile (loudest layer)**
- direction: `/workspace/loudness_evaluation/qwen_p2_layer_profile_fixnorm/figures/{jlens,logitlens}_loudness_per_layer.png`
- grid: `/workspace/loudness_evaluation/qwen_p2_grid_layer_profile_fixnorm/figures/{jlens,logitlens}_loudness_per_layer.png`
- notebooks: `wrappers/qwen_analysis/1_loudest_layer/qwen_p2_loudest_layer_fixnorm.ipynb` and
  `wrappers/qwen_analysis/grid/1_loudest_layer/qwen_p2_grid_loudest_layer_fixnorm.ipynb`

| | old argmax | new argmax |
|---|---|---|
| J-lens direction | L27 (−5.249, next L19 −5.495) | **L27** (−7.278), but **L23 is tied** (−7.280) |
| J-lens grid | L27 (−3.776, next L31 −4.016) | **L27** (−4.123), but **L23 is tied** (−4.126) |
| logit lens direction | L38 | **L2** (L0–L3 on top). Probably an early-layer flat-distribution effect. |
| logit lens grid | L38 | **L2** (−5.773), then L31 and L30 |

L27 is still the J-lens argmax for both signals, so nothing downstream changes layer. It no longer
stands out, though. The old direction profile sampled 20% of tokens; this one sampled 5%.

**Loudness evaluation (held-out 70, every reasoning token, L27)**
- direction (next action, local belief): `/workspace/results/qwen_p2_local_belief/heldout_fixnorm/figures/` (8 figures)
  from `wrappers/qwen_analysis/4_loudness_evaluation/probe_accuracy_by_loudness_decile_fixnorm.ipynb`.
  Accuracy rises with J-lens loudness for every arm. For the MLP probes, decile 1 → decile 10 is
  jlens .286 → .468, logitlens .304 → .457, random .323 → .420. The old table never had this
  notebook run on it, so there is no before/after for direction.
- grid **multiclass**: `/workspace/results/qwen_p2_grid/heldout_multiclass_fixnorm/figures/` (8 figures, drawn in the
  direction notebook's style) from `wrappers/qwen_analysis/grid/4_loudness_evaluation/probe_accuracy_by_loudness_decile_multiclass_fixnorm.ipynb`.
  This result is new: the multiclass table had never been built. It matches the evaluator's JSONs
  (identical ground truth, at most 4 of 29.2M cells differ). The effect is weak. Loudest minus quietest
  J-lens decile, MLP: jlens +.034, logitlens −.002, random +.036. Under logit-lens loudness all three
  gain +.02 to +.03.
- grid binary: `/workspace/results/qwen_p2_grid/heldout_fixnorm/figures/` (3 figures). This was run
  before binary was dropped, and is kept only for reference.
- The held-out tables are the old per-token tables with their 8 loudness columns replaced through
  `join_signal_loudness.py`. All 1,169,734 rows joined, and the probe columns are unchanged.

**Lens agreement (eval 52, L27, 120,159 tokens, same sample as before)**
- `/workspace/results/lens_agreement_fixnorm/` (4 PNG and 4 PDF, plus `loudness_agreement_summary.csv`) from
  `wrappers/lens_agreement/jlens_vs_logitlens_loudness_fixnorm.ipynb`. Only the Qwen cells were run;
  gpt-oss is unaffected by the bug.
- Spearman ρ, old → new: direction jlens-vs-logitlens .555 → .526; grid .604 → .564;
  direction-vs-grid under J-lens .039 → .009; under the logit lens .033 → .088.

## 3. New wrapper

`wrappers/qwen_analysis/grid/4_loudness_evaluation/score_multiclass_probes_heldout_per_token.sh` builds the multiclass
per-token table (`score_probes_per_token.py --probe-type grid_multiclass`, with the corrected `--lens-dir`). It had
been recorded only as a string in the notebook. The existing `eval_multiclass_probes_heldout.sh` was also run for the
first time, writing to `/workspace/results/qwen_p2_grid/heldout_multiclass/`.

## 4. Not done

- **README step 3:** the `ICLR log.txt` entry and the `claude_session_readme.md` header, which still
  warns not to build on the Qwen numbers.
- **README step 4:** the overlap between the recorded jlens/logitlens picks and the corrected top-60, which
  decides whether the lens-arm probes need re-gathering and retraining. This is CPU only; the inputs are
  the `eval52_*` trees.
- Nothing was committed.

The helper scripts used for the notebook copies, column swap and profile join are in
`/workspace/logs/qwen_fixnorm/session_scripts/`.
