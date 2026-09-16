# probe-folder-reorg: what is still open (re-verified 2026-09-16)

Branch state: **4 ahead of `reasoning_theatre`, 0 behind, working tree clean, fully pushed to
`origin/probe-folder-reorg`.** Nothing technical blocks the merge. Note the merge *target* is
dirty: `reasoning_theatre`'s working tree has 31 uncommitted changes, including the 276 new
lines of `ICLR log.txt` that are entries 57 and 58.

## Closed since the last note

- **The cross-selection run landed.** All 50 cells present (`jlens/logitlens/random/eos` p1 = 8
  each, `jlens/logitlens/random` p2 = 6 each), 0 failures, `RESULTS.md` written 2026-09-15
  22:33 under `/workspace/reasoning_theatre/cross_selection_eval/`. It reports **balanced**
  accuracy, so it post-dates the `eval_local_belief.py` fix in `bc9d8d8`. Entry 58's "in
  flight, 20/50" is stale, and what remains is a log append, not a run.

---

## 1. Entry 58 has no results — and the run produced a finding entry 58 does not mention

`ICLR log.txt` (main checkout, uncommitted) line ~4326 still reads *"STATUS AT TIME OF WRITING:
in flight, 20/50 complete"*. Append from `RESULTS.md`:

- the 8x7 matrix, balanced accuracy, all 14 diagonal cells reproducing their checkpoint's
  `final_balanced_accuracy` to 4 dp;
- **the diagonal wins every column, 14 of 14** — specialisation is not a jlens or a loudness
  property, the `eos` arm shows it as sharply as any;
- **the random control is the strongest non-native probe in 5 of the 7 columns** — the
  generalist is the one trained on uniformly drawn tokens;
- and item 2 below, which is a separate finding and probably wants its own entry.

## 2. NEW, and it reaches published numbers: `best_balanced_accuracy` is a high-water mark

The trainer keeps a running max over epochs and **never checkpoints the epoch that achieved
it** — the weights on disk are the last epoch's.

- `telos_interp/commands/train_next_action_probe/train_next_action_probe_fn.py:628,636,686`
- `telos_interp/commands/train_cognitive_map_probe/train_cognitive_map_probe_fn.py` — same
  pattern, with `# model.load_state_dict(best_model_state)` commented out at :1352-1353 and the
  print immediately below it still announcing `FINAL EVALUATION (best model)`. **So the grid
  probes are affected too, not just the belief/action ones.**
- `scripts/build_probe_inventory.py:78` feeds `best_balanced_accuracy` into the CSV's
  `eval720_bal_acc_OWN_tokens`; :912 documents it as exactly that.

Measured across the 14 diagonal cells: every published figure is **0.00 to 0.93 pp high, never
low**, and the bias is not uniform across arms, so it does not cancel in a between-arm gap of
that size.

**The decision.** Either checkpoint the best epoch — which changes what "the probe" *is* and
forces a re-measure of everything published — or switch every quotation to
`final_balanced_accuracy`, which is already in the same results dict on every checkpoint and
costs nothing. The second is the cheap, honest one. Either way, fix the "best model" print.
Nothing in the cross-selection matrix is affected: every cell there is one evaluation of the
saved weights.

## 3. `reproduce_all.sh` reports 4 stages MISSING that are on disk — and the symlink does not mask one

`LIST=1 bash scripts/reproduce_all.sh` on this host: **27 done, 4 missing, and all four are
false negatives.** A run without `FORCE=` would re-do ~45m + ~2h + ~6h + ~7h of GPU work that
is already finished.

| line | stage | tests | why it fails |
|---|---|---|---|
| :177 | `local_belief_probes` | `$RT/local_belief_probes/probes` | directory deleted |
| :184 | `belief_baseline_probes` | `$PROBES/local_belief_baselines` | directory deleted |
| :187 | `more_belief_probes` | `$PROBES/local_belief_baselines` | directory deleted |
| :190 | `equal_n_belief_probes` | `find "$PROBES/local_belief_equalN" -name '*.pt'` | **the symlink does not mask this one** |

That last row corrects entry 58, which claims the compatibility symlink "masks all five".
`find` given a symlink with no `-L` and no trailing slash does not descend it — measured:
`find <link>` → 0 `.pt`, `find <link>/` → 20, `find -L <link>` → 20. So the stage has been
reading MISSING since the rename.

`$PROBES/grid` (:192, :682) is also absent, but that stage is `GRID_ROUND=1` and the round
genuinely never ran. Correct as-is.

## 4. The `local_belief_equalN` symlink: three code references left

- `scripts/train_equal_n_belief_arms.sh:60` — a *write* path (`mkdir -p`, `--output-path`);
  resolves through the symlink, so it works but files probes under a name nothing else uses.
- `scripts/eval_belief_arms_heldout.sh:51` → used as globs at :100; globs follow the symlink,
  so it works.
- `scripts/reproduce_all.sh:190` — **broken now**, see above.

Prose references, lower stakes: `scripts/INVENTORY.md:94,115`, `research_summary.md:484,650`,
`claude_session_readme.md:31`. `ICLR log.txt:3601` names it too and **must not be edited** —
the log is append-only and that line was true when written.

Decision unchanged from entry 58: repoint the three and delete the symlink, or keep it
deliberately and say so. But :190 needs `-L` (or a trailing slash) either way, so keeping the
symlink alone does not close this.

## 5. `eval_belief_arms_heldout.sh` has three dead roots, not one

- `:48 LB=$RT/local_belief_probes/probes` — deleted
- `:49 MASS=$PROBES/next_action_mass_l15` — **missed by the reorg commit.** That directory is
  now `next_action_l15/p2`; `reproduce_all.sh` and `build_sixteen_probe_loudness_report.sh`
  were both repointed in `bc9d8d8`, this one was not.
- `:50 BASELINES=$PROBES/local_belief_baselines` — deleted

It guards every probe path with `[ -e "$p" ] || exit 1` at :121, so it fails loudly rather than
silently — but only when the round's CSV is absent, and both rounds' CSVs exist, so it is
dormant rather than fixed.

## 6. `build_sixteen_probe_loudness_report.sh`: six of the sixteen probes are gone for good

`:35 LB` and `:37 NEW` both name deleted directories (`:36 MASS` was repointed and is fine).
Of the sixteen probes it reads:

- **4 live** — the mass arms under `next_action_l15/p2/`.
- **6 have live homes** under `probes/local_belief_action_l15/p2/`: `local_belief_p2_{lr,mlp}`
  → `p2/next_action_probe_jlens_*`, `logitlens_p2_*` → `p2/next_action_probe_logitlens_*`,
  `random_belief_*` → `p2/next_action_probe_random_*`. (Timestamps confirm these are the
  entry-45/49 originals, moved, not rebuilt.)
- **6 are gone from disk entirely** — entry 45's `local_belief_p1_{lr,mlp}` and
  `local_belief_p1_top20_{lr,mlp}`, plus entry 49's `logitlens_p1_{lr,mlp}`. No `.pt` by those
  names exists anywhere under `/workspace`; only the logs survive
  (`reasoning_theatre/{local_belief_probes,entry49_baselines}/logs/`).

Their nearest replacements are the **entry-52 equal-N rebuilds** in `p1/` and `p1-top20/` — a
different vintage, trained on different data. Substituting them silently changes what the page
means, which is why this is a decision and not a rename. Currently dormant: `report.html`
exists so the stage reads "done"; it bites on a rebuild or a fresh host.

Related inconsistency inside the same commit: `scripts/build_sixteen_probe_report_page.py`
had its 4 mass `file=` fields repointed (:78-244 diff in `bc9d8d8`) but its 12 belief ones
still name the deleted directories — so the published page now cites 4 live paths and 12 dead
ones.

## 7. Stragglers

- `scripts/audit_trajectory_sets.py:108` reads
  `PROBES/next_action_mass_l15/heldout360_per_token.csv`; the file is now at
  `next_action_l15/p2/heldout360_per_token.csv`. `from_csv` prints "MISSING" and returns
  (:78), so this quietly drops entry 37 from the audit instead of failing.
- `scripts/belief_baselines_status.sh:17` defaults `PROBES` to the deleted
  `local_belief_baselines`.
- `scripts/pull_artifacts_to_laptop.sh:123` rsyncs `reasoning_theatre/local_belief_probes/probes/`
  (deleted); :138 excludes the same dead path.
- `scripts/train_belief_baseline_probes.sh:35` and `scripts/train_more_belief_arms.sh:49`
  default `PROBES` to `local_belief_baselines` — a re-run would **recreate** the deleted
  directory and re-split the belief probes across two roots, which is the thing the reorg
  exists to prevent.
- `scripts/compare_equal_n_arms.py` **still exists on this branch** (:31-33 name two deleted
  roots). Entry 58 says it was deleted, but that deletion lives only in `reasoning_theatre`'s
  uncommitted working tree — merging this branch keeps the file alive.
- `claude_session_readme.md`'s header pointer still reads "As of 2026-09-10 … log entry 56".
  CLAUDE.md requires it to be current; it is two entries behind and never mentions this branch.

### One trap while fixing 5 and 7

`next_action_mass_l15` has three unrelated senses. Only the **probe-directory** one is dead.
Do not touch:

- the prepared-dataset prefix — `/workspace/prepared/next_action_mass_l15_eval_names.txt`,
  `_random`, `_jlens` all exist and are live (`build_mass_era_split.sh`, `verify_mass_era_split.py`,
  `push_to_huggingface.sh`, `train_*_arms.sh`, `tests/test_mass_era_split.py`);
- the frozen CSV column keys — `join_rollouts.py:107-110`,
  `build_probe_inventory.py` H26 (:425-439, left spelling the old folders on purpose,
  because the join is by name).
