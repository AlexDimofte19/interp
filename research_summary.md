# Research summary — the jlens → probe → rollout line, end to end

Companion to `ICLR log.txt` (49 entries, chronological, append-only). That file is the
narrative; **this one is the inventory**: every probe, dataset, selection mechanism, rollout
arm, report and figure set produced so far, with what each was trained on and what it scores.

Written 2026-09-03, at the close of ICLR log entry 49. Where a number is not yet measured the
cell says so rather than being left blank.

---

## 0. The question, and the three axes the round-49 work separated

The model (gpt-oss-20b) reasons in text before emitting a grid action. A **linear/MLP probe** on
the layer-15 residual stream can recover that action. Three things independently determine how
well:

| axis | choices | what it decides |
| --- | --- | --- |
| **selection** | jlens · logitlens · random · eos · every-token | *which tokens* the probe trains on |
| **label** | final `agent_action` · **local belief** | *what the probe is asked to predict* |
| **layer** | pooled 7:23 · pinned L15 | which representation space |

Entry 48 could not separate selection from label — every belief-trained probe was jlens-selected,
and the only non-jlens selection existed only with the final-action label. **Entry 49 filled the
missing cells.** All three axes are now measured independently (§6, §7).

---

## 1. The pipeline

```
trajectory JSONs
   ├─ jlens_reasoning_tokens.py ─→ per-token .pt  +  {stem}_{lens}_analysis.csv
   │                                              +  {stem}_{lens}_direction_mass.csv (+ .meta.json)
   │                                              +  {stem}_jlens_selection.json   (the arms)
   ├─ run_inference.py (truncation strategies) ─→ per-trajectory rollout JSON = the LOCAL BELIEF label
   ├─ prepare_activations_for_probing ─→ manifest.json (v3, token-major)
   ├─ relabel_manifest_from_rollout.py ─→ same tokens, belief label
   ├─ split_next_action_manifest.py ─→ train/eval, split over TRAJECTORY NAMES
   └─ train_next_action_probe ─→ probe .pt ─→ eval_probe_per_token.py ─→ per-token CSV ─→ report
```

---

## 2. Trajectory sets — and why cross-era numbers do not compare

Three canonical sets, verified mutually disjoint by `scripts/audit_trajectory_sets.py`
(`train ∩ eval = 0`, `(train ∪ eval) ∩ heldout360 = 0`).

| set | n | role |
| --- | --- | --- |
| **training 2,880** | 2,880 | probe training; distribution references only, never an accuracy number |
| **eval 720** | 720 | the matched split (`next_action_mass_l15_eval_names.txt`); 14,391 tokens at 20/traj |
| **heldout 360** | 360 | a disjoint *tree*, no shared draw at all; 87,221 reasoning tokens |

> **⚠ The count-era and mass-era trees are nearly disjoint.** Entry 36's own correction: two
> gathers with identical `--per-combo 100 --seed 0` overlapped by only **348 of 3600**, and their
> 720-trajectory eval sets overlap by **21** (md5 `5190d21d084d` vs `a5aea3ff418e`).
> **Never difference a count-era eval-720 number against a mass-era one.** Within an era, and for
> everything pinned by `--eval-names`, the split is identical (md5 `32d1feac1464`, verified for the
> entry-45, entry-38 and entry-49 arms alike).

---

## 3. Selection mechanisms — which was used for what

**Registry:** `telos_interp/jlens_utils/methods.py` (`jlens`, `logitlens`, `random`), scoring in
`jlens_utils/scoring.py`, one shared `top_filter` across the three consumers.

| mechanism | how a token scores | used for |
| --- | --- | --- |
| `jlens` | Jacobian-lens direction-loading at L15 | the primary arm in every round |
| `logitlens` | logit-lens direction-loading at L15 | matched second lens (entries 24/26/28/31, and round 49) |
| `random` | seeded uniform draw, **no scores recorded** | the matched control — its absence of scores is what makes `split_next_action_manifest.py` sample rather than rank |
| `eos` | every sentence end | entry 29/30 — the punctuation baseline |
| `every_token` | no selection | entry 48's dense heldout grid |

**Score modes** (`--direction-score`): `count` (original) · `logprob_mass` (top-k window) ·
`logprob_sum` (literal but wrong — it multiplies probabilities) · `logprob_mass_full` (the
direction-mass table, whole vocabulary).

> **The negative result worth remembering (entry 38c): mass is NOT a better training-set selector
> than count.** L15 jlens count `.6872 ± .0016 / .7289 ± .0041` vs mass `.6889 / .7229` — inside
> one seed-sd. Mass *is* the better instrument for **asking where the signal is** at evaluation
> time. Do not re-gather to convert a tree from count to mass; do write the mass table on any new
> gather.

> **The most robust finding in the whole line (entry 38a): the jlens ranks better than the logit
> lens, 26/26 probes.** Top-20 bucket mean over all 26 probes: jlens ranking `.6800`, logitlens
> ranking `.6217`, **+.0584 and the jlens wins every single probe** — including probes *trained* on
> logitlens-selected tokens, the case that could have gone the other way.

---

## 4. Activation trees

| tree | layers | contents | round |
| --- | --- | --- | --- |
| `jlens_reasoning_tokens` | 7:23 | count-era, pruned to jlens ∪ random | 1–2 |
| `jlens_mass_l15` | CSV 7:23, `.pt` L15 | mass-era, 3600 traj, arms `jlens` + `random` | 3 (entry 36) |
| **`logitlens_mass_l15`** | CSV 7:23, `.pt` L15 | **new (entry 49)** — same 3600 pinned by name, arm `logitlens` | 49 |
| `argmax_per_sentence_l15` | L15 | jlens per-sentence-loudest positions, 75,012 `.pt` | 45 |
| **`logitlens_argmax_per_sentence_l15`** | L15 | **new (entry 49)** — 74,975 `.pt`, 0 NaN, 66 min | 49 |
| `heldout360_l15` | L15 | every reasoning token of the 360 (87,221) | 37 |
| `heldout360_lens` | 7:23 | both lenses' CSVs + both mass tables, no `.pt` | 37 |
| `activations_train_single_step*` | — | round-1 legacy trees | 1 |

---

## 5. Rollout arms (truncation strategies) — the local-belief label

`scripts/inference_oss/truncation_strategies.py`. Cut the chain at position *p*, append the
final-channel prefix, read the single action token. Everything else is identical between arms, so
any difference **is** the cut points.

| arm | strategy | lens | set | files | duration |
| --- | --- | --- | --- | --- | --- |
| `rollout_strategies/jlens_argmax_per_sentence` | per-sentence loudest | jlens | 3600 | 3600 | 6h04m |
| `rollout_strategies/jlens_top_k_global` | global top-20 | jlens | 3600 | 3600 | 3h55m |
| `rollout_strategies_heldout360/every_token` | **every** token | jlens (recorded only) | 360 | 360 | 6h13m |
| **`rollout_strategies_baselines/recorded_selection`** | replay recorded picks | jlens (recorded only) | 3600 | 3600 | **4h05m** |
| **`rollout_strategies_baselines/jlens_argmax_per_sentence`** | per-sentence loudest | **logitlens** | 3600 | 3600 | **6h01m** |
| **`rollout_strategies_baselines/jlens_top_k_global`** | global top-20 | **logitlens** | 3600 | 3600 | **4h07m** |

> **Naming trap.** The two logitlens arms *reuse the jlens strategy names* — the strategy is "cut
> at each sentence's loudest token", the lens is a separate flag — so their directories say
> `jlens_*` while holding logitlens results. Every result file records `strategy.lens`;
> `entry49_status.sh` prints it. They live under `rollout_strategies_**baselines**/`; entries 43/44's
> jlens arms under `rollout_strategies/` are untouched (September-1 mtimes).

**Label sanity anchors**, measured per arm (`model_action` == the step's action):

| arm | `no_reasoning` | selected tokens | `end_of_reasoning` |
| --- | --- | --- | --- |
| recorded random | .375 | .842 | **1.000** |
| logitlens per-sentence | .400 | .847 | **1.000** |
| logitlens top-20 | .350 | .869 | **1.000** |

The 1.000 is the anchor — with the full chain the model always reproduces the trajectory's action.
*(These agreement levels come from head-of-list subsamples and are indicative only; see §10.)*

> **A rollout label has a noise floor.** The same prompt re-run in a different batch shape agrees
> 100% at `end_of_reasoning` but only ~80% at `no_reasoning` (mean |Δp| .105) — batching and padding
> move low-confidence logits. Intrinsic arm-vs-arm disagreement is ~3.3%. Do not read a few points
> between arms as signal.

---

## 6. Prepared datasets

| dataset | samples | tokens/traj | label | tree |
| --- | --- | --- | --- | --- |
| `next_action_mass_l15_jlens` | 71,913 | 20 | final | jlens_mass_l15 |
| `next_action_mass_l15_random` | 71,913 | 20 | final | jlens_mass_l15 |
| `local_belief_p1_local` | 75,008 | 20.8 | belief | argmax_per_sentence_l15 |
| `local_belief_p2_local` | 71,913 | 20 | belief | jlens_mass_l15 |
| **`entry49_random_belief`** | **71,913** | 20 | **belief** | jlens_mass_l15 |
| **`entry49_logitlens_p1_local`** | **74,972** | 20.8 | **belief** | logitlens_argmax_per_sentence_l15 |
| **`entry49_logitlens_p2_local`** | **71,913** | 20 | **belief** | logitlens_mass_l15 |

Relabel joins were exact: **0 rows with no matching cutoff** in any of the three entry-49 arms
(3 rows dropped in P1 for a null `model_action`, out of 74,975).

Label movement (local belief ≠ final action), same eval split:

| selection | differ |
| --- | --- |
| random | 24.1% |
| logitlens top-20 | 21.6% |
| jlens top-20 | 19.5% |
| jlens per-sentence | 30.3% |
| logitlens per-sentence | 30.2% |

---

## 7. THE PROBE TABLE

Balanced accuracy. **eval-720** = that era's matched 720-trajectory split (see §2 — cross-era cells
do not compare). **heldout-360** = all 87,221 reasoning tokens of the disjoint 360, scored against
the *local belief* / against the *final action*.

### 7a. Round 1–2, count-era tree (`jlens_reasoning_tokens`)

| probe dir / stem | selection | layers | label | eval-720 lr | eval-720 mlp |
| --- | --- | --- | --- | --- | --- |
| `next_action/jlens_topall` | jlens count | 7:23 pooled | final | .6332 | .7097 |
| `next_action/logitlens_topall` | logitlens count | 7:23 pooled | final | .6026 | .6668 |
| `next_action/random_topall` | random | L15 (pinned) | final | .5675 | .6132 |
| `next_action_l15/jlens_topall` | jlens count | **L15** | final | **.6800** | **.7329** |
| `next_action_eos/eos_topall` | eos | L15 | final | .4632 | .5143 |

Pinning L15 is worth **+4.7 lr / +2.3 mlp** over pooling 7:23 (entry 28); entry 38b puts the pooled
group ~10 (lr) / ~6 (mlp) below single-layer across all 26 probes, *and* its seed sd is .084 against
.002–.008 — pooling destabilises as well as lowers.

**eos is below the uniform control** — .4632/.5143 against random's .5675/.6132. Taking every
sentence end is *worse* than 20 random tokens from the same chains.

### 7b. Three-seed sweep at L15 (`next_action_seeds`, entry 31)

| selection | lr mean ± sd | mlp mean ± sd |
| --- | --- | --- |
| jlens | .6801 ± .0005 | .7315 ± .0015 |
| logitlens | .6533 ± .0005 | .7107 ± .0013 |
| random | .5668 ± .0007 | .6123 ± .0011 |
| eos | .4623 ± .0009 | .5122 ± .0026 |

Max seed sd across all eight cells is **0.26 pp**; effects claimed are 2–12 pp. Mean gaps:
jlens−logitlens **+2.69 / +2.08**, jlens−random **+11.33 / +11.92**, logitlens−random **+8.65 / +9.84**.

### 7c. Round 3, mass-era tree (`jlens_mass_l15`), final-action label

| probe | selection | eval-720 lr | eval-720 mlp | heldout-360 vs local | vs final |
| --- | --- | --- | --- | --- | --- |
| `next_action_mass_l15/jlens_topall` | jlens mass, 20/traj | .6994 | .7446 | .4359 | .3809 |
| `next_action_mass_l15/random_topall` | random, 20/traj | .5744 | .6248 | .4569 | .4128 |
| `next_action_mass_l15/jlens_top1` | top-1 loudest | .8722 | .8726 | — | — |
| `next_action_mass_l15/jlens_top2` | top-2 loudest | .8153 | .8369 | — | — |

Within-run gap **+12.50 / +11.98**, matching the count era's +11.25/+11.97 — mass buys what count
bought. The top-1/top-2 numbers are on far smaller, far louder token sets and are not comparable to
the top-20 rows.

### 7d. Entry 45 — the local-belief relabel (same tokens, new label)

| probe | selection | label | eval-720 lr | eval-720 mlp | heldout-360 vs local | vs final |
| --- | --- | --- | --- | --- | --- | --- |
| `local_belief_p1` (full) | jlens per-sentence | belief | .606 | .678 | .4634 | .3751 |
| `local_belief_p1_top20` | ↑ thinned to 20 | belief | .710 | .774 | .4572 | .3760 |
| `local_belief_p2` | jlens top-20 | belief | **.802** | **.862** | **.4869** | .3895 |

P2 vs the `next_action_mass_l15` baseline is **identical tokens, layer, split and hyperparameters —
only the label differs**: `.745 → .862` mlp, **+11.7 pp**.

### 7e. Entry 49 — the missing cells (NEW)

| probe | selection | label | eval-720 lr | eval-720 mlp | heldout-360 vs local | vs final |
| --- | --- | --- | --- | --- | --- | --- |
| `local_belief_baselines/random_belief` | random 20/traj | belief | .6595 | .7234 | **.5274 / .5886** | .4129 / .4624 |
| `local_belief_baselines/logitlens_p1` | logitlens per-sentence (uncapped) | belief | .5737 | .6438 | .4861 / .5483 | .3929 / .4370 |
| `local_belief_baselines/logitlens_p2` | logitlens top-20 | belief | .7297 | .7963 | .5018 / .5498 | .3936 / .4379 |

*(heldout-360 cells are lr / mlp, over all 87,221 tokens.)*

**Regression check passed:** in the 16-probe rebuild, all ten entry-48 probes reproduce their
published heldout-360 numbers to **4 dp, delta +0.0000 on every one**. That is the check that the
six added arms sit on the same measurement rather than a shifted one. The join was clean: 0 mass
mismatches, 0 final-label mismatches, 1 row of 87,221 with no valid rollout action.

### 7f. The 2×2 that entry 49 exists to produce (eval-720, identical 71,913 samples and split)

| tokens | **final** label | **belief** label | label effect |
| --- | --- | --- | --- |
| random 20 | .574 / .625 | **.660 / .723** | **+8.6 / +9.8** |
| jlens top-20 | .699 / .745 | .802 / .862 | **+10.3 / +11.7** |
| **selection effect** | **+12.5 / +12.0** | **+14.3 / +13.9** | |

**The label and the selection are independent and roughly additive.** Relabelling buys ~+9 to +12 pp
whatever the selection; loud selection buys ~+12 to +14 pp whatever the label.

### 7g. Selection quality, label held fixed (belief, 20/traj, identical n and split)

| selection | lr | mlp |
| --- | --- | --- |
| random | .660 | .723 |
| **logitlens top-20** | **.730** | **.796** |
| jlens top-20 | .802 | .862 |

Monotone, near-evenly spaced. Both lenses beat a uniform draw; the jlens is the better instrument.
`logitlens_p2` (.796 mlp) **beats the jlens final-action baseline** (.745) despite the weaker lens —
the label is worth more than the lens gap.

At the matched per-sentence rule: jlens P1-full .606/.678 vs logitlens P1-full **.574/.644**, a
stable −3.2/−3.4 pp.

### 7h. ⚠ The ordering INVERTS on the held-out set — and that is distribution shift, not a defeat

The same three belief-trained arms, same probes, read on **all 87,221 heldout-360 tokens** instead
of on their own 20-per-trajectory regime:

| selection (belief label) | eval-720 mlp *(its own loud regime)* | heldout-360 mlp *(every token)* |
| --- | --- | --- |
| random | .723 *(worst)* | **.5886** *(best of all 16)* |
| logitlens top-20 | .796 | .5498 |
| jlens top-20 | **.862** *(best)* | .5243 |

**A perfect mirror image.** This is entry 37(d)'s effect, now reproduced with the belief label and
with a third arm in the middle: a loud-selected probe only ever saw high-mass tokens and is
*specialised* to them; the random-trained control saw a uniform draw and so generalises across the
whole chain. The two orderings cross exactly where the training distributions do.

Read them together, and the deployment question decides which one matters:

- *"score a token the lens told me is loud"* → the jlens arm, and the jlens ordering (§7g).
- *"score every token in the chain"* → the random arm.

The **logit lens sits between random and jlens in BOTH orderings**, which is a useful consistency
check: it is not that one population flatters one lens, it is that selection strength trades against
coverage monotonically.

The label effect, by contrast, **survives the population change intact** — on the same tokens and
the disjoint tree:

| pair (identical tokens, only the label differs) | heldout-360 mlp |
| --- | --- |
| random: final `.4870` → belief `.5886` | **+10.2 pp** |
| jlens top-20: final `.4556` → belief `.5243` | **+6.9 pp** |
| logitlens top-20 vs the jlens final-action baseline | `.4556` → `.5498`, **+9.4 pp** |

So of the two axes entry 49 separated, **the label effect generalises and the selection effect is
population-dependent.** Any claim about selection must name the population it was measured on;
the label claim does not need that caveat.

---

## 8. Reports, artifacts and figures

| report dir | entry | dataset | figures | tables | artifact |
| --- | --- | --- | --- | --- | --- |
| `loudness/` — *Where the Lens Is Loud* | 42 | training 2,880 | 22 | 15 | [a26ec417](https://claude.ai/code/artifact/a26ec417-feae-42eb-a09d-af529f603c06) |
| `probe_vs_rollout/` — *The Commitment Boundary* | 39–41 | heldout 360 | 15 | 22 | [fbb52bdd](https://claude.ai/code/artifact/fbb52bdd-4980-415d-9d94-93177a54539d) |
| `probe_loudness/` — *What Loudness Buys the Probe* | 46 | eval 720 | 20 | 18 | [2e873b12](https://claude.ai/code/artifact/2e873b12-7c49-4ac6-b861-9c2a7aa707f0) |
| `probe_vs_rollout_lb/` — the belief-probe clone | 47 | heldout 360 | — | 12 | [95a74d99](https://claude.ai/code/artifact/95a74d99-bb0a-440f-839c-e6e4dbac8c65) |
| `probe_loudness_heldout360/` — selection removed | 48 | heldout 360 | 20 | 18 | [a62f826d](https://claude.ai/code/artifact/a62f826d-34bd-44eb-a13f-606fe2d49c6c) |
| `probe_loudness_heldout360_16probes/` | **49** | heldout 360 | 20 | 18 | [cd8900f2](https://claude.ai/code/artifact/cd8900f2-5bb4-4e56-b1e4-8e13ed80f47b) |

The entry-49 page is the only one whose **every figure carries its own provenance underneath** —
which probes, the tree and selection each was trained on, the training label and n, its eval-720 and
heldout-360 numbers, the binning, the evaluation label and the hyperparameters. It is generated by
`scripts/entry49_build_report.py` from a `PROBES` and a `FIGURES` registry rather than hand-written,
so a caption cannot drift from its figure.

`probes/heldout360_all_probes.csv` (entry 38) holds all 26 count/mass-era probes on the 87,221
tokens under **both** rankings.

---

## 9. Findings, in dependency order

1. **Pin one layer.** Pooling 7:23 costs ~10 pp (lr) and destabilises across arms (sd .084 vs .005).
2. **The jlens ranks better than the logit lens — 26/26 probes**, including logitlens-trained ones.
3. **Mass and count pick equally good training tokens**; mass is the better *evaluation-time* instrument.
4. **eos is below the uniform control** — punctuation is the wrong grid.
5. **Loudness orders decodability monotonically**, and it is loudness, not sentence position
   (2.5–3× the gradient across loudness at fixed position vs across position at fixed loudness).
6. **Verbalization is not the mechanism.** Within the selected top-20, whether the token *is* a
   direction word adds nothing (.6878 vs .6884). Confirmed twice more since.
7. **The probe reads the model's current belief, not its final answer.** Relabelling to local belief
   moves L15 mlp `.745 → .862`; on disagreement rows the probe follows the belief .77 to .14.
8. **The commitment happens inside one sentence**, and belief-trained probes open a switch sentence
   holding the *previous* answer, crossing over mid-sentence (entry 47 revised entry 39 here).
9. **Entry 49: label and selection are separable and additive**, and lens quality orders
   random < logitlens < jlens with the label held fixed — *on the loud regime*.
10. **Entry 49: the label effect generalises, the selection effect does not.** On all 87,221
    heldout tokens the selection ordering inverts (random .589 > logitlens .550 > jlens .524 mlp)
    because loud-selected probes are specialised to loud tokens, while the label is still worth
    +6.9 to +10.2 pp on the same tokens. Name the population whenever claiming a selection effect.

---

## 9b. The two loudness rulers, compared (entry 49)

**Say which lens.** "Loudness" is always *jlens loudness* (Jacobian lens) or *logitlens loudness*
(logit lens) — never bare. At layer 15 their top-20 token sets overlap only ~50% and the logit lens
puts ~6x less mass on direction words (mean log-mass −5.23 vs −3.38). Every page before entry 49
binned by **jlens loudness alone**, hard-coded in `build_probe_loudness_heldout.py`, which meant the
two logitlens-selected arms were being ordered by a ruler they were never selected by. `--mass-column`
now selects it; both cuts exist.

Balanced-accuracy rise from the quietest to the loudest decile, heldout 360, all 87,221 tokens:

| probe | jlens loudness Δ | logitlens loudness Δ | better ruler |
| --- | --- | --- | --- |
| P2 jlens top-20, mlp | +.358 | +.154 | jlens by .204 |
| baseline jlens top-20 final, mlp | +.290 | +.119 | jlens by .171 |
| random control final, mlp | — | — | jlens by .140 |
| random selection belief, mlp | — | — | jlens by .125 |
| **logitlens P1, mlp** | — | — | **jlens by .061** |
| **logitlens P2, mlp** | — | — | **jlens by .068** |

**jlens loudness orders decodability better for all 12 probes — including the two the logit lens
selected.** That extends entry 38(a)'s 26/26 to the belief-label arms. The margin is much narrower on
the logitlens-selected arms (+.04 to +.07) than on the jlens ones (+.15 to +.20), which is the
expected shape: the logit lens is closest to competitive on the tokens it chose itself, and still
does not win there. Zero reversals under either ruler.

---

## 10. Landmines

- **Do not subsample an analysis over trajectories.** Entry 44(d)'s two arms were first read on ~100
  trajectories and both overstated the effect ~2×. Reproduced in entry 49: reading the first 40
  files *alphabetically* gave 4.9 cutoffs/traj for the per-sentence arm against a true **23.9**,
  because the head of the list is all `size11_comp0.0` (shortest chains). Sample randomly.
- **`abs_pos` is prompt-inclusive; `token_idx`/`token_id` indexes `output_tokens`.** Joining on the
  wrong one gives an empty or wrong join, never an error.
- **`sentence_frac` means two different things.** In the probe-loudness CSVs it is position *within*
  a sentence; in the commitment CSV that quantity is `frac_in_sentence` and `sentence_frac` is
  `sentence_idx / n_sentences`.
- **Never say "loudness" unqualified.** It means *jlens loudness* by default in this repo's tooling
  and that default is invisible. Say jlens/logitlens, and show both where the column exists.
- **Never read a mass table without its `.meta.json`** — two vocabularies point at the same trees.
- **Token-major manifests leak under a row-level split.** Use `split_next_action_manifest.py`, which
  splits over unique trajectory names.
- **Read lens CSVs with `csv.DictReader`, never pandas** — decoded tokens include `"NA"`, empty
  strings, embedded commas and newlines.
- **A control arm's draw cannot be re-made** once the tree is pruned; replay it from the record
  (`recorded_selection`).
- **`--per-combo`/`--seed` does not reproduce a previous draw** (348/3600 overlap measured). Pin by
  name with `--names-file`.
- **A `uv run` without `--extra gpu` syncs `accelerate` back OUT** of a venv that had it.

---

## 11. Open threads

- **The ±k proximity window (entry 42e) is the last uncontrolled confound** — a token two positions
  before `" up"` is still "the model about to say up". Position-in-sentence is controlled; proximity
  is not.
- **No logitlens P1-top20 arm**, so the logitlens per-sentence rule has no counterpart to the capped
  jlens row that actually beat the baseline. One split + two trainings, ~10 min, no GPU rollout.
- **No random-position truncation arm** — the matched control for entry 44(d).
- **Layer 15 only** since entry 36; `--direction-classes` is `all`, so nothing can be split by class.
- **Entry 44(d)'s figures cover all 3,600** (train and eval mixed) — a different population from
  every other result. Re-cut to eval 720 before comparing to any probe number.
- `--names-file` on `jlens_reasoning_tokens.py` and its `has_work()` guard are **uncommitted**, mixed
  with ~1,180 lines of older working state in the same two files.
