# J-lens token localization → ICLR 2027: novelty check and idea menu

Working doc for the brainstorm (Aug 21, 2026). Deadlines: **abstract Sep 18, full paper Sep 25** —
four and five weeks out. Everything below is calibrated to what the pipeline on
`reasoning_theatre` can already produce plus what can realistically be added in that window.

## 0. The honest verdict up front

The roadmap as written (j-lens-selected tokens/layers → next-action probe → random baseline →
short paper) is **real work but thin for ICLR main track**. It is one method comparison, in one
environment, on one model, with a headline number that a reviewer can deflate in one sentence:
*"your selector picks the literal token ' left', so the probe is reading the transcript."* The
session readme already flags exactly this.

The good news: the *setup* is unusually strong — a ground-truth agentic environment (optimal
action from A*, full map state, parametric difficulty), a J-lens refit on the task distribution,
and a per-(token, layer) sweep already engineered. Most of what would turn this into a defensible
paper is **analysis over artifacts the pipeline already produces**, not new infrastructure. The
gap between "workshop note" and "ICLR submission" here is 2–3 additional experiments, chosen well.

My recommendation in one line: **keep j-lens selection as the instrument, but make the paper about
when/where the decision forms in the reasoning chain — validated causally — with selection
efficiency as the practical payoff.** Details in §5.

## 1. Where the current plan sits in the literature

What I could establish from the paper landscape (fetching is proxy-blocked in this session, so
secondary coverage + abstracts; worth re-verifying the two or three closest ones by hand):

| Neighbor | What it already claims | Overlap with our plan |
|---|---|---|
| **Anthropic J-lens / workspace** (transformer-circuits 2026, arXiv 2607.15495) | J-space carries pre-verbal intermediate steps (math steps appear in order before being said); steering/patching j-space modulates behavior; five workspace properties | The tool itself, and "decision info is in j-space before it's said" — **not novel for us to re-show** |
| **Reasoning Theater** (Goodfire, arXiv 2603.05488) | Attention probes over CoT activations decode the final answer early; on **GPT-OSS-120B** and R1; early-exit saves 68% of tokens; easy-vs-hard differences | "Probe the reasoning chain to predict the final answer/action" is established, on our model family |
| **Thought Anchors** (arXiv 2506.19143) | Which reasoning *sentences* matter, via 100× counterfactual resampling, receiver heads, attention suppression | The "locate the important positions" question — but expensive, sentence-level, no lens |
| **Beyond the Commitment Boundary** (arXiv 2606.13603) | Sharp commitment point mid-chain; post-commitment CoT is epiphenomenal; attention probes decode answer-formation stages | Decision-formation dynamics over the chain |
| **Latent horizon / How far ahead do LLMs plan** (arXiv 2602.02103) | Probes on hidden states show myopic incremental planning | Planning-depth questions |
| **Our prior paper** (arXiv 2602.08964) | GPT-OSS-20B grid navigation is goal-directed; cell/distance probes partially decode the map from prompt-token activations | The environment, model, and probe stack |
| **From Text to Space** (arXiv 2502.16690), **Do LLMs Build Spatial World Models** (arXiv 2604.10690) | Spatial probing of LLMs in grid tasks | Environment-type novelty is **already spent** |
| **Token-/layer-selective probing** (e.g. arXiv 2601.13288; hallucination-probe literature) | Best probe position varies per sample; heuristics (last token, exact-answer token) or brute search | The "where to probe" problem is *recognized*; no one uses a lens readout as the selector |

So the two genuinely open slots our plan can occupy:

1. **A lens readout as a *selection policy* for probe training (tokens and layers).** Nobody does
   this. But as a pure method claim it must beat the obvious alternatives (see §3, objection 2).
2. **Causal validation that lens-scored positions are the load-bearing ones.** Anthropic showed
   j-space steering works *in general*; Thought Anchors buys positional importance with 100
   resamples per sentence. Nobody has shown a **forward-pass-cheap lens score predicts
   counterfactual importance of positions in an agentic CoT**. That claim is open, and we are
   unusually well-placed to test it (ground-truth action flips, short action space).

And one slot that is closed: "probes can decode the upcoming action from reasoning tokens" — that
alone is Reasoning Theater on a smaller model.

## 2. The three objections the paper must survive

**Objection 1 — "You selected the word 'left'."** Top-scored tokens are literally the direction
words. A probe trained there reads a verbalized decision; accuracy is uninteresting and the
random baseline is a strawman. *This objection is fatal if unaddressed and a gift if addressed:*
the interesting result is the signal at **non-direction tokens** and at tokens **before the first
verbalized direction mention** (§4, A).

**Objection 2 — "Why select at all?"** Attention probes (Reasoning Theater) *learn* where to look;
training on all tokens needs no selector; last-token heuristics are free. The selection story
must therefore be about something selection uniquely buys: sample efficiency at matched budget,
interpretability of *where* the signal lives, cross-size robustness, or cheap causal targeting —
and it must include an attention-probe or all-token baseline at matched capacity, not only
`random`. (§4, E)

**Objection 3 — "One environment, one model, correlational."** Partially mitigable in-budget:
key-door variant as a second task flavor (data already on HF), causal patching for the
correlational gap, honesty about single-model scope. A second model (e.g. Qwen) is probably out
of the window; say so in limitations rather than pretend.

Also expect: **"Is the J-lens even necessary?"** — a mandatory control is running the identical
direction-count selector on plain **logit lens** (J = I; the code path exists — layer 23 already
is the identity case) and on the **WikiText-fit lens** we replaced. If gridenv-J-lens ≈ logit
lens for selection, the honest paper says so and the contribution shifts to the selection
framework itself; if it wins, the domain-refit is a real finding. Either way we need the number.

## 3. What we already have working (inventory)

- J-lens refit on the *training* distribution (1000 grid prompts, convergence tracked,
  `skip_first` measured, roundtrip-checked). Test set untouched. ✔ hygiene
- One-pass sweep: per-(reasoning token, layer 7–23) top-20 j-space tokens + action ranks +
  logprobs, CSV per trajectory + gather_activations-compatible `.pt` tree. ✔
- 539-token multilingual direction vocabulary (lexical + semantic pipeline, hand-reviewed). ✔
- Selection machinery: `jlens_direction`/`random` × token/layer, budget-matched, seeded,
  manifest provenance. ✔ (unit-tested; GPU end-to-end pending)
- Next-action probe training; eval via separate manifest exists, per-position eval curves need a
  thin wrapper. Cell-identity + distance probes from the prior paper. ✔
- Not yet run at scale: the GPU sweep over the train set; probe train/compare; everything in §4.

## 4. Idea menu

Ranked by (scientific value × feasibility). "Cost" assumes the sweep artifacts exist.

### Tier 1 — do these regardless; they are analyses, not new infrastructure

**A. Pre-verbalization probing (turn the fatal caveat into the headline).**
Stratify every selected token: (i) literal direction tokens, (ii) non-direction tokens with
direction-loaded j-space, (iii) tokens *before the first verbalized direction mention in the
chain*. Train/eval probes per stratum at matched budget. The number that matters: probe accuracy
on (iii) — decision information present before any decision is written down, located by the lens.
Additionally bin tokens by direction-count decile and show probe accuracy is monotone in the
score — that validates the score as a *measure of signal richness*, which is the paper's central
construct, far more convincingly than one top-N-vs-random comparison.
*Cost: selection-code tweaks + probe runs. Kill risk: signal at (iii) may be weak — but that is
itself a finding ("the lens mostly finds echoes"), and we'd want to know before writing anything.*

**B. Decision-formation curves with ground truth (what QA benchmarks cannot do).**
Per trajectory: j-space rank of the taken action + probe prediction, as a function of reasoning
position. Define a *first-decodable position* (earliest position where the probe/lens crosses a
threshold and stays); compare to the position of the first verbalized mention. Then exploit what
Reasoning Theater/commitment-boundary papers don't have — **exact ground truth and parametric
difficulty**: does the decision form earlier on easy grids (low A* distance, low complexity,
small size)? And on steps where the agent moves *suboptimally*, does j-space track the **taken**
action or the **optimal** action, and when does it diverge? The suboptimal-step analysis is a
genuinely new lens on "is the CoT causally upstream of the action."
Bonus: reasoning chains contain verbalized corrections ("wait, that's a wall"); check whether the
j-space action signal flips *at* the correction or *before it* (theater vs. genuine deliberation,
token-resolved).
*Cost: pure analysis of the CSV + probe outputs. This is the most paper-shaping cheap item.*

**C. Verbalizable vs. decodable layer profiles.**
Per layer: probe accuracy (linear decodability) vs. direction-count (j-lens verbalizability).
If probes decode the action at layers where the lens shows nothing, we quantify a
"pre-workspace" stage in an agentic setting — a crisp, citable delta to the Anthropic paper
rather than a re-demonstration of it. Also directly answers roadmap 3.2 (layer selection) with a
figure instead of a convention.
*Cost: pure analysis.*

### Tier 2 — one of these is what makes it an ICLR paper; pick by appetite

**D. Causal validation of lens localization (my top pick).**
At matched budget, intervene at (a) j-lens-selected positions, (b) random positions, (c) literal
direction-token positions, (d) top positions by an attention-based saliency; measure action-flip
rate / logit shift on the final action. Interventions that fit the window: zero/mean-ablate the
residual at those positions, or patch from a counterfactual trajectory (same grid, different
optimal action — constructible in this environment on purpose). If lens-selected positions are
causally load-bearing beyond (c), the selection story stops being correlational and becomes:
**a forward-pass lens finds the tokens that matter, at ~zero marginal cost** — validate against a
Thought-Anchors-style resampling importance on a ~50-trajectory subset to make the comparison
explicit (they pay 100 generations per sentence; we pay one forward pass total).
*Cost: new but modest intervention script (hook + forward, infra exists), plus a bounded
resampling run. Kill risk: effects may concentrate at the literal tokens — again, a finding, and
publishable as a negative result about lens-guided localization.*

**E. The monitor-efficiency story (the practical payoff).**
Budget-matched learning curves: probe accuracy vs. number of training (token, layer) samples for
jlens-selected / random / all-token / last-token / attention-probe-at-matched-capacity. Then the
OOD axis this environment gives for free: train on sizes {5,7,9}, evaluate on {11,13,15}. Claim
shape: *selected tokens yield monitors that are more sample-efficient and more size-robust*.
This is the version of "j-lens improves probing" that survives objection 2, and it is the framing
a safety-motivated reviewer actually wants (cheap, robust monitors).
*Cost: several probe trainings over existing data + one attention-probe implementation
(~a day; pooling over positions, standard).*

**F. Environment-aware positions done properly (roadmap item 2, upgraded).**
Build the second vocabulary (digits/coordinates, "wall/walls", "goal", "row/column", cell-name
tokens — single-token constraint is satisfied by all of these), score positions the same way, and
probe *map* targets (cell identity, agent/goal position — probes exist from the prior paper) at
map-loaded vs. direction-loaded positions. The interesting question is structural: **are
map-reading and decision-making localized at different positions/layers, in a consistent order
(look → decide)?** That is a workspace-structure claim in an agent, not a re-run of the
cognitive-map probe.
*Cost: vocabulary notebook rerun with new seeds + selection + probe runs. Fits the existing
roadmap line, so no scope fight with the supervisor.*

### Tier 3 — cheap hedges and controls (mostly mandatory anyway)

- **G. Lens ablation for selection:** gridenv-J-lens vs. WikiText-J-lens vs. logit lens (J=I)
  driving the *same* selector. Mandatory control (§2); also the only place our lens *refit*
  becomes a measured contribution rather than an engineering note.
- **H. Key-door second task flavor:** trajectories + activations already on the HF org; vocab =
  {key, door} + directions; shows the selector isn't direction-specific. Cheap breadth.
- **I. Second model:** out of window; write as limitation, not promise.

## 5. Two possible papers, and which one I'd write

**Framing 1 — method paper:** "Lens-guided probe placement: locating signal-rich tokens and
layers in reasoning chains." Contribution = the selector + efficiency/robustness wins (E) +
causal validation (D). Risk: lives or dies on beating attention probes and the literal-token
strawman; reviewers grade method papers on baselines, and ours is a 20B single-model study.

**Framing 2 — science paper:** "Where decisions live in a reasoning agent: locating and
validating decision formation with a Jacobian lens." Contribution = A + B + C (+D): decision
information appears at identifiable non-verbal positions before it is written, at layers below
verbalizability, tracks the taken (not optimal) action from position X on, and those positions
are causally load-bearing; lens-guided selection then falls out as the *application* (E), not the
thesis. Risk: needs the Tier-1 results to actually come out with structure — run A/B first, cheap.

I'd write **Framing 2 with D and E as its second half**. Reasons: (1) it occupies the open slots
(§1) instead of the closed one; (2) every crowded neighbor (Reasoning Theater, commitment
boundary) is on QA benchmarks without ground-truth state or difficulty control — our environment
is the differentiator, so the paper should stand where we are strongest; (3) it degrades
gracefully — if selection only finds echoes, framing 1 collapses but framing 2 still has a paper
("the verbalizable workspace in an agent is echo-dominated; probes see more than the lens"), with
the same figures.

The supervisor's instinct that "this is enough for a paper" is right about *volume* — sweep +
selection + baselines + probes is a full pipeline — but volume is not the ICLR bar; the bar is a
claim that survives §2. Tier 1 alone (one week of analysis) tells us which claim that is.

## 6. Rigor checklist (cheap to do now, expensive to be caught without)

- Logit-lens and WikiText-lens selection controls (G). Non-negotiable.
- Budget matching everywhere (same N tokens × M layers per condition, same seeds protocol).
- **Trajectory-grouped splits**: current next-action train/eval split is i.i.d. over samples;
  tokens from the same trajectory must not straddle the split (test-set eval is fine, but the
  in-training eval numbers will be optimistic; fix is one grouping flag).
- Cluster bootstrap by trajectory for every headline number; per-size breakdown in appendix.
- Report the label distribution: next-action classes are imbalanced (RIGHT/DOWN dominate on
  some sizes); balanced accuracy or per-class F1, not raw accuracy.
- Exclude the final-channel action token itself from all probing (trivial leak).
- The `torch.allclose` cross-check of sweep activations vs. gather_activations (open thread in
  the session readme) before any probe result is trusted.
- Fixed direction vocabulary version, committed, with the notebook that built it (done — commit
  `2698522`; freeze it).

## 7. Five weeks, concretely

| Week | Focus |
|---|---|
| Aug 21–27 | GPU sweep over train set (already scripted); `allclose` check; Tier-1 A+B analysis on first slice of data → **go/no-go on framing by end of week** |
| Aug 28–Sep 3 | Full probe matrix (selected/random/all/last-token + strata from A); C layer profiles; G lens controls |
| Sep 4–10 | D (interventions; resampling subset) *or* E (attention probe + OOD curves) — pick one, start the other only if ahead |
| Sep 11–17 | F/H if ahead; freeze numbers; write. **Abstract due Sep 18** |
| Sep 18–25 | Full draft, ablation appendix, repo cleanup (roadmap item 5). **Paper due Sep 25** |

The single most important scheduling fact: **Tier 1 is upstream of the framing decision and costs
about a week.** Run it before committing to any storyline with the supervisor.

## 8. Links

- J-lens / workspace paper: transformer-circuits.pub/2026/workspace · arXiv 2607.15495 · code: github.com/anthropics/jlens
- Reasoning Theater (Goodfire): arXiv 2603.05488 · goodfire.ai/research/reasoning-theater
- Thought Anchors: arXiv 2506.19143 · Commitment boundary: arXiv 2606.13603 · Latent horizon: arXiv 2602.02103
- When CoT Fails: arXiv 2604.23351 · Latent-CoT dynamics: arXiv 2602.08783 · Probe-filtered RL ("Drop the Act"): arXiv 2605.11467
- Spatial: prior paper arXiv 2602.08964 · From Text to Space arXiv 2502.16690 · Spatial world models arXiv 2604.10690
- Token/layer-selective probing: arXiv 2601.13288 · J-lens engineering analysis: lesswrong.com/posts/vHxGD5HKsFuBStirq
- ICLR 2027: abstract **Sep 18**, paper **Sep 25** (AoE)
