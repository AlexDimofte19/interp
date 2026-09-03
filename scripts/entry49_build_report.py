#!/usr/bin/env python3
"""Build the 16-probe loudness report, with FULL PROVENANCE under every figure.

Entry 48's page was hand-written HTML. This generates it instead, from two registries:

``PROBES``   one entry per probe -- the .pt it came from, the activation tree and token
             selection it was TRAINED on, how many samples that was, and which label.
``FIGURES``  one entry per .png -- what it plots, which probes appear in it, how the x
             axis is binned, and any filter applied to the rows.

Every caption is assembled from those two, so a figure can never drift from its own
description: change the registry, and the caption changes with it.

WHY THIS MATTERS HERE. Three things about this page are counter-intuitive enough that a
reader who guesses will guess wrong:

  1. THE ROWSET NAMES NO LONGER SELECT TOKENS. Since entry 48 ``p1_full``, ``p1_top20``
     and ``p2`` hold IDENTICAL rows -- every reasoning token of the held-out 360. The name
     now selects WHICH PROBES ARE READ, nothing else. A caption that said "p1_full" and
     stopped would read as a token subset. Each one says so explicitly.
  2. TRAINING AND EVALUATION SELECTION ARE DIFFERENT THINGS. Every probe was trained on a
     SELECTED ~20 tokens per trajectory; every probe is READ here on ALL 87,221 tokens,
     selected by nothing. That gap is the entire content of entry 49(c).
  3. THERE ARE TWO BELIEF LABELS IN THIS PROJECT AND THIS PAGE USES ONE OF THEM. Here
     ``label_local`` is the AT-TOKEN belief: the chain is cut at that exact token and the
     model asked for its action (the ``every_token`` rollout). The commitment-boundary
     pages (entries 39/47) instead score agreement at a SENTENCE END. Same idea, different
     cut point, not interchangeable.

Usage:
    python scripts/entry49_build_report.py            # -> <out>/report.html
"""

import argparse
import base64
import json
from pathlib import Path

OUT_DEFAULT = Path("/workspace/reasoning_theatre/probe_loudness_heldout360_16probes")

# --------------------------------------------------------------------------------------
# What every figure on this page shares. Stated once, then referenced by every caption.
# --------------------------------------------------------------------------------------
COMMON = {
    "eval_set": "held-out 360 trajectories — a disjoint tree, zero overlap with the 3,600 any probe trained on",
    "eval_tokens": "ALL 87,221 reasoning tokens. No selection: every analysis-tagged token of every step",
    "eval_layer": "layer 15 residual stream (<code>heldout360_l15</code>), the only layer gathered there",
    "eval_label_local": (
        "<b>local belief, at-token</b> — the chain is truncated at <i>that exact token</i>, the fixed "
        "final-channel prefix appended, and the single action token the model emits is the label "
        "(<code>every_token</code> rollout, 87,581 cutoff-evals). <b>Not</b> the sentence-end answer used "
        "by the commitment-boundary pages"
    ),
    "eval_label_final": "<b>final action</b> — the trajectory's own <code>agent_action</code>, i.e. where it ended up",
    "loudness_jlens": (
        "<b>jlens loudness</b> — layer-15 full-vocabulary direction mass under the <b>Jacobian lens</b>, "
        "<code>log Σ p(t)</code> over the 446-token <code>direction_tokens_full.json</code> "
        "(<code>jlens_mass_L15</code>)"
    ),
    "loudness_logitlens": (
        "<b>logitlens loudness</b> — the same quantity under the <b>logit lens</b> "
        "(<code>logitlens_mass_L15</code>). At layer 15 the two lenses' top-20 token sets overlap only "
        "~50%, and the logit lens puts ~6x less mass on direction words (mean log-mass −5.23 vs −3.38), "
        "so these are genuinely different rulers"
    ),
    "train_set": "2,880 training trajectories (the 720 eval trajectories are held out of training and are a different set again)",
    "hparams": "lr / mlp(1024), lr 3e-4, wd 1e-3, dropout 0, 50 epochs, batch 512, class-weight balanced, normalized, seed 42",
}

# --------------------------------------------------------------------------------------
# One entry per probe. `sel` is the TRAINING selection; every probe is EVALUATED on all tokens.
# --------------------------------------------------------------------------------------
P = dict
PROBES: dict[str, dict] = {
    "p1_lr": P(
        label="P1 all — jlens per-sentence, lr",
        file="local_belief_probes/probes/local_belief_p1_lr.pt",
        tree="argmax_per_sentence_l15",
        sel="jlens: each sentence's LOUDEST token, uncapped (~20.8/traj, up to 230 from one chain)",
        n="75,008 samples",
        lab="local belief (at-token), from the jlens <code>jlens_argmax_per_sentence</code> rollout",
        e720=".606",
        held=".4634",
        entry="45",
    ),
    "p1_mlp": P(
        label="P1 all — jlens per-sentence, mlp",
        file="local_belief_probes/probes/local_belief_p1_mlp.pt",
        tree="argmax_per_sentence_l15",
        sel="jlens: each sentence's LOUDEST token, uncapped (~20.8/traj)",
        n="75,008 samples",
        lab="local belief (at-token), jlens per-sentence rollout",
        e720=".678",
        held=".5276",
        entry="45",
    ),
    "p1t20_lr": P(
        label="P1 top-20 — thinned, lr",
        file="local_belief_probes/probes/local_belief_p1_top20_lr.pt",
        tree="argmax_per_sentence_l15",
        sel="jlens per-sentence loudest, then THINNED to the 20 loudest per trajectory",
        n="~57,600 samples",
        lab="local belief (at-token), jlens per-sentence rollout",
        e720=".710",
        held=".4572",
        entry="45",
    ),
    "p1t20_mlp": P(
        label="P1 top-20 — thinned, mlp",
        file="local_belief_probes/probes/local_belief_p1_top20_mlp.pt",
        tree="argmax_per_sentence_l15",
        sel="jlens per-sentence loudest, thinned to top-20/traj",
        n="~57,600 samples",
        lab="local belief (at-token), jlens per-sentence rollout",
        e720=".774",
        held=".4989",
        entry="45",
    ),
    "p2_lr": P(
        label="P2 — jlens global top-20, lr",
        file="local_belief_probes/probes/local_belief_p2_lr.pt",
        tree="jlens_mass_l15",
        sel="jlens: the 20 loudest tokens of the whole chain (<code>logprob_mass_full</code> @L15)",
        n="71,913 samples",
        lab="local belief (at-token), from <code>jlens_top_k_global</code>",
        e720=".802",
        held=".4869",
        entry="45",
    ),
    "p2_mlp": P(
        label="P2 — jlens global top-20, mlp",
        file="local_belief_probes/probes/local_belief_p2_mlp.pt",
        tree="jlens_mass_l15",
        sel="jlens global top-20 by <code>logprob_mass_full</code> @L15",
        n="71,913 samples",
        lab="local belief (at-token), <code>jlens_top_k_global</code>",
        e720=".862",
        held=".5243",
        entry="45",
    ),
    "base_lr": P(
        label="baseline — jlens top-20, FINAL label, lr",
        file="probes/next_action_mass_l15/next_action_probe_jlens_topall_lr.pt",
        tree="jlens_mass_l15",
        sel="jlens global top-20 — <b>the same tokens as P2</b>",
        n="71,913 samples",
        lab="<b>final action</b> (<code>agent_action</code>) — the label contrast against P2",
        e720=".699",
        held=".4359",
        entry="37/38",
    ),
    "base_mlp": P(
        label="baseline — jlens top-20, FINAL label, mlp",
        file="probes/next_action_mass_l15/next_action_probe_jlens_topall_mlp.pt",
        tree="jlens_mass_l15",
        sel="jlens global top-20 — the same tokens as P2",
        n="71,913 samples",
        lab="<b>final action</b> — the label contrast against P2",
        e720=".745",
        held=".4556",
        entry="37/38",
    ),
    "rand_lr": P(
        label="random control — FINAL label, lr",
        file="probes/next_action_mass_l15/next_action_probe_random_topall_lr.pt",
        tree="jlens_mass_l15",
        sel="<b>random</b>: 20 tokens/traj, seeded uniform draw (seed 42), no loudness used",
        n="71,913 samples",
        lab="<b>final action</b>",
        e720=".574",
        held=".4569",
        entry="37/38",
    ),
    "rand_mlp": P(
        label="random control — FINAL label, mlp",
        file="probes/next_action_mass_l15/next_action_probe_random_topall_mlp.pt",
        tree="jlens_mass_l15",
        sel="<b>random</b>: 20 tokens/traj, seeded uniform draw (seed 42)",
        n="71,913 samples",
        lab="<b>final action</b>",
        e720=".625",
        held=".4870",
        entry="37/38",
    ),
    # ---- entry 49 ----
    "randb_lr": P(
        label="NEW random selection — BELIEF label, lr",
        file="probes/local_belief_baselines/next_action_probe_random_belief_lr.pt",
        tree="jlens_mass_l15",
        sel="<b>random</b>: <b>the identical 20 tokens/traj the random control above used</b>, replayed from the selection record",
        n="71,913 samples",
        lab="local belief (at-token), from <code>recorded_selection</code>",
        e720=".6595",
        held=".5274",
        entry="49",
    ),
    "randb_mlp": P(
        label="NEW random selection — BELIEF label, mlp",
        file="probes/local_belief_baselines/next_action_probe_random_belief_mlp.pt",
        tree="jlens_mass_l15",
        sel="<b>random</b>: identical tokens to the random control — only the label differs",
        n="71,913 samples",
        lab="local belief (at-token), <code>recorded_selection</code>",
        e720=".7234",
        held=".5886",
        entry="49",
    ),
    "ll1_lr": P(
        label="NEW logitlens P1 — per-sentence, lr",
        file="probes/local_belief_baselines/next_action_probe_logitlens_p1_lr.pt",
        tree="logitlens_argmax_per_sentence_l15",
        sel="<b>logit lens</b>: each sentence's loudest token, uncapped (~20.8/traj)",
        n="74,972 samples",
        lab="local belief (at-token), logitlens per-sentence rollout",
        e720=".5737",
        held=".4861",
        entry="49",
    ),
    "ll1_mlp": P(
        label="NEW logitlens P1 — per-sentence, mlp",
        file="probes/local_belief_baselines/next_action_probe_logitlens_p1_mlp.pt",
        tree="logitlens_argmax_per_sentence_l15",
        sel="<b>logit lens</b>: each sentence's loudest token, uncapped",
        n="74,972 samples",
        lab="local belief (at-token), logitlens per-sentence rollout",
        e720=".6438",
        held=".5483",
        entry="49",
    ),
    "ll2_lr": P(
        label="NEW logitlens P2 — global top-20, lr",
        file="probes/local_belief_baselines/next_action_probe_logitlens_p2_lr.pt",
        tree="logitlens_mass_l15",
        sel="<b>logit lens</b>: the 20 loudest of the chain (<code>logprob_mass_full</code> @L15)",
        n="71,913 samples",
        lab="local belief (at-token), logitlens <code>jlens_top_k_global</code> arm",
        e720=".7297",
        held=".5018",
        entry="49",
    ),
    "ll2_mlp": P(
        label="NEW logitlens P2 — global top-20, mlp",
        file="probes/local_belief_baselines/next_action_probe_logitlens_p2_mlp.pt",
        tree="logitlens_mass_l15",
        sel="<b>logit lens</b>: the 20 loudest of the chain @L15",
        n="71,913 samples",
        lab="local belief (at-token), logitlens global-top-20 arm",
        e720=".7963",
        held=".5498",
        entry="49",
    ),
}

ROWSET_NOTE = (
    "<b>Rowset <code>{rs}</code>.</b> The three rowsets on this page hold <b>identical rows</b> — every "
    "reasoning token of the held-out 360. The rowset name selects <b>which probes are read</b>, not which "
    "tokens. Kept only so entry 46's figures have direct counterparts."
)

FIGURES: dict[str, dict] = {
    "loudness_distribution.png": P(
        title="Loudness distribution — jlens vs logitlens",
        shows="Distribution of layer-15 direction mass over the evaluated tokens, against the all-token reference — left under <b>jlens loudness</b>, right under <b>logitlens loudness</b>.",
        probes=[],
        rowset=None,
        x="direction mass, log and probability form, under the lens named above each panel",
        note="Three coincident curves — since entry 48 the rowsets hold the same rows, so their distributions are identical by construction. The two panels differ because the lenses do: logitlens loudness sits ~6x lower in mass than jlens loudness.",
    ),
    "{rs}_by_mass_decile.png": P(
        title="Balanced accuracy by loudness decile — jlens vs logitlens",
        shows="Each probe's balanced accuracy against the at-token local belief, within deciles of loudness — left panel binned by <b>jlens loudness</b>, right panel by <b>logitlens loudness</b>.",
        x="loudness decile (9 = loudest), equal-count bins — under the lens named above each panel",
        note="The headline shape: does a louder token decode better, with no selection truncating the axis? Ask it of each lens separately — the two rulers disagree about which tokens are loud.",
    ),
    "{rs}_by_sentence_frac.png": P(
        title="Balanced accuracy by position within the sentence",
        shows="The same accuracy, binned by where the token sits inside its own sentence.",
        x="position in its sentence, decile (0 = first token) — this is <code>frac_in_sentence</code>, NOT <code>sentence_idx/n_sentences</code>",
        note="The control for entry 44(e): loudness and sentence position are correlated under <i>both</i> lenses, so the position axis must be read beside the loudness one.",
    ),
    "{rs}_by_rel_sentence.png": P(
        title="Balanced accuracy around the commitment boundary",
        shows="Accuracy by sentence index relative to the sentence where the model becomes convinced.",
        x="sentence_idx − convinced_idx, clipped to ±6",
        note="Rows with no defined commitment sentence are dropped, so this figure uses a subset of the 87,221.",
    ),
    "{rs}_mass_x_position.png": P(
        title="Loudness × position (one probe)",
        shows="Accuracy on a loudness-tercile × sentence-position-tercile grid, for a single mlp probe — loudness under the lens named above each panel.",
        x="loudness tercile (columns, under the panel's own lens) × position tercile (rows)",
        note="Separates the two correlated axes: movement across that panel's loudness at fixed position, vs across position at fixed loudness.",
    ),
    "{rs}_follows_by_mass.png": P(
        title="Which label the probe follows, by loudness — jlens vs logitlens",
        shows="On rows where the local belief and the final action DISAGREE: how often the probe predicts the belief, the final action, or neither.",
        x="loudness decile, under the lens named above each panel",
        note="Restricted to disagreement rows only. This is where a final-action-trained probe can be seen switching to reporting the current belief.",
    ),
    "{rs}_chain_length_control.png": P(
        title="Loudness gradient inside chain-length quartiles — jlens vs logitlens",
        shows="The loudness→accuracy gradient re-cut within quartiles of reasoning-chain length, under each lens.",
        x="loudness tercile (that panel's lens), one sub-panel per chain-length quartile",
        note="Entry 46's landmine: inside a fixed top-K arm, <b>jlens loudness</b> correlates with chain length (r = +0.42) and the two effects cancel. On this page selection is removed and the confound is gone (r = −0.010) — the panel is kept as the check. Both r values were measured under jlens loudness.",
    ),
    "p2_label_comparison.png": P(
        title="The two labels, side by side",
        shows="Each probe scored against the at-token local belief and against the final action.",
        probes=None,
        rowset="p2",
        x="probe",
        note="The gap between the two bars for one probe is the label effect on this population.",
    ),
}


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def probe_rows(keys: list[str]) -> str:
    if not keys:
        return ""
    out = [
        "<div class='scroll'><table class='prov'><thead><tr><th>probe</th>"
        "<th>trained on — tree &amp; selection</th>"
        "<th>training label</th><th>train n</th><th>eval-720</th><th>heldout-360</th></tr></thead><tbody>"
    ]
    for k in keys:
        m = PROBES.get(k)
        if not m:
            continue
        out.append(
            f"<tr><td><b>{m['label']}</b><br><code>{m['file']}</code></td>"
            f"<td><code>{m['tree']}</code><br>{m['sel']}</td>"
            f"<td>{m['lab']}</td><td>{m['n']}</td><td>{m['e720']}</td><td>{m['held']}</td></tr>"
        )
    return "\n".join(out) + "</tbody></table></div>"


def caption(fname: str, spec: dict, rs: str | None, keys: list[str]) -> str:
    rows = probe_rows(keys)
    rsnote = f"<p class='rs'>{ROWSET_NOTE.format(rs=rs)}</p>" if rs else ""
    return f"""
<div class="cap">
  <p class="what"><b>What this shows.</b> {spec["shows"]}</p>
  {rsnote}
  <div class="scroll"><table class="prov settings"><tbody>
    <tr><th>x axis / binning</th><td>{spec["x"]}</td></tr>
    <tr><th>evaluation set</th><td>{COMMON["eval_set"]}</td></tr>
    <tr><th>evaluation tokens</th><td>{COMMON["eval_tokens"]}</td></tr>
    <tr><th>evaluation label</th><td>{COMMON["eval_label_local"]}</td></tr>
    <tr><th>comparison label</th><td>{COMMON["eval_label_final"]}</td></tr>
    <tr><th>activations</th><td>{COMMON["eval_layer"]}</td></tr>
    <tr><th>left panel — loudness</th><td>{COMMON["loudness_jlens"]}</td></tr>
    <tr><th>right panel — loudness</th><td>{COMMON["loudness_logitlens"]}</td></tr>
    <tr><th>metric</th><td>balanced accuracy — mean per-class recall over LEFT/UP/RIGHT/DOWN present in the bin; chance = .25</td></tr>
    <tr><th>training set</th><td>{COMMON["train_set"]}</td></tr>
    <tr><th>training hyperparameters</th><td>{COMMON["hparams"]}</td></tr>
    <tr><th>files</th><td><code>plots/{fname}</code> (jlens loudness) &middot; <code>plots_logitlens_loudness/{fname}</code> (logitlens loudness)</td></tr>
  </tbody></table></div>
  <p class="note"><b>Note.</b> {spec["note"]}</p>
  {rows}
</div>"""


CSS = """
/* Palette is taken from the figures themselves: the belief probes plot in indigo, the
   entry-49 arms in magenta, the final-label controls in ochre. The page uses the same three
   so a reader moves between a figure and its caption without re-learning the colour code.
   Neutrals carry a slight blue bias toward the indigo rather than sitting at pure grey. */
:root{
  --bg:#ffffff; --card:#f3f6fa; --sunk:#e9eef5; --line:#d8e0ea;
  --fg:#15181e; --mut:#586074;
  --indigo:#1f3f7a; --magenta:#8a1f5e; --ochre:#a8541c;
  --warn-bg:#fdf3e7; --key-bg:#f5eef4;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#0f1216; --card:#181c23; --sunk:#212630; --line:#2c333f;
    --fg:#e7eaf1; --mut:#98a2b6;
    --indigo:#7ea3d8; --magenta:#cf86b6; --ochre:#d79a63;
    --warn-bg:#241d13; --key-bg:#221925;
  }
}
:root[data-theme="dark"]{
  --bg:#0f1216; --card:#181c23; --sunk:#212630; --line:#2c333f;
  --fg:#e7eaf1; --mut:#98a2b6;
  --indigo:#7ea3d8; --magenta:#cf86b6; --ochre:#d79a63;
  --warn-bg:#241d13; --key-bg:#221925;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:15px;line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:44px 22px 96px;display:flex;flex-direction:column;gap:0}
h1,h2,figure h3{font-family:"Source Serif 4",Georgia,serif;text-wrap:balance}
h1{font-size:38px;line-height:1.15;margin:0 0 8px;font-weight:600;letter-spacing:-.01em}
h2{font-size:23px;font-weight:600;margin:52px 0 12px;padding-top:20px;border-top:2px solid var(--line)}
.sub{color:var(--mut);margin:0 0 26px;max-width:66ch}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--magenta);margin:0 0 6px}
figure{margin:28px 0 0;padding:20px;background:var(--card);border:1px solid var(--line);border-radius:8px}
figure h3{margin:0;font-size:18px;font-weight:600}
figure img{width:100%;height:auto;display:block;background:#fff;border:1px solid var(--line);
  border-radius:5px;margin:6px 0 4px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:12px 0 4px}
@media (max-width:820px){.pair{grid-template-columns:1fr}}
.panel{min-width:0}
.lens{margin:0;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;
  letter-spacing:.08em;text-transform:uppercase;color:var(--magenta);font-weight:600}
.lenshint{margin:2px 0 0;font-size:11.5px;color:var(--mut)}
.cap{font-size:13.5px}
.what{margin:10px 0}
.rs{margin:10px 0;padding:9px 12px;background:var(--warn-bg);border-left:3px solid var(--ochre);border-radius:4px}
.scroll{overflow-x:auto;margin:12px 0}
table.prov{width:100%;min-width:640px;border-collapse:collapse;font-size:12.5px;
  font-variant-numeric:tabular-nums}
table.prov th,table.prov td{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
table.prov thead th{background:var(--sunk);font-weight:600}
table.settings{min-width:520px}
table.settings th{width:210px;background:var(--sunk);color:var(--mut);font-weight:600;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;letter-spacing:.02em}
.note{margin:12px 0 0;color:var(--mut)}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;background:var(--sunk);
  padding:1px 5px;border-radius:3px;font-size:12px}
.key{background:var(--key-bg);border-left:3px solid var(--magenta);padding:14px 16px;
  border-radius:4px;margin:18px 0}
table.res{width:100%;min-width:560px;border-collapse:collapse;font-size:13.5px;
  font-variant-numeric:tabular-nums}
table.res th,table.res td{border:1px solid var(--line);padding:8px 10px;text-align:left}
table.res thead th{background:var(--sunk);font-weight:600}
a{color:var(--indigo)}
:focus-visible{outline:2px solid var(--magenta);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


DW_FIGURES = [
    (
        "q1_direction_share_by_loudness.png",
        "Direction words concentrate in the loud tokens — and the lenses differ",
        "Share of evaluated tokens that are themselves direction words, per loudness decile, against the "
        "6.2% base rate over all 87,221. <b>jlens loudness runs 0.0% → 44.3%</b> across its deciles; "
        "<b>logitlens loudness 0.7% → 26.9%</b>. Both concentrate verbalized tokens; the Jacobian lens does "
        "it far harder — 7.1x the base rate in its top decile against 4.3x. The two are nearly equal at "
        "decile 8 (12.6% vs 13.2%) and only part at the very top.",
    ),
    (
        "q2_gradient_nondirection.png",
        "The gradient survives with every direction word removed",
        "The headline claim, asked of the 81,819 <b>non-direction</b> tokens alone. Under <b>jlens "
        "loudness</b> accuracy still climbs across all ten deciles — jlens-top-20-belief .399→.729 "
        "(+.331), the final-label baseline .343→.597 (+.255), logitlens-P2-belief .478→.688 (+.210), "
        "random-belief .511→.684 (+.174). Under <b>logitlens loudness</b> the same probes climb far less "
        '(+.104, +.076, +.138, +.043). So loudness is not a proxy for "the token says up" — but only '
        "jlens loudness orders the non-verbalized tokens well.",
    ),
    (
        "q3_gradient_direction.png",
        "The same gradient on direction words alone",
        "The 5,401 tokens that ARE direction words. They thin out fast in the quiet deciles — under jlens "
        "loudness the bottom deciles hold almost none, which is finding 1 restated — so cells below 120 "
        "rows are computed but not plotted.",
    ),
    (
        "q3_direction_word_gap.png",
        "Is the direction-word advantage just loudness?",
        "Accuracy on direction words minus accuracy on non-direction words: over all tokens, and again "
        "inside the loudest decile only, where loudness is held roughly fixed. Over all tokens the "
        "advantage is large (+.13 to +.23) and identical under both lenses — it must be, since no binning "
        "is involved. <b>Matching on jlens loudness collapses it</b> (+.064 to +.127, roughly halved or "
        "better); <b>matching on logitlens loudness barely moves it</b> (+.134 to +.193). jlens loudness "
        "captures most of what makes a direction word easy to decode; logitlens loudness does not.",
    ),
    (
        "q5_lens_agreement.png",
        "How much do the two rulers actually agree — and is it the direction words?",
        "<b>jlens loudness on x, logitlens loudness on y</b>, every one of the 87,221 held-out tokens, "
        "as a log-count hexbin. Left: all tokens, <b>Pearson r = 0.437, Spearman ρ = 0.411</b>. Right: the "
        "same with all 5,401 direction words removed, <b>r = 0.368, ρ = 0.346</b>. "
        "<br><br>Two things follow. First, the rulers are only <i>moderately</i> related — ρ ≈ .41 means "
        "each lens is largely its own measurement, which is why the ~50% top-20 overlap and the inverted "
        "orderings elsewhere on this page are not anomalies. Second, agreement <b>falls</b> when the "
        "direction words go, so part of what agreement there is comes from the easy case both lenses "
        "catch: the token the model literally typed. On everything else they agree less still. Spearman ρ "
        "is the figure to quote — every use of loudness in this project is a ranking, not a regression.",
    ),
    (
        "q4_label_effect_nondirection.png",
        "The label effect does not need verbalized tokens",
        "Entry 49's headline — relabelling from the final action to the at-token local belief — "
        "recomputed on <b>non-direction words only</b>. Random selection <b>+10.3 pp</b> (.4772 → .5806) "
        "and jlens top-20 <b>+6.6 pp</b> (.4440 → .5099), against +10.2 and +6.9 over all tokens. "
        "Essentially unchanged: the label effect is not carried by tokens the model already typed. "
        "Identical under both lenses because no loudness binning enters this quantity.",
    ),
]


def direction_word_section(root: Path) -> str:
    """The direction-word analysis. Each figure is ALREADY two-panel (jlens | logitlens),
    so it is emitted once rather than paired the way the loudness figures are."""
    plots = root / "direction_words" / "plots"
    if not plots.exists():
        return ""
    out = [
        "<h2>Isolating the direction words</h2>",
        "<p class='sub'>The standing confound: a lens calls a token loud partly because the model has "
        "already <i>written</i> a direction word there, so a probe scoring well on loud tokens might only "
        "be reading that word. Tested twice before — entry 37(2) inside the selected top-20 against the "
        "final label, entry 46 on the eval-720 split — and here with the selection gone, the belief label, "
        "sixteen probes, and both rulers.</p>",
        "<div class='key'><b>The answer, in one line.</b> Both lenses do concentrate direction words in "
        "their loud tokens, the Jacobian lens far harder (44.3% vs 26.9% of its top decile against a 6.2% "
        "base rate) — but the loudness→accuracy gradient survives with every direction word removed, and "
        "the label effect survives essentially untouched. What does <i>not</i> survive is the assumption "
        "that the two rulers are interchangeable: matching on jlens loudness collapses the direction-word "
        "advantage, matching on logitlens loudness does not.</div>",
    ]
    for fname, title, body in DW_FIGURES:
        f = plots / fname
        if not f.exists():
            continue
        out.append(
            f"<figure><h3>{title}</h3>"
            f"<p class='lens'>{'left: all tokens &nbsp;·&nbsp; right: direction words removed' if fname.startswith('q5') else 'left: jlens loudness &nbsp;·&nbsp; right: logitlens loudness'}</p>"
            f"<img src='data:image/png;base64,{b64(f)}' alt='{title}'>"
            f"<div class='cap'><p class='what'>{body}</p>"
            "<div class='scroll'><table class='prov settings'><tbody>"
            f"<tr><th>evaluation set</th><td>{COMMON['eval_set']}</td></tr>"
            f"<tr><th>evaluation tokens</th><td>81,819 non-direction + 5,401 direction = all 87,221</td></tr>"
            f"<tr><th>evaluation label</th><td>{COMMON['eval_label_local']}</td></tr>"
            "<tr><th>direction vocabulary</th><td>the 446-token <code>direction_tokens_full.json</code> — "
            "a token counts as a direction word if the decoded token string is in it</td></tr>"
            "<tr><th>metric</th><td>balanced accuracy; cells below 120 rows computed but not plotted</td></tr>"
            f"<tr><th>file</th><td><code>direction_words/plots/{fname}</code></td></tr>"
            "</tbody></table></div></div></figure>"
        )
    return "\n".join(out)


def lens_comparison(sum_j: dict, sum_l: dict) -> str:
    """Per-probe loudness gradient under each ruler, side by side.

    ``delta`` is the balanced-accuracy rise from the quietest decile to the loudest, and
    ``reversals`` counts decile-to-decile drops -- so a bigger delta with zero reversals is a
    ruler that orders decodability better. The question this table answers is whether the
    logit lens orders the probes' own tokens as cleanly as the Jacobian lens does, including
    for the two arms the logit lens actually selected.
    """
    rows_j = sum_j.get("rowsets", {}).get("p2", {}).get("mass_monotonicity", {})
    rows_l = sum_l.get("rowsets", {}).get("p2", {}).get("mass_monotonicity", {})
    if not rows_j or not rows_l:
        return "<p class='note'>Lens comparison unavailable — one of the two summaries is missing.</p>"
    out = [
        "<div class='scroll'><table class='res'><thead><tr><th>probe</th>"
        "<th>jlens loudness — quietest → loudest</th><th>Δ</th><th>rev</th>"
        "<th>logitlens loudness — quietest → loudest</th><th>Δ</th><th>rev</th>"
        "<th>which ruler orders it better</th></tr></thead><tbody>"
    ]
    for k in rows_j:
        if k not in rows_l:
            continue
        j, l = rows_j[k], rows_l[k]
        name = PROBES.get(k, {}).get("label", k)
        win = "jlens" if j["delta"] > l["delta"] else "logitlens"
        margin = abs(j["delta"] - l["delta"])
        out.append(
            f"<tr><td>{name}</td>"
            f"<td>{j['first']:.3f} → {j['last']:.3f}</td><td>{j['delta']:+.3f}</td><td>{j['n_reversals']}</td>"
            f"<td>{l['first']:.3f} → {l['last']:.3f}</td><td>{l['delta']:+.3f}</td><td>{l['n_reversals']}</td>"
            f"<td><b>{win}</b> by {margin:.3f}</td></tr>"
        )
    return "\n".join(out) + "</tbody></table></div>"


def pair(plots_j: Path, plots_l: Path, fname: str, spec: dict, rs: str | None, keys: list[str]) -> str:
    """One figure, drawn twice: binned by jlens loudness and by logitlens loudness.

    Same probes, same rows, same metric -- the ONLY difference between the two panels is
    which lens's layer-15 direction mass defines the x axis. Showing them apart would let a
    reader carry a conclusion from one ruler to the other, which is exactly the mistake the
    single-lens version of this page invited for the two logitlens-selected arms.
    """
    panels = []
    for lens, root, note in (
        ("jlens loudness", plots_j, "the Jacobian lens — the ruler every page before entry 49 used"),
        ("logitlens loudness", plots_l, "the logit lens — the ruler the ll1 / ll2 arms were actually selected by"),
    ):
        f = root / fname
        if not f.exists():
            continue
        panels.append(
            f"<div class='panel'><p class='lens'>{lens}</p>"
            f"<img src='data:image/png;base64,{b64(f)}' alt='{spec['title']} — {lens}'>"
            f"<p class='lenshint'>{note}</p></div>"
        )
    if not panels:
        return ""
    return (
        f"<figure><h3>{spec['title']}</h3><div class='pair'>"
        + "".join(panels)
        + "</div>"
        + caption(fname, spec, rs, keys)
        + "</figure>"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    plots_j = args.out / "plots"
    plots_l = args.out / "plots_logitlens_loudness"
    summary = json.loads((args.out / "summary.json").read_text()) if (args.out / "summary.json").exists() else {}
    sum_l_path = args.out / "logitlens_loudness" / "summary.json"
    summary_l = json.loads(sum_l_path.read_text()) if sum_l_path.exists() else {}

    rowset_probes = {rs: list(b.get("overall", {})) for rs, b in summary.get("rowsets", {}).items()}

    parts = [
        "<title>Sixteen Probes on the Held-Out 360</title>",
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        "family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&"
        'family=Source+Serif+4:opsz,wght@8..60,600&display=swap">',
        f"<style>{CSS}</style>",
        "<div class='wrap'>",
        "<p class='eyebrow'>ICLR log entry 49 &middot; probe loudness, selection removed</p>",
        "<h1>Sixteen Probes on the Held-Out 360</h1>",
        "<p class='sub'>Entry 48's page with the three entry-49 arms added — random selection with the "
        "belief label, and both logit-lens arms. Every probe is read on the identical 87,221 tokens, and "
        "<b>every figure is drawn twice: once binned by jlens loudness, once by logitlens loudness</b>. "
        "Each carries its full provenance underneath: which probes, trained on what, with which label, "
        "evaluated against what.</p>",
        "<div class='key'><b>Loudness is never unqualified on this page.</b> It is always <i>jlens "
        "loudness</i> (Jacobian lens) or <i>logitlens loudness</i> (logit lens) — two different rulers "
        "whose top-20 token sets overlap only ~50% at layer 15. Every page before this one binned by "
        "jlens loudness alone, which meant the two logitlens-selected arms were being ordered by a ruler "
        "they were never selected by. Both are shown side by side throughout, left and right.</div>",
        "<div class='key'><b>Read this first.</b> Every probe here was <b>trained</b> on a selected ~20 "
        "tokens per trajectory, and every probe is <b>evaluated</b> on <b>all</b> 87,221 reasoning tokens "
        "of a disjoint set of trajectories, selected by nothing. That gap between the training and "
        "evaluation distributions is not a flaw in the design — it is the finding (see the closing "
        "section). Numbers here are therefore <b>not</b> comparable to the eval-720 numbers quoted in the "
        "per-probe tables; only shapes are.</div>",
    ]

    order = ["p1_full", "p1_top20", "p2"]
    parts.append("<h2>Loudness distribution</h2>")
    f = "loudness_distribution.png"
    parts.append(pair(plots_j, plots_l, f, FIGURES[f], None, []))

    titles = {
        "p1_full": "Rowset p1_full — the two uncapped jlens per-sentence probes",
        "p1_top20": "Rowset p1_top20 — the two thinned jlens per-sentence probes",
        "p2": "Rowset p2 — twelve probes: the jlens top-20 pair, both final-label controls, and all six entry-49 arms",
    }
    for rs in order:
        keys = rowset_probes.get(rs, [])
        parts.append(f"<h2>{titles[rs]}</h2>")
        for tmpl in [
            "{rs}_by_mass_decile.png",
            "{rs}_by_sentence_frac.png",
            "{rs}_by_rel_sentence.png",
            "{rs}_mass_x_position.png",
            "{rs}_follows_by_mass.png",
            "{rs}_chain_length_control.png",
        ]:
            fname = tmpl.format(rs=rs)
            if not (plots_j / fname).exists() and not (plots_l / fname).exists():
                continue
            spec = FIGURES[tmpl]
            shown = keys
            if "mass_x_position" in tmpl:
                shown = [k for k in keys if k.endswith("mlp")][:1]
            elif "follows_by_mass" in tmpl or "chain_length" in tmpl:
                shown = [k for k in keys if k.endswith("mlp")]
            parts.append(pair(plots_j, plots_l, fname, spec, rs, shown))

    f = "p2_label_comparison.png"
    parts.append("<h2>The two labels, side by side</h2>")
    parts.append(pair(plots_j, plots_l, f, FIGURES[f], "p2", rowset_probes.get("p2", [])))

    parts.append("<h2>The two rulers, compared</h2>")
    parts.append(lens_comparison(summary, summary_l))
    parts.append(direction_word_section(args.out))

    parts.append("<h2>What the numbers say</h2>")
    parts.append(
        "<div class='scroll'><table class='res'><thead><tr><th>selection (belief label)</th>"
        "<th>eval-720 mlp — its own loud regime</th><th>heldout-360 mlp — every token</th></tr></thead><tbody>"
        "<tr><td>random 20/traj</td><td>.723 <i>(worst)</i></td><td><b>.5886 (best of all 16)</b></td></tr>"
        "<tr><td>logitlens top-20</td><td>.796</td><td>.5498</td></tr>"
        "<tr><td>jlens top-20</td><td><b>.862 (best)</b></td><td>.5243</td></tr>"
        "</tbody></table></div>"
        "<div class='key'>The selection ordering <b>inverts</b> between the two populations — a mirror "
        "image. A loud-selected probe only ever saw high-mass tokens and is specialised to them; the "
        "random control saw a uniform draw and generalises. The logit lens sits between the other two in "
        "<b>both</b> orderings, which is what says this is a selection-strength / coverage trade rather "
        "than one population flattering one lens.<br><br>The <b>label</b> effect, by contrast, survives "
        "the population change intact: on the same tokens and this disjoint tree, +10.2 pp (random, "
        ".4870→.5886) and +6.9 pp (jlens top-20, .4556→.5243). <b>So a selection claim must always name "
        "its population; a label claim need not.</b></div>"
        "<p class='note'>All ten entry-48 probes reproduce their published held-out numbers to 4 dp "
        "(delta +0.0000) in this rebuild, which is the check that the six added arms sit on the same "
        "measurement. Join: 0 mass mismatches, 0 final-label mismatches, 1 row of 87,221 with no valid "
        "rollout action.</p>"
        "<p class='note'><b>Not settled here:</b> the ±k proximity window of entry 42(e) — a token two "
        'positions before <code>" up"</code> is still "the model about to say up". Layer 15 only. '
        "There is no logitlens top-20 counterpart to the capped jlens per-sentence arm.</p>"
    )
    parts.append("</div>")

    dest = args.out / "report.html"
    dest.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {dest}  ({dest.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
