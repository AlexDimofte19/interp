# Script renames, 2026-09-04

Six scripts were named after the ICLR log entry that produced them (`entry49_*`). A log number
says when something was written, not what it does, and it stops meaning anything once the log
moves on. They now carry descriptive names and titles. The log entry survives as a
parenthetical cross-reference inside each header, never as the name.

## The renames

| old | new | what it does |
|---|---|---|
| `scripts/entry49_baseline_arms.sh` | `scripts/rollout_belief_baseline_arms.sh` | The belief-baseline rollout arms: random replay, and the two logit-lens loud arms |
| `scripts/entry49_baseline_probes.sh` | `scripts/train_belief_baseline_probes.sh` | The belief-baseline probes: label and selection separated |
| `scripts/entry49_baseline_report.sh` | `scripts/build_sixteen_probe_loudness_report.sh` | Sixteen probes on the held-out 360, under both loudness rulers |
| `scripts/entry49_status.sh` | `scripts/belief_baselines_status.sh` | Status of the belief-baseline round |
| `scripts/entry49_build_report.py` | `scripts/build_sixteen_probe_report_page.py` | Builds that report's HTML page from the probe/figure registries |
| `scripts/entry49_direction_word_analysis.py` | `scripts/analyze_direction_word_isolation.py` | Isolating the direction words under both loudness rulers |

Done with `git mv`, so `git log --follow` still reaches the whole history of each file.

## What did NOT change: the data paths

Three on-disk locations keep their original `entry49` spelling, because renaming them would
strand the run already on disk -- every resumability marker, every relabel report and every
provenance caption in the sixteen-probe page points at them:

```
/workspace/prepared/entry49_*                     the six belief-baseline datasets and splits
/workspace/reasoning_theatre/entry49_baselines    relabel reports and per-step logs
/workspace/logs/entry49                           the round's driver logs
```

They are no longer written as literals. Each is reached through a named variable, so a fresh
host can point them anywhere and the old spelling is a default, not a fact:

| variable | default |
|---|---|
| `BELIEF_BASELINE_PREPARED_PREFIX` | `/workspace/prepared/entry49` |
| `BELIEF_BASELINE_ROOT` | `/workspace/reasoning_theatre/entry49_baselines` |
| `BELIEF_BASELINE_LOGS` | `/workspace/logs/entry49` |

Rename the directories later by setting these; nothing else needs to change.

## Also folded in

`build_sixteen_probe_loudness_report.sh` previously stopped after the jlens-loudness figures,
and the second ruler was run by hand (the stray `analyze_logitlens_loudness.log` and
`plot_logitlens_loudness.log` in that output directory are the evidence). The two-ruler pass,
the direction-word isolation and the page build are now steps 6a'-8 of the script, so the
report rebuilds in one command.

## Not renamed

`ICLR log.txt` is append-only history and keeps every entry number, including in text that
names the old scripts. That is correct: it records what was true when it was written.
