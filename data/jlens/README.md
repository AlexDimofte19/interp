# Lens signal vocabularies

The two token vocabularies the lens scoring is measured against. Both are copies of what is
deployed on the GPU host at `/workspace/jlens/`, taken 2026-09-04, and both are consumed as
`--signal-json` / `--direction-mass-json` / `--direction-tokens-path`.

| file | classes | tokens | used by |
|---|---|---|---|
| `direction_tokens_full.json` | UP, DOWN, LEFT, RIGHT | **446** | every completed experiment in the direction / next-action line |
| `grid_tokens_full.json` | WALL, OPEN, GOAL, AGENT, AXIS, STATUS | **1258** | the grid-probing round only -- which never ran |

Deploy them with `scripts/reproduce_all.sh` stage `deploy_vocabularies`, or by hand:

```bash
cp data/jlens/*.json /workspace/jlens/
```

Both are built **from the model vocabulary alone** by `notebooks/direction_tokens.ipynb` and
`notebooks/grid_tokens.ipynb` -- never from what is frequent in the j-space, since the j-space
is what they are used to measure. Every seed and anchor must itself be a gpt-oss-20b token
(`admissible()`, cell 2 of both notebooks). Keep that separation.

## Known discrepancy: the grid vocabulary is the pre-rule version

ICLR log entry 19 records the model-token-only rule reducing **direction 540 -> 446** and
**grid 1258 -> 927**. The direction file here is the post-rule 446. The grid file here is
**1258**, i.e. the version from *before* the rule was applied, even though the host copy is
dated after it.

It is committed as-is on purpose: it is what is deployed, so it is what a re-run would use,
and rewriting it would change the vocabulary silently. Nothing published depends on it -- the
only consumer is the grid-probing round, which never started (log entries 32-35), so no result
was ever measured against it.

Before spending GPU on the grid round, regenerate it by running `notebooks/grid_tokens.ipynb`
start to finish and confirm the count lands at 927. Note that log entry 20 flags ~95 junk
tokens that the dropped dictionary now admits, concentrated in ten seeds
(`objectif`, `pared`, `livre`, `agent`, `ligne`, `position`, `barrier`, `blocked`, ...) -- the
recommended fix is to drop those seeds and re-run, and it was never applied. So a naive
regeneration produces a *third* version, not the 927 of entry 19. Decide which is wanted
before gathering against it.

The direction vocabulary needs none of this: 446 is the post-rule count, and it is what every
number in the research summary was measured with.
