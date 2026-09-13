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

## The Qwen twins

`notebooks/direction_tokens_qwen.ipynb` and `notebooks/grid_tokens_qwen.ipynb` are the same two
notebooks against `Qwen/Qwen3.6-35B-A3B` (248,077 tokens, 200,358 stripped forms, against
gpt-oss-20b's 200,019 and 150,233). **`MODEL_ID` is the only thing that differs** -- every seed,
anchor, threshold and guard is byte-identical -- so any difference between a gpt-oss vocabulary
and its Qwen twin is the model and nothing else.

Every artifact name in them carries a `MODEL_SLUG` (`qwen3-6-35b-a3b`), because the four they
touch must not collide: `vocab_all_tokens.npy` is loaded from cache with no check on which
tokenizer wrote it, so an unslugged run would silently re-emit the gpt-oss vocabulary under a
Qwen label, and the output JSON would overwrite the 446-token file above. They write
`direction_tokens_full_qwen3-6-35b-a3b.json` / `grid_tokens_full_qwen3-6-35b-a3b.json`; neither
has been run to completion or committed yet.

The model-token-only rule costs Qwen about what it costs gpt-oss -- seed coverage is comparable
or slightly better in every class (direction LEFT keeps 8 lexical seeds against gpt-oss's 5,
RIGHT 14 against 11, UP 12 against 14) -- but `SEM_T` is a percentile tuned by eye on the
gpt-oss bands, so re-read the bands the notebook prints before trusting the inherited `0.9999`.

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
