"""Place a reasoning token inside its sentence, and label it against the commitment boundary.

The rollouts written by `scripts/inference_oss/run_inference.py` truncate a reasoning chain at
every sentence end and ask the model to answer, giving one `sentence_evals` entry per sentence
with an `eos_token_pos` and whether that truncation answered correctly. From those, `commitment_
metrics` derives the two indices this module labels against:

  `first_correct_sentence_idx`  the first sentence whose truncation is correct
  `convinced_sentence_idx`      the first sentence from which *every* later truncation is
                                correct, i.e. correct-from-here-on. `None` when the final
                                truncation is wrong, which is "never convinced", not "convinced
                                late".

**"Convinced" is defined against ground truth, not against the model settling down.** A chain
that flips to the right answer and stays there is convinced at the flip; one that holds a stable
wrong belief is never convinced at all. Do not read it as "stopped changing its mind".

Two index spaces meet here and they are not the same, which is the standing trap:

  `eos_token_pos`   indexes `step["output_tokens"]` directly.
  `reasoning_pos`   indexes the *analysis-tagged* tokens, i.e. what the lens CSVs and the
                    direction-mass tables are keyed by.

The gap between them is `reasoning_offset(eos)` == `eos[0] + 1`, the `<|channel|>analysis
<|message|>` header. It happens to be 3 on every trajectory checked so far, and
`scripts/build_probe_rollout_join.py` hardcodes that 3 — do not copy it. Read the offset per
trajectory; a chain whose header tokenizes differently would be silently shifted otherwise.

Sentence 0 is that header and owns no reasoning tokens, so every reasoning token lands in a
sentence with index >= 1. That is what makes `convinced_label` collapse to one expression across
all three commitment regimes.

Stdlib only, like the rest of `jlens_utils`: deciding what a token is labelled should not require
importing torch.
"""

import json
from dataclasses import dataclass
from pathlib import Path

#: A trajectory-step's commitment regime, from `convinced_sentence_idx`.
COHORT_NEVER = "never"
COHORT_IMMEDIATE = "immediate"
COHORT_NORMAL = "normal"


def rollout_path(probs_root: str | Path, name: str) -> Path:
    """`{probs_root}/size{S}/{name}.json`, with the grid size read out of the stem itself.

    The `size{S}` directory is redundant with the stem, so the stem alone is the join key
    across every artifact in this project.

    >>> rollout_path("/p", "together_ai_openai_gpt-oss-20b_size11_comp0.0_1").as_posix()
    '/p/size11/together_ai_openai_gpt-oss-20b_size11_comp0.0_1.json'
    """
    size = name.split("_size")[1].split("_")[0]
    return Path(probs_root) / f"size{size}" / f"{name}.json"


def read_rollout_steps(path: str | Path) -> dict[int, dict]:
    """A rollout JSON's steps, keyed by `step_id` so a mass table's `step` can index them."""
    with open(path, encoding="utf-8") as f:
        return {step["step_id"]: step for step in json.load(f)["steps"]}


def eos_positions(step: dict) -> list[int]:
    """The truncation points of one step, in `step["output_tokens"]` coordinates."""
    return [entry["eos_token_pos"] for entry in step["sentence_evals"]]


def reasoning_offset(eos: list[int]) -> int:
    """Output-token index of `reasoning_pos` 0, i.e. the first analysis-tagged token.

    Always `eos[0] + 1`: sentence 0 ends on the last token of the `<|channel|>analysis
    <|message|>` header, so the reasoning proper starts one past it. Read it, never assume 3.

    >>> reasoning_offset([2, 5, 8])
    3
    """
    return eos[0] + 1


def sentence_of_token(eos: list[int]) -> dict[int, int]:
    """output-token index -> sentence index, for sentences 1.. (sentence 0 is the header).

    Sentences are indexed by their RIGHT endpoint and the span is inclusive of it, so the eos
    token itself belongs to the sentence it closes.

    >>> sentence_of_token([2, 5, 8])
    {3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}
    """
    span: dict[int, int] = {}
    prev = eos[0]
    for si, e in enumerate(eos):
        if si == 0:
            continue
        for t in range(prev + 1, e + 1):
            span[t] = si
        prev = e
    return span


def cohort(convinced_idx: int | None) -> str:
    """Which commitment regime a trajectory-step is in.

    Kept as a column rather than a filter: `immediate` steps contribute only positive rows and
    `never` steps only negative ones, so a model trained on all three should be able to be
    scored without them without anything being rebuilt.

    >>> cohort(None), cohort(0), cohort(4)
    ('never', 'immediate', 'normal')
    """
    if convinced_idx is None:
        return COHORT_NEVER
    return COHORT_IMMEDIATE if convinced_idx == 0 else COHORT_NORMAL


def convinced_label(convinced_idx: int | None, sentence_idx: int) -> int:
    """1 when this token's sentence is at or past the commitment boundary.

    Reasoning tokens always sit in `sentence_idx >= 1`, so this one expression already covers
    all three regimes: `None` labels every token 0, `0` labels every token 1, and a boundary at
    `k >= 1` splits the chain at `k`.

    >>> convinced_label(3, 2), convinced_label(3, 3), convinced_label(3, 4)
    (0, 1, 1)
    >>> convinced_label(0, 1)
    1
    >>> convinced_label(None, 9)
    0
    """
    if convinced_idx is None:
        return 0
    return int(sentence_idx >= convinced_idx)


@dataclass(frozen=True)
class Placement:
    """Where one reasoning token sits, in its sentence and relative to the boundary.

    `sentence_frac` is 0 at a sentence's first token and 1 at its last; `rel_sentence` is 0 for
    the convinced sentence itself, so it occupies `x_sentence` in [-1, 0] and the sentence after
    it [0, +1]. `rel_sentence`/`x_sentence` are None exactly when the step was never convinced.
    """

    sentence_idx: int
    pos_in_sentence: int
    sentence_len: int
    sentence_frac: float
    is_sentence_end: int
    rel_sentence: int | None
    x_sentence: float | None


def place_token(
    tok_idx: int,
    eos: list[int],
    span: dict[int, int],
    convinced_idx: int | None = None,
) -> Placement | None:
    """Locate one output-token index within its sentence. None if it is in no sentence.

    A one-token sentence has no interior to interpolate over, so its `sentence_frac` is 1.0 —
    the token is simultaneously the sentence's first and last, and it is always an eos.

    >>> span = sentence_of_token([2, 5, 8])
    >>> place_token(3, [2, 5, 8], span, 2)
    Placement(sentence_idx=1, pos_in_sentence=0, sentence_len=3, sentence_frac=0.0, is_sentence_end=0, rel_sentence=-1, x_sentence=-2.0)
    >>> place_token(8, [2, 5, 8], span, 2).is_sentence_end
    1
    >>> place_token(99, [2, 5, 8], span, 2) is None
    True
    """
    si = span.get(tok_idx)
    if si is None:
        return None
    start = eos[si - 1] + 1
    sentence_len = eos[si] - start + 1
    pos_in_sentence = tok_idx - start
    sentence_frac = (pos_in_sentence / (sentence_len - 1)) if sentence_len > 1 else 1.0
    rel = None if convinced_idx is None else si - convinced_idx
    return Placement(
        sentence_idx=si,
        pos_in_sentence=pos_in_sentence,
        sentence_len=sentence_len,
        sentence_frac=sentence_frac,
        is_sentence_end=int(tok_idx == eos[si]),
        rel_sentence=rel,
        x_sentence=None if rel is None else rel - 1 + sentence_frac,
    )
