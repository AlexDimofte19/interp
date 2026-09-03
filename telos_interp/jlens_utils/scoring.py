"""How a token's direction evidence at one layer becomes a number.

The original score was a **count**: of a row's top-k lens predictions, how many are
direction words. It throws away everything the lens actually said — a direction word the
lens ranked first with p=0.4 counts exactly as much as one it ranked 20th with p=1e-6.
`scripts/jlens_reasoning_tokens.py` now writes a `top_{i}_logprob` beside every `top_{i}`,
so the score can be the probability mass the lens puts on direction words instead.

Like `methods.py`, this is a **registry** rather than a branch in each consumer: a score
mode is an entry in `SCORES`, and the CSV reader, the top filter, the pruner and the
probe-data preparer all take one by name.

Two artifacts, and `source` picks between them
----------------------------------------------
`source="topk"` scores the analysis CSV: the direction words among a row's top-k lens
predictions. That is all a top-k table can see, and it is a **truncation** — direction mass
sitting at rank 21 is invisible, and a token whose direction belief is spread thinly over
many words can look empty.

`source="mass"` reads the direction-mass table, which
`scripts/jlens_reasoning_tokens.py` computes at the source: while the `[b, vocab]` logits
are still on the device it gathers *every* direction token id and logsumexps them, so the
number is the total direction probability over the whole vocabulary rather than over a
top-20 window. It costs one gather and one reduction per (chunk, layer) — nothing next to
the unembed that produced the logits.

The trade is when the vocabulary is fixed. A top-k score can be recomputed against a
different direction vocabulary from the CSV alone; a mass table is baked at gather time,
which is why each one is written with a sidecar naming the vocabulary that produced it.

Two aggregations, not one
-------------------------
A score has to combine twice, and the two are separate on purpose:

`per_layer`
    the matched logprobs of one (token, layer) row -> that cell's score.

`across_layers`
    a token's per-layer scores -> the single number `rank_tokens` orders tokens by.

For `count` both are sums and nothing is interesting. For a logprob score they are not:
`logprob_mass` combines with **logsumexp** at both levels, because adding probabilities is
what "how much direction mass is here" means, and doing it in log space is the only way to
keep it monotone — more (or better-ranked) direction words can then only raise the score.

One caveat about the second level. `per_layer` is a real log-probability and so is always
≤ 0. `across_layers` adds probabilities belonging to *different* distributions — one per
layer — so its result is not a probability and can exceed 0 on a token that is
direction-loaded at several layers at once. That is fine and intended: it is only ever used
to order tokens against each other, and the ordering is what has to be monotone. Do not read
a token's total as `log P(anything)`.

`logprob_sum` is the literal reading — add the log probabilities — and it is available, but
know what it does before using it: adding logs multiplies probabilities, so a token whose
lens emits *five* direction words at logprob -3 (sum -15) scores below one that emits a
single direction word at -2. It rewards confident-and-lonely over direction-saturated,
which is the opposite of what the count did. Prefer `logprob_mass` unless you specifically
want the product.

Every score is "higher is better", including the negative ones, so every ranker downstream
(`rank_tokens`, `rank_layers_by_direction`, `split_next_action_manifest.py`) sorts by
`-score` in all modes without knowing which is in use.

Stdlib only, like the rest of this package.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

# What a (token, layer) cell scores when *no* top-k prediction is a direction word. For a
# count that is genuinely 0; for a logprob it is log(0), and -inf would poison every sum,
# every mean and `json.dump` alike. -40 is ~4e-18, orders of magnitude below any logprob a
# top-20 entry of a 200k vocabulary can carry, so it floors without ever competing.
NO_MATCH_LOGPROB = -40.0


def _logsumexp(values: list[float]) -> float:
    """log(sum(exp(v))) without overflowing, in stdlib.

    >>> round(_logsumexp([-1.0, -1.0]), 6) == round(-1.0 + math.log(2), 6)
    True
    """
    peak = max(values)
    return peak + math.log(math.fsum(math.exp(v - peak) for v in values))


@dataclass(frozen=True)
class DirectionScore:
    """One named way of turning direction evidence into a comparable number.

    `empty` is the value of "nothing matched" and is also what a layer a token has no row
    for contributes, so a token can be scored over a candidate layer set it does not fully
    cover.

    `source` names which artifact the per-layer numbers come from — `"topk"` computes them
    from an analysis CSV's top-k columns, `"mass"` reads them ready-made out of a
    direction-mass table. `combine_matches` is unused for a `"mass"` score: there is nothing
    to combine, the cell is already one number.
    """

    name: str
    needs_logprobs: bool
    empty: float
    combine_matches: Callable[[list[float]], float]
    combine_layers: Callable[[list[float]], float]
    description: str = ""
    source: str = "topk"

    def per_layer(self, matched_logprobs: list[float]) -> float:
        """Score one (token, layer) row from the logprobs of its direction hits.

        >>> get_score("count").per_layer([-1.0, -2.0])
        2
        >>> round(get_score("logprob_sum").per_layer([-1.0, -2.0]), 6)
        -3.0
        >>> get_score("logprob_mass").per_layer([]) == NO_MATCH_LOGPROB
        True
        """
        return self.combine_matches(matched_logprobs) if matched_logprobs else self.empty

    def across_layers(self, per_layer_scores: list[float]) -> float:
        """Combine a token's per-layer scores into the number it is ranked by.

        >>> get_score("count").across_layers([3, 1])
        4
        >>> get_score("logprob_mass").across_layers([]) == NO_MATCH_LOGPROB
        True
        """
        return self.combine_layers(per_layer_scores) if per_layer_scores else self.empty


SOURCES = ("topk", "mass")

SCORES: dict[str, DirectionScore] = {
    "count": DirectionScore(
        "count",
        needs_logprobs=False,
        empty=0,
        combine_matches=len,
        combine_layers=sum,
        description="How many of the top-k lens predictions are direction words. The original score.",
    ),
    "logprob_mass": DirectionScore(
        "logprob_mass",
        needs_logprobs=True,
        empty=NO_MATCH_LOGPROB,
        combine_matches=_logsumexp,
        combine_layers=_logsumexp,
        description="log of the total lens probability on direction words -- the count, weighted by belief.",
    ),
    "logprob_sum": DirectionScore(
        "logprob_sum",
        needs_logprobs=True,
        empty=NO_MATCH_LOGPROB,
        combine_matches=math.fsum,
        combine_layers=math.fsum,
        description="Sum of the matched logprobs, i.e. the product of their probabilities. See the caveat above.",
    ),
    "logprob_mass_full": DirectionScore(
        "logprob_mass_full",
        needs_logprobs=False,  # reads the mass table, not the top_i_logprob columns
        empty=NO_MATCH_LOGPROB,
        combine_matches=_logsumexp,  # unused: a mass table cell is already one number
        combine_layers=_logsumexp,
        source="mass",
        description="logprob_mass over the WHOLE direction vocabulary, read from the mass table.",
    ),
}

DEFAULT_SCORE = "count"


def get_score(name: str) -> DirectionScore:
    """Look a score mode up by name, with a message that lists the alternatives.

    >>> get_score("logprob_mass").needs_logprobs
    True
    >>> get_score("count").needs_logprobs
    False
    """
    try:
        return SCORES[name]
    except KeyError:
        raise ValueError(f"Unknown direction score {name!r}; available: {sorted(SCORES)}") from None


def score_names() -> list[str]:
    """Every registered score mode, in registry order.

    >>> score_names()
    ['count', 'logprob_mass', 'logprob_sum', 'logprob_mass_full']
    """
    return list(SCORES)


def mass_scores() -> list[str]:
    """The modes that read a direction-mass table rather than an analysis CSV.

    >>> mass_scores()
    ['logprob_mass_full']
    """
    return [name for name, score in SCORES.items() if score.source == "mass"]


__all__ = [
    "DEFAULT_SCORE",
    "NO_MATCH_LOGPROB",
    "SCORES",
    "SOURCES",
    "DirectionScore",
    "get_score",
    "mass_scores",
    "score_names",
]
