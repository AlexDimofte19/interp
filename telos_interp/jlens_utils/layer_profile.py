"""Which single layer carries the most direction information, over a whole dataset.

Per-token layer selection — each token at its own top-`M` layers — pools rows that are not
in a shared basis: layer 7 and layer 22 are different spaces, and one probe weight vector
cannot read both the same way. The alternative is to fix **one** layer for the whole
dataset, and the natural choice is the layer whose direction score is highest on average
across every reasoning token of every trajectory.

That average has to be taken over the *full* (token, layer) table, which is the CSV — not
over a prepared manifest. A manifest only holds the layers that were selected, and a layer
appears there only when it scored into some token's top-`M`, so its mean over the entries
present is conditioned on having won. Layer 15 is worse still: it is force-kept for every
token (see `top_filter.DEFAULT_ALWAYS_LAYERS`), so it alone is present unconditionally.
Averaging a manifest would compare a conditional mean against an unconditional one.

`LayerProfile` therefore accumulates straight from `read_direction_scores` output, where
every token has a row at every layer the lens covers, and a token with no direction hits at
a layer contributes that score's `empty` value rather than being skipped.

Stdlib only, like the rest of this package.
"""

from dataclasses import dataclass, field

from .jlens_csv import TokenScore
from .scoring import DEFAULT_SCORE, get_score


@dataclass
class LayerProfile:
    """Running mean of the per-layer direction score over an arbitrary set of tokens.

    >>> profile = LayerProfile(score_mode="count")
    >>> profile.add({(0, 5): TokenScore("a", {7: 1, 15: 4}), (0, 6): TokenScore("b", {7: 3, 15: 2})})
    >>> profile.means()
    {7: 2.0, 15: 3.0}
    >>> profile.best_layer()
    15
    >>> profile.tokens
    2
    """

    score_mode: str = DEFAULT_SCORE
    tokens: int = 0
    totals: dict[int, float] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)

    def add(self, scores: dict[tuple[int, int], TokenScore], layers: list[int] | None = None) -> None:
        """Fold one trajectory's `read_direction_scores` result into the running totals.

        `layers` pins the layer set so every token is counted at every one of them, missing
        rows included at the score's `empty`. Left out, each token contributes only the
        layers it has rows for — fine when the CSV covers all of them uniformly, which it
        does within a single lens.
        """
        empty = get_score(self.score_mode).empty
        for token_score in scores.values():
            self.tokens += 1
            wanted = layers if layers is not None else sorted(token_score.per_layer)
            for layer in wanted:
                value = token_score.per_layer.get(layer, empty)
                self.totals[layer] = self.totals.get(layer, 0.0) + value
                self.counts[layer] = self.counts.get(layer, 0) + 1

    def means(self) -> dict[int, float]:
        """{layer: mean direction score}, in ascending layer order."""
        return {layer: self.totals[layer] / self.counts[layer] for layer in sorted(self.counts)}

    def best_layer(self) -> int | None:
        """The layer with the highest mean score, or None if nothing was accumulated.

        Ties break on the **lower** layer index, matching `rank_layers_by_direction`.
        """
        means = self.means()
        if not means:
            return None
        return min(means, key=lambda layer: (-means[layer], layer))

    def to_dict(self) -> dict:
        """JSON-serialisable summary — what a profile run writes out and a script reads back."""
        means = self.means()
        return {
            "score_mode": self.score_mode,
            "tokens": self.tokens,
            "best_layer": self.best_layer(),
            "layers": [{"layer": layer, "mean": means[layer], "rows": self.counts[layer]} for layer in sorted(means)],
        }

    def merge(self, other: "LayerProfile") -> None:
        """Fold another profile in — used to combine per-worker or per-size accumulations."""
        if other.score_mode != self.score_mode:
            raise ValueError(f"cannot merge score_mode={other.score_mode!r} into {self.score_mode!r}")
        self.tokens += other.tokens
        for layer, total in other.totals.items():
            self.totals[layer] = self.totals.get(layer, 0.0) + total
            self.counts[layer] = self.counts.get(layer, 0) + other.counts[layer]


def format_profile(profile: LayerProfile) -> str:
    """A fixed-width table of the per-layer means, best layer first-marked.

    >>> print(format_profile(LayerProfile("count", 2, {7: 4.0, 15: 6.0}, {7: 2, 15: 2})))
    layer     mean  rows
        7   2.0000     2
       15   3.0000     2  <- best
    """
    means = profile.means()
    best = profile.best_layer()
    lines = [f"{'layer':>5} {'mean':>8} {'rows':>5}"]
    for layer in sorted(means):
        mark = "  <- best" if layer == best else ""
        lines.append(f"{layer:>5} {means[layer]:>8.4f} {profile.counts[layer]:>5}{mark}")
    return "\n".join(lines)


__all__ = ["LayerProfile", "format_profile"]
