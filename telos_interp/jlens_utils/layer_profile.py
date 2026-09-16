"""Which single layer carries the most direction information, over a whole dataset.

Per-token layer selection — each token at its own top-`M` layers — pools rows that are not
in a shared basis: layer 7 and layer 22 are different spaces, and one probe weight vector
cannot read both the same way. The alternative is to fix **one** layer for the whole
dataset, and the natural choice is the layer whose direction score is highest on average
across every reasoning token of every trajectory.

That average has to be taken over the *full* (token, layer) table, which is the CSV — not
over a prepared manifest. A manifest only holds the layers that were *selected*, and a
layer appears there only when it scored into some token's top-`M`, so its mean over the
entries present is conditioned on having won. Layer 15 is worse still: it is force-kept for
every token (see `top_filter.DEFAULT_ALWAYS_LAYERS`), so it alone is present
unconditionally. Averaging a manifest would compare a conditional mean against an
unconditional one.

`LayerProfile` therefore accumulates straight from `read_direction_scores` output, where
every token has a row at every layer the lens covers, and a token with no direction hits at
a layer contributes that score's `empty` value rather than being skipped.

THE ARGMAX IS A DECISION, SO IT NEEDS AN ERROR BAR
--------------------------------------------------
The mean alone says which layer won; it does not say whether winning meant anything. Two
adjacent layers of the same model are strongly correlated, and on a *sampled* gather
(`build_loudness_tables.py --data_sample_p`) the means carry sampling noise on top of that,
so an argmax that sits half a standard error above its neighbour is a coin flip dressed up
as a result — and the layer it names is then pinned into every downstream arm.

**The denominator is trajectories, not tokens.** The ~18k reasoning tokens of one chain are
nowhere near independent: they share a prompt, a grid and a train of thought, so a standard
error over tokens is too small by whatever the intra-chain correlation is, often by a large
factor. Each `add()` call is therefore treated as one **cluster** — `profile_tree` calls it
once per trajectory — and `standard_errors()` is the spread of the per-cluster means over
the clusters. `separation()` goes further and pairs the two layers *within* each cluster
before taking the spread, which is the comparison that actually matters: layers move
together from trajectory to trajectory, and the paired difference cancels that shared
movement instead of counting it as noise in both.

The pooled token mean stays what `means()` reports, unchanged, because it is the number
already written into every `layer_profile.json` on disk. The cluster machinery only adds
the uncertainty around it.

Stdlib only, like the rest of this package.
"""

import math
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

    One `add()` is one cluster, so a profile built one trajectory at a time can say how
    much of the gap between two layers is trajectory-to-trajectory noise:

    >>> profile.clusters
    1
    """

    score_mode: str = DEFAULT_SCORE
    tokens: int = 0
    totals: dict[int, float] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)
    # One entry per add(): {layer: that call's own mean}. 40 layers x a few thousand
    # trajectories is a few hundred KB, and keeping the vectors rather than running sums
    # is what lets separation() pair two layers *within* a cluster after the fact.
    cluster_means: list[dict[int, float]] = field(default_factory=list)

    @property
    def clusters(self) -> int:
        """How many `add()` calls were folded in — trajectories, under `profile_tree`."""
        return len(self.cluster_means)

    def add(self, scores: dict[tuple[int, int], TokenScore], layers: list[int] | None = None) -> None:
        """Fold one trajectory's `read_direction_scores` result into the running totals.

        `layers` pins the layer set so every token is counted at every one of them, missing
        rows included at the score's `empty`. Left out, each token contributes only the
        layers it has rows for — fine when the CSV covers all of them uniformly, which it
        does within a single lens.

        An empty `scores` adds no cluster: a trajectory that contributed no tokens is not a
        draw from the population, it is a trajectory that was not read.
        """
        empty = get_score(self.score_mode).empty
        local_totals: dict[int, float] = {}
        local_counts: dict[int, int] = {}
        for token_score in scores.values():
            self.tokens += 1
            wanted = layers if layers is not None else sorted(token_score.per_layer)
            for layer in wanted:
                value = token_score.per_layer.get(layer, empty)
                self.totals[layer] = self.totals.get(layer, 0.0) + value
                self.counts[layer] = self.counts.get(layer, 0) + 1
                local_totals[layer] = local_totals.get(layer, 0.0) + value
                local_counts[layer] = local_counts.get(layer, 0) + 1
        if local_counts:
            self.cluster_means.append({layer: local_totals[layer] / local_counts[layer] for layer in local_counts})

    def means(self) -> dict[int, float]:
        """{layer: mean direction score}, in ascending layer order.

        Pooled over tokens, so a long chain weighs more than a short one. This is the
        number every `layer_profile.json` on disk already holds.
        """
        return {layer: self.totals[layer] / self.counts[layer] for layer in sorted(self.counts)}

    def standard_errors(self) -> dict[int, float]:
        """{layer: standard error of that layer's mean}, over clusters.

        The spread of the per-cluster means divided by sqrt(k), which treats one trajectory
        as one observation rather than one token. Empty until there are two clusters to
        take a spread of, and a layer missing from some cluster is only averaged over the
        clusters that have it.

        >>> p = LayerProfile("count")
        >>> p.add({(0, 5): TokenScore("a", {7: 1.0})})
        >>> p.add({(0, 6): TokenScore("b", {7: 3.0})})
        >>> p.standard_errors()
        {7: 1.0}
        """
        if len(self.cluster_means) < 2:
            return {}
        errors = {}
        for layer in sorted(self.counts):
            values = [cluster[layer] for cluster in self.cluster_means if layer in cluster]
            if len(values) < 2:
                continue
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            errors[layer] = math.sqrt(variance / len(values))
        return errors

    def separation(self) -> dict | None:
        """How far the best layer beats the runner-up, in standard errors of the difference.

        Paired **within** each cluster before the spread is taken: two layers of one model
        rise and fall together across trajectories, so the unpaired difference of two noisy
        means would charge that shared movement to both and understate the separation.

        Returns `{best, runner_up, gap, se, z, clusters}`, or `None` when there are fewer
        than two layers or two clusters to compare. `z` is the gap in standard errors —
        under 2, the argmax is not distinguishable from its neighbour and the layer should
        be chosen on some other ground (cost, an existing arm, the force-kept layer).

        >>> p = LayerProfile("count")
        >>> p.add({(0, 5): TokenScore("a", {7: 1.0, 15: 4.0})})
        >>> p.add({(0, 6): TokenScore("b", {7: 3.0, 15: 6.0})})
        >>> sep = p.separation()
        >>> sep["best"], sep["runner_up"], sep["gap"]
        (15, 7, 3.0)
        >>> sep["se"], sep["z"]  # the two layers move together, so the pairing is exact
        (0.0, inf)
        """
        means = self.means()
        best = self.best_layer()
        if best is None or len(means) < 2 or len(self.cluster_means) < 2:
            return None
        runner_up = min((layer for layer in means if layer != best), key=lambda layer: (-means[layer], layer))
        diffs = [c[best] - c[runner_up] for c in self.cluster_means if best in c and runner_up in c]
        if len(diffs) < 2:
            return None
        mean_diff = sum(diffs) / len(diffs)
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
        se = math.sqrt(variance / len(diffs))
        return {
            "best": best,
            "runner_up": runner_up,
            "gap": means[best] - means[runner_up],
            "se": se,
            "z": (means[best] - means[runner_up]) / se if se else math.inf,
            "clusters": len(diffs),
        }

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
        errors = self.standard_errors()
        return {
            "score_mode": self.score_mode,
            "tokens": self.tokens,
            "clusters": self.clusters,
            "best_layer": self.best_layer(),
            "separation": self.separation(),
            "layers": [
                {"layer": layer, "mean": means[layer], "rows": self.counts[layer], "se": errors.get(layer)}
                for layer in sorted(means)
            ],
        }

    def merge(self, other: "LayerProfile") -> None:
        """Fold another profile in — used to combine per-worker or per-size accumulations.

        The cluster vectors concatenate, so splitting a tree across workers and merging
        gives byte-identical error bars to one sequential pass.
        """
        if other.score_mode != self.score_mode:
            raise ValueError(f"cannot merge score_mode={other.score_mode!r} into {self.score_mode!r}")
        self.tokens += other.tokens
        for layer, total in other.totals.items():
            self.totals[layer] = self.totals.get(layer, 0.0) + total
            self.counts[layer] = self.counts.get(layer, 0) + other.counts[layer]
        self.cluster_means.extend(other.cluster_means)


def format_profile(profile: LayerProfile) -> str:
    """A fixed-width table of the per-layer means, best layer first-marked.

    The `se` column is over trajectories, not tokens, and is blank until there are two
    clusters to take a spread of.

    >>> print(format_profile(LayerProfile("count", 2, {7: 4.0, 15: 6.0}, {7: 2, 15: 2})))
    layer     mean       se  rows
        7   2.0000        -     2
       15   3.0000        -     2  <- best
    """
    means = profile.means()
    errors = profile.standard_errors()
    best = profile.best_layer()
    lines = [f"{'layer':>5} {'mean':>8} {'se':>8} {'rows':>5}"]
    for layer in sorted(means):
        mark = "  <- best" if layer == best else ""
        se = f"{errors[layer]:.4f}" if layer in errors else "-"
        lines.append(f"{layer:>5} {means[layer]:>8.4f} {se:>8} {profile.counts[layer]:>5}{mark}")
    return "\n".join(lines)


def format_separation(profile: LayerProfile) -> str:
    """One line on whether the argmax is distinguishable from the layer below it.

    >>> p = LayerProfile("count")
    >>> p.add({(0, 5): TokenScore("a", {7: 1.0, 15: 4.0})})
    >>> p.add({(0, 6): TokenScore("b", {7: 3.0, 15: 5.0})})
    >>> print(format_separation(p))
    layer 15 beats layer 7 by 2.5000 +/- 0.5000 (z=5.0, paired over 2 trajectories)
    """
    sep = profile.separation()
    if sep is None:
        return "separation: not enough layers or trajectories to say"
    z = "inf" if sep["z"] == math.inf else f"{sep['z']:.1f}"
    return (
        f"layer {sep['best']} beats layer {sep['runner_up']} by {sep['gap']:.4f} "
        f"+/- {sep['se']:.4f} (z={z}, paired over {sep['clusters']} trajectories)"
    )


__all__ = ["LayerProfile", "format_profile", "format_separation"]
