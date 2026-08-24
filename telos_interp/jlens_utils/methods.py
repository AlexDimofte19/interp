"""The registry of token/layer selection methods.

A *method* is one named recipe for choosing which reasoning tokens — and which layers of
them — are worth keeping on disk and training a probe on. Every consumer takes the method
by name, so adding one is an entry in `METHODS` rather than a new branch in four files:

  * `scripts/jlens_reasoning_tokens.py` decides what to *write*,
  * `scripts/delete_non_jlens_selected.py` decides what to *keep*,
  * `prepare_activations_for_probing` decides what a probe *trains on*.

Two axes matter, and they are properties of the method rather than flags a caller can set
independently:

`csv_suffix`
    Which `{stem}{suffix}` analysis CSV the method scores, or `None` for a method that
    needs no CSV at all. `jlens` and `logitlens` differ *only* here: the two CSVs share a
    schema, so all the scoring and coordinate code is already lens-agnostic. What differs
    is upstream — the Jacobian lens transports a layer's residual stream into the layer-23
    space before the unembed, the logit lens unembeds it directly.

`scored`
    Whether tokens are ranked by direction count (and layers by
    `rank_layers_by_direction`) or drawn uniformly at random. A scored arm records its
    counts; an unscored one records **none**, deliberately —
    `scripts/split_next_action_manifest.py` decides whether to rank or sample a token's
    layers by whether a count is present, so a control that recorded counts would silently
    collapse onto its lowest-scoring layer.

Stdlib only, like the rest of this package.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SelectionMethod:
    """One named recipe for choosing (token, layers) out of a reasoning chain.

    `abbrev` is what `prepare_activations_for_probing` puts in an auto-generated dataset
    directory name. It lives here so adding a method forces a choice, instead of a
    hand-maintained lookup elsewhere going stale -- which is exactly how the last method
    added produced a `KeyError` at dirname time.
    """

    name: str
    csv_suffix: str | None
    scored: bool
    abbrev: str
    description: str = ""


METHODS: dict[str, SelectionMethod] = {
    "jlens": SelectionMethod(
        "jlens",
        "_jlens_analysis.csv",
        scored=True,
        abbrev="jl",
        description="Top tokens by direction count in the Jacobian lens' top-k predictions.",
    ),
    "logitlens": SelectionMethod(
        "logitlens",
        "_logitlens_analysis.csv",
        scored=True,
        abbrev="ll",
        description="The same ranking over the logit lens: unembed each layer directly, no transport.",
    ),
    "random": SelectionMethod(
        "random",
        None,
        scored=False,
        abbrev="rnd",
        description="Seeded uniform draw over the whole chain -- the matched control.",
    ),
}

# What a gather or prune run selects when nothing is asked for: the jlens arm plus its
# control. A lens arm without a matched control means nothing (see this package's README).
DEFAULT_METHODS = ("jlens", "random")


def get_method(name: str) -> SelectionMethod:
    """Look a method up by name, with a message that lists the alternatives.

    >>> get_method("logitlens").csv_suffix
    '_logitlens_analysis.csv'
    >>> get_method("jlens").scored
    True
    """
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(f"Unknown selection method {name!r}; available: {sorted(METHODS)}") from None


def method_names() -> list[str]:
    """Every registered method, in registry order."""
    return list(METHODS)


def scored_methods() -> list[str]:
    """The methods that rank by a signal count rather than sampling.

    >>> scored_methods()
    ['jlens', 'logitlens']
    """
    return [name for name, method in METHODS.items() if method.scored]


def parse_methods(spec: str) -> list[str]:
    """Comma-separated method names, order preserved, duplicates dropped.

    Validates as it goes, so a typo fails at argument-parsing time rather than after a
    multi-hour forward pass.

    >>> parse_methods("jlens,random")
    ['jlens', 'random']
    >>> parse_methods(" logitlens , logitlens ")
    ['logitlens']
    >>> parse_methods("")
    []
    """
    out: list[str] = []
    for part in spec.split(","):
        name = part.strip()
        if not name or name in out:
            continue
        get_method(name)
        out.append(name)
    return out


def analysis_csv_path(trajectory_folder: Path, method: str) -> Path | None:
    """`<folder>/<folder name>{suffix}` for a scored method; `None` for an unscored one.

    Mirrors how the gather script names its artefacts, so the filter, the pruner and the
    probe-data preparer all look in the same place.
    """
    suffix = get_method(method).csv_suffix
    return trajectory_folder / f"{trajectory_folder.name}{suffix}" if suffix else None


__all__ = [
    "DEFAULT_METHODS",
    "METHODS",
    "SelectionMethod",
    "analysis_csv_path",
    "get_method",
    "method_names",
    "parse_methods",
    "scored_methods",
]
