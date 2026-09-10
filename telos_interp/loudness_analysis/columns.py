"""One name per quantity, and one label per axis.

THE PROBLEM THIS SOLVES. The same number -- log P(any signal word) at a layer -- was written
under three different column names by three different producers:

    dir_logmass_L15          build_sentence_loudness.py
    dir_logmass              build_probe_loudness.py, build_probe_loudness_heldout.py
    {lens}_logmass_L{layer}  build_token_loudness_x_infered_action_probability.py
    {lens}_mass_L{layer}     eval_probe_per_token.py

Readers hard-coded the variant they expected, which is why the four builders could not be one
builder. The canonical name is `{lens}_{signal}_logmass_L{layer}` -- it states the lens that
read it, the vocabulary it was read against, and the layer it was read at, none of which the
old names carried in full. `resolve` accepts every legacy spelling, so **every CSV already on
disk still loads**.

TWO OVERLOADED NAMES ARE FIXED HERE TOO, both of which have already cost a debugging round:

  * `sentence_frac` means position WITHIN a sentence in the loudness tables, but
    `sentence_idx / n_sentences` in `probe_vs_rollout/per_token.csv`, whose within-sentence
    column is `frac_in_sentence`. Reading the wrong one destroys any position control without
    raising. Canonically: `frac_in_sentence` and `frac_of_chain`.
  * `rowset` selects which TOKENS in `build_probe_loudness.py` but only which PROBES in
    `build_probe_loudness_heldout.py`, where all three rowsets hold identical rows.
    Canonically `probe_set`, since the token sense no longer exists.

Stdlib only -- the analysis layer imports this without pulling in torch.
"""

from __future__ import annotations

__all__ = [
    "AMBIGUOUS",
    "LENS_LABEL",
    "axis_label",
    "legacy_score_columns",
    "loudness_column",
    "membership_column",
    "prob_column",
    "resolve",
    "resolve_ambiguous",
    "score_columns",
]

# How a lens is spelled in a figure. The registry key is the code name; this is the prose one.
LENS_LABEL = {"jlens": "J-lens", "logitlens": "logitlens"}


def loudness_column(lens: str, signal: str, layer: int | str) -> str:
    """The canonical loudness column.

    `layer` may be an int, or "best" for a token's argmax layer.

    >>> loudness_column("jlens", "direction", 15)
    'jlens_direction_logmass_L15'
    >>> loudness_column("logitlens", "grid", 7)
    'logitlens_grid_logmass_L7'
    >>> loudness_column("jlens", "direction", "best")
    'jlens_direction_logmass_Lbest'
    """
    return f"{lens}_{signal}_logmass_L{layer}"


def prob_column(lens: str, signal: str, layer: int | str) -> str:
    """The probability twin of `loudness_column` -- exp() of it.

    >>> prob_column("jlens", "direction", 15)
    'jlens_direction_prob_L15'
    """
    return f"{lens}_{signal}_prob_L{layer}"


def membership_column(signal: str) -> str:
    """"The model emitted a word of this vocabulary here". Lens-independent.

    >>> membership_column("direction")
    'is_direction_token'
    >>> membership_column("grid")
    'is_grid_token'
    """
    return f"is_{signal}_token"


def candidates(lens: str, signal: str, layer: int | str) -> list[str]:
    """Every spelling of this loudness column, canonical first.

    The legacy names are tried in the order that makes a wrong hit least likely: the ones
    carrying both a lens and a layer come before the bare `dir_logmass`, which carries
    neither and is only correct because the direction vocabulary was the only one in use
    when it was written.

    >>> candidates("jlens", "direction", 15)[0]
    'jlens_direction_logmass_L15'
    >>> "dir_logmass" in candidates("jlens", "direction", 15)
    True
    >>> "dir_logmass" in candidates("jlens", "grid", 15)
    False
    """
    names = [
        loudness_column(lens, signal, layer),
        f"{lens}_logmass_L{layer}",
        f"{lens}_mass_L{layer}",
    ]
    if signal == "direction":
        # Written before any vocabulary but `direction` existed, so "dir" identifies the
        # signal and nothing else. Only valid for that signal.
        names += [f"dir_logmass_L{layer}", "dir_logmass"]
    return names


def resolve(fieldnames, lens: str, signal: str, layer: int | str) -> str:
    """The loudness column actually present in `fieldnames`.

    >>> resolve(["name", "jlens_direction_logmass_L15"], "jlens", "direction", 15)
    'jlens_direction_logmass_L15'
    >>> resolve(["name", "jlens_mass_L15"], "jlens", "direction", 15)
    'jlens_mass_L15'
    >>> resolve(["name", "dir_logmass"], "jlens", "direction", 15)
    'dir_logmass'
    >>> resolve(["name", "whatever"], "jlens", "direction", 15)
    Traceback (most recent call last):
        ...
    KeyError: "no jlens/direction loudness column at layer 15; looked for ['jlens_direction_logmass_L15', 'jlens_logmass_L15', 'jlens_mass_L15', 'dir_logmass_L15', 'dir_logmass'], table has ['name', 'whatever']"
    """
    present = set(fieldnames)
    wanted = candidates(lens, signal, layer)
    for name in wanted:
        if name in present:
            return name
    raise KeyError(
        f"no {lens}/{signal} loudness column at layer {layer}; "
        f"looked for {wanted}, table has {sorted(present)}"
    )


# Canonical name -> the spellings that have meant it, canonical first. Unlike the loudness
# columns these are not parameterised, so one flat map covers them.
AMBIGUOUS: dict[str, tuple[str, ...]] = {
    # Position WITHIN a sentence, 0 at its first token and 1 at its last.
    "frac_in_sentence": ("frac_in_sentence", "sentence_frac"),
    # Which sentence, as a fraction of the chain. NOT the above.
    "frac_of_chain": ("frac_of_chain", "sentence_frac_of_chain"),
    # Which probes a row carries predictions for.
    "probe_set": ("probe_set", "rowset"),
}


def resolve_ambiguous(fieldnames, canonical: str) -> str:
    """The column in `fieldnames` holding `canonical`'s quantity.

    Never guesses across the `sentence_frac` collision: `frac_in_sentence` accepts
    `sentence_frac` because that is what the loudness tables call it, and `frac_of_chain`
    does not, because in `probe_vs_rollout/per_token.csv` `sentence_frac` means the other
    thing entirely.

    >>> resolve_ambiguous(["sentence_frac"], "frac_in_sentence")
    'sentence_frac'
    >>> resolve_ambiguous(["rowset"], "probe_set")
    'rowset'
    >>> resolve_ambiguous(["sentence_frac"], "frac_of_chain")
    Traceback (most recent call last):
        ...
    KeyError: "no column for 'frac_of_chain'; looked for ('frac_of_chain', 'sentence_frac_of_chain'), table has ['sentence_frac']"
    """
    present = set(fieldnames)
    wanted = AMBIGUOUS[canonical]
    for name in wanted:
        if name in present:
            return name
    raise KeyError(f"no column for {canonical!r}; looked for {wanted}, table has {sorted(present)}")


def axis_label(lens: str, signal: str, layer: int | str | None = None) -> str:
    """The loudness axis label, identical in every table and every figure.

    Loudness is never unqualified: a figure that does not name its lens and its vocabulary
    cannot be compared with the one beside it, and at layer 15 the two lenses' top-20 sets
    overlap only about half.

    >>> axis_label("jlens", "direction")
    'J-lens direction logmass'
    >>> axis_label("logitlens", "grid", 7)
    'logitlens grid logmass (L7)'
    >>> axis_label("newlens", "shape")
    'newlens shape logmass'
    """
    base = f"{LENS_LABEL.get(lens, lens)} {signal} logmass"
    return f"{base} (L{layer})" if layer is not None else base


def score_columns(lens: str, signal: str, layer: int | str) -> list[str]:
    """The four columns an evaluator writes per lens, canonical spelling.

    `count` is the top-k score (how many of the lens' top-20 predictions were signal words);
    the three `logmass` columns come from the full-vocabulary mass table. `best` is the
    token's argmax layer, which is NOT the same question as the fixed-layer cell -- a token
    can be quiet at layer 15 and loud somewhere else.

    >>> score_columns("jlens", "direction", 15)
    ['jlens_direction_count', 'jlens_direction_logmass_L15', 'jlens_direction_logmass_best_layer', 'jlens_direction_logmass_best']
    """
    return [
        f"{lens}_{signal}_count",
        loudness_column(lens, signal, layer),
        f"{lens}_{signal}_logmass_best_layer",
        f"{lens}_{signal}_logmass_best",
    ]


def legacy_score_columns(lens: str, layer: int | str) -> list[str]:
    """The same four as `eval_probe_per_token.py` wrote them, for reading old CSVs.

    >>> legacy_score_columns("jlens", 15)
    ['jlens_count', 'jlens_mass_L15', 'jlens_mass_best_layer', 'jlens_mass_best']
    """
    return [
        f"{lens}_count",
        f"{lens}_mass_L{layer}",
        f"{lens}_mass_best_layer",
        f"{lens}_mass_best",
    ]
