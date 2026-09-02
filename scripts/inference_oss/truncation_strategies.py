"""Where to cut a reasoning chain before asking the model for its action.

``run_inference.py`` re-runs gpt-oss at a series of *cutoffs* inside a step's own
reasoning: keep ``output_tokens[:pos + 1]``, append the fixed final-channel prefix
(``<|end|>...{\\n  "action": "``), and read the action the model then emits. Which
positions ``pos`` runs over is the experiment, and this module is the registry of
those choices. Every strategy returns a list of :class:`Cutoff` in ascending
position order; nothing else about the rollout changes between them.

Four strategies, selected by name:

``eos``
    The original: every reasoning sentence end (``get_indices_for_eos_tokens``
    restricted to the analysis region), plus a no-reasoning cutoff at the analysis
    header and the end of reasoning. This is the grid every rollout on disk was
    measured on.

``jlens_argmax_per_sentence``
    One cutoff per sentence, as ``eos``, but placed at the sentence's *loudest*
    token instead of its final one. Motivated by log entry 42: loudness falls ~2.2x
    through a sentence and peaks in its second decile, so a sentence end is
    systematically the quietest place in it, and entry 41(c) put the commitment at
    the first tokens of a sentence -- a median ~7 tokens before ``eos`` can see it.

``jlens_top_k_global``
    The K loudest tokens of the whole step's reasoning, wherever they fall, so the
    grid follows the lens rather than the punctuation.

``every_token``
    No selection at all: cut at EVERY reasoning token (or every ``stride``-th).
    The other three each sample the chain somewhere, which means a downstream
    analysis only ever sees the model's belief where that strategy chose to look;
    this one measures it everywhere, so a per-token label exists for any token an
    analysis wants to ask about. It is the dense grid the loudness-vs-probe work
    needs on a held-out set, and it is the only strategy whose cutoffs do not
    depend on the lens -- loudness is still attached per cutoff when a mass table
    is there, but it never decides where to cut.

LOUDNESS is the layer-15 full-vocabulary direction mass of log entry 42:
``sum(exp(logprob(t)))`` over the 446-token ``direction_tokens_full.json``. The
direction-mass table beside each analysis CSV already holds its log, so this module
ranks on ``L15`` directly (exp is monotone) and only exponentiates for the record.
Per CLAUDE.md a mass table is never read without its ``.meta.json`` sidecar: the
vocabulary it was baked against is verified and reported, not assumed.
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import exp
from pathlib import Path

from telos_interp.commands.gather_activations.gather_activations_utils import get_indices_for_eos_tokens

DEFAULT_LENS_ROOT = Path("/workspace/activations/jlens_mass_l15")
DEFAULT_LAYER = 15
DEFAULT_TOP_K = 20

# Cutoff kinds, recorded per eval so a downstream join can tell them apart.
KIND_NO_REASONING = "no_reasoning"
KIND_SENTENCE_END = "sentence_end"
KIND_END_OF_REASONING = "end_of_reasoning"
KIND_LOUDEST_IN_SENTENCE = "loudest_in_sentence"
KIND_LOUD_TOP_K = "loud_top_k"
KIND_EVERY_TOKEN = "every_token"


class LoudnessUnavailable(Exception):
    """No usable direction-mass table for this trajectory (or step)."""


@dataclass(frozen=True)
class Cutoff:
    """One truncation point, as an index into ``step["output_tokens"]``.

    ``sentence_idx``/``pos_in_sentence``/``sentence_len`` place the cutoff on the
    ``eos`` grid even when the strategy did not use it, so any two strategies can be
    compared in the same sentence coordinates. ``logmass`` is the layer-15 log
    direction mass at ``pos`` (``None`` when the strategy never scored it).
    """

    pos: int
    kind: str
    sentence_idx: int | None = None
    pos_in_sentence: int | None = None
    sentence_len: int | None = None
    logmass: float | None = None

    @property
    def prob_mass(self) -> float | None:
        return None if self.logmass is None else exp(self.logmass)


# --------------------------------------------------------------------------------------
# Token-region helpers (shared by every strategy; ``run_inference`` re-exports them).
# --------------------------------------------------------------------------------------


def find_action_cut(output_tokens: list[dict]) -> int | None:
    """Index of the first final-channel action token, or ``None`` if there is none.

    Everything before it is the reasoning plus the final-channel header
    (``...<|channel|>final<|message|>{\\n  "action": "``).
    """
    for i, token in enumerate(output_tokens):
        if {"final", "action"} <= set(token.get("token_groups", [])):
            return i
    return None


def analysis_positions(output_tokens: list[dict]) -> list[int]:
    """Indices of the analysis/reasoning tokens in ``output_tokens``."""
    return [i for i, t in enumerate(output_tokens) if "analysis" in t.get("token_groups", [])]


def get_final_prefix_ids(output_tokens: list[dict]) -> list[int] | None:
    """Token ids of the fixed ``<|end|>...{\\n  "action": "`` slice.

    Lifted verbatim from the data (``output_tokens[last_analysis + 1 : action_cut]``)
    rather than re-tokenized. ``None`` if reasoning or the action value is missing.
    """
    cut = find_action_cut(output_tokens)
    ana = analysis_positions(output_tokens)
    if cut is None or not ana:
        return None
    return [t["token_id"] for t in output_tokens[max(ana) + 1 : cut]]


def reasoning_eos_positions(output_tokens: list[dict]) -> list[int]:
    """Positions of reasoning sentence-ends, via gather_activations' EOS detector.

    Restricted to the analysis region. Always includes a no-reasoning cutoff (the
    analysis-header ``<|message|>``, just before the first reasoning token) as the first
    position, so index 0 is inference with zero reasoning. The end-of-reasoning position
    is always included as the final cutoff even if it does not end in sentence
    punctuation.
    """
    ana = analysis_positions(output_tokens)
    if not ana:
        return []
    last_analysis = max(ana)
    ana_set = set(ana)
    positions = [p for p in get_indices_for_eos_tokens(output_tokens) if p in ana_set]
    if last_analysis not in positions:
        positions.append(last_analysis)
    no_reasoning = min(ana) - 1  # analysis header <|message|>; keeps empty analysis channel
    if no_reasoning >= 0:
        positions.append(no_reasoning)
    return sorted(set(positions))


def sentence_spans(eos_positions: list[int]) -> list[tuple[int, int]]:
    """``[(start, end)]`` inclusive token ranges of sentences 1.., aligned with ``eos_positions[1:]``.

    ``eos_positions[0]`` is the no-reasoning cutoff (the analysis header) and owns no
    reasoning tokens, so sentence *i* spans ``eos[i - 1] + 1 .. eos[i]``. Mirrors
    ``scripts/build_sentence_loudness.py::sentence_of_token``. That first entry is always
    present here because harmony output always opens the channel with
    ``<|channel|>analysis<|message|>``, so reasoning never starts at index 0.
    """
    return [(prev + 1, end) for prev, end in zip(eos_positions, eos_positions[1:], strict=False)]


# --------------------------------------------------------------------------------------
# Loudness
# --------------------------------------------------------------------------------------


@dataclass
class MassTableLoudness:
    """Layer-15 direction mass per reasoning token, read from the gather's mass tables.

    Lookup is ``{lens_root}/size{N}/{name}/{name}_{lens}_direction_mass.csv``, whose
    ``reasoning_pos`` indexes the ANALYSIS-TAGGED tokens of ``step["output_tokens"]``;
    the caller adds ``min(analysis_positions(...))`` to land back in ``output_tokens``
    coordinates (verified token-for-token against the trajectory on load).

    The last trajectory read is cached, since ``run_inference`` walks one file at a time.
    """

    lens_root: Path = DEFAULT_LENS_ROOT
    lens: str = "jlens"
    layer: int = DEFAULT_LAYER
    _cache_name: str | None = field(default=None, init=False, repr=False)
    _cache: dict[int, dict[int, float]] = field(default_factory=dict, init=False, repr=False)
    _meta: dict = field(default_factory=dict, init=False, repr=False)

    def table_path(self, name: str) -> Path:
        """Mass-table path for a trajectory stem, e.g. ``..._size11_comp0.0_1``."""
        size = name.split("_size")[1].split("_")[0] if "_size" in name else "*"
        direct = self.lens_root / f"size{size}" / name / f"{name}_{self.lens}_direction_mass.csv"
        if direct.exists():
            return direct
        matches = sorted(self.lens_root.glob(f"*/{name}/{name}_{self.lens}_direction_mass.csv"))
        if not matches:
            raise LoudnessUnavailable(f"no {self.lens} direction-mass table for {name} under {self.lens_root}")
        return matches[0]

    def _read_meta(self, path: Path) -> dict:
        """Read the ``.meta.json`` sidecar; a mass table without one must not be read."""
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not meta_path.exists():
            raise LoudnessUnavailable(
                f"{path} has no .meta.json sidecar, so the vocabulary it was baked against is unknown"
            )
        return json.loads(meta_path.read_text())

    @property
    def meta(self) -> dict:
        """Sidecar of the most recently loaded table (vocabulary, lens, layers)."""
        return self._meta

    def load(self, name: str) -> dict[int, dict[int, float]]:
        """``{step_id: {reasoning_pos: log_direction_mass}}`` for one trajectory."""
        if name == self._cache_name:
            return self._cache
        path = self.table_path(name)
        meta = self._read_meta(path)
        col = f"L{self.layer}"
        # csv.DictReader, never pandas: decoded tokens include "NA", commas and newlines.
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or col not in reader.fieldnames:
                raise LoudnessUnavailable(f"{path} has no column {col} (layers: {meta.get('layers')})")
            scores: dict[int, dict[int, float]] = {}
            for row in reader:
                scores.setdefault(int(row["step"]), {})[int(row["reasoning_pos"])] = float(row[col])
        self._cache_name, self._cache, self._meta = name, scores, meta
        return scores

    def step_scores(self, name: str, step_id: int, offset: int) -> dict[int, float]:
        """``{output_token_idx: log_direction_mass}`` for one step of one trajectory."""
        by_step = self.load(name)
        if step_id not in by_step:
            raise LoudnessUnavailable(f"{name}: no mass-table rows for step {step_id}")
        return {pos + offset: v for pos, v in by_step[step_id].items()}

    def describe(self) -> dict:
        """Config worth recording in the results JSON."""
        return {
            "lens_root": str(self.lens_root),
            "lens": self.lens,
            "layer": self.layer,
            "signal_json": self._meta.get("signal_json"),
            "direction_classes": self._meta.get("direction_classes"),
            "num_direction_tokens": self._meta.get("num_direction_tokens"),
        }


# --------------------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------------------


class TruncationStrategy(ABC):
    """Chooses the cutoffs for one step. Subclasses implement :meth:`cutoffs`."""

    name: str = "base"
    # Whether ``build_strategy`` wires a MassTableLoudness into the constructor...
    uses_loudness: bool = False
    # ...and whether a trajectory without a usable table is fatal for that trajectory.
    # The two differ for ``every_token``, which records loudness but never ranks on it.
    needs_loudness: bool = False

    @abstractmethod
    def cutoffs(self, trajectory: dict, step: dict, traj_name: str) -> list[Cutoff]:
        """Ascending-position cutoffs for ``step``; ``[]`` when the step is unusable.

        Raise :class:`LoudnessUnavailable` if the strategy needs loudness this
        trajectory does not have; ``run_inference`` skips the file and says so.
        """

    def config(self) -> dict:
        """Strategy configuration, recorded in the results JSON."""
        return {"strategy": self.name}


class EosStrategy(TruncationStrategy):
    """Every reasoning sentence end, plus the no-reasoning and end-of-reasoning cutoffs."""

    name = "eos"

    def cutoffs(self, trajectory: dict, step: dict, traj_name: str) -> list[Cutoff]:
        eos = reasoning_eos_positions(step["output_tokens"])
        if not eos:
            return []
        spans = sentence_spans(eos)
        out = [Cutoff(pos=eos[0], kind=KIND_NO_REASONING)]
        for si, (start, end) in enumerate(spans, start=1):
            kind = KIND_END_OF_REASONING if si == len(spans) else KIND_SENTENCE_END
            out.append(
                Cutoff(
                    pos=end,
                    kind=kind,
                    sentence_idx=si,
                    pos_in_sentence=end - start,
                    sentence_len=end - start + 1,
                )
            )
        return out


class LoudnessStrategy(TruncationStrategy):
    """Shared plumbing for the strategies that rank tokens by layer-15 direction mass."""

    uses_loudness = True
    needs_loudness = True

    def __init__(self, loudness: MassTableLoudness, *, include_endpoints: bool = True) -> None:
        self.loudness = loudness
        # Keeping the no-reasoning and full-reasoning cutoffs makes every arm's first and
        # last eval the same prompt, so `final_sentence_accuracy` and `convinced_*`
        # compare across strategies rather than measuring different endpoints.
        self.include_endpoints = include_endpoints

    def config(self) -> dict:
        return {"strategy": self.name, "include_endpoints": self.include_endpoints, **self.loudness.describe()}

    def _scores(self, step: dict, traj_name: str) -> tuple[dict[int, float], list[int]]:
        """``({output_idx: logmass}, eos_positions)`` for a step, in output-token coordinates."""
        output_tokens = step["output_tokens"]
        ana = analysis_positions(output_tokens)
        if not ana:
            return {}, []
        scores = self.loudness.step_scores(traj_name, step["step_id"], offset=min(ana))
        return scores, reasoning_eos_positions(output_tokens)

    @staticmethod
    def _place(pos: int, spans: list[tuple[int, int]]) -> tuple[int | None, int | None, int | None]:
        """``(sentence_idx, pos_in_sentence, sentence_len)`` of ``pos`` on the eos grid."""
        for si, (start, end) in enumerate(spans, start=1):
            if start <= pos <= end:
                return si, pos - start, end - start + 1
        return None, None, None

    @staticmethod
    def _end_of_reasoning(spans: list[tuple[int, int]], scores: dict[int, float]) -> Cutoff:
        """The full-reasoning cutoff: the last reasoning token, i.e. the eos arm's final eval."""
        start, end = spans[-1]
        return Cutoff(
            pos=end,
            kind=KIND_END_OF_REASONING,
            sentence_idx=len(spans),
            pos_in_sentence=end - start,
            sentence_len=end - start + 1,
            logmass=scores.get(end),
        )


class JlensArgmaxPerSentenceStrategy(LoudnessStrategy):
    """One cutoff per sentence, at that sentence's loudest token.

    Same number of cutoffs as ``eos`` and the same sentence grid; only the position
    inside each sentence moves. A sentence whose tokens are all missing from the mass
    table falls back to its eos token (counted as ``sentence_end``), so a partial table
    degrades to the baseline rather than dropping the sentence.
    """

    name = "jlens_argmax_per_sentence"

    def cutoffs(self, trajectory: dict, step: dict, traj_name: str) -> list[Cutoff]:
        scores, eos = self._scores(step, traj_name)
        if not eos:
            return []
        spans = sentence_spans(eos)
        out: list[Cutoff] = []
        if self.include_endpoints:
            out.append(Cutoff(pos=eos[0], kind=KIND_NO_REASONING))
        for si, (start, end) in enumerate(spans, start=1):
            scored = [(p, scores[p]) for p in range(start, end + 1) if p in scores]
            if scored:
                # max() keeps the FIRST of equal scores, so ties resolve to the earlier token.
                pos, logmass = max(scored, key=lambda pv: pv[1])
                kind = KIND_LOUDEST_IN_SENTENCE
            else:
                pos, logmass, kind = end, None, KIND_SENTENCE_END
            out.append(
                Cutoff(
                    pos=pos,
                    kind=kind,
                    sentence_idx=si,
                    pos_in_sentence=pos - start,
                    sentence_len=end - start + 1,
                    logmass=logmass,
                )
            )
        if self.include_endpoints:
            out.append(self._end_of_reasoning(spans, scores))
        return _dedupe(out)


class JlensTopKGlobalStrategy(LoudnessStrategy):
    """The K loudest reasoning tokens of the step, wherever in the chain they fall.

    Ranked over the whole step's reasoning rather than within a sentence, so a quiet
    sentence contributes no cutoff and a loud one may contribute several. Steps with
    fewer than K reasoning tokens simply yield all of them.
    """

    name = "jlens_top_k_global"

    def __init__(self, loudness: MassTableLoudness, *, top_k: int = DEFAULT_TOP_K, include_endpoints: bool = True):
        super().__init__(loudness, include_endpoints=include_endpoints)
        self.top_k = top_k

    def config(self) -> dict:
        return {**super().config(), "top_k": self.top_k}

    def cutoffs(self, trajectory: dict, step: dict, traj_name: str) -> list[Cutoff]:
        scores, eos = self._scores(step, traj_name)
        if not eos:
            return []
        spans = sentence_spans(eos)
        # Sort by -score then position: deterministic, and ties break toward the earlier token.
        top = sorted(scores.items(), key=lambda pv: (-pv[1], pv[0]))[: self.top_k]
        out: list[Cutoff] = []
        if self.include_endpoints:
            out.append(Cutoff(pos=eos[0], kind=KIND_NO_REASONING))
        for pos, logmass in sorted(top):
            si, pis, slen = self._place(pos, spans)
            out.append(
                Cutoff(
                    pos=pos,
                    kind=KIND_LOUD_TOP_K,
                    sentence_idx=si,
                    pos_in_sentence=pis,
                    sentence_len=slen,
                    logmass=logmass,
                )
            )
        if self.include_endpoints:
            out.append(self._end_of_reasoning(spans, scores))
        return _dedupe(out)


class EveryTokenStrategy(LoudnessStrategy):
    """Cut at every reasoning token (or every ``stride``-th), selecting nothing.

    The other strategies each answer "what does the model believe *where I chose to
    look*". This one removes the choice: with ``stride=1`` there is a cutoff at every
    analysis-tagged token, so a downstream join has a measured belief for any token it
    wants to ask about and no selection sits between the loudness axis and the label.

    It subclasses :class:`LoudnessStrategy` for the sentence-placement plumbing and to
    record ``logmass`` per cutoff, but ``needs_loudness`` is False: the cutoffs are the
    tokens themselves, so a trajectory with no mass table still yields the full grid --
    with ``logmass=None`` -- rather than being skipped. Loudness here is a recorded
    covariate, never a selector.

    ``stride`` thins the grid uniformly (``stride=2`` is every other token) for a run that
    cannot afford the dense one; the endpoints are kept whatever the stride, so the first
    and last eval stay the same prompt as every other arm's.
    """

    name = "every_token"
    needs_loudness = False

    def __init__(
        self,
        loudness: MassTableLoudness | None = None,
        *,
        stride: int = 1,
        include_endpoints: bool = True,
    ) -> None:
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        super().__init__(loudness or MassTableLoudness(), include_endpoints=include_endpoints)
        self.stride = stride

    def config(self) -> dict:
        return {**super().config(), "stride": self.stride}

    def _scores(self, step: dict, traj_name: str) -> tuple[dict[int, float], list[int]]:
        """As the parent, but a missing or unreadable mass table costs the covariate, not the run."""
        try:
            return super()._scores(step, traj_name)
        except LoudnessUnavailable:
            return {}, reasoning_eos_positions(step["output_tokens"])

    def cutoffs(self, trajectory: dict, step: dict, traj_name: str) -> list[Cutoff]:
        scores, eos = self._scores(step, traj_name)
        ana = analysis_positions(step["output_tokens"])
        if not eos or not ana:
            return []
        spans = sentence_spans(eos)
        out: list[Cutoff] = []
        if self.include_endpoints:
            out.append(Cutoff(pos=eos[0], kind=KIND_NO_REASONING))
        # Strided from the FIRST reasoning token, so the grid a stride>1 run keeps is a
        # subset of the dense one and the two are directly comparable.
        for pos in ana[:: self.stride]:
            si, pis, slen = self._place(pos, spans)
            out.append(
                Cutoff(
                    pos=pos,
                    kind=KIND_EVERY_TOKEN,
                    sentence_idx=si,
                    pos_in_sentence=pis,
                    sentence_len=slen,
                    logmass=scores.get(pos),
                )
            )
        if self.include_endpoints:
            # _dedupe lets this relabel the last token when the stride happens to land on it.
            out.append(self._end_of_reasoning(spans, scores))
        return _dedupe(out)


def _dedupe(cutoffs: list[Cutoff]) -> list[Cutoff]:
    """Sort by position and keep one cutoff per position.

    Endpoint cutoffs are added around strategy-chosen ones and can coincide with them
    (a step whose loudest token IS its last). The later entry wins, so the endpoint
    label survives and the eval count stays honest -- the same prompt is never run twice.
    """
    by_pos: dict[int, Cutoff] = {}
    for c in cutoffs:
        by_pos[c.pos] = c
    return [by_pos[p] for p in sorted(by_pos)]


STRATEGIES: dict[str, type[TruncationStrategy]] = {
    EosStrategy.name: EosStrategy,
    JlensArgmaxPerSentenceStrategy.name: JlensArgmaxPerSentenceStrategy,
    JlensTopKGlobalStrategy.name: JlensTopKGlobalStrategy,
    EveryTokenStrategy.name: EveryTokenStrategy,
}


def build_strategy(
    name: str,
    *,
    lens_root: Path = DEFAULT_LENS_ROOT,
    lens: str = "jlens",
    layer: int = DEFAULT_LAYER,
    top_k: int = DEFAULT_TOP_K,
    stride: int = 1,
    include_endpoints: bool = True,
) -> TruncationStrategy:
    """Instantiate a strategy by name, wiring loudness only for the ones that use it.

    >>> build_strategy("eos").name
    'eos'
    >>> sorted(STRATEGIES)
    ['eos', 'every_token', 'jlens_argmax_per_sentence', 'jlens_top_k_global']
    """
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose one of {sorted(STRATEGIES)}") from None
    if not cls.uses_loudness:
        return cls()
    loudness = MassTableLoudness(lens_root=Path(lens_root), lens=lens, layer=layer)
    if cls is JlensTopKGlobalStrategy:
        return JlensTopKGlobalStrategy(loudness, top_k=top_k, include_endpoints=include_endpoints)
    if cls is EveryTokenStrategy:
        return EveryTokenStrategy(loudness, stride=stride, include_endpoints=include_endpoints)
    return cls(loudness, include_endpoints=include_endpoints)  # type: ignore[call-arg]
