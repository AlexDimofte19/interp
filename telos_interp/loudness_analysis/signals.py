"""What counts as the SIGNAL a lens is loud about -- a fourth registry, in the shape of the
other three.

`jlens_utils.methods.METHODS` says *which lens*, `jlens_utils.scoring.SCORES` says *how a
token scores*, `rollouts.truncation_strategies.STRATEGIES` says *where to cut*. This one says
*what we are measuring the probability mass of*.

A signal is a vocabulary: a JSON mapping class name to a list of decoded gpt-oss-20b token
strings, exactly as `data/jlens/direction_tokens_full.json` and `grid_tokens_full.json` are
shaped. The registry is seeded with those two because they are committed and deployed, but
**nothing here is limited to them** -- `resolve` accepts any conforming JSON under any name,
so a new signal is a file plus `--signal-name`, not a code change.

The vocabulary is deliberately NOT loaded at import: the registry entry names a file, and the
deployed copy lives at a different path (`/workspace/jlens/`) from the committed one
(`data/jlens/`). The caller supplies the directory; the entry supplies the filename.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from telos_interp.jlens_utils.jlens_csv import load_direction_tokens

__all__ = [
    "DEFAULT_SIGNAL",
    "SIGNALS",
    "Signal",
    "get_signal",
    "resolve",
    "signal_names",
]


@dataclass(frozen=True)
class Signal:
    """One signal vocabulary.

    `name` is what appears in every column and label this module writes, so it is also the
    thing that must stay stable across runs -- a table saying `jlens_direction_logmass_L15`
    is only readable because "direction" means one vocabulary forever.
    """

    name: str
    filename: str
    description: str = ""

    @property
    def membership_column(self) -> str:
        """The column flagging "the model emitted a word from this vocabulary here".

        Lens-independent -- membership is a property of the token, not of how it was read.

        >>> get_signal("direction").membership_column
        'is_direction_token'
        """
        return f"is_{self.name}_token"

    def path(self, root: str | Path) -> Path:
        """This vocabulary's JSON under `root`.

        >>> get_signal("grid").path("/workspace/jlens").as_posix()
        '/workspace/jlens/grid_tokens_full.json'
        """
        return Path(root) / self.filename

    def load(self, path: str | Path, classes: str = "all") -> set[str]:
        """The vocabulary as a flat set of decoded token strings.

        `classes` is "all" or a comma-separated subset ("UP,DOWN"); unknown class names
        raise rather than silently narrowing the vocabulary to nothing.
        """
        return load_direction_tokens(path, classes)

    def classes(self, path: str | Path) -> tuple[str, ...]:
        """The class names this vocabulary defines, in file order."""
        with open(path, encoding="utf-8") as fh:
            return tuple(json.load(fh).keys())

    def fingerprint(self, path: str | Path) -> str:
        """Short content hash of the vocabulary file, for `run_config.json`.

        Two tables are only comparable if they were baked against the same vocabulary, and
        this repo points several vocabularies at the same trees -- so provenance records the
        content, not just the filename.
        """
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


SIGNALS: dict[str, Signal] = {
    "direction": Signal(
        "direction",
        "direction_tokens_full.json",
        description="Movement words (UP/DOWN/LEFT/RIGHT) -- the signal every published result uses.",
    ),
    "grid": Signal(
        "grid",
        "grid_tokens_full.json",
        description="Grid-world nouns (WALL/OPEN/GOAL/AGENT/AXIS/STATUS) for the cognitive-map line.",
    ),
}

DEFAULT_SIGNAL = "direction"


def signal_names() -> list[str]:
    """Registered signal names, in registry order.

    >>> signal_names()
    ['direction', 'grid']
    """
    return list(SIGNALS)


def get_signal(name: str) -> Signal:
    """Look up a registered signal.

    >>> get_signal("direction").name
    'direction'
    >>> get_signal("nope")
    Traceback (most recent call last):
        ...
    ValueError: Unknown signal 'nope'; available: ['direction', 'grid']
    """
    try:
        return SIGNALS[name]
    except KeyError:
        raise ValueError(f"Unknown signal {name!r}; available: {sorted(SIGNALS)}") from None


def resolve(name: str | None = None, json_path: str | Path | None = None) -> Signal:
    """The signal a CLI's `--signal-name` / `--signal-json` pair means.

    Registered names resolve through `SIGNALS`. An unregistered name is accepted as an
    ad-hoc signal, which is the whole point -- a new vocabulary is a file, not a patch.
    With no name at all the filename decides, so `foo_tokens_full.json` is signal "foo".

    >>> resolve("direction").name
    'direction'
    >>> resolve(None, "/workspace/jlens/grid_tokens_full.json").name
    'grid'
    >>> resolve("shape", "/tmp/whatever.json").name
    'shape'
    >>> resolve(None, "/tmp/whatever.json").name
    'whatever'
    """
    if name:
        if name in SIGNALS:
            return SIGNALS[name]
        return Signal(name, Path(json_path).name if json_path else f"{name}_tokens_full.json")
    if json_path is None:
        return SIGNALS[DEFAULT_SIGNAL]
    stem = Path(json_path).name
    for signal in SIGNALS.values():
        if signal.filename == stem:
            return signal
    inferred = stem.split(".")[0]
    for suffix in ("_tokens_full", "_tokens"):
        if inferred.endswith(suffix):
            inferred = inferred[: -len(suffix)]
            break
    return Signal(inferred, stem)
