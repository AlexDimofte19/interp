"""Per-token probe verdicts on a PREPARED SLICE, in the count form every loudness table uses.

`score_probes_per_token.py` scores every reasoning token of a tree. A prepared slice -- a
selection arm's val split -- is a sparse set of tokens, and the question "how does accuracy on
the uniformly drawn eval tokens vary with loudness" needs its verdicts per token. The two
evaluators (`eval_grid_probe.py`, `rollouts/eval_local_belief.py`) already score a slice; with
`--per-token-out` they also write one row per entry through `write_counts`, and
`build_slice_per_token_table.py` merges those files, one per probe, into a table that
`join_signal_loudness.py` can widen.

ONE FORM FOR BOTH LABELS. A grid entry carries many cells, an action entry one label. Both are
written as `n_true_{c}` / `correct_{c}` counts per class, so an action entry is one "cell" whose
class is its label, and pooling those counts is exactly per-row macro recall. The downstream
statistic (`stats.bal_acc_from_counts`) then cannot differ between signals.

No torch here: the evaluators hand over plain arrays.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

__all__ = ["COUNT_KEY", "decode_bpe", "write_counts"]

# Every per-token file is keyed on these, in this order; abs_pos is added by the merge, which
# is the step that has the trajectory JSON.
COUNT_KEY = ("name", "step", "token_idx")


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's `bytes_to_unicode`, the byte-level BPE alphabet gpt-oss and Qwen share."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


_BYTE_DECODER = _byte_decoder()


def decode_bpe(token: str) -> str:
    """A raw byte-level BPE token (`Ġrow`) as the decoded text a signal vocabulary holds (` row`).

    A piece that is not valid UTF-8 on its own (half of a multi-byte character) is returned
    unchanged: it cannot be a vocabulary word either way.

    >>> decode_bpe("Ġrow"), decode_bpe("Ċ"), decode_bpe("ĉUP"), decode_bpe("We")
    (' row', '\\n', '\\tUP', 'We')
    """
    try:
        return bytes(_BYTE_DECODER[ch] for ch in token).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return token


def write_counts(path: Path, entries: Sequence[dict], classes: Sequence[int], n_true, correct) -> None:
    """One row per manifest entry: its key, then `n_true_{c}` and `correct_{c}` per class.

    `n_true` and `correct` are `(len(entries), len(classes))` integer arrays; `classes` are
    the ORIGINAL label ids (cell ids, action ids), which is what `n_true_{c}` means in every
    per-token table.
    """
    if len(n_true) != len(entries) or len(correct) != len(entries):
        raise ValueError(f"{len(entries)} entries but {len(n_true)} / {len(correct)} count rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([*COUNT_KEY, *(f"n_true_{c}" for c in classes), *(f"correct_{c}" for c in classes)])
        for e, t, k in zip(entries, n_true, correct, strict=True):
            w.writerow([e["name"], e["step"], e["token_id"], *(int(x) for x in t), *(int(x) for x in k)])
    print(f"  -> {path} ({len(entries)} rows)")
