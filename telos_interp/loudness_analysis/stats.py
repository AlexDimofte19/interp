"""The statistics every loudness analysis shares, written once.

TWO BALANCED ACCURACIES, AND THEY ARE NOT THE SAME NUMBER. Both were in use and both are
kept, named apart, because collapsing them would silently change published figures:

  * `bal_acc` -- mean per-class recall over ROWS, where one row is one prediction. What the
    next-action analyses use, and what `train_next_action_probe::_evaluate` reports.
  * `bal_acc_from_counts` -- mean per-class recall rebuilt from per-class COUNT columns
    (`n_true_{c}`, `{probe}_correct_{c}`), where one row already summarises many predictions.
    What the grid analyses use, because a grid row carries a whole step's cells.

Averaging per-token accuracies instead of pooling per-class counts weights a token with two
cells the same as one with twenty-five. Reach for the counts form whenever a row is a bucket.

BOOTSTRAPS RESAMPLE TRAJECTORY NAMES, NEVER ROWS. Tokens inside a trajectory share a chain, a
sentence structure and a commitment boundary, so a row-level bootstrap reports every band
several times too narrow. This was already the rule in all four analysis scripts; it was just
implemented four times.

A class with no support in a bucket is DROPPED, not scored 0 -- it was never asked about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "bal_acc",
    "bal_acc_from_counts",
    "boot_bal_acc",
    "clustered_band",
    "plain_accuracy",
    "qbin",
    "spearman",
]


def bal_acc(truth: np.ndarray, pred: np.ndarray, classes) -> float:
    """Mean per-class recall over the classes present. nan on an empty bin.

    `classes` comes from the probe type (`probes.PROBE_TYPES[...].classes`), so this works
    for four actions or eight grid cells without knowing which it was handed.

    >>> t = np.array([0, 0, 1, 1])
    >>> bal_acc(t, np.array([0, 0, 1, 1]), [0, 1])
    1.0
    >>> bal_acc(t, np.array([0, 1, 0, 1]), [0, 1])
    0.5
    >>> bal_acc(t, np.array([0, 0, 1, 1]), [0, 1, 2])  # class 2 absent, dropped not zeroed
    1.0
    """
    recalls = []
    for a in classes:
        m = truth == a
        if m.sum():
            recalls.append(float((pred[m] == truth[m]).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def bal_acc_from_counts(df: pd.DataFrame, probe: str, classes) -> tuple[float, dict]:
    """Balanced accuracy over a bucket of rows carrying per-class counts, plus the recalls.

    Pools `n_true_{c}` and `{probe}_correct_{c}` across the bucket before dividing, so a row
    covering many predictions carries its proper weight.
    """
    recalls: dict[int, float] = {}
    for c in classes:
        tcol, ccol = f"n_true_{c}", f"{probe}_correct_{c}"
        if tcol not in df.columns or ccol not in df.columns:
            continue
        n_true = float(df[tcol].sum())
        if n_true > 0:
            recalls[c] = float(df[ccol].sum()) / n_true
    ba = float(np.mean(list(recalls.values()))) if recalls else float("nan")
    return ba, recalls


def plain_accuracy(df: pd.DataFrame, probe: str) -> float:
    """Unbalanced accuracy from the same count columns. Moves with the bucket's class mix,
    which is why nothing reports it alone -- it is carried beside `bal_acc_from_counts` so a
    gap between the two is visible."""
    n_correct = float(df[f"{probe}_n_correct"].sum())
    n_cells = float(df["n_cells"].sum())
    return n_correct / n_cells if n_cells else float("nan")


def boot_bal_acc(df: pd.DataFrame, truth_col: str, pred_col: str, n: int, rng, classes) -> tuple[float, float, float]:
    """Balanced accuracy with a 95% CI from resampling trajectory names.

    Returns `(nan, nan)` for the interval when there is only one trajectory or `n == 0` --
    a CI over a single cluster is not a CI.
    """
    point = bal_acc(df[truth_col].to_numpy(), df[pred_col].to_numpy(), classes)
    names = df["name"].to_numpy()
    uniq = np.unique(names)
    if len(uniq) < 2 or n == 0:
        return point, float("nan"), float("nan")
    idx_of = {u: np.flatnonzero(names == u) for u in uniq}
    t, p = df[truth_col].to_numpy(), df[pred_col].to_numpy()
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_of[uniq[k]] for k in pick])
        draws.append(bal_acc(t[rows], p[rows], classes))
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def qbin(s: pd.Series, q: int, labels=None) -> pd.Series:
    """Quantile bins that survive ties.

    Loudness is heavy-tailed and has a repeated floor -- every token with no vocabulary hit
    sits at NO_MATCH_LOGPROB -- so `qcut` can fail on duplicate edges. Falling back to
    equal-width bins keeps the bin count rather than dropping the analysis.
    """
    try:
        return pd.qcut(s, q, labels=labels, duplicates="drop")
    except ValueError:
        return pd.cut(s, q, labels=labels)


def spearman(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    """Rank correlation between two columns."""
    return float(df[x_col].corr(df[y_col], method="spearman"))


def clustered_band(df: pd.DataFrame, col: str, bin_id: np.ndarray, n_bins: int, seed: int = 0, n_boot: int = 300):
    """Mean of `col` per bin, plus a 95% band from resampling TRAJECTORIES, not rows.

    `bin_id` is a per-row bin index; negative means "drop this row". Returns
    `(point, lo, hi)`, each an array of length `n_bins`, with nan in bins nothing landed in.
    """
    vals = df[col].to_numpy(float)
    codes, _ = pd.factorize(df["name"])
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(codes.max() + 1))
    ends = np.append(starts[1:], len(order))
    n_traj = len(starts)

    def means(idx: np.ndarray) -> np.ndarray:
        b = bin_id[idx]
        keep = b >= 0
        b, v = b[keep], vals[idx][keep]
        cnt = np.bincount(b, minlength=n_bins)
        tot = np.bincount(b, weights=v, minlength=n_bins)
        with np.errstate(invalid="ignore"):
            return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    point = means(np.arange(len(vals)))
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, n_bins))
    for b in range(n_boot):
        pick = rng.integers(0, n_traj, n_traj)
        boots[b] = means(np.concatenate([order[starts[i] : ends[i]] for i in pick]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return point, lo, hi
