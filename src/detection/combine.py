"""Rule-based highlight candidate generation and evaluation.

This is the heart of Phase 4: turning raw signals (visual motion, audio
energy) into a single excitement score, thresholding it to produce
candidate highlight windows, and measuring how well those candidates line
up with the ground-truth labels.

The design decision from Phase 3 is encoded here: audio energy is weak on
its own for MMA, so motion dominates the combined score (`alpha` weights
motion more heavily) and audio acts only as a supporting signal.
"""

from __future__ import annotations

import numpy as np


def bin_aggregate(
    times: np.ndarray, values: np.ndarray, bin_s: float, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Average `values` into fixed-width time bins of `bin_s` seconds.

    Returns (bin_times, binned) where bin_times are the centres of each
    bin and binned holds the mean of all values falling in that bin
    (0.0 if a bin has no samples). This lets us place the motion series
    (5 fps) and audio series (2 fps) on a common 0.5 s grid.
    """
    bins = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    idx = np.floor(times / bin_s).astype(np.int64)
    valid = (idx >= 0) & (idx < n_bins)
    np.add.at(bins, idx[valid], values[valid])
    np.add.at(counts, idx[valid], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        binned = np.where(counts > 0, bins / np.maximum(counts, 1), 0.0)
    bin_times = (np.arange(n_bins) + 0.5) * bin_s
    return bin_times, binned


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing-window mean with variable length at the start of the series."""
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    cum = np.concatenate(([0.0], np.cumsum(values)))
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = (cum[i + 1] - cum[lo]) / (i - lo + 1)
    return out


def _trailing_mean_exclusive(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of the `window` bins *before* each index (excludes the current).

    Unlike `rolling_mean`, this does not include the current value, so a
    spike at index i does not inflate its own baseline. At the very start
    of the series there are no prior bins, so we fall back to the current
    value (which yields a boost of ~1.0 there).
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    cum = np.concatenate(([0.0], np.cumsum(values)))
    for i in range(n):
        lo = max(0, i - window)
        if lo == i:
            out[i] = values[i]
        else:
            out[i] = (cum[i] - cum[lo]) / (i - lo)
    return out


def relative_boost(
    values: np.ndarray, window: int, min_floor: float = 1e-6
) -> np.ndarray:
    """Energy relative to a trailing baseline that EXCLUDES the current bin.

    Values near 1.0 mean "about as energetic as the recent average";
    values well above 1.0 mean "a sudden burst". Excluding the current bin
    from its own baseline makes true spikes stand out more sharply. The
    `min_floor` guards against dividing by (near) zero during quiet passages.
    """
    base = _trailing_mean_exclusive(values, window)
    return values / np.maximum(base, min_floor)


def combined_score(
    motion_boost: np.ndarray,
    audio_boost: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """Weighted combination of motion and audio boosts.

    alpha weights motion (the stronger signal); (1 - alpha) weights audio.
    Both inputs are 'boost' ratios centred near 1.0, so a plain weighted
    sum is meaningful on a shared scale.
    """
    return alpha * motion_boost + (1.0 - alpha) * audio_boost


def find_windows(
    score: np.ndarray,
    threshold: float,
    bin_s: float,
    min_gap_s: float = 1.0,
    min_len_s: float = 0.5,
) -> list[tuple[float, float]]:
    """Find candidate highlight windows in a score series.

    A window is a contiguous run of bins where score >= threshold. Runs
    separated by a gap of <= `min_gap_s` are merged, and runs shorter than
    `min_len_s` are dropped. Returns a list of (start_seconds, end_seconds).
    """
    above = score >= threshold
    n = len(above)
    min_gap_bins = max(0, int(round(min_gap_s / bin_s)))
    min_len_bins = max(1, int(round(min_len_s / bin_s)))

    windows: list[tuple[float, float]] = []
    start: int | None = None
    for i in range(n):
        if above[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                windows.append((start, i - 1))
                start = None
    if start is not None:
        windows.append((start, n - 1))

    # Merge runs separated by a small gap.
    merged: list[tuple[int, int]] = []
    for a, b in windows:
        if merged and a - merged[-1][1] - 1 <= min_gap_bins:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    return [
        (s * bin_s, e * bin_s)
        for s, e in merged
        if (e - s + 1) >= min_len_bins
    ]


def evaluate_windows(
    windows: list[tuple[float, float]],
    label_seconds: np.ndarray,
    tolerance_s: float = 3.0,
) -> dict:
    """Score candidate windows against ground-truth label times.

    Uses a greedy **one-to-one** match: each label is assigned to its
    nearest, still-unused candidate within `tolerance_s`. This avoids
    inflating recall when several candidates land on the same label.
    A candidate with no label is a false positive; a label with no candidate
    is a false negative. Returns precision, recall, F1 and tp/fp/fn counts.
    """
    label_seconds = np.asarray(label_seconds, dtype=np.float64)
    n_cand = len(windows)
    n_label = len(label_seconds)

    if n_cand == 0:
        return {
            "candidates": 0, "tp": 0, "fp": 0, "fn": n_label,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }

    centers = np.array([(s + e) / 2.0 for s, e in windows])
    used_cand = np.zeros(n_cand, dtype=bool)
    used_label = np.zeros(n_label, dtype=bool)
    pairs = 0
    for li in range(n_label):
        dists = np.abs(centers - label_seconds[li])
        dists[used_cand] = np.inf
        j = int(np.argmin(dists))
        if dists[j] <= tolerance_s:
            used_cand[j] = True
            used_label[li] = True
            pairs += 1

    tp = pairs
    fp = n_cand - pairs
    fn = n_label - pairs
    precision = tp / n_cand if n_cand else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "candidates": n_cand,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
