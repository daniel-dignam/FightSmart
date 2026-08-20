"""Tests for src.detection.combine."""

from __future__ import annotations

import numpy as np

from src.detection.combine import (
    bin_aggregate,
    combined_score,
    evaluate_windows,
    find_windows,
    relative_boost,
    rolling_mean,
)


def test_rolling_mean_constant():
    out = rolling_mean(np.full(10, 5.0), window=3)
    assert np.allclose(out, 5.0)


def test_rolling_mean_trailing():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    out = rolling_mean(v, window=3)
    # window 3: [1], [1.5], [(1+2+3)/3], [(2+3+4)/3]
    assert np.allclose(out, [1.0, 1.5, 2.0, 3.0])


def test_relative_boost_baseline_is_one():
    # Flat series => boost ~1.
    boost = relative_boost(np.full(20, 2.0), window=5)
    assert np.allclose(boost, 1.0)


def test_relative_boost_spike_detected():
    v = np.ones(20)
    v[10] = 4.0
    boost = relative_boost(v, window=5)
    assert boost[10] > 1.0


def test_bin_aggregate_means():
    times = np.array([0.1, 0.3, 0.4, 0.6, 0.9])
    values = np.array([1.0, 3.0, 2.0, 5.0, 7.0])
    bt, binned = bin_aggregate(times, values, bin_s=0.5, n_bins=2)
    # bin0: times 0.1,0.3,0.4 => mean 2.0 ; bin1: 0.6,0.9 => mean 6.0
    assert np.allclose(binned, [2.0, 6.0])
    assert np.allclose(bt, [0.25, 0.75])


def test_combined_score_weights_motion():
    motion = np.ones(5) * 2.0
    audio = np.ones(5) * 1.0
    score = combined_score(motion, audio, alpha=0.7)
    assert np.allclose(score, 0.7 * 2.0 + 0.3 * 1.0)


def test_find_windows_basic():
    score = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    wins = find_windows(score, threshold=0.5, bin_s=0.5, min_gap_s=0.5, min_len_s=0.5)
    # Runs: bins 1-2, bin 4, bin 7. The gap between bin2 and bin4 is one bin
    # (= 0.5s == min_gap), so they merge into 1-4. Bin 7 stays separate.
    assert wins == [(0.5, 2.0), (3.5, 3.5)]


def test_find_windows_merges_close_runs():
    score = np.array([1.0, 0.0, 1.0])  # gap of one bin
    wins = find_windows(score, threshold=0.5, bin_s=0.5, min_gap_s=1.0, min_len_s=0.5)
    # gap of 1 bin = 0.5s <= min_gap 1.0s => merged into one window 0..1.0
    assert wins == [(0.0, 1.0)]


def test_find_windows_none_above_threshold():
    score = np.zeros(10)
    assert find_windows(score, threshold=0.5, bin_s=0.5, min_gap_s=0.5, min_len_s=0.5) == []


def test_relative_boost_excludes_current_bin():
    v = np.ones(20)
    v[10] = 5.0
    boost = relative_boost(v, window=5)
    # baseline at i=10 uses only prior bins (all 1.0), so the spike is 5.0x.
    assert np.isclose(boost[10], 5.0)
    # a non-spike bin stays near 1.0.
    assert np.isclose(boost[5], 1.0, atol=1e-2)


def test_evaluate_windows_one_to_one_matching():
    # Two candidates both within tolerance of the single label: only one is a
    # true positive; the other is a false positive (no recall inflation).
    wins = [(5.0, 5.5), (5.2, 5.7)]
    labels = np.array([5.3])
    res = evaluate_windows(wins, labels, tolerance_s=3.0)
    assert res["tp"] == 1 and res["fp"] == 1 and res["fn"] == 0
    assert res["precision"] == 0.5 and res["recall"] == 1.0


def test_evaluate_windows_perfect():
    wins = [(5.0, 5.5)]
    labels = np.array([5.2])
    res = evaluate_windows(wins, labels, tolerance_s=3.0)
    assert res["precision"] == 1.0 and res["recall"] == 1.0 and res["f1"] == 1.0


def test_evaluate_windows_misses():
    wins = [(5.0, 5.5)]
    labels = np.array([50.0])
    res = evaluate_windows(wins, labels, tolerance_s=3.0)
    assert res["tp"] == 0 and res["fn"] == 1 and res["precision"] == 0.0 and res["recall"] == 0.0
