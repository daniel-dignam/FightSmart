"""Tests for src.features.dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.dataset import (
    build_feature_matrix,
    clean_events,
    parse_time_to_sec,
)


def test_parse_time_to_sec_simple():
    assert parse_time_to_sec("3:46") == 226


def test_parse_time_to_sec_range_uses_start():
    assert parse_time_to_sec("4:45-4:48") == 285


def test_clean_events_drops_empty_and_fixes_typo():
    df = pd.DataFrame(
        {
            "VIDEO_TIME": ["3:46", "6:33", "", None, "8:47"],
            "EVENT_TYPE": ["fight_start", "significant_stike", "x", "y", "round_1_end"],
        }
    )
    out = clean_events(df)
    assert len(out) == 3
    assert out["event_type"].tolist() == ["fight_start", "significant_strike", "round_1_end"]


def test_clean_events_marks_highlights():
    df = pd.DataFrame(
        {
            "VIDEO_TIME": ["3:46", "6:38", "8:47"],
            "EVENT_TYPE": ["fight_start", "knockdown", "round_1_end"],
        }
    )
    out = clean_events(df)
    assert out["is_highlight"].tolist() == [False, True, False]


def test_build_feature_matrix_shapes_and_labels():
    # 10 seconds at 0.5 s bins => 20 bins.
    times = np.arange(0, 10, 0.5)
    motion = np.ones_like(times)
    audio = np.ones_like(times)
    events = pd.DataFrame({"start_sec": [5.0], "is_highlight": [True]})
    names, X, y, bin_times = build_feature_matrix(
        times, motion, times, audio, events, bin_s=0.5, tol_s=1.0
    )
    assert names == ["motion", "motion_boost", "audio_rms", "audio_boost"]
    assert X.shape == (20, 4)
    assert y.shape == (20,)
    # A highlight at 5.0 s with tolerance 1.0 s => bins centred in [4, 6] are positive.
    assert y.sum() > 0
    # Bins far away are negative.
    assert y[0] == 0 and y[-1] == 0


def test_build_feature_matrix_excludes_boundary_events():
    times = np.arange(0, 10, 0.5)
    motion = np.ones_like(times)
    events = pd.DataFrame({"start_sec": [5.0], "is_highlight": [False]})
    _, _, y, _ = build_feature_matrix(times, motion, times, motion, events, bin_s=0.5, tol_s=3.0)
    assert y.sum() == 0
