"""Tests for src.video.motion using a small synthetic video I generate."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.video.motion import motion_series


@pytest.fixture
def static_then_moving_video(tmp_path):
    """A 12-frame video: static for the first half, then a moving white square."""
    path = tmp_path / "motion_test.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (100, 100)
    )
    for frame_idx in range(12):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        x = 10 + (frame_idx * 5 if frame_idx >= 6 else 0)  # move only after frame 6
        cv2.rectangle(img, (x, 30), (x + 20, 70), (255, 255, 255), -1)
        writer.write(img)
    writer.release()
    return path


def test_static_region_has_near_zero_motion(static_then_moving_video):
    times, motion = motion_series(static_then_moving_video, sample_every=1, max_frames=12)
    # First motion values (during the static phase, after the very first frame)
    # should be ~0. motion[0] compares frame 1 to frame 0 (both static).
    assert motion[0] < 0.02


def test_moving_region_has_nonzero_motion(static_then_moving_video):
    times, motion = motion_series(static_then_moving_video, sample_every=1, max_frames=12)
    # Once the square starts moving (frame >= 6), motion should be clearly > 0.
    assert motion[6] > 0.01
    assert motion.max() > 0.01


def test_times_increase_monotonically(static_then_moving_video):
    times, _ = motion_series(static_then_moving_video, sample_every=1, max_frames=12)
    assert np.all(np.diff(times) > 0)
    # At 30 fps with sample_every=1, times are 1/30 apart.
    assert np.allclose(np.diff(times), 1 / 30, atol=1e-3)
