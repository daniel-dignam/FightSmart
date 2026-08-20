"""Tests for src.video.loader using the real fight video in data/raw."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.video.loader import bgr_to_rgb, frame_at_second, open_video, video_metadata

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
VIDEO = RAW_DIR / "Watch Islam Makhachev vs Alexander Volkanovski 1 UFC.mp4"


@pytest.fixture(scope="module")
def video_path():
    if not VIDEO.exists():
        pytest.skip(f"Raw fight video not present: {VIDEO}")
    return VIDEO


def test_metadata_has_expected_fields(video_path):
    meta = video_metadata(video_path)
    assert meta["width"] == 1920
    assert meta["height"] == 1080
    assert meta["fps"] == pytest.approx(30, abs=1)
    assert meta["frame_count"] > 0
    assert meta["duration_s"] > 0
    assert isinstance(meta["codec"], str) and meta["codec"]


def test_frame_at_second_returns_expected_shape(video_path):
    frame = frame_at_second(video_path, 3.0)
    assert frame is not None
    # OpenCV frames are (height, width, channels) in BGR.
    assert frame.shape == (1080, 1920, 3)


def test_bgr_to_rgb_swaps_channels(video_path):
    frame = frame_at_second(video_path, 3.0)
    rgb = bgr_to_rgb(frame)
    assert rgb.shape == frame.shape


def test_open_video_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_video(tmp_path / "does_not_exist.mp4")


def test_frame_at_second_past_end_is_none(video_path):
    # 1,000,000 seconds is far past the ~36 minute video.
    assert frame_at_second(video_path, 1_000_000) is None
