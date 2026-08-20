"""Tests for src.pipeline using a tiny synthetic video."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.pipeline import detect_highlights, extract_clip


def _make_motion_video(path, fps=30, seconds=6, move_after=3):
    """Static for the first `move_after` s, then a white square moving right."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (100, 100)
    )
    for f in range(fps * seconds):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        x = 10 + ((f - move_after * fps) * 2) if f >= move_after * fps else 10
        cv2.rectangle(img, (x, 30), (x + 20, 70), (255, 255, 255), -1)
        writer.write(img)
    writer.release()


def test_detect_highlights_returns_expected_structure(tmp_path):
    video = tmp_path / "fight.mp4"
    _make_motion_video(video)
    res = detect_highlights(
        video, threshold=1.0, high_conf_peak=1.0, min_dur_s=0.5, pad_s=0.0
    )
    assert set(res) >= {"duration_s", "n_windows", "windows"}
    assert res["duration_s"] > 0
    assert isinstance(res["windows"], list)
    for w in res["windows"]:
        assert {"start_s", "end_s", "duration_s", "peak_score", "start_mmss", "end_mmss"} <= set(w)
        assert w["end_s"] >= w["start_s"]


def test_detect_highlights_finds_motion_window(tmp_path):
    video = tmp_path / "fight.mp4"
    _make_motion_video(video)
    res = detect_highlights(
        video, threshold=1.0, high_conf_peak=1.0, min_dur_s=0.5, pad_s=0.0
    )
    assert res["n_windows"] >= 1


def test_detect_highlights_pads_windows(tmp_path):
    video = tmp_path / "fight.mp4"
    _make_motion_video(video)
    res_no_pad = detect_highlights(
        video, threshold=1.0, high_conf_peak=1.0, min_dur_s=0.5, pad_s=0.0
    )
    res_pad = detect_highlights(
        video, threshold=1.0, high_conf_peak=1.0, min_dur_s=0.5, pad_s=1.0
    )
    assert res_no_pad["windows"] and res_pad["windows"]
    # Padding extends both sides: start earlier (or clamped to 0), end later.
    assert res_pad["windows"][0]["start_s"] <= res_no_pad["windows"][0]["start_s"]
    assert res_pad["windows"][0]["end_s"] >= res_no_pad["windows"][0]["end_s"]


def test_extract_clip_builds_shell_safe_args(monkeypatch, tmp_path):
    """extract_clip must pass a list (no shell=True) with the right ffmpeg args."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        Path(cmd[-1]).touch()
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("src.pipeline.subprocess.run", fake_run)
    video = tmp_path / "v.mp4"
    out = tmp_path / "clip.mp4"

    result = extract_clip(video, start_s=10.0, end_s=14.5, out_path=out)

    assert result == out
    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg"
    assert "-ss" in cmd and "10.00" in cmd
    assert "-i" in cmd and str(video) in cmd
    assert "-t" in cmd and "4.50" in cmd
    assert cmd[-1] == str(out)
    # Must never use a shell (protects against injection).
    assert captured["kwargs"].get("shell") is not True
