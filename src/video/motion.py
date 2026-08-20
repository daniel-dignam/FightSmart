"""Visual motion feature extraction.

Phase 4's core signal is **visual motion**: the amount of change between
consecutive video frames. High motion usually means significant fight
activity (strikes, scrambles, takedowns); low motion means stalemates,
feinting, or rest between rounds.

I measure motion as the **mean absolute difference** between downscaled
grayscale frames. Downscaling keeps the computation fast and ignores
high-frequency noise, and grayscale lets me focus on spatial change rather
than colour shifts (which are often caused by broadcast lighting/cuts).

I sample a subset of frames (default every 6th frame at 30 fps => 5 fps)
because adjacent frames are nearly identical — motion is a low-frequency
signal, and this cuts the workload ~6x with negligible loss.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.video.loader import open_video


def _downscale_gray(frame: cv2.Mat, target_width: int) -> np.ndarray:
    """Convert a BGR frame to downscaled grayscale float32 in [0, 1]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w != target_width:
        scale = target_width / w
        gray = cv2.resize(gray, (target_width, int(h * scale)))
    return gray.astype(np.float32) / 255.0


def motion_series(
    path: str | Path,
    sample_every: int = 6,
    target_width: int = 480,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a per-sampled-frame motion score.

    Returns (times_seconds, motion) where each motion value is the mean
    absolute pixel difference (0..1) between a sampled frame and the one
    before it. `sample_every` controls how many frames I skip (6 @ 30 fps
    => 5 samples/second). `target_width` downscales before differencing for
    speed. `max_frames` limits how many source frames I read (useful for
    tests and quick checks).
    """
    cap = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev: np.ndarray | None = None
    times: list[float] = []
    scores: list[float] = []

    frames_read = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frames_read % sample_every == 0:
                small = _downscale_gray(frame, target_width)
                t = frames_read / fps
                if prev is not None:
                    scores.append(float(np.mean(np.abs(small - prev))))
                    times.append(t)
                prev = small
            frames_read += 1
            if max_frames is not None and frames_read >= max_frames:
                break
    finally:
        cap.release()

    return np.asarray(times, dtype=np.float32), np.asarray(scores, dtype=np.float32)
