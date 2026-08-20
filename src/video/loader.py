"""Reusable video loading and inspection helpers.

Keeping this logic in `src/` (not the notebook) means exploration stays
light and the same functions can be reused and unit-tested by later phases
(feature extraction, rule-based highlight detection).

OpenCV reads frames as BGR (blue-green-red) colour images, which is why I
flag that convention everywhere it matters. Matplotlib expects RGB, so I
expose a small BGR->RGB conversion for display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2


def open_video(path: str | Path) -> cv2.VideoCapture:
    """Open a video file and verify it is readable.

    Raises FileNotFoundError if the file does not exist or OpenCV cannot
    open it, so callers fail loudly instead of silently processing nothing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file does not exist: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"OpenCV could not open video: {path}")
    return cap


def video_metadata(path: str | Path) -> dict:
    """Return a dictionary of the video's basic properties.

    Includes FPS, frame count, computed duration, resolution, and the
    fourcc codec string. FPS/frame_count can legitimately read as 0 for
    some files, so I surface the raw values and let the caller decide.
    """
    cap = open_video(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_raw = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()

    codec = "".join(chr((fourcc_raw >> (8 * i)) & 0xFF) for i in range(4)) if fourcc_raw else "unknown"
    duration_s = frame_count / fps if fps > 0 else 0.0

    return {
        "path": str(path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": round(duration_s, 2),
        "duration_min": round(duration_s / 60, 2),
        "codec": codec,
    }


def frame_at_second(path: str | Path, second: float) -> Optional[cv2.Mat]:
    """Read the frame at a given time (in seconds).

    I seek by setting CAP_PROP_POS_MSEC rather than by frame index, which
    is more intuitive (callers think in time, not frame numbers) and works
    with variable-frame-rate content. Returns None if the seek lands past
    the end of the video.
    """
    cap = open_video(path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok else None


def bgr_to_rgb(frame: cv2.Mat) -> cv2.Mat:
    """Convert an OpenCV BGR frame to RGB for display (e.g. with matplotlib)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
