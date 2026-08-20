"""End-to-end highlight detection pipeline for a video file (Phase 8).

Turns a raw fight video into a list of timestamped, high-confidence
highlight windows ready for a user-facing app.

Design notes (documented so they can be defended):
- Uses the **rule-based motion detector** (motion boost vs a rolling
  baseline). This generalises to any video without needing training data.
  The Phase 6 Random Forest was a single-fight experiment, so the app keeps
  the generalisable path as its default.
- **Audio is deliberately excluded.** Phase 3/4 showed audio energy hurts
  MMA highlight detection (compressed broadcast mix, constant crowd), so the
  pipeline is motion-only by default.
- Windows are **padded by `pad_s`** (default 1.5 s) on each side so an
  extracted clip shows the full action sequence rather than ending just
  before the action lands.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.detection.combine import (
    bin_aggregate,
    find_windows,
    pad_windows,
    relative_boost,
)
from src.video.motion import motion_series


def extract_clip(video_path: str | Path, start_s: float, end_s: float, out_path: str | Path) -> Path:
    """Cut a segment out of the video with ffmpeg (stream copy, no re-encode).

    Uses a list-based subprocess call (no shell), so the times/paths can't
    be interpreted by a shell. Returns the output path.
    """
    out_path = Path(out_path)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{start_s:.2f}", "-i", str(video_path),
            "-t", f"{max(0.1, end_s - start_s):.2f}", "-c", "copy", str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return out_path


def _peak_score_in_window(score: np.ndarray, start_s: float, end_s: float, bin_s: float) -> float:
    # `hi = int(end_s/bin_s) + 1` makes the slice inclusive of the bin that
    # contains end_s (bins are half-open [i*bin_s, (i+1)*bin_s)).
    lo = int(start_s / bin_s)
    hi = int(end_s / bin_s) + 1
    return float(score[lo:hi].max())


def detect_highlights(
    video_path: str | Path,
    bin_s: float = 0.5,
    threshold: float = 1.6,
    min_gap_s: float = 1.0,
    min_len_s: float = 0.5,
    high_conf_peak: float = 1.8,
    min_dur_s: float = 1.0,
    pad_s: float = 1.5,
    boost_window: int = 12,
    work_dir: str | Path | None = None,
) -> dict:
    """Detect highlight windows in a video.

    Returns a dict with metadata and a list of windows, each as:
      {"start_s", "end_s", "duration_s", "peak_score", "start_mmss", "end_mmss"}
    """
    video_path = Path(video_path)
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="fightsmart_"))
    work.mkdir(parents=True, exist_ok=True)

    # Motion is the primary signal.
    mt, motion = motion_series(video_path, sample_every=6, target_width=480)
    if len(mt) == 0:
        raise ValueError("No motion samples were extracted from the video")

    n_bins = int(mt[-1] / bin_s) + 1
    _, motion_bin = bin_aggregate(mt, motion, bin_s, n_bins)
    score = relative_boost(motion_bin, boost_window)

    # Candidate windows above the base threshold.
    candidates = find_windows(score, threshold, bin_s, min_gap_s=min_gap_s, min_len_s=min_len_s)

    # Keep only the high-confidence, sustained windows (what a user sees),
    # then pad so clips show the full sequence.
    windows = []
    for s, e in candidates:
        peak = _peak_score_in_window(score, s, e, bin_s)
        if peak >= high_conf_peak and (e - s) >= min_dur_s:
            ps, pe = pad_windows([(s, e)], pad_s)[0]
            windows.append(
                {
                    "start_s": round(ps, 2),
                    "end_s": round(pe, 2),
                    "duration_s": round(pe - ps, 2),
                    "peak_score": round(peak, 2),
                    "start_mmss": _mmss(ps),
                    "end_mmss": _mmss(pe),
                }
            )

    windows.sort(key=lambda w: w["start_s"])
    return {"duration_s": round(float(mt[-1]), 2), "n_windows": len(windows), "windows": windows}


def _mmss(t: float) -> str:
    t = int(t)
    return f"{t // 60:02d}:{t % 60:02d}"
