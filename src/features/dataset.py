"""Build a supervised dataset from raw signals and manual event labels.

Phase 5 turns the ground-truth labels and the signal series into a form a
classifier can learn from. Each 0.5 s time bin becomes one row of features
(raw signal values plus their relative boost), labelled positive if a
highlight event falls within a tolerance window of that bin.

Key design decisions (documented so they can be revisited):
- "highlight" is defined as any non-boundary event. Round/fight boundary
  markers (fight_start, round_N_start/end) are structural, not highlights,
  so they are excluded from the positive class.
- The label tolerance (default 3 s) matches the evaluation tolerance used in
  Phase 4, so Phase 6 can be compared directly against the rule-based
  baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.detection.combine import bin_aggregate, relative_boost

# Structural markers that should NOT be treated as highlights.
BOUNDARY_MARKERS = {"fight_start"}

# Known typos / variants in the raw manual labels.
EVENT_TYPE_FIXES = {
    "significant_stike": "significant_strike",
}


def _is_round_marker(event_type: str) -> bool:
    return str(event_type).startswith("round_")


def parse_time_to_sec(txt) -> float:
    """Parse 'MM:SS' or 'MM:SS-MM:SS' to the start second."""
    txt = str(txt).split("-")[0].strip()
    m, s = txt.split(":")
    return int(m) * 60 + int(s)


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the raw manual-labels CSV into a usable events table.

    - Drops rows without a time and empty rows.
    - Parses VIDEO_TIME into start_seconds.
    - Fixes known EVENT_TYPE typos.
    - Adds `is_highlight` (True for all non-boundary events).
    """
    out = df.copy()
    out = out.dropna(subset=["VIDEO_TIME"])
    out = out[out["VIDEO_TIME"].astype(str).str.strip() != ""]
    out["start_sec"] = out["VIDEO_TIME"].apply(parse_time_to_sec)
    out["event_type"] = out["EVENT_TYPE"].astype(str).replace(EVENT_TYPE_FIXES)
    out["is_highlight"] = ~(
        out["event_type"].map(_is_round_marker) | out["event_type"].isin(BOUNDARY_MARKERS)
    )
    return out


def build_feature_matrix(
    motion_times: np.ndarray,
    motion: np.ndarray,
    audio_times: np.ndarray,
    audio_rms: np.ndarray,
    events: pd.DataFrame,
    bin_s: float = 0.5,
    n_bins: int | None = None,
    tol_s: float = 3.0,
    boost_window: int = 12,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Build (feature_names, X, y, bin_times) for the whole signal timeline.

    Features per bin: raw motion, motion boost, raw audio RMS, audio boost.
    Labels: 1 if any highlight event is within `tol_s` seconds of the bin.
    """
    if n_bins is None:
        if len(motion_times) == 0 and len(audio_times) == 0:
            raise ValueError("Cannot size the feature matrix from empty signal series")
        last_t = 0.0
        if len(motion_times) > 0:
            last_t = max(last_t, float(motion_times[-1]))
        if len(audio_times) > 0:
            last_t = max(last_t, float(audio_times[-1]))
        n_bins = int(last_t / bin_s) + 1

    bin_times, motion_bin = bin_aggregate(motion_times, motion, bin_s, n_bins)
    _, audio_bin = bin_aggregate(audio_times, audio_rms, bin_s, n_bins)
    motion_boost = relative_boost(motion_bin, boost_window)
    audio_boost = relative_boost(audio_bin, boost_window)

    X = np.column_stack([motion_bin, motion_boost, audio_bin, audio_boost])
    feature_names = ["motion", "motion_boost", "audio_rms", "audio_boost"]

    y = np.zeros(n_bins, dtype=np.int64)
    highlights = events[events["is_highlight"]]
    for start in highlights["start_sec"].values:
        y[np.abs(bin_times - start) <= tol_s] = 1

    return feature_names, X.astype(np.float32), y, bin_times
