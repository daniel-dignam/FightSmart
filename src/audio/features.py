"""Audio feature extraction helpers for FightSmart.

The central idea for Phase 3 is to use **audio energy** as a cheap proxy
for crowd/commentary excitement: loud, energetic moments in the broadcast
tend to coincide with significant fight action (big strikes, takedowns,
submissions) and the crowd reacting to them.

I keep the feature math here (testable, reusable) and leave the
exploration and interpretation to the notebook.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile


def load_wav_mono(path: str | Path) -> tuple[int, np.ndarray]:
    """Load a WAV file and return (sample_rate, samples) as float in [-1, 1].

    Multi-channel files are averaged down to mono, and int16 samples are
    normalised to floats in [-1, 1] so downstream math is consistent and
    independent of the source bit depth.
    """
    rate, data = wavfile.read(str(path))
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        samples = data.astype(np.float32) / 32768.0
    elif data.dtype == np.float32 or data.dtype == np.float64:
        samples = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV dtype: {data.dtype}")
    return int(rate), np.ascontiguousarray(samples, dtype=np.float32)


def windowed_rms(
    samples: np.ndarray,
    sample_rate: int,
    window_s: float,
    hop_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute root-mean-square energy over sliding windows.

    Returns (times_seconds, rms) where each RMS value is the average
    power of the audio in a `window_s`-second window, and times are the
    centre of each window. `hop_s` controls overlap (defaults to the
    window size, i.e. non-overlapping windows).

    RMS is a simple, robust energy measure: a loud, exciting moment shows
    up as a spike. I choose this over more complex features to establish
    a strong baseline before adding spectral features if needed.
    """
    window = int(window_s * sample_rate)
    hop = int((hop_s or window_s) * sample_rate)
    if len(samples) < window:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    frames = sliding_window_view(samples, window)[::hop]
    rms = np.sqrt(np.mean(frames**2, axis=1)).astype(np.float32)
    times = (np.arange(len(rms)) * hop + window / 2) / sample_rate
    return times, rms


def rms_to_db(rms: np.ndarray, reference: float | None = None) -> np.ndarray:
    """Convert RMS amplitude to decibels relative to `reference`.

    Defaults to a reference of 1.0 (full-scale digital), which puts values
    in a familiar range (roughly -60 dB to 0 dB for normal audio). I add a
    tiny epsilon to avoid log(0).
    """
    ref = reference if reference is not None else 1.0
    eps = 1e-10
    return 20.0 * np.log10(np.maximum(rms, eps) / ref)
