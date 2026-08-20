"""Tests for src.audio.features using a small synthetic WAV I generate."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import wavfile

from src.audio.features import load_wav_mono, rms_to_db, windowed_rms


@pytest.fixture
def sine_wav(tmp_path):
    """Write a 1 kHz sine wave at 22,050 Hz, 2 seconds, int16."""
    sr = 22050
    t = np.arange(sr * 2) / sr
    samples = (np.sin(2 * np.pi * 1000 * t) * 10000).astype(np.int16)
    path = tmp_path / "sine.wav"
    wavfile.write(str(path), sr, samples)
    return path, sr


def test_load_wav_mono_normalises_to_unit_amplitude(sine_wav):
    path, sr = sine_wav
    rate, samples = load_wav_mono(path)
    assert rate == sr
    assert samples.dtype == np.float32
    # Peak of 10000/32768 ~= 0.305, within [-1, 1].
    assert samples.max() <= 1.0
    assert samples.min() >= -1.0
    assert np.isclose(np.abs(samples).max(), 10000 / 32768, atol=0.01)


def test_windowed_rms_non_overlapping(sine_wav):
    path, sr = sine_wav
    _, samples = load_wav_mono(path)
    times, rms = windowed_rms(samples, sr, window_s=1.0)  # 2 windows of 1s
    assert len(rms) == 2
    # RMS of a sine of amplitude A is A/sqrt(2). Here A~=0.305.
    expected = (10000 / 32768) / np.sqrt(2)
    assert np.allclose(rms, expected, atol=0.01)


def test_windowed_rms_times_are_window_centres(sine_wav):
    path, sr = sine_wav
    _, samples = load_wav_mono(path)
    times, _ = windowed_rms(samples, sr, window_s=1.0)
    assert np.allclose(times, [0.5, 1.5])


def test_rms_to_db_negative_for_quiet_audio(sine_wav):
    _, samples = load_wav_mono(sine_wav[0])
    rms = samples.astype(np.float64) ** 2  # arbitrary small positive values
    db = rms_to_db(rms * 1e-6)
    assert np.all(db < 0)


def test_silence_has_very_low_energy(tmp_path):
    sr = 22050
    zeros = np.zeros(sr, dtype=np.int16)
    path = tmp_path / "silence.wav"
    wavfile.write(str(path), sr, zeros)
    _, samples = load_wav_mono(path)
    _, rms = windowed_rms(samples, sr, window_s=1.0)
    assert np.all(rms < 1e-5)
