"""Tiny energy-based VAD.

Not as good as Silero/WebRTC VAD but zero-dependency and adequate for MVP.
Returns True if the trailing window is below the energy threshold for at
least `silence_ms` milliseconds.
"""
from __future__ import annotations

import numpy as np


def _rms(pcm16: bytes) -> float:
    if len(pcm16) == 0:
        return 0.0
    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def is_silence(
    pcm16_tail: bytes,
    sample_rate: int,
    silence_ms: int,
    energy_threshold: float,
) -> bool:
    if len(pcm16_tail) == 0:
        return False

    needed_samples = int(sample_rate * (silence_ms / 1000.0))
    needed_bytes = needed_samples * 2
    if len(pcm16_tail) < needed_bytes:
        return False

    tail = pcm16_tail[-needed_bytes:]
    return _rms(tail) < energy_threshold


def has_speech(
    pcm16: bytes,
    sample_rate: int,
    energy_threshold: float,
    min_speech_ms: int = 300,
    window_ms: int = 30,
) -> bool:
    """True if the buffer contains at least `min_speech_ms` of audio above
    `energy_threshold`. Slides a small window across the buffer and counts
    windows that exceed the threshold.

    This is the gate before calling ASR — without it, silent / noise-only
    buffers get sent to the model and it hallucinates ("Welcome.", "Thank
    you.", etc).
    """
    if len(pcm16) == 0:
        return False

    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return False

    window_samples = max(1, int(sample_rate * (window_ms / 1000.0)))
    if samples.size < window_samples:
        return _rms(pcm16) >= energy_threshold

    # Compute RMS per window
    n_windows = samples.size // window_samples
    trimmed = samples[: n_windows * window_samples].reshape(n_windows, window_samples)
    rms_per_window = np.sqrt(np.mean(trimmed ** 2, axis=1))
    voiced_windows = int((rms_per_window >= energy_threshold).sum())
    voiced_ms = voiced_windows * window_ms
    return voiced_ms >= min_speech_ms
