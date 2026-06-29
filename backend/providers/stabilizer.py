"""Subtitle stabilizer.

Tracks the difference between an in-flight "live" line (revisable) and
finalized history (immutable). The backend calls `update_partial` as
chunks come in, and `finalize` when VAD says the speaker paused or a
sentence boundary is detected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class StabilizedSegment:
    source_text: str
    translated_text: str
    is_final: bool


@dataclass
class SubtitleStabilizer:
    # Buffer accumulating PCM bytes since last finalize
    audio_buffer: bytearray = field(default_factory=bytearray)
    # Most recent partial transcripts (overwritten as new ones arrive)
    last_partial_source: str = ""
    last_partial_translated: str = ""
    # Finalized history (kept server-side mostly for debugging / context)
    history: List[StabilizedSegment] = field(default_factory=list)

    def append_audio(self, pcm: bytes) -> None:
        self.audio_buffer.extend(pcm)

    def buffer_bytes(self) -> bytes:
        return bytes(self.audio_buffer)

    def buffer_duration_ms(self, sample_rate: int) -> float:
        # 16-bit mono => 2 bytes per sample
        samples = len(self.audio_buffer) / 2
        return (samples / sample_rate) * 1000.0

    def update_partial(self, source: str, translated: str) -> StabilizedSegment:
        self.last_partial_source = source
        self.last_partial_translated = translated
        return StabilizedSegment(
            source_text=source, translated_text=translated, is_final=False
        )

    def finalize(self, source: str, translated: str) -> StabilizedSegment:
        seg = StabilizedSegment(
            source_text=source, translated_text=translated, is_final=True
        )
        self.history.append(seg)
        self.audio_buffer.clear()
        self.last_partial_source = ""
        self.last_partial_translated = ""
        return seg
