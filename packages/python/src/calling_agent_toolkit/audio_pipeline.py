# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Vasik Mansuri
"""Shared audio path between a media adapter and STT.

decode -> validate -> resample -> fixed framing -> processors (noise suppression, speaker isolation).
The pipeline reports exactly which processors ran; it never claims suppression that did not happen.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from .audio import decode_alaw, decode_mulaw
from .errors import ProviderError
from .types import SUPPORTED_SAMPLE_RATES, AudioFormat, AudioFrame, MediaDirection, TimestampKind

_PROVIDER = "audio_pipeline"


def _config_error(message: str) -> ProviderError:
    return ProviderError(code="invalid_configuration", provider=_PROVIDER, message=message)


def pcm16_to_samples(data: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(data) // 2}h", data[: len(data) // 2 * 2]))


def samples_to_pcm16(samples: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


class LinearResampler:
    """Streaming linear-interpolation resampler, integer-exact so Python and TypeScript agree bit for bit.

    Adequate for upsampling telephony audio (8 kHz -> 16 kHz). It does not low-pass filter, so
    downsampling can alias; plug in a better resampler for that case.
    """

    def __init__(self, from_rate: int, to_rate: int) -> None:
        if from_rate <= 0 or to_rate <= 0:
            raise _config_error(f"Invalid resample rates {from_rate} -> {to_rate}.")
        self.from_rate = from_rate
        self.to_rate = to_rate
        self._buffer: list[int] = []
        self._base = 0
        self._out_index = 0

    def push(self, samples: Sequence[int]) -> list[int]:
        self._buffer.extend(samples)
        return self._drain(final=False)

    def flush(self) -> list[int]:
        out = self._drain(final=True)
        self._buffer = []
        return out

    def _drain(self, final: bool) -> list[int]:
        src, dst = self.from_rate, self.to_rate
        if src == dst:
            out = self._buffer
            self._base += len(out)
            self._out_index += len(out)
            self._buffer = []
            return out
        out = []
        buf = self._buffer
        while True:
            num = self._out_index * src
            local = num // dst - self._base
            r = num % dst
            if local >= len(buf) or (not final and local + 1 >= len(buf)):
                break
            a = buf[local]
            b = buf[local + 1] if local + 1 < len(buf) else a
            out.append((2 * (a * (dst - r) + b * r) + dst) // (2 * dst))
            self._out_index += 1
        keep_from = (self._out_index * src) // dst - self._base
        if keep_from > 0:
            drop = min(keep_from, len(buf))
            del buf[:drop]
            self._base += drop
        return out


ProcessorKind = Literal["noise_suppression", "speaker_isolation", "other"]
PipelineWarning = Literal["stacked_noise_suppression", "downsampling_without_antialiasing", "processor_bypassed"]


@dataclass(frozen=True)
class ProcessorHealth:
    active: bool
    detail: str | None = None


class AudioProcessor(Protocol):
    """A frame processor, e.g. an RNNoise or DeepFilterNet binding. Input and output are PCM16 LE mono."""

    name: str
    kind: ProcessorKind
    sample_rate: int
    frame_samples: int

    def process(self, pcm: bytes) -> bytes: ...


@dataclass(frozen=True)
class ProcessorStatus:
    name: str
    kind: ProcessorKind
    active: bool
    bypassed: bool
    error: str | None


@dataclass(frozen=True)
class PipelineStatus:
    target_format: AudioFormat
    frame_samples: int
    processors: tuple[ProcessorStatus, ...]
    noise_suppression_active: bool
    speaker_isolation_active: bool
    provider_noise_suppression: bool
    warnings: tuple[PipelineWarning, ...]


@dataclass
class _Slot:
    processor: AudioProcessor
    bypassed: bool = False
    error: str | None = None


@dataclass
class AudioPipeline:
    target_sample_rate: int
    frame_ms: int = 20
    #: Only frames of this direction are processed; others are ignored. Default inbound (the caller).
    direction: MediaDirection = "inbound"
    processors: Sequence[AudioProcessor] = ()
    #: bypass (default): a failing processor is skipped and reported; throw: fail the frame.
    on_processor_error: Literal["bypass", "throw"] = "bypass"
    #: Set when the provider already suppresses noise (e.g. Plivo noise_cancellation).
    provider_noise_suppression: bool = False
    _slots: list[_Slot] = field(init=False)
    _resampler: LinearResampler | None = field(init=False, default=None)
    _pending: list[int] = field(init=False, default_factory=list)
    _template: AudioFrame | None = field(init=False, default=None)
    _emitted: int = field(init=False, default=0)
    _downsampling: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        rate = self.target_sample_rate
        if rate not in SUPPORTED_SAMPLE_RATES:
            raise _config_error(f"Unsupported target rate {rate} Hz.")
        if (rate * self.frame_ms) % 1000 or self.frame_ms <= 0:
            raise _config_error(f"frameMs {self.frame_ms} does not give a whole number of samples at {rate} Hz.")
        for p in self.processors:
            if p.sample_rate != rate or p.frame_samples != self.frame_samples:
                raise _config_error(
                    f"Processor {p.name} needs {p.frame_samples} samples at {p.sample_rate} Hz; "
                    f"the pipeline produces {self.frame_samples} at {rate} Hz."
                )
        self._slots = [_Slot(p) for p in self.processors]

    @property
    def frame_samples(self) -> int:
        return self.target_sample_rate * self.frame_ms // 1000

    @property
    def target_format(self) -> AudioFormat:
        return AudioFormat("pcm_s16le", self.target_sample_rate, 1)

    def push(self, frame: AudioFrame) -> list[AudioFrame]:
        """Accept one provider frame; return zero or more complete, processed frames at the target format."""
        if frame.direction != self.direction:
            return []
        if frame.format.channels != 1:
            raise ProviderError(code="unsupported_format", provider=_PROVIDER, message="Only mono audio is supported.")
        if frame.format.encoding == "mulaw":
            samples = pcm16_to_samples(decode_mulaw(frame.payload))
        elif frame.format.encoding == "alaw":
            samples = pcm16_to_samples(decode_alaw(frame.payload))
        elif frame.format.encoding == "pcm_s16le":
            samples = pcm16_to_samples(frame.payload)
        else:
            raise ProviderError(
                code="unsupported_format", provider=_PROVIDER, message=f"No codec for {frame.format.encoding} yet."
            )
        if self._resampler is None:
            self._resampler = LinearResampler(frame.format.sample_rate, self.target_sample_rate)
            self._downsampling = frame.format.sample_rate > self.target_sample_rate
            self._template = frame
        elif self._resampler.from_rate != frame.format.sample_rate:
            raise ProviderError(
                code="protocol_violation",
                provider=_PROVIDER,
                message=f"Sample rate changed mid-stream from {self._resampler.from_rate} to {frame.format.sample_rate} Hz.",
            )
        self._pending.extend(self._resampler.push(samples))
        return self._drain_frames()

    def flush(self) -> list[AudioFrame]:
        """End the stream: emit remaining complete frames, then any short tail unprocessed (is_complete_frame False)."""
        if self._resampler is None:
            return []
        self._pending.extend(self._resampler.flush())
        out = self._drain_frames()
        if self._pending:
            out.append(self._make_frame(samples_to_pcm16(self._pending), complete=False))
            self._pending = []
        return out

    def status(self) -> PipelineStatus:
        processors = tuple(
            ProcessorStatus(
                name=s.processor.name,
                kind=s.processor.kind,
                bypassed=s.bypassed,
                error=s.error,
                active=not s.bypassed and _health(s.processor).active,
            )
            for s in self._slots
        )
        active_ns = sum(p.kind == "noise_suppression" and p.active for p in processors)
        warnings: list[PipelineWarning] = []
        if active_ns > 1 or (active_ns > 0 and self.provider_noise_suppression):
            warnings.append("stacked_noise_suppression")
        if self._downsampling:
            warnings.append("downsampling_without_antialiasing")
        if any(p.bypassed for p in processors):
            warnings.append("processor_bypassed")
        return PipelineStatus(
            target_format=self.target_format,
            frame_samples=self.frame_samples,
            processors=processors,
            noise_suppression_active=active_ns > 0,
            speaker_isolation_active=any(p.kind == "speaker_isolation" and p.active for p in processors),
            provider_noise_suppression=self.provider_noise_suppression,
            warnings=tuple(warnings),
        )

    def _drain_frames(self) -> list[AudioFrame]:
        out = []
        n = self.frame_samples
        while len(self._pending) >= n:
            chunk, self._pending = self._pending[:n], self._pending[n:]
            out.append(self._make_frame(self._process(samples_to_pcm16(chunk)), complete=True))
        return out

    def _process(self, pcm: bytes) -> bytes:
        current = pcm
        for slot in self._slots:
            if slot.bypassed:
                continue
            try:
                result = slot.processor.process(current)
                if len(result) != len(current):
                    raise ValueError(f"returned {len(result)} bytes for a {len(current)}-byte frame")
                current = result
            except Exception as err:  # noqa: BLE001 - any engine failure must degrade, not kill the call
                if self.on_processor_error == "throw":
                    raise ProviderError(
                        code="unknown", provider=_PROVIDER, message=f"Processor {slot.processor.name} failed: {err}"
                    ) from err
                slot.bypassed = True
                slot.error = str(err)
        return current

    def _make_frame(self, payload: bytes, complete: bool) -> AudioFrame:
        t = self._template
        assert t is not None
        kind: TimestampKind = "none" if t.timestamp_ms is None else t.timestamp_kind
        index = self._emitted
        self._emitted += 1
        return AudioFrame(
            call_id=t.call_id,
            session_id=t.session_id,
            provider=t.provider,
            direction=t.direction,
            format=self.target_format,
            payload=payload,
            timestamp_kind=kind,
            timestamp_ms=None if kind == "none" or t.timestamp_ms is None else t.timestamp_ms + index * self.frame_ms,
            sequence_number=None,
            chunk=index,
            is_complete_frame=complete,
        )


def _health(processor: AudioProcessor) -> ProcessorHealth:
    health = getattr(processor, "health", None)
    return health() if callable(health) else ProcessorHealth(active=True)
