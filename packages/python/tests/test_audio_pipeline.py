# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Vasik Mansuri
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pytest

from calling_agent_toolkit import AudioFormat, AudioFrame, ProviderError
from calling_agent_toolkit.audio_pipeline import (
    AudioPipeline,
    LinearResampler,
    ProcessorHealth,
    pcm16_to_samples,
    samples_to_pcm16,
)

CONTRACTS = Path(__file__).resolve().parents[3] / "contracts"
CASES = json.loads((CONTRACTS / "fixtures/audio/resample.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_resampler_vectors(case: dict[str, Any]) -> None:
    one = LinearResampler(case["from"], case["to"])
    assert one.push(case["input"]) + one.flush() == case["output"]
    streamed = LinearResampler(case["from"], case["to"])
    out: list[int] = []
    start = 0
    for end in [*case["chunks"], len(case["input"])]:
        out += streamed.push(case["input"][start:end])
        start = end
    assert out + streamed.flush() == case["output"]


def test_streaming_matches_one_shot() -> None:
    data = [round(math.sin(i / 7) * 20000) for i in range(1000)]
    a = LinearResampler(8000, 16000)
    whole = a.push(data) + a.flush()
    b = LinearResampler(8000, 16000)
    parts: list[int] = []
    for i in range(0, len(data), 37):
        parts += b.push(data[i : i + 37])
    assert parts + b.flush() == whole
    assert len(whole) == 2000


MULAW_8K = AudioFormat("mulaw", 8000, 1)
PCM_16K = AudioFormat("pcm_s16le", 16000, 1)


def frame(payload: bytes, **overrides: Any) -> AudioFrame:
    base = AudioFrame(
        call_id="c", session_id="s", provider="test", direction="inbound", format=MULAW_8K, payload=payload,
        timestamp_kind="stream_relative_ms", timestamp_ms=100, sequence_number=1, chunk=1,
    )
    return dataclasses.replace(base, **overrides)


@dataclasses.dataclass
class Gain:
    factor: int
    kind: Any = "noise_suppression"
    name: str = "gain"
    sample_rate: int = 16000
    frame_samples: int = 320
    active: bool = True

    def process(self, pcm: bytes) -> bytes:
        return samples_to_pcm16([s * self.factor for s in pcm16_to_samples(pcm)])

    def health(self) -> ProcessorHealth:
        return ProcessorHealth(self.active)


class Broken(Gain):
    def process(self, pcm: bytes) -> bytes:
        raise RuntimeError("model missing")


def test_mulaw_8k_becomes_20ms_16k_frames() -> None:
    pipeline = AudioPipeline(target_sample_rate=16000)
    silence = b"\xff" * 160
    out = pipeline.push(frame(silence)) + pipeline.push(frame(silence)) + pipeline.flush()
    assert [len(f.payload) for f in out] == [640, 640]
    assert all(f.is_complete_frame and f.format == PCM_16K for f in out)
    assert [f.timestamp_ms for f in out] == [100, 120]


def test_alaw_8k_becomes_20ms_16k_frames() -> None:
    pipeline = AudioPipeline(target_sample_rate=16000)
    alaw = AudioFormat("alaw", 8000, 1)
    quiet_positive = b"\xd5" * 160  # A-law 0xD5 decodes to +8
    out = pipeline.push(frame(quiet_positive, format=alaw)) + pipeline.push(frame(quiet_positive, format=alaw)) + pipeline.flush()
    assert [len(f.payload) for f in out] == [640, 640]
    assert all(f.is_complete_frame and f.format == PCM_16K for f in out)
    assert all(s == 8 for f in out for s in pcm16_to_samples(f.payload))


def test_processors_run_and_report_active() -> None:
    pipeline = AudioPipeline(target_sample_rate=16000, processors=[Gain(2)])
    [out] = pipeline.push(frame(samples_to_pcm16([10] * 320), format=PCM_16K))
    assert pcm16_to_samples(out.payload) == [20] * 320
    status = pipeline.status()
    assert status.noise_suppression_active and status.warnings == ()


def test_failing_processor_is_bypassed_and_not_claimed() -> None:
    pipeline = AudioPipeline(target_sample_rate=16000, processors=[Broken(1)])
    pcm = samples_to_pcm16([7] * 320)
    [out] = pipeline.push(frame(pcm, format=PCM_16K))
    assert out.payload == pcm
    status = pipeline.status()
    assert not status.noise_suppression_active
    assert (status.processors[0].bypassed, status.processors[0].active, status.processors[0].error) == (True, False, "model missing")
    assert "processor_bypassed" in status.warnings


def test_throw_mode() -> None:
    pipeline = AudioPipeline(target_sample_rate=16000, processors=[Broken(1)], on_processor_error="throw")
    with pytest.raises(ProviderError, match="failed"):
        pipeline.push(frame(samples_to_pcm16([7] * 320), format=PCM_16K))


def test_health_overrides_presence() -> None:
    assert not AudioPipeline(target_sample_rate=16000, processors=[Gain(1, active=False)]).status().noise_suppression_active


def test_stacked_noise_suppression_warning() -> None:
    assert AudioPipeline(target_sample_rate=16000, processors=[Gain(1)], provider_noise_suppression=True).status().warnings == (
        "stacked_noise_suppression",
    )
    assert AudioPipeline(target_sample_rate=16000, processors=[Gain(1), Gain(1)]).status().warnings == (
        "stacked_noise_suppression",
    )


def test_speaker_isolation_reported_separately() -> None:
    status = AudioPipeline(target_sample_rate=16000, processors=[Gain(1, kind="speaker_isolation")]).status()
    assert (status.noise_suppression_active, status.speaker_isolation_active) == (False, True)


def test_rejects_mismatched_processor_shape() -> None:
    with pytest.raises(ProviderError, match="480 samples at 48000"):
        AudioPipeline(target_sample_rate=16000, processors=[Gain(1, sample_rate=48000, frame_samples=480)])


def test_direction_codec_and_rate_change() -> None:
    pipeline = AudioPipeline(target_sample_rate=16000)
    assert pipeline.push(frame(b"\x00" * 160, direction="outbound")) == []
    with pytest.raises(ProviderError, match="No codec"):
        pipeline.push(frame(b"\x00" * 160, format=AudioFormat("opus", 8000, 1)))  # type: ignore[arg-type]
    pipeline.push(frame(b"\x00" * 160))
    with pytest.raises(ProviderError, match="changed mid-stream"):
        pipeline.push(frame(b"\x00" * 320, format=PCM_16K))


def test_downsampling_warning_and_tail() -> None:
    pipeline = AudioPipeline(target_sample_rate=8000)
    out = pipeline.push(frame(samples_to_pcm16([1] * 100), format=PCM_16K)) + pipeline.flush()
    assert len(out) == 1 and not out[0].is_complete_frame
    assert "downsampling_without_antialiasing" in pipeline.status().warnings
