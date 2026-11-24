# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Vasik Mansuri
from __future__ import annotations

import json
import math
import os
import struct
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from calling_agent_toolkit import AudioFormat, AudioFrame, ProviderError
from calling_agent_toolkit.audio_pipeline import AudioPipeline
from calling_agent_toolkit.processors.rnnoise import RnnoiseProcessor

CONTRACTS = Path(__file__).resolve().parents[3] / "contracts"
FIXTURE = json.loads((CONTRACTS / "fixtures/audio/rnnoise.json").read_text())
FRAME_BYTES = FIXTURE["frameSamples"] * 2

# The real-engine tests need librnnoise; they run when RNNOISE_LIBRARY points at it and are skipped otherwise.
NATIVE = os.environ.get("RNNOISE_LIBRARY")
needs_native = pytest.mark.skipif(not NATIVE, reason="set RNNOISE_LIBRARY to the librnnoise shared library")


def noise_frames(seed: int, amplitude: int, frames: int) -> Iterator[bytes]:
    for _ in range(frames):
        samples = []
        for _ in range(FIXTURE["frameSamples"]):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            samples.append(math.floor((seed / 2**32 - 0.5) * amplitude + 0.5))
        yield struct.pack(f"<{len(samples)}h", *samples)


def energy(pcm: bytes) -> int:
    return sum(s * s for s in struct.unpack(f"<{len(pcm) // 2}h", pcm))


class FakeLib:
    """Stands in for librnnoise: same C-style calls, real ctypes float arrays, a transform chosen by the test."""

    def __init__(self, transform: Callable[[float], float], *, fail_create: bool = False) -> None:
        self.transform = transform
        self.fail_create = fail_create
        self.destroyed: list[object] = []

    def rnnoise_create(self, model: object) -> int:
        assert model is None
        return 0 if self.fail_create else 7

    def rnnoise_destroy(self, state: object) -> None:
        self.destroyed.append(state)

    def rnnoise_process_frame(self, state: object, out: Any, inp: Any) -> float:
        assert state == 7
        for i in range(480):
            out[i] = self.transform(inp[i])
        return 0.25


def test_declares_the_shape_rnnoise_needs() -> None:
    rnnoise = RnnoiseProcessor(library=FakeLib(lambda s: s))
    assert (rnnoise.name, rnnoise.kind, rnnoise.sample_rate, rnnoise.frame_samples) == (
        "rnnoise",
        "noise_suppression",
        FIXTURE["sampleRate"],
        FIXTURE["frameSamples"],
    )
    assert rnnoise.health().active is True


def test_converts_pcm16_to_float_and_back_with_rounding_and_clamping() -> None:
    rnnoise = RnnoiseProcessor(library=FakeLib(lambda s: s * 2.5))
    samples = [1, -1, 3, -3, 20000, -20000, 0] + [0] * 473
    out = struct.unpack("<480h", rnnoise.process(struct.pack("<480h", *samples)))
    # 2.5 -> 3, -2.5 -> -2 (round half up), 7.5 -> 8, -7.5 -> -7, then clamped
    assert list(out[:7]) == [3, -2, 8, -7, 32767, -32768, 0]
    assert rnnoise.voice_probability == 0.25


def test_rejects_frames_that_are_not_480_samples() -> None:
    rnnoise = RnnoiseProcessor(library=FakeLib(lambda s: s))
    for size in (0, 2, FRAME_BYTES - 2, FRAME_BYTES + 2):
        with pytest.raises(ProviderError) as info:
            rnnoise.process(bytes(size))
        assert info.value.code == "malformed_message"


def test_dispose_frees_the_engine_and_refuses_further_use() -> None:
    lib = FakeLib(lambda s: s)
    rnnoise = RnnoiseProcessor(library=lib)
    rnnoise.dispose()
    rnnoise.dispose()
    assert lib.destroyed == [7]
    assert rnnoise.health().active is False
    with pytest.raises(ProviderError) as info:
        rnnoise.process(bytes(FRAME_BYTES))
    assert info.value.code == "session_closed"


def test_inactive_when_the_library_cannot_allocate() -> None:
    rnnoise = RnnoiseProcessor(library=FakeLib(lambda s: s, fail_create=True))
    health = rnnoise.health()
    assert health.active is False
    assert health.detail is not None and "rnnoise_create" in health.detail
    with pytest.raises(ProviderError) as info:
        rnnoise.process(bytes(FRAME_BYTES))
    assert info.value.code == "invalid_configuration"


def test_inactive_when_the_library_file_is_missing(tmp_path: Path) -> None:
    missing = tmp_path / "librnnoise.so"
    rnnoise = RnnoiseProcessor(missing)
    health = rnnoise.health()
    assert health.active is False
    assert health.detail is not None and "RNNoise engine not loaded" in health.detail and str(missing) in health.detail


def test_inactive_when_no_library_can_be_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RNNOISE_LIBRARY", raising=False)
    monkeypatch.setattr("ctypes.util.find_library", lambda name: None)
    health = RnnoiseProcessor().health()
    assert health.active is False
    assert health.detail is not None and "RNNOISE_LIBRARY" in health.detail


def test_library_path_environment_variable_is_used(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RNNOISE_LIBRARY", str(tmp_path / "from-env.so"))
    detail = RnnoiseProcessor().health().detail
    assert detail is not None and "from-env.so" in detail


MULAW_8K = AudioFormat("mulaw", 8000, 1)


def mulaw_frame(payload: bytes) -> AudioFrame:
    return AudioFrame("c", "s", "twilio", "inbound", MULAW_8K, payload, "stream_relative_ms", 0)


def call_audio() -> list[bytes]:
    return [f[:80] for f in noise_frames(9, 20000, 100)]  # 80 mu-law bytes = 10 ms at 8 kHz


def test_pipeline_runs_at_48k_10ms_and_reports_suppression_active() -> None:
    pipeline = AudioPipeline(target_sample_rate=48000, frame_ms=10, processors=[RnnoiseProcessor(library=FakeLib(lambda s: s))])
    out = [f for payload in call_audio() for f in pipeline.push(mulaw_frame(payload))]
    assert len(out) > 90
    assert all(len(f.payload) == FRAME_BYTES and f.format.sample_rate == 48000 for f in out)
    status = pipeline.status()
    assert status.noise_suppression_active is True
    assert status.warnings == ()


def test_pipeline_refuses_a_16k_configuration() -> None:
    with pytest.raises(ProviderError, match="480 samples at 48000"):
        AudioPipeline(target_sample_rate=16000, processors=[RnnoiseProcessor(library=FakeLib(lambda s: s))])


def test_pipeline_never_claims_suppression_when_the_engine_is_missing(tmp_path: Path) -> None:
    pipeline = AudioPipeline(target_sample_rate=48000, frame_ms=10, processors=[RnnoiseProcessor(tmp_path / "missing.so")])
    out = [f for payload in call_audio() for f in pipeline.push(mulaw_frame(payload))]
    assert len(out) > 90
    status = pipeline.status()
    assert status.noise_suppression_active is False
    assert "processor_bypassed" in status.warnings
    assert (status.processors[0].name, status.processors[0].active, status.processors[0].bypassed) == ("rnnoise", False, True)


@needs_native
def test_real_engine_suppresses_noise_only_input() -> None:
    rnnoise = RnnoiseProcessor(NATIVE)
    assert rnnoise.health().active is True
    cfg = FIXTURE["noiseOnly"]
    in_energy = out_energy = 0
    for n, frame in enumerate(noise_frames(cfg["seed"], cfg["amplitude"], cfg["frames"])):
        out = rnnoise.process(frame)
        assert len(out) == FRAME_BYTES
        if n >= cfg["warmupFrames"]:
            in_energy += energy(frame)
            out_energy += energy(out)
    assert out_energy / in_energy < cfg["maxEnergyRatio"]
    assert rnnoise.voice_probability is not None and 0 <= rnnoise.voice_probability <= 1
    rnnoise.dispose()


@needs_native
def test_real_engine_keeps_state_per_instance() -> None:
    frames = list(noise_frames(3, 6000, 5))
    first = RnnoiseProcessor(NATIVE)
    expected = [first.process(f) for f in frames]
    second = RnnoiseProcessor(NATIVE)
    assert [second.process(f) for f in frames] == expected
    first.dispose()
    second.dispose()
