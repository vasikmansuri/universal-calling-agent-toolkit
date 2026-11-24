# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Vasik Mansuri
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from calling_agent_toolkit import AudioFormat, AudioFrame, ProviderError
from calling_agent_toolkit.audio_pipeline import AudioPipeline
from calling_agent_toolkit.email_capture import EmailCapture
from calling_agent_toolkit.providers.openai import (
    OPENAI_TRANSCRIPTION_URL,
    OpenAiResultMapper,
    OpenAiStreamingSession,
    openai_capabilities,
    openai_session_update,
)
from calling_agent_toolkit.stt import SttAudioBridge, SttCapabilities, TranscriptReconciler

CONTRACTS = Path(__file__).resolve().parents[3] / "contracts"
FIXTURE = json.loads((CONTRACTS / "fixtures/providers/openai/messages.json").read_text())

PCM_24K = AudioFormat("pcm_s16le", 24000, 1)
MULAW_8K = AudioFormat("mulaw", 8000, 1)


@pytest.mark.parametrize("case", FIXTURE["sequences"], ids=lambda c: c["name"])
def test_message_sequences(case: dict[str, Any]) -> None:
    mapper = OpenAiResultMapper()
    got = [[r.to_wire() for r in mapper.map(json.dumps(m))] for m in case["messages"]]
    assert got == case["expect"]


@pytest.mark.parametrize("case", FIXTURE["errors"], ids=lambda c: c["name"])
def test_message_errors(case: dict[str, Any]) -> None:
    raw = case.get("raw") or json.dumps(case["message"])
    with pytest.raises(ProviderError) as info:
        OpenAiResultMapper().map(raw)
    assert info.value.code == case["expectError"]
    assert info.value.provider == "openai"
    if "expectProviderCode" in case:
        assert info.value.provider_code == case["expectProviderCode"]
    if "expectMessage" in case:
        assert case["expectMessage"] in info.value.message


def test_mapper_accepts_a_parsed_object() -> None:
    msg = {"type": "conversation.item.input_audio_transcription.delta", "item_id": "i", "content_index": 0, "delta": "hi"}
    assert [r.to_wire() for r in OpenAiResultMapper().map(msg)] == [{"type": "interim", "text": "hi", "confidence": None}]


def test_failed_item_drops_its_partial_text() -> None:
    mapper = OpenAiResultMapper()
    key = {"item_id": "item_1", "content_index": 0}
    mapper.map({"type": "conversation.item.input_audio_transcription.delta", **key, "delta": "stale"})
    with pytest.raises(ProviderError):
        mapper.map({"type": "conversation.item.input_audio_transcription.failed", **key, "error": {"message": "x"}})
    fresh = mapper.map({"type": "conversation.item.input_audio_transcription.delta", **key, "delta": "fresh"})
    assert [r.to_wire() for r in fresh] == [{"type": "interim", "text": "fresh", "confidence": None}]


def test_never_invents_timestamps_or_confidence() -> None:
    out = []
    for case in FIXTURE["sequences"]:
        mapper = OpenAiResultMapper()
        for m in case["messages"]:
            out += mapper.map(json.dumps(m))
    assert out
    assert all(r.type == "turn_end" or r.confidence is None for r in out)
    assert all(r.start_ms is None for r in out if r.type == "segment_final")


def test_transcription_url_has_no_model() -> None:
    assert OPENAI_TRANSCRIPTION_URL == "wss://api.openai.com/v1/realtime?intent=transcription"


def _options(raw: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in raw.items() if k != "turnDetection"}
    if "turnDetection" in raw:
        out["turn_detection"] = raw["turnDetection"]
    return out


@pytest.mark.parametrize("case", FIXTURE["sessionUpdates"], ids=lambda c: c["name"])
def test_session_update(case: dict[str, Any]) -> None:
    assert openai_session_update(**_options(case["options"])) == case["expect"]


@pytest.mark.parametrize("case", FIXTURE["sessionUpdateErrors"], ids=lambda c: c["name"])
def test_session_update_errors(case: dict[str, Any]) -> None:
    with pytest.raises(ProviderError) as info:
        openai_session_update(**_options(case["options"]))
    assert info.value.code == case["expectError"]


def test_capabilities_are_honest() -> None:
    assert openai_capabilities() == SttCapabilities(
        provider="openai",
        interim_results=True,
        word_timestamps=False,
        confidence=False,
        diarization=False,
        input_formats=(PCM_24K,),
    )


class FakeSocket:
    def __init__(self) -> None:
        self.binary: list[bytes] = []
        self.text: list[str] = []
        self.closed = False

    async def send_binary(self, data: bytes) -> None:
        self.binary.append(data)

    async def send_text(self, data: str) -> None:
        self.text.append(data)

    async def close(self) -> None:
        self.closed = True


def frame(fmt: AudioFormat, payload: bytes) -> AudioFrame:
    return AudioFrame("c", "s", "twilio", "inbound", fmt, payload, "stream_relative_ms", 0)


@pytest.mark.parametrize("fmt", [AudioFormat("pcm_s16le", 16000, 1), MULAW_8K, AudioFormat("pcm_s16le", 24000, 2)])
def test_session_rejects_other_formats(fmt: AudioFormat) -> None:
    with pytest.raises(ProviderError) as info:
        OpenAiStreamingSession(FakeSocket(), fmt)
    assert info.value.code == "unsupported_format"


async def test_configure_then_base64_audio() -> None:
    socket = FakeSocket()
    session = OpenAiStreamingSession(socket, PCM_24K)
    await session.configure("gpt-4o-transcribe", turn_detection=None)
    await session.send_audio(frame(PCM_24K, bytes([1, 2, 3, 4])))
    assert socket.binary == []
    assert [json.loads(t) for t in socket.text] == [
        openai_session_update("gpt-4o-transcribe", turn_detection=None),
        {"type": "input_audio_buffer.append", "audio": "AQIDBA=="},
    ]


async def test_wrong_format_frame_is_rejected_without_sending() -> None:
    socket = FakeSocket()
    with pytest.raises(ProviderError) as info:
        await OpenAiStreamingSession(socket, PCM_24K).send_audio(frame(MULAW_8K, b"\x00"))
    assert info.value.code == "unsupported_format"
    assert socket.text == []


async def test_commit_on_finish_default_and_override() -> None:
    a = FakeSocket()
    await OpenAiStreamingSession(a, PCM_24K).finish()
    assert a.text == ['{"type": "input_audio_buffer.commit"}']
    b = FakeSocket()
    vad = OpenAiStreamingSession(b, PCM_24K, commit_on_finish=False)
    await vad.finish()
    assert b.text == []
    await vad.commit()
    assert b.text == ['{"type": "input_audio_buffer.commit"}']


async def test_receive_surfaces_provider_errors_and_keeps_working() -> None:
    session = OpenAiStreamingSession(FakeSocket(), PCM_24K)
    with pytest.raises(ProviderError, match="boom"):
        session.receive(json.dumps({"type": "error", "error": {"type": "server_error", "message": "boom"}}))
    done = {"type": "conversation.item.input_audio_transcription.completed", "item_id": "i", "content_index": 0, "transcript": "ok"}
    session.receive(json.dumps(done))
    await session.close()
    assert [r.type async for r in session.results()] == ["segment_final", "turn_end"]


async def test_close_closes_the_connection() -> None:
    socket = FakeSocket()
    await OpenAiStreamingSession(socket, PCM_24K).close()
    assert socket.closed


async def test_stt_receives_the_pipelines_24k_frames() -> None:
    socket = FakeSocket()
    session = OpenAiStreamingSession(socket, PCM_24K, commit_on_finish=False)
    bridge = SttAudioBridge(AudioPipeline(target_sample_rate=24000), session, PCM_24K)
    silence = frame(MULAW_8K, b"\xff" * 160)
    await bridge.push(silence)
    await bridge.push(silence)
    await bridge.end()
    sent = [json.loads(t) for t in socket.text]
    assert all(m["type"] == "input_audio_buffer.append" for m in sent)
    assert [len(base64.b64decode(m["audio"])) for m in sent] == [960, 960]


def test_bridge_rejects_a_pipeline_that_is_not_24k() -> None:
    with pytest.raises(ProviderError):
        SttAudioBridge(AudioPipeline(target_sample_rate=16000), OpenAiStreamingSession(FakeSocket(), PCM_24K), PCM_24K)


async def test_email_from_streamed_deltas_and_completed() -> None:
    session = OpenAiStreamingSession(FakeSocket(), PCM_24K)
    delta = "conversation.item.input_audio_transcription.delta"
    for text in ("my email is john dot ", "smith at gmail dot com"):
        session.receive(json.dumps({"type": delta, "item_id": "item_1", "content_index": 0, "delta": text}))
    session.receive(
        json.dumps(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_1",
                "content_index": 0,
                "transcript": "my email is john dot smith at gmail dot com",
            }
        )
    )
    await session.close()
    reconciler = TranscriptReconciler("t1", "call-1", now=lambda: 0)
    capture = EmailCapture()
    results = []
    async for result in session.results():
        for event in reconciler.accept(result):
            results += await capture.handle(event)
    assert results == [
        {"type": "candidate", "email": "john.smith@gmail.com", "literal": False, "turnId": "turn-1", "replacesConfirmed": False}
    ]
