# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Vasik Mansuri
"""OpenAI Realtime transcription (wss://api.openai.com/v1/realtime?intent=transcription).

Docs: https://developers.openai.com/api/docs/guides/realtime-transcription,
      https://developers.openai.com/api/docs/guides/realtime-websocket,
      https://developers.openai.com/api/docs/api-reference/realtime-client-events/session/update,
      https://developers.openai.com/api/docs/api-reference/realtime-server-events/error,
      https://developers.openai.com/api/docs/api-reference/realtime-server-events/conversation/item/input_audio_transcription/failed
The host opens the WebSocket (with ``Authorization: Bearer <key>``) and passes it in as an SttConnection,
so API keys never pass through this module.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

from ..errors import ProviderError
from ..parse import JsonObject, MessageReader
from ..queue import BoundedQueue
from ..stt import SttCapabilities, SttResult
from ..types import AudioFormat, AudioFrame
from .deepgram import SttConnection

OPENAI = "openai"

#: Server-side transcription endpoint. The docs say to omit ``model`` from the URL and set it in ``session.update``.
OPENAI_TRANSCRIPTION_URL = "wss://api.openai.com/v1/realtime?intent=transcription"

#: The only input OpenAI documents for ``audio/pcm``: "Only a 24kHz sample rate is supported."
_INPUT_FORMAT = AudioFormat("pcm_s16le", 24000, 1)

TurnDetection = Literal["server_vad", "semantic_vad"] | None


class _Unset:
    """Sentinel: the turn_detection field is not sent at all (distinct from None, which sends null)."""


UNSET = _Unset()


def _require_input_format(fmt: AudioFormat) -> None:
    if fmt != _INPUT_FORMAT:
        raise ProviderError(
            code="unsupported_format",
            provider=OPENAI,
            message=f"OpenAI transcription takes {_INPUT_FORMAT.encoding}/{_INPUT_FORMAT.sample_rate} mono, "
            f"got {fmt.encoding}/{fmt.sample_rate}/{fmt.channels}ch. Set the pipeline's target_sample_rate to 24000.",
        )


def openai_session_update(
    model: str,
    *,
    language: str | None = None,
    languages: Sequence[str] | None = None,
    prompt: str | None = None,
    turn_detection: TurnDetection | _Unset = UNSET,
) -> dict[str, Any]:
    """Build the ``session.update`` event that configures a transcription session.

    ``turn_detection``: left unset, the field is not sent; ``None`` sends null (VAD off, the host commits turns);
    otherwise the VAD type, with OpenAI's default settings.
    """
    if not model:
        raise ProviderError(code="invalid_configuration", provider=OPENAI, message="A transcription model is required.")
    if language is not None and languages is not None:
        raise ProviderError(
            code="invalid_configuration",
            provider=OPENAI,
            message="Pass either language or languages, not both: OpenAI says not to send the singular field with languages.",
        )
    if languages is not None and len(languages) == 0:
        raise ProviderError(code="invalid_configuration", provider=OPENAI, message="languages needs at least one code.")
    transcription: dict[str, Any] = {"model": model}
    if language is not None:
        transcription["language"] = language
    if languages is not None:
        transcription["languages"] = list(languages)
    if prompt is not None:
        transcription["prompt"] = prompt
    audio_input: dict[str, Any] = {"format": {"type": "audio/pcm", "rate": _INPUT_FORMAT.sample_rate}, "transcription": transcription}
    if turn_detection is None:
        audio_input["turn_detection"] = None
    elif not isinstance(turn_detection, _Unset):
        audio_input["turn_detection"] = {"type": turn_detection}
    return {"type": "session.update", "session": {"type": "transcription", "audio": {"input": audio_input}}}


def openai_capabilities() -> SttCapabilities:
    return SttCapabilities(
        provider=OPENAI,
        interim_results=True,
        # The docs list no word timestamps or confidence for transcription events, so none are reported.
        word_timestamps=False,
        confidence=False,
        diarization=False,
        input_formats=(_INPUT_FORMAT,),
    )


class OpenAiResultMapper:
    """Translate OpenAI server events into SttResults.

    - ``...input_audio_transcription.delta`` -> ``interim``. Deltas are appended per item for display only. OpenAI
      calls them "newly available transcript text" and does not say whether later deltas may revise earlier text,
      so the interim text is never committed.
    - ``...input_audio_transcription.completed`` -> ``segment_final`` (key ``item_id:content_index``) then
      ``turn_end``. ``transcript`` is authoritative. Each completed item is one committed audio unit (a VAD turn,
      or what the host committed).
    - ``...input_audio_transcription.failed`` and ``error`` raise a ProviderError. Other events yield [].
    Confidence and start time are None: the events carry neither.
    """

    def __init__(self) -> None:
        self._pending: dict[str, str] = {}
        self._r = MessageReader(OPENAI)

    def map(self, message: str | JsonObject) -> list[SttResult]:
        msg = self._r.json(message, "type") if isinstance(message, str) else message
        kind = msg.get("type")
        if kind == "conversation.item.input_audio_transcription.delta":
            key = self._item_key(msg)
            text = self._pending.get(key, "") + self._r.string(msg, "delta")
            self._pending[key] = text
            return [SttResult("interim", text, None)]
        if kind == "conversation.item.input_audio_transcription.completed":
            key = self._item_key(msg)
            text = self._r.string(msg, "transcript")
            self._pending.pop(key, None)
            return [SttResult("segment_final", text, None, key, None), SttResult("turn_end")]
        if kind == "conversation.item.input_audio_transcription.failed":
            key = self._item_key(msg)
            self._pending.pop(key, None)
            raise self._provider_error(self._r.object(msg, "error"), f"Transcription failed for item {key}")
        if kind == "error":
            raise self._provider_error(self._r.object(msg, "error"), "OpenAI reported an error")
        return []

    def _item_key(self, msg: JsonObject) -> str:
        return f"{self._r.id(msg, 'item_id')}:{self._r.int(msg, 'content_index')}"

    @staticmethod
    def _provider_error(error: JsonObject, prefix: str) -> ProviderError:
        def detail(key: str) -> str | None:
            value = error.get(key)
            return value if isinstance(value, str) else None

        return ProviderError(
            code="unknown",
            provider=OPENAI,
            message=f"{prefix}: {detail('message') or 'no message'}",
            provider_code=detail("code") or detail("type"),
        )


class OpenAiStreamingSession:
    def __init__(
        self,
        connection: SttConnection,
        fmt: AudioFormat,
        *,
        max_queued_results: int = 1000,
        commit_on_finish: bool = True,
    ) -> None:
        """``commit_on_finish``: send ``input_audio_buffer.commit`` from ``finish()``. Default True, which is what you
        need when turn detection is off. With VAD on, set False if everything is already committed: committing an
        empty buffer makes OpenAI send an ``error``."""
        _require_input_format(fmt)
        self.format = fmt
        self._connection = connection
        self._commit_on_finish = commit_on_finish
        self._mapper = OpenAiResultMapper()
        self._queue: BoundedQueue[SttResult] = BoundedQueue(max_queued_results, "reject", OPENAI)

    async def configure(
        self,
        model: str,
        *,
        language: str | None = None,
        languages: Sequence[str] | None = None,
        prompt: str | None = None,
        turn_detection: TurnDetection | _Unset = UNSET,
    ) -> None:
        """Send ``session.update``. Call once after the socket opens, before any audio."""
        event = openai_session_update(model, language=language, languages=languages, prompt=prompt, turn_detection=turn_detection)
        await self._connection.send_text(json.dumps(event))

    def receive(self, message: str) -> None:
        """Feed each text message from the OpenAI socket here. Raises ProviderError for ``error`` and transcription ``failed`` events."""
        for result in self._mapper.map(message):
            self._queue.push(result)

    async def send_audio(self, frame: AudioFrame) -> None:
        if frame.format != self.format:
            raise ProviderError(
                code="unsupported_format",
                provider=OPENAI,
                message=f"Session expects {self.format.encoding}/{self.format.sample_rate}, "
                f"got {frame.format.encoding}/{frame.format.sample_rate}.",
            )
        audio = base64.b64encode(frame.payload).decode("ascii")
        await self._connection.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": audio}))

    async def commit(self) -> None:
        """Commit the audio buffered so far, ending the current turn when turn detection is off."""
        await self._connection.send_text(json.dumps({"type": "input_audio_buffer.commit"}))

    async def finish(self) -> None:
        if self._commit_on_finish:
            await self.commit()

    def results(self) -> AsyncIterator[SttResult]:
        return self._queue

    async def close(self) -> None:
        self._queue.close()
        await self._connection.close()
