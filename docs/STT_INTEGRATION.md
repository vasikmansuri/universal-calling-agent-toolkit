# STT integration

## Contracts

- **`SttResult`:** what a provider adapter emits, already normalized:
  - `interim` (may still change);
  - `segment_final` (committed text, with a provider-stable `segmentKey` and optional `startMs`);
  - `turn_end` (the speaker finished).
  Confidence is `null` when the provider gives none. It is never invented.
- **`StreamingSttSession` / `StreamingSttAdapter`, `BatchSttAdapter`:** provider-neutral session and adapter interfaces.
- **`TranscriptReconciler`:** turns `SttResult`s into `TranscriptEvent`s for email capture and RAG:
  - interim text is shown on top of committed segments as `partial` events, and is never committed;
  - segment finals are de-duplicated by `segmentKey` and ordered by `startMs`;
  - each turn ends with exactly one `final` event (`revision 0`), whose confidence is the minimum segment confidence (or `null` if any segment lacks one);
  - a `turn_end` with no committed text emits nothing, so double end-of-speech signals are harmless.
- **`SttAudioBridge`:** feeds pipeline output to an STT session and enforces matching formats.

## Provider adapters

| Provider | Status | Notes |
|---|---|---|
| Deepgram live (`/v1/listen`) | **implemented, not integration-tested** | Message mapping follows Deepgram's docs: `Results` with `is_final=false` → `interim`; `is_final=true` → `segment_final` (key = start:duration in ms); `speech_final=true` → also `turn_end`; `UtteranceEnd` → `turn_end`. `deepgramListenUrl` builds the URL and enforces Deepgram's rule that `utterance_end_ms` needs `interim_results=true`. The host opens the socket with `Authorization: Token <key>`, so keys never pass through the toolkit |
| OpenAI Realtime transcription (`wss://api.openai.com/v1/realtime?intent=transcription`) | **implemented, not integration-tested** | See below |
| Google Cloud STT, Azure Speech, Amazon Transcribe, ElevenLabs, local engines | planned | Not implemented; each needs its API verified first. The `SttResult` contract is designed so their final/interim models map onto it |

### OpenAI Realtime transcription

`openai.ts` / `openai.py`, built from OpenAI's [realtime transcription](https://developers.openai.com/api/docs/guides/realtime-transcription), [WebSocket](https://developers.openai.com/api/docs/guides/realtime-websocket) and [session reference](https://developers.openai.com/api/docs/api-reference/realtime-client-events/session/update) docs. As with Deepgram, the host opens the socket (`Authorization: Bearer <key>`), so keys never pass through the toolkit.

- **Input:** 24 kHz mono PCM16 only, the one rate OpenAI documents for `audio/pcm`. Run the pipeline with `targetSampleRate: 24000`; `SttAudioBridge` rejects any other rate. OpenAI also lists `audio/pcmu` and `audio/pcma` but gives no sample rate for them, so they are not offered.
- **Setup:** `OPENAI_TRANSCRIPTION_URL` (no `model` in the URL), then `session.configure({ model, ... })` sends `session.update`. The model is required, with no default. `turnDetection: null` turns VAD off, and `language` and `languages` are mutually exclusive, as OpenAI requires.
- **Audio:** each frame goes out as `input_audio_buffer.append` with base64 audio, as a text message. `commit()` sends `input_audio_buffer.commit`. `finish()` commits by default, which is what you need with VAD off. With VAD on, pass `commitOnFinish: false` once everything is committed.
- **Results:**
  - `…transcription.delta` → `interim`. Deltas are appended per `item_id:content_index` for display only. OpenAI describes them as new text and does not say whether later deltas can revise earlier text, so interim text is never committed.
  - `…transcription.completed` → `segment_final` (key `item_id:content_index`), then `turn_end`. `transcript` is authoritative. One completed item is one committed audio unit, either a VAD turn or what the host committed. An address spoken across two VAD turns therefore arrives as two turns, unlike Deepgram's `is_final` segments.
  - `confidence` and `startMs` are always `null`: the events carry neither, and the toolkit does not invent them. Capabilities report no word timestamps, confidence or diarization.
  - `…transcription.failed` and `error` throw a `ProviderError` (`unknown`, with OpenAI's `code` or `type` as `providerCode`) from `receive()`. The failed item's partial text is dropped. Other events, including `input_audio_buffer.*`, are ignored.
- **Not implemented:** keep-alive (none documented), diarization models, `keywords`, `noise_reduction`, and WebRTC.
- **Not verified live:** see `RELEASE_CHECKLIST.md`.

Deepgram's `KeepAlive` and `CloseStream` messages are sent as `{"type":"KeepAlive"}` and `{"type":"CloseStream"}`. These two message shapes come from Deepgram's SDK conventions, not from the two documentation pages reviewed. Verify them with a live key before production use.

## Verified end to end (fixtures)

- **Audio reaches STT:** the bytes sent to the STT socket equal the processed pipeline frames (`stt.test.ts`, `test_stt.py`).
- **Email from fragments:** an address spoken across two Deepgram segments is captured as one candidate (`john.smith@gmail.com`).
