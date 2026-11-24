# Changelog

## Unreleased (0.0.0, not published)

### Added
- **Contracts:** JSON Schema for the core types, plus shared fixtures that both language packages run.
- **Core:** typed errors, capability declaration and checks, adapter interfaces, provider registry, bounded queues, G.711 μ-law and A-law codecs (bit-exact with CPython `audioop`).
- **Twilio:**
  - Media Streams adapter (in/out, marks, clear);
  - webhook and handshake signature validation (checked against twilio-python);
  - REST call control: calls, hangup, transfer, unidirectional streams, status callbacks.
- **Plivo:**
  - Audio Streams adapter (in/out, checkpoints, clear, DTMF);
  - V3 signature validation (checked against plivo-python);
  - REST call control: calls, hangup, transfer, DTMF, streams with native noise cancellation, callbacks.
- **Telnyx:** media streaming adapter.
- **Vonage:** WebSocket adapter, with handshake authorization (custom header or HS256 JWT).
- **Generic transports:** configurable JSON WebSocket parser; raw PCM WebSocket session; RTP parser, packetizer and session; WAV reading and writing; recording framing.
- **Conformance suite:** reusable, framework-agnostic, for media adapters.
- **Audio pipeline:** streaming resampler, framing, processor chain with honest status and stacking warnings.
- **Email capture:** spoken-form normalization, clarification instead of guessing, refusal and deferral, confirmation protection, persistence, isolation.
- **STT:** contracts, transcript reconciler, pipeline-to-STT bridge, Deepgram live adapter.
- **RAG:** safe retriever (tenant and ACL, timeout, cancellation, citations, untrusted marking), retrieval coordinator (final-only, stale protection), LangChain and LlamaIndex wrappers, in-memory retriever.
- **Examples and benchmarks:** end-to-end examples and micro-benchmarks in both languages.
