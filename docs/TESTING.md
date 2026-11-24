# Testing

## Shared contract tests

`contracts/` is the source of truth for cross-language behavior. Both packages run the same fixtures:

| Fixture | Checks |
|---|---|
| `fixtures/core/schema-cases.json` | Valid and invalid documents for each schema type |
| `fixtures/core/capability-checks.json` | `requireCapability` outcomes and error codes |
| `fixtures/core/frame-validation.json` | Payload size, alignment and sample-rate validation |
| `fixtures/core/g711-mulaw.json` | All 256 μ-law decode values and encode vectors |

The test suites also check that each language's enums (`AudioEncoding`, `Capability`, `ProviderErrorCode`) are identical to the schema's.

The G.711 vectors were generated from `audioop-lts` (CPython's `audioop`), an implementation independent of both packages. To regenerate them:

```sh
uv run --no-project --python 3.13 --with audioop-lts python -I -c '
import audioop, json, struct
dec=[struct.unpack("<h",audioop.ulaw2lin(bytes([i]),2))[0] for i in range(256)]
samples=[-32768,-32124,-10000,-1000,-100,-1,0,1,2,31,100,1000,4096,10000,32124,32635,32767]
enc=[[s,audioop.lin2ulaw(struct.pack("<h",s),2)[0]] for s in samples]
json.dump({"description":"ITU-T G.711 mu-law reference vectors generated with audioop-lts (CPython audioop). decode[i] is the PCM16 value of mu-law byte i; encode entries are [pcm16, mulaw].","source":"audioop-lts on PyPI","decode":dec,"encode":enc},open("contracts/fixtures/core/g711-mulaw.json","w"),separators=(",",":"))'
```

Note: CPython encodes on 14-bit samples (`pcm >> 2`, arithmetic shift), so -1 encodes to `0x7E`. The common bias-0x84 16-bit encoder gives `0x7F`. Both packages follow CPython.

## Running

```sh
# TypeScript (Node >= 20)
cd packages/typescript && npm ci && npm run typecheck && npm test && npm run build

# Python (>= 3.10)
cd packages/python && uv sync && uv run pytest && uv run mypy
```

## Provider fixtures

| Fixture | Checks |
|---|---|
| `fixtures/providers/{twilio,plivo}/signatures.json` | Webhook and handshake signatures. Every expected result was produced by the **official SDK** (twilio-python `RequestValidator`, plivo-python `signature_v3`), so the toolkit is tested against vendor code, not against itself. Regenerate with `contracts/fixtures/providers/generate_signature_fixtures.py`. |
| `fixtures/providers/{twilio,plivo}/sessions.json` | Scripted sessions: inbound messages, outbound actions, the exact JSON each adapter must send, the error code for each bad step, and the full normalized event stream. Each emitted event is also validated against `TelephonyEvent` in the schema. |

## Conformance suite

`mediaConformanceChecks(target)` (TS) and `media_conformance_checks(target)` (Python) are exported and framework-agnostic. A third-party adapter supplies a target with three message builders (start, media and optional stop) and gets these checks:
- every capability declared;
- non-JSON and invalid UTF-8 rejected;
- media before start rejected without changing state;
- start and media normalized with matching ids and format;
- payload size limit enforced;
- audio dropped, not thrown, when the consumer lags;
- idempotent close emitting exactly one `stream_stopped`;
- concurrent session isolation;
- playback refused on a unidirectional stream with nothing written;
- provider stop ending the stream.

Twilio and Plivo run it in both languages.

## Other shared fixtures

| Fixture | Checks |
|---|---|
| `fixtures/providers/{twilio,plivo}/rest.json` | Exact REST requests (method, URL, auth, body) and error mapping, against a mock HTTP client |
| `fixtures/providers/{telnyx,vonage}/sessions.json` | Telnyx and Vonage sessions, including Vonage binary frames |
| `fixtures/providers/deepgram/messages.json` | Deepgram live messages → normalized STT results |
| `fixtures/stt/reconciler.json` | Interim, final and turn-end reconciliation |
| `fixtures/email/{extraction,sessions}.json` | 24 extraction cases and 11 multi-step capture sessions |
| `fixtures/audio/resample.json` | Integer-exact resampler vectors, one-shot and streamed |
| `fixtures/audio/rnnoise.json` | RNNoise processor behaviour (property check, not exact samples; engines differ). Noise-only input must be attenuated below a set energy ratio |
| `fixtures/rag/retrieval.json` | Tenant, ACL, filters, ranking, citations and escaped context formatting |
| `fixtures/transports/rtp-wav.json` | 10 RTP packet vectors and 8 WAV vectors |

## Results recorded so far

2026-10-23, macOS arm64 (Apple M4):

| Runtime | Result |
|---|---|
| Node 26.10 / TypeScript 5 | typecheck clean; **268/268** tests passed (9 files); build OK |
| Python 3.13 | **271/271** passed (includes the real `langchain-core` test) |
| Python 3.10 | **271/271** passed; `mypy --strict` clean (22 source files) |
| Packaging | The wheel installs into an empty Python 3.12 venv with **no runtime dependencies**, and every module imports. The npm tarball installs into an empty project with **no dependencies**, and all 7 entry points import |
| Examples | `examples/typescript/end-to-end.mjs` and `examples/python/end_to_end.py` produce identical results: 50 frames / 32,000 bytes reach STT, the email candidate is captured, and the support question is grounded with a citation |

Mutation check (Phase 3a): two bugs injected into the Python adapters failed 3 and 2 tests respectively.

## Benchmarks

Measured with `benchmarks/typescript/bench.mjs` and `benchmarks/python/bench.py` on 2026-10-23, Apple M4. These are single-threaded micro-benchmarks, not a capacity guarantee:

| Operation | TypeScript (Node 26) | Python 3.10 |
|---|---|---|
| μ-law decode, 20 ms frame | 0.25 µs | 4.8 µs |
| Resample 8 → 16 kHz, 20 ms frame | 6.6 µs | 70.5 µs |
| AudioPipeline (decode + resample + framing), 20 ms frame | 7.9 µs | 85.2 µs |
| Twilio media message → AudioFrame | 3.4 µs | 7.3 µs |
| Email extraction, per utterance | 0.9 µs | 3.9 µs |

A 20 ms frame arrives every 20,000 µs, so the measured per-stream processing cost is well under 1% of real time in both runtimes. These figures exclude noise-suppression engines, network I/O and STT. RNNoise has its own benchmark, `benchmarks/typescript/rnnoise-bench.mjs`, with figures in [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#measured-cost).

**RNNoise tests:** the TypeScript suite runs the real WebAssembly engine (a dev dependency). The Python suite tests the processor's logic against a stand-in library; the two tests that need the real `librnnoise` run only when `RNNOISE_LIBRARY` points at it, and are skipped otherwise.

## Not yet done

- **Live provider tests:** run behind explicit opt-in environment variables with dedicated test accounts. These resolve the open questions in PROVIDER_COMPATIBILITY.md and RELEASE_CHECKLIST.md.
- **Noise-suppression quality:** STT word error rate with and without an engine, on real call audio. RNNoise's Python latency and CPU, and the Python processor against the real `librnnoise`, are also unmeasured.
- **Retrieval relevance:** evaluation on a labelled set.
